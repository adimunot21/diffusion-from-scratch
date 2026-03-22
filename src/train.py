"""
DDPM Training Loop — Teaching the U-Net to Predict Noise.

THE TRAINING ALGORITHM (one step):
    1. Sample a batch of clean images x₀ from the dataset
    2. Sample random timesteps t ~ Uniform(0, T-1) for each image
    3. Sample random noise ε ~ N(0, I)
    4. Create noised images: x_t = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε
    5. Predict the noise: ε_pred = network(x_t, t)
    6. Loss = MSE(ε_pred, ε)  — "how well did we predict the noise?"
    7. Backpropagate and update weights

That's it. No labels, no rewards, no adversarial training. Just:
"I added this noise. Can you figure out what I added?"

WHY PREDICT NOISE (not the clean image)?
Mathematically, predicting noise ε, predicting the clean image x₀, and
predicting the "score" ∇log p(x_t) are all equivalent — you can convert
between them. But predicting noise is the most numerically stable in practice.
The noise ε is always drawn from N(0, I), so the target has a consistent
scale regardless of the image content. Predicting x₀ directly would have
targets that vary wildly in scale across different images and timesteps.

EMA (EXPONENTIAL MOVING AVERAGE):
We maintain a shadow copy of the model weights that's a running average:
    θ_ema = decay · θ_ema + (1 - decay) · θ_current

With decay = 0.9999, the EMA weights change very slowly — they smooth out
the noise in SGD updates. We use the EMA weights for sampling (generation)
because they produce more stable, higher-quality outputs. Training still
updates the main weights normally.

This is the same concept as Polyak averaging, used in many places:
- Target networks in DQN (your RL project used this)
- Teacher networks in self-supervised learning
The difference: DQN copies weights periodically, EMA blends them continuously.
"""

import os
import time
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.diffusion import DiffusionSchedule
from src.unet import UNet


# -----------------------------------------------------------------------
# EMA (Exponential Moving Average)
# -----------------------------------------------------------------------

class EMA:
    """
    Maintains an exponential moving average of model parameters.

    After each training step:
        θ_ema = decay · θ_ema + (1 - decay) · θ_model

    With decay = 0.9999:
        - Each update moves θ_ema 0.01% toward the current weights
        - Effectively averages over the last ~10,000 updates
        - Smooths out SGD noise → better generation quality

    Usage:
        ema = EMA(model, decay=0.9999)
        # ... training step ...
        ema.update()           # blend current weights into EMA
        ema.apply_shadow()     # swap in EMA weights (for sampling)
        # ... sample images ...
        ema.restore()          # swap back to training weights
    """

    def __init__(self, model, decay=0.9999):
        self.decay = decay
        # Store a deep copy of the initial weights
        self.shadow = {name: param.clone().detach()
                       for name, param in model.named_parameters()}
        # Backup storage for swapping
        self.backup = {}

    def update(self, model):
        """Blend current model weights into the EMA shadow weights."""
        for name, param in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model):
        """Replace model weights with EMA weights (for sampling/evaluation)."""
        self.backup = {name: param.data.clone()
                       for name, param in model.named_parameters()}
        for name, param in model.named_parameters():
            param.data.copy_(self.shadow[name])

    def restore(self, model):
        """Restore original training weights after sampling."""
        for name, param in model.named_parameters():
            param.data.copy_(self.backup[name])
        self.backup = {}


# -----------------------------------------------------------------------
# Training Config
# -----------------------------------------------------------------------

MNIST_CONFIG = {
    # Data
    "dataset": "mnist",
    "image_size": 28,
    "in_channels": 1,

    # Diffusion
    "num_timesteps": 1000,
    "schedule_type": "linear",
    "beta_start": 1e-4,
    "beta_end": 0.02,

    # U-Net
    "channel_list": [64, 128, 256],
    "time_dim": 128,
    "num_res_blocks": 2,

    # Training
    "epochs": 50,
    "batch_size": 128,
    "lr": 2e-4,
    "grad_clip": 1.0,
    "ema_decay": 0.9999,

    # Logging
    "log_every": 100,           # print loss every N steps
    "sample_every": 5,          # sample images every N epochs
    "checkpoint_every": 10,     # save model every N epochs
}


# -----------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------

def train(config=None, device="cpu"):
    """
    Train the DDPM on MNIST.

    Args:
        config: dict of hyperparameters (uses MNIST_CONFIG if None)
        device: "cpu" or "cuda"

    Returns:
        model: trained U-Net
        ema: EMA wrapper with smoothed weights
        schedule: the diffusion schedule (needed for sampling)
        history: dict with loss curves
    """
    cfg = MNIST_CONFIG.copy()
    if config:
        cfg.update(config)

    print("=" * 60)
    print("DDPM Training — MNIST")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Epochs: {cfg['epochs']}")
    print(f"Batch size: {cfg['batch_size']}")
    print(f"Learning rate: {cfg['lr']}")
    print(f"Timesteps: {cfg['num_timesteps']}")
    print(f"Schedule: {cfg['schedule_type']}")
    print()

    # ── Dataset ──
    transform = transforms.Compose([
        transforms.ToTensor(),  # [0, 255] → [0.0, 1.0]
        # Scale to [-1, 1] — standard for diffusion models.
        # The forward process adds Gaussian noise (mean 0), so centering
        # the data around 0 makes the signal-to-noise ratio symmetric.
        # At t=T, x_T ~ N(0, I) regardless of the original image.
        # If images were in [0, 1], x_T would be shifted (mean ~0.5 + noise).
        transforms.Normalize((0.5,), (0.5,)),  # (x - 0.5) / 0.5 → [-1, 1]
    ])

    train_dataset = datasets.MNIST(root="./data", train=True, download=True,
                                   transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"],
                              shuffle=True, drop_last=True, num_workers=0)

    print(f"Training set: {len(train_dataset)} images")
    print(f"Batches per epoch: {len(train_loader)}")

    # ── Model ──
    model = UNet(
        in_channels=cfg["in_channels"],
        out_channels=cfg["in_channels"],  # predict noise with same shape as input
        channel_list=cfg["channel_list"],
        time_dim=cfg["time_dim"],
        num_res_blocks=cfg["num_res_blocks"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # ── Diffusion schedule ──
    schedule = DiffusionSchedule(
        num_timesteps=cfg["num_timesteps"],
        schedule_type=cfg["schedule_type"],
        beta_start=cfg["beta_start"],
        beta_end=cfg["beta_end"],
    )

    # ── Optimizer ──
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    # ── EMA ──
    ema = EMA(model, decay=cfg["ema_decay"])

    # ── Training ──
    history = {"loss": [], "epoch_loss": []}
    global_step = 0
    t_start = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        epoch_losses = []

        for batch_idx, (images, _labels) in enumerate(train_loader):
            # _labels are digit classes — we ignore them for unconditional generation.
            # The network learns to denoise without knowing WHAT digit it is.
            images = images.to(device)
            batch_size_actual = images.shape[0]

            # ── Step 1: Sample random timesteps ──
            # Each image in the batch gets a different random timestep
            t = torch.randint(0, cfg["num_timesteps"], (batch_size_actual,),
                              device=device)

            # ── Step 2: Sample noise and create noised images ──
            noise = torch.randn_like(images)

            # Move schedule tensors to the right device for gathering
            sqrt_alpha = schedule.sqrt_alphas_cumprod.to(device)
            sqrt_one_minus_alpha = schedule.sqrt_one_minus_alphas_cumprod.to(device)

            # x_t = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε
            x_t = (sqrt_alpha[t].view(-1, 1, 1, 1) * images +
                   sqrt_one_minus_alpha[t].view(-1, 1, 1, 1) * noise)

            # ── Step 3: Predict the noise ──
            noise_pred = model(x_t, t)

            # ── Step 4: Compute loss ──
            # Simple MSE between predicted noise and actual noise
            loss = F.mse_loss(noise_pred, noise)

            # ── Step 5: Backprop and update ──
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping — prevents rare large gradients from
            # destabilizing training. Clips the global norm of all gradients.
            if cfg["grad_clip"] > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])

            optimizer.step()

            # ── Step 6: Update EMA ──
            ema.update(model)

            # ── Logging ──
            loss_val = loss.item()
            epoch_losses.append(loss_val)
            history["loss"].append(loss_val)
            global_step += 1

            if global_step % cfg["log_every"] == 0:
                elapsed = time.time() - t_start
                print(f"  Step {global_step:5d} | Epoch {epoch}/{cfg['epochs']} | "
                      f"Loss: {loss_val:.4f} | "
                      f"Time: {elapsed:.0f}s")

        # ── End of epoch ──
        avg_loss = np.mean(epoch_losses)
        history["epoch_loss"].append(avg_loss)
        elapsed = time.time() - t_start
        print(f"Epoch {epoch}/{cfg['epochs']} complete | "
              f"Avg loss: {avg_loss:.4f} | Time: {elapsed:.0f}s")

        # ── Sample images periodically ──
        if epoch % cfg["sample_every"] == 0 or epoch == 1:
            sample_and_save(model, ema, schedule, epoch, device, cfg)

        # ── Save checkpoint ──
        if epoch % cfg["checkpoint_every"] == 0 or epoch == cfg["epochs"]:
            os.makedirs("checkpoints", exist_ok=True)
            path = f"checkpoints/ddpm_mnist_epoch{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "model_state": model.state_dict(),
                "ema_shadow": ema.shadow,
                "optimizer_state": optimizer.state_dict(),
                "config": cfg,
                "history": history,
            }, path)
            print(f"  Saved checkpoint: {path}")

    # ── Final summary ──
    total_time = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"TRAINING COMPLETE — {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"{'=' * 60}")
    print(f"Final avg loss: {history['epoch_loss'][-1]:.4f}")
    print(f"Total steps: {global_step}")

    # Plot loss curve
    plot_loss(history, save_path="notebooks/training_loss.png")

    return model, ema, schedule, history


# -----------------------------------------------------------------------
# Quick Sampling (for training progress visualization)
# -----------------------------------------------------------------------

@torch.no_grad()
def quick_sample(model, schedule, num_images=16, device="cpu"):
    """
    Generate images using DDPM sampling (full 1000-step reverse process).
    This is a simple version — the full sampler is built in Phase 4.

    The reverse process:
        Start with x_T ~ N(0, I)  (pure noise)
        For t = T-1, T-2, ..., 0:
            Predict noise: ε_pred = model(x_t, t)
            Compute mean: μ = (1/√α_t)(x_t - β_t/√(1-ᾱ_t) · ε_pred)
            Sample: x_{t-1} = μ + σ_t · z   (z ~ N(0,I), except at t=0)
    """
    model.eval()

    # Move schedule tensors to device
    betas = schedule.betas.to(device)
    sqrt_recip_alphas = schedule.sqrt_recip_alphas.to(device)
    sqrt_one_minus_alphas_cumprod = schedule.sqrt_one_minus_alphas_cumprod.to(device)
    posterior_variance = schedule.posterior_variance.to(device)

    # Start from pure noise
    x = torch.randn(num_images, 1, 28, 28, device=device)

    # Reverse process: denoise step by step
    for t_val in reversed(range(schedule.num_timesteps)):
        t = torch.full((num_images,), t_val, device=device, dtype=torch.long)

        # Predict noise
        noise_pred = model(x, t)

        # Compute the mean of p(x_{t-1} | x_t)
        # μ_θ(x_t, t) = (1/√α_t) · (x_t - β_t/√(1-ᾱ_t) · ε_θ(x_t, t))
        mean = sqrt_recip_alphas[t_val] * (
            x - betas[t_val] / sqrt_one_minus_alphas_cumprod[t_val] * noise_pred
        )

        if t_val > 0:
            # Add noise for all steps except the last one
            noise = torch.randn_like(x)
            x = mean + torch.sqrt(posterior_variance[t_val]) * noise
        else:
            # At t=0, don't add noise — this is the final clean image
            x = mean

    # Rescale from [-1, 1] back to [0, 1] for visualization
    x = (x + 1) / 2
    x = x.clamp(0, 1)

    return x


def sample_and_save(model, ema, schedule, epoch, device, cfg):
    """Sample images using EMA weights and save a grid."""
    ema.apply_shadow(model)  # swap in EMA weights

    images = quick_sample(model, schedule, num_images=16, device=device)

    ema.restore(model)  # swap back to training weights

    # Create a 4×4 grid
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for i in range(16):
        ax = axes[i // 4, i % 4]
        ax.imshow(images[i, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")

    plt.suptitle(f"DDPM Samples — Epoch {epoch}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    os.makedirs("notebooks", exist_ok=True)
    path = f"notebooks/samples_epoch{epoch:03d}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved samples to {path}")


def plot_loss(history, save_path="notebooks/training_loss.png"):
    """Plot training loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Per-step loss (smoothed)
    ax = axes[0]
    losses = history["loss"]
    # Smooth with a running average for readability
    window = min(100, len(losses) // 10)
    if window > 1:
        smoothed = np.convolve(losses, np.ones(window)/window, mode="valid")
        ax.plot(smoothed, "b-", linewidth=1, alpha=0.8)
    else:
        ax.plot(losses, "b-", linewidth=1, alpha=0.8)
    ax.set_title("Training Loss (per step, smoothed)", fontweight="bold")
    ax.set_xlabel("Step")
    ax.set_ylabel("MSE Loss")
    ax.grid(True, alpha=0.3)

    # Per-epoch loss
    ax = axes[1]
    epochs = range(1, len(history["epoch_loss"]) + 1)
    ax.plot(epochs, history["epoch_loss"], "r-o", linewidth=2, markersize=4)
    ax.set_title("Training Loss (per epoch)", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Avg MSE Loss")
    ax.grid(True, alpha=0.3)

    plt.suptitle("DDPM Training Progress", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved loss plot to {save_path}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, ema, schedule, history = train(device=device)

    print("\n✓ Training complete!")
    print(f"  Check notebooks/ for sample images and loss curves")
    print(f"  Check checkpoints/ for saved models")