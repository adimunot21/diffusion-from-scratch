"""
DDPM Training on CIFAR-10 — Unconditional RGB Image Generation.

Run on Kaggle T4 GPU. Expected time: ~3-4 hours for 100 epochs.

What changes from MNIST:
    - 3 channels (RGB) instead of 1 (grayscale)
    - 32×32 instead of 28×28
    - Larger U-Net: [128, 256, 256, 512] with self-attention
    - ~28M params instead of ~4.5M
    - Cosine noise schedule (better for higher resolution)
    - More epochs needed (100) — CIFAR is much harder than MNIST
      (10 diverse object classes vs 10 similar digit shapes)

What stays the same:
    - Training objective: predict the noise ε
    - Loss: MSE(ε_pred, ε)
    - EMA for stable sampling
    - Same diffusion math

Usage (Kaggle notebook):
    !git clone https://github.com/adimunot21/diffusion-from-scratch.git
    %cd diffusion-from-scratch
    !pip install "numpy<2" -q
    !python -m src.train_cifar
"""

import os
import time
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
from src.train import EMA


# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------

CIFAR_CONFIG = {
    # Diffusion
    "num_timesteps": 1000,
    "schedule_type": "cosine",       # cosine works better for RGB images

    # U-Net (larger than MNIST)
    "channel_list": [128, 256, 256, 512],
    "time_dim": 256,
    "num_res_blocks": 2,
    "attention_levels": [False, True, True, False],  # attention at 16×16 and 8×8
    "num_heads": 4,

    # Training
    "epochs": 100,
    "batch_size": 128,
    "lr": 2e-4,
    "grad_clip": 1.0,
    "ema_decay": 0.9999,
}


# -----------------------------------------------------------------------
# Sampling (CIFAR version — 3 channels, 32×32)
# -----------------------------------------------------------------------

@torch.no_grad()
def sample_cifar(model, schedule, num_images=16, ddim_steps=50, device="cpu"):
    """
    Generate CIFAR-10 images using DDIM sampling.
    Uses DDIM (not DDPM) because 1000-step DDPM is too slow for
    periodic sampling during training with a 28M param model.
    """
    model.eval()

    step_size = schedule.num_timesteps // ddim_steps
    timestep_seq = list(range(0, schedule.num_timesteps, step_size))
    alphas_cumprod = schedule.alphas_cumprod.to(device)

    x = torch.randn(num_images, 3, 32, 32, device=device)

    for i in reversed(range(len(timestep_seq))):
        t_val = timestep_seq[i]
        t = torch.full((num_images,), t_val, device=device, dtype=torch.long)

        alpha_bar_t = alphas_cumprod[t_val]
        alpha_bar_t_prev = (alphas_cumprod[timestep_seq[i - 1]]
                            if i > 0 else torch.tensor(1.0, device=device))

        noise_pred = model(x, t)
        x0_pred = ((x - torch.sqrt(1 - alpha_bar_t) * noise_pred) /
                    torch.sqrt(alpha_bar_t))

        dir_xt = torch.sqrt(1 - alpha_bar_t_prev) * noise_pred
        x = torch.sqrt(alpha_bar_t_prev) * x0_pred + dir_xt

    # Rescale [-1, 1] → [0, 1]
    x = (x + 1) / 2
    x = x.clamp(0, 1)
    return x


def save_cifar_grid(images, path, nrow=8, title=None):
    """Save a grid of RGB images."""
    n = images.shape[0]
    ncol = min(nrow, n)
    nrow_actual = (n + ncol - 1) // ncol

    fig, axes = plt.subplots(nrow_actual, ncol,
                             figsize=(1.5 * ncol, 1.5 * nrow_actual))
    if nrow_actual == 1:
        axes = axes[np.newaxis, :]

    for i in range(nrow_actual):
        for j in range(ncol):
            idx = i * ncol + j
            ax = axes[i, j]
            if idx < n:
                # (C, H, W) → (H, W, C) for matplotlib
                img = images[idx].cpu().permute(1, 2, 0).numpy()
                ax.imshow(img)
            ax.axis("off")

    if title:
        plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# -----------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------

def train_cifar():
    """Train DDPM on CIFAR-10."""

    # ── Device ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    cfg = CIFAR_CONFIG

    # ── Dataset ──
    # Standard CIFAR augmentation: random horizontal flip
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),   # data augmentation
        transforms.ToTensor(),               # [0, 255] → [0, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # → [-1, 1]
    ])

    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True,
                                     transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"],
                              shuffle=True, drop_last=True, num_workers=2,
                              pin_memory=True)

    # CIFAR-10 class names for reference
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    print(f"Training set: {len(train_dataset)} images, {len(classes)} classes")
    print(f"Classes: {classes}")
    print(f"Batches per epoch: {len(train_loader)}")

    # ── Model ──
    model = UNet(
        in_channels=3, out_channels=3,
        channel_list=cfg["channel_list"],
        time_dim=cfg["time_dim"],
        num_res_blocks=cfg["num_res_blocks"],
        attention_levels=cfg["attention_levels"],
        num_heads=cfg["num_heads"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # ── Diffusion schedule ──
    schedule = DiffusionSchedule(
        num_timesteps=cfg["num_timesteps"],
        schedule_type=cfg["schedule_type"],
    )

    # Move schedule tensors to GPU once
    sqrt_alpha_gpu = schedule.sqrt_alphas_cumprod.to(device)
    sqrt_one_minus_alpha_gpu = schedule.sqrt_one_minus_alphas_cumprod.to(device)

    # ── Optimizer & EMA ──
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    ema = EMA(model, decay=cfg["ema_decay"])

    # ── Training loop ──
    history = {"epoch_loss": []}
    t_start = time.time()

    print(f"\nStarting training: {cfg['epochs']} epochs")
    print(f"{'=' * 60}")

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        epoch_losses = []

        for images, _ in train_loader:
            images = images.to(device, non_blocking=True)
            bs = images.shape[0]

            # Random timesteps
            t = torch.randint(0, cfg["num_timesteps"], (bs,), device=device)

            # Forward process
            noise = torch.randn_like(images)
            x_t = (sqrt_alpha_gpu[t].view(-1, 1, 1, 1) * images +
                   sqrt_one_minus_alpha_gpu[t].view(-1, 1, 1, 1) * noise)

            # Predict noise
            noise_pred = model(x_t, t)
            loss = F.mse_loss(noise_pred, noise)

            # Update
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            ema.update(model)

            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        history["epoch_loss"].append(avg_loss)
        elapsed = time.time() - t_start
        print(f"Epoch {epoch:3d}/{cfg['epochs']} | Loss: {avg_loss:.4f} | "
              f"Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

        # ── Sample every 20 epochs ──
        if epoch % 20 == 0 or epoch == 1:
            ema.apply_shadow(model)
            images_gen = sample_cifar(model, schedule, num_images=16,
                                      ddim_steps=50, device=device)
            ema.restore(model)

            os.makedirs("notebooks", exist_ok=True)
            save_cifar_grid(images_gen,
                            f"notebooks/cifar_samples_epoch{epoch:03d}.png",
                            title=f"CIFAR-10 Samples — Epoch {epoch}")
            print(f"  Saved samples to notebooks/cifar_samples_epoch{epoch:03d}.png")

        # ── Checkpoint every 50 epochs ──
        if epoch % 50 == 0 or epoch == cfg["epochs"]:
            os.makedirs("checkpoints", exist_ok=True)
            path = f"checkpoints/ddpm_cifar_epoch{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "ema_shadow": ema.shadow,
                "optimizer_state": optimizer.state_dict(),
                "config": cfg,
                "history": history,
            }, path)
            print(f"  Saved checkpoint: {path}")

    # ── Final results ──
    total_time = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"TRAINING COMPLETE — {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"{'=' * 60}")

    # Final samples (64 images)
    print("Generating final samples (64 images)...")
    ema.apply_shadow(model)
    final_images = sample_cifar(model, schedule, num_images=64,
                                ddim_steps=50, device=device)
    ema.restore(model)

    save_cifar_grid(final_images, "notebooks/cifar_final_samples.png",
                    nrow=8, title="CIFAR-10 — Final Samples (64)")

    # Loss plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(history["epoch_loss"]) + 1), history["epoch_loss"],
            "b-o", linewidth=2, markersize=3)
    ax.set_title("CIFAR-10 Training Loss", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Avg MSE Loss")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("notebooks/cifar_training_loss.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved notebooks/cifar_training_loss.png")

    # Save final checkpoint
    torch.save({
        "epoch": cfg["epochs"],
        "model_state": model.state_dict(),
        "ema_shadow": ema.shadow,
        "config": cfg,
        "history": history,
    }, "checkpoints/ddpm_cifar_final.pt")
    print("Saved checkpoints/ddpm_cifar_final.pt")


if __name__ == "__main__":
    train_cifar()