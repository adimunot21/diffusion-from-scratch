"""
Class-Conditional DDPM with Classifier-Free Guidance.

This file contains:
1. ConditionalUNet — U-Net that accepts class labels as input
2. Training loop with random label dropout (10%)
3. Classifier-free guidance sampling

HOW CLASS CONDITIONING WORKS:
    The class label (0-9) gets embedded into a vector (like word embeddings
    in your Transformer project), then ADDED to the time embedding.
    Every ResBlock receives time_emb + class_emb, so the network knows
    both "how noisy is this?" and "what class should this be?"

    class_label → nn.Embedding → (batch, time_dim)
                                        ↓
    timestep → SinusoidalEmbed → (batch, time_dim) → ADD → injected into ResBlocks

HOW CLASSIFIER-FREE GUIDANCE WORKS:
    Training:
        - 90% of the time: feed the real class label
        - 10% of the time: feed a "null" label (num_classes, i.e., index 10)
        - The model learns: "if I get a real label, generate that class.
          If I get null, generate anything."

    Sampling:
        - Run the model TWICE per denoising step:
          1. ε_cond   = model(x_t, t, class_label)     "noise for this class"
          2. ε_uncond = model(x_t, t, null_label)       "noise for any class"
        - Combine: ε = ε_uncond + w × (ε_cond - ε_uncond)
        - w > 1 amplifies the class signal → stronger class adherence

    WHY THIS WORKS:
        (ε_cond - ε_uncond) isolates WHAT THE CLASS LABEL CHANGES about
        the prediction. Multiplying by w > 1 exaggerates that change.
        It's like asking "what's different about a car vs a random image?"
        and then pushing harder in the "car direction."

    WHY NOT JUST ALWAYS CONDITION?
        If the model never sees null labels during training, it can't do
        ε_uncond at inference. The 10% dropout teaches it both modes.
        Without unconditional mode, guidance is impossible.

Usage (Kaggle):
    !git clone https://github.com/adimunot21/diffusion-from-scratch.git
    %cd diffusion-from-scratch
    !pip install "numpy<2" -q
    !python -m src.conditional
"""

import os
import time
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
from src.unet import SinusoidalTimeEmbedding, ResBlock, SelfAttention, Downsample, Upsample
from src.train import EMA


# -----------------------------------------------------------------------
# CIFAR-10 class names
# -----------------------------------------------------------------------
CIFAR_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck']


# -----------------------------------------------------------------------
# Conditional U-Net
# -----------------------------------------------------------------------

class ConditionalUNet(nn.Module):
    """
    U-Net with class conditioning for classifier-free guidance.

    Identical to UNet from unet.py, except:
    1. Adds a class embedding layer: nn.Embedding(num_classes + 1, time_dim)
       The +1 is for the "null" class used during unconditional training.
    2. Adds class_emb to time_emb before injecting into ResBlocks.

    Everything else — encoder, decoder, skip connections, attention — is
    exactly the same.
    """

    def __init__(self, in_channels=3, out_channels=3,
                 channel_list=(128, 256, 256, 512), time_dim=256,
                 num_res_blocks=2, attention_levels=None, num_heads=4,
                 num_classes=10):
        super().__init__()
        self.channel_list = channel_list
        self.num_res_blocks = num_res_blocks
        self.num_classes = num_classes

        if attention_levels is None:
            attention_levels = [False] * len(channel_list)
        self.attention_levels = attention_levels

        # ── Time embedding ──
        self.time_embed = SinusoidalTimeEmbedding(time_dim)

        # ── Class embedding ──
        # num_classes + 1: indices 0-9 are real classes, index 10 is "null" (unconditional)
        # This is a learned lookup table: class_id → vector of size time_dim
        # Same concept as word embeddings in your Transformer project
        self.class_embed = nn.Embedding(num_classes + 1, time_dim)

        # ── Input convolution ──
        self.conv_in = nn.Conv2d(in_channels, channel_list[0], kernel_size=3, padding=1)

        # ── Encoder ──
        self.encoder_blocks = nn.ModuleList()
        self.encoder_attns = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        for i, ch_out in enumerate(channel_list):
            ch_in = channel_list[i - 1] if i > 0 else channel_list[0]
            level_blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                block_in = ch_in if j == 0 else ch_out
                level_blocks.append(ResBlock(block_in, ch_out, time_dim))
            self.encoder_blocks.append(level_blocks)

            if attention_levels[i]:
                self.encoder_attns.append(SelfAttention(ch_out, num_heads))
            else:
                self.encoder_attns.append(nn.Identity())

            if i < len(channel_list) - 1:
                self.downsamplers.append(Downsample(ch_out))
            else:
                self.downsamplers.append(nn.Identity())

        # ── Bottleneck ──
        bottleneck_ch = channel_list[-1]
        self.bottleneck = ResBlock(bottleneck_ch, bottleneck_ch, time_dim)

        # ── Decoder ──
        self.decoder_blocks = nn.ModuleList()
        self.decoder_attns = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        reversed_channels = list(reversed(channel_list))
        reversed_attention = list(reversed(attention_levels))

        for i, ch_out in enumerate(reversed_channels):
            skip_ch = ch_out
            if i == 0:
                ch_in_first = bottleneck_ch + skip_ch
            else:
                ch_in_first = reversed_channels[i - 1] + skip_ch

            level_blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                block_in = ch_in_first if j == 0 else ch_out
                level_blocks.append(ResBlock(block_in, ch_out, time_dim))
            self.decoder_blocks.append(level_blocks)

            if reversed_attention[i]:
                self.decoder_attns.append(SelfAttention(ch_out, num_heads))
            else:
                self.decoder_attns.append(nn.Identity())

            if i < len(channel_list) - 1:
                self.upsamplers.append(Upsample(ch_out))
            else:
                self.upsamplers.append(nn.Identity())

        # ── Output ──
        self.out_norm = nn.GroupNorm(num_groups=min(8, channel_list[0]),
                                    num_channels=channel_list[0])
        self.out_conv = nn.Conv2d(channel_list[0], out_channels, kernel_size=1)

    def forward(self, x, t, class_labels):
        """
        Args:
            x: noised images, shape (batch, C, H, W)
            t: timesteps, shape (batch,)
            class_labels: class indices, shape (batch,)
                          values 0-9 for real classes, 10 for null/unconditional

        Returns:
            noise_pred: shape (batch, C, H, W)
        """
        # ── Conditioning: time + class ──
        t_emb = self.time_embed(t)                # (batch, time_dim)
        c_emb = self.class_embed(class_labels)    # (batch, time_dim)
        cond = t_emb + c_emb                      # combined conditioning signal

        # ── Input ──
        h = self.conv_in(x)

        # ── Encoder ──
        skip_connections = []
        for i, level_blocks in enumerate(self.encoder_blocks):
            for block in level_blocks:
                h = block(h, cond)  # pass combined time+class conditioning
            h = self.encoder_attns[i](h)
            skip_connections.append(h)
            if i < len(self.channel_list) - 1:
                h = self.downsamplers[i](h)

        # ── Bottleneck ──
        h = self.bottleneck(h, cond)

        # ── Decoder ──
        for i, level_blocks in enumerate(self.decoder_blocks):
            skip = skip_connections[-(i + 1)]
            h = torch.cat([h, skip], dim=1)
            for block in level_blocks:
                h = block(h, cond)
            h = self.decoder_attns[i](h)
            if i < len(self.channel_list) - 1:
                h = self.upsamplers[i](h)

        # ── Output ──
        h = self.out_norm(h)
        h = F.silu(h)
        h = self.out_conv(h)
        return h


# -----------------------------------------------------------------------
# Classifier-Free Guidance Sampling
# -----------------------------------------------------------------------

@torch.no_grad()
def guided_sample(model, schedule, class_labels, guidance_scale=5.0,
                  ddim_steps=50, device="cpu"):
    """
    Generate images for specific classes using classifier-free guidance.

    At each denoising step:
        1. Run model with class label    → ε_cond
        2. Run model with null label (10) → ε_uncond
        3. Combine: ε = ε_uncond + w × (ε_cond - ε_uncond)

    This requires TWO forward passes per step (2× slower than unconditional),
    but produces much stronger class adherence.

    Args:
        model: trained ConditionalUNet
        schedule: DiffusionSchedule
        class_labels: tensor of class indices, shape (num_images,), values 0-9
        guidance_scale: w — how strongly to follow the class label
                        1.0 = no guidance, 3.0 = moderate, 7.0+ = strong
        ddim_steps: number of DDIM sampling steps
        device: "cpu" or "cuda"

    Returns:
        images: (num_images, 3, 32, 32) in [0, 1]
    """
    model.eval()
    num_images = class_labels.shape[0]

    step_size = schedule.num_timesteps // ddim_steps
    timestep_seq = list(range(0, schedule.num_timesteps, step_size))
    alphas_cumprod = schedule.alphas_cumprod.to(device)

    # The "null" class label for unconditional prediction
    null_labels = torch.full((num_images,), model.num_classes,
                             device=device, dtype=torch.long)

    # Start from pure noise
    x = torch.randn(num_images, 3, 32, 32, device=device)

    for i in reversed(range(len(timestep_seq))):
        t_val = timestep_seq[i]
        t = torch.full((num_images,), t_val, device=device, dtype=torch.long)

        alpha_bar_t = alphas_cumprod[t_val]
        alpha_bar_t_prev = (alphas_cumprod[timestep_seq[i - 1]]
                            if i > 0 else torch.tensor(1.0, device=device))

        # ── Two forward passes ──
        # 1. Conditional: "what noise for this specific class?"
        noise_cond = model(x, t, class_labels)
        # 2. Unconditional: "what noise for any class?"
        noise_uncond = model(x, t, null_labels)

        # ── Classifier-free guidance ──
        # Amplify the difference between conditional and unconditional
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

        # ── DDIM update ──
        x0_pred = ((x - torch.sqrt(1 - alpha_bar_t) * noise_pred) /
                    torch.sqrt(alpha_bar_t))
        dir_xt = torch.sqrt(1 - alpha_bar_t_prev) * noise_pred
        x = torch.sqrt(alpha_bar_t_prev) * x0_pred + dir_xt

    x = (x + 1) / 2
    x = x.clamp(0, 1)
    return x


# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------

CONDITIONAL_CONFIG = {
    # Diffusion
    "num_timesteps": 1000,
    "schedule_type": "cosine",

    # U-Net (same architecture as unconditional CIFAR)
    "channel_list": [128, 256, 256, 512],
    "time_dim": 256,
    "num_res_blocks": 2,
    "attention_levels": [False, True, True, False],
    "num_heads": 4,
    "num_classes": 10,

    # Training
    "epochs": 100,
    "batch_size": 128,
    "lr": 2e-4,
    "grad_clip": 1.0,
    "ema_decay": 0.9999,

    # Classifier-free guidance
    "uncond_prob": 0.1,    # 10% chance of dropping the label during training
}


# -----------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------

def train_conditional():
    """Train class-conditional DDPM with classifier-free guidance."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    cfg = CONDITIONAL_CONFIG

    # ── Dataset ──
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True,
                                     transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"],
                              shuffle=True, drop_last=True, num_workers=2,
                              pin_memory=True)

    print(f"Training set: {len(train_dataset)} images")
    print(f"Classes: {CIFAR_CLASSES}")

    # ── Model ──
    model = ConditionalUNet(
        in_channels=3, out_channels=3,
        channel_list=cfg["channel_list"],
        time_dim=cfg["time_dim"],
        num_res_blocks=cfg["num_res_blocks"],
        attention_levels=cfg["attention_levels"],
        num_heads=cfg["num_heads"],
        num_classes=cfg["num_classes"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # ── Diffusion schedule ──
    schedule = DiffusionSchedule(
        num_timesteps=cfg["num_timesteps"],
        schedule_type=cfg["schedule_type"],
    )
    sqrt_alpha_gpu = schedule.sqrt_alphas_cumprod.to(device)
    sqrt_one_minus_alpha_gpu = schedule.sqrt_one_minus_alphas_cumprod.to(device)

    # ── Optimizer & EMA ──
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    ema = EMA(model, decay=cfg["ema_decay"])

    # ── Training loop ──
    history = {"epoch_loss": []}
    t_start = time.time()
    null_label = cfg["num_classes"]  # index 10 = "no class"

    print(f"\nStarting training: {cfg['epochs']} epochs")
    print(f"Unconditional dropout probability: {cfg['uncond_prob']}")
    print(f"{'=' * 60}")

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        epoch_losses = []

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            bs = images.shape[0]

            # ── Random label dropout for classifier-free guidance ──
            # With probability uncond_prob, replace the real label with null_label
            # This teaches the model to generate unconditionally
            drop_mask = torch.rand(bs, device=device) < cfg["uncond_prob"]
            labels = torch.where(drop_mask, torch.full_like(labels, null_label), labels)

            # ── Forward diffusion ──
            t = torch.randint(0, cfg["num_timesteps"], (bs,), device=device)
            noise = torch.randn_like(images)
            x_t = (sqrt_alpha_gpu[t].view(-1, 1, 1, 1) * images +
                   sqrt_one_minus_alpha_gpu[t].view(-1, 1, 1, 1) * noise)

            # ── Predict noise (conditioned on time AND class) ──
            noise_pred = model(x_t, t, labels)
            loss = F.mse_loss(noise_pred, noise)

            # ── Update ──
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
            _sample_class_grid(model, schedule, epoch, device)
            ema.restore(model)

        # ── Checkpoint ──
        if epoch % 50 == 0 or epoch == cfg["epochs"]:
            os.makedirs("checkpoints", exist_ok=True)
            path = f"checkpoints/ddpm_conditional_epoch{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "ema_shadow": ema.shadow,
                "config": cfg,
                "history": history,
            }, path)
            print(f"  Saved checkpoint: {path}")

    # ── Final results ──
    total_time = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"TRAINING COMPLETE — {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"{'=' * 60}")

    # Final samples with different guidance scales
    ema.apply_shadow(model)

    print("\nGenerating class grid (10 classes × 8 samples)...")
    _sample_class_grid(model, schedule, "final", device)

    print("Generating guidance scale comparison...")
    _guidance_comparison(model, schedule, device)

    ema.restore(model)

    # Loss plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(history["epoch_loss"]) + 1), history["epoch_loss"],
            "b-o", linewidth=2, markersize=3)
    ax.set_title("Conditional CIFAR-10 Training Loss", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Avg MSE Loss")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("notebooks/conditional_training_loss.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Save final checkpoint
    torch.save({
        "epoch": cfg["epochs"],
        "model_state": model.state_dict(),
        "ema_shadow": ema.shadow,
        "config": cfg,
        "history": history,
    }, "checkpoints/ddpm_conditional_final.pt")
    print("Saved checkpoints/ddpm_conditional_final.pt")


# -----------------------------------------------------------------------
# Visualization helpers
# -----------------------------------------------------------------------

def _sample_class_grid(model, schedule, epoch_label, device):
    """Generate a 10×8 grid: one row per class, 8 samples each."""
    os.makedirs("notebooks", exist_ok=True)

    all_images = []
    for class_idx in range(10):
        labels = torch.full((8,), class_idx, device=device, dtype=torch.long)
        images = guided_sample(model, schedule, labels,
                               guidance_scale=5.0, ddim_steps=50, device=device)
        all_images.append(images)

    # Stack into (80, 3, 32, 32)
    all_images = torch.cat(all_images, dim=0)

    fig, axes = plt.subplots(10, 8, figsize=(12, 15))
    for i in range(10):
        for j in range(8):
            idx = i * 8 + j
            ax = axes[i, j]
            img = all_images[idx].cpu().permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.axis("off")
            if j == 0:
                ax.set_ylabel(CIFAR_CLASSES[i], fontsize=10, rotation=0,
                              labelpad=60, va="center")

    plt.suptitle(f"Conditional CIFAR-10 — Epoch {epoch_label} (w=5.0)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = f"notebooks/conditional_grid_epoch{epoch_label}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def _guidance_comparison(model, schedule, device):
    """
    Show how guidance scale affects generation quality.
    Same noise, same class, different guidance scales.
    """
    scales = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
    # Pick 3 classes to show
    test_classes = [0, 3, 8]  # airplane, cat, ship

    fig, axes = plt.subplots(len(test_classes), len(scales),
                             figsize=(2.5 * len(scales), 2.5 * len(test_classes)))

    for row, class_idx in enumerate(test_classes):
        for col, w in enumerate(scales):
            torch.manual_seed(42 + class_idx)  # same noise for each scale
            labels = torch.full((1,), class_idx, device=device, dtype=torch.long)
            img = guided_sample(model, schedule, labels,
                                guidance_scale=w, ddim_steps=50, device=device)
            ax = axes[row, col]
            ax.imshow(img[0].cpu().permute(1, 2, 0).numpy())
            ax.axis("off")
            if row == 0:
                ax.set_title(f"w={w}", fontsize=10)
            if col == 0:
                ax.set_ylabel(CIFAR_CLASSES[class_idx], fontsize=10,
                              rotation=0, labelpad=50, va="center")

    plt.suptitle("Classifier-Free Guidance Scale Comparison",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("notebooks/guidance_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved notebooks/guidance_comparison.png")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    train_conditional() 