"""
Kaggle/Colab training script for DDPM on MNIST.

Run this on a T4 GPU — 50 epochs takes ~15-20 minutes.

Usage (in a Kaggle notebook cell):
    !git clone https://github.com/adimunot21/diffusion-from-scratch.git
    %cd diffusion-from-scratch
    !pip install "numpy<2" -q
    !python -m src.train_kaggle
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
from src.train import EMA, quick_sample


def train_mnist_gpu():
    """Train DDPM on MNIST with GPU acceleration."""

    # ── Device ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: No GPU found. This will be slow.")

    # ── Config ──
    cfg = {
        "num_timesteps": 1000,
        "schedule_type": "linear",
        "beta_start": 1e-4,
        "beta_end": 0.02,
        "channel_list": [64, 128, 256],
        "time_dim": 128,
        "num_res_blocks": 2,
        "epochs": 50,
        "batch_size": 128,
        "lr": 2e-4,
        "grad_clip": 1.0,
        "ema_decay": 0.9999,
    }

    # ── Dataset ──
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True,
                                   transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"],
                              shuffle=True, drop_last=True, num_workers=2,
                              pin_memory=True)

    print(f"Training set: {len(train_dataset)} images")
    print(f"Batches per epoch: {len(train_loader)}")

    # ── Model ──
    model = UNet(
        in_channels=1, out_channels=1,
        channel_list=cfg["channel_list"],
        time_dim=cfg["time_dim"],
        num_res_blocks=cfg["num_res_blocks"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # ── Diffusion schedule ──
    schedule = DiffusionSchedule(
        num_timesteps=cfg["num_timesteps"],
        schedule_type=cfg["schedule_type"],
        beta_start=cfg["beta_start"],
        beta_end=cfg["beta_end"],
    )

    # Move schedule tensors to GPU once (avoid repeated transfers)
    sqrt_alpha_gpu = schedule.sqrt_alphas_cumprod.to(device)
    sqrt_one_minus_alpha_gpu = schedule.sqrt_one_minus_alphas_cumprod.to(device)

    # ── Optimizer & EMA ──
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    ema = EMA(model, decay=cfg["ema_decay"])

    # ── Training loop ──
    history = {"loss": [], "epoch_loss": []}
    global_step = 0
    t_start = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        epoch_losses = []

        for images, _ in train_loader:
            images = images.to(device, non_blocking=True)
            bs = images.shape[0]

            # Random timesteps
            t = torch.randint(0, cfg["num_timesteps"], (bs,), device=device)

            # Forward process on GPU
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
            global_step += 1

        avg_loss = np.mean(epoch_losses)
        history["epoch_loss"].append(avg_loss)
        elapsed = time.time() - t_start
        print(f"Epoch {epoch:3d}/{cfg['epochs']} | Loss: {avg_loss:.4f} | Time: {elapsed:.0f}s")

        # Sample every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            ema.apply_shadow(model)
            images = quick_sample(model, schedule, num_images=16, device=device)
            ema.restore(model)

            fig, axes = plt.subplots(4, 4, figsize=(6, 6))
            for i in range(16):
                ax = axes[i // 4, i % 4]
                ax.imshow(images[i, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
                ax.axis("off")
            plt.suptitle(f"Epoch {epoch}", fontsize=14, fontweight="bold")
            plt.tight_layout()
            os.makedirs("notebooks", exist_ok=True)
            plt.savefig(f"notebooks/samples_epoch{epoch:03d}.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved samples to notebooks/samples_epoch{epoch:03d}.png")

    # ── Save final checkpoint ──
    total_time = time.time() - t_start
    print(f"\nTraining complete: {total_time:.0f}s ({total_time/60:.1f} min)")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "epoch": cfg["epochs"],
        "global_step": global_step,
        "model_state": model.state_dict(),
        "ema_shadow": ema.shadow,
        "config": cfg,
        "history": history,
    }, "checkpoints/ddpm_mnist_final.pt")
    print("Saved checkpoints/ddpm_mnist_final.pt")

    # ── Loss plot ──
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(history["epoch_loss"]) + 1), history["epoch_loss"],
            "b-o", linewidth=2, markersize=4)
    ax.set_title("DDPM MNIST Training Loss", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Avg MSE Loss")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("notebooks/training_loss.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved notebooks/training_loss.png")

    # ── Final samples (64 images, larger grid) ──
    print("\nGenerating final samples (64 images)...")
    ema.apply_shadow(model)
    images = quick_sample(model, schedule, num_images=64, device=device)
    ema.restore(model)

    fig, axes = plt.subplots(8, 8, figsize=(10, 10))
    for i in range(64):
        ax = axes[i // 8, i % 8]
        ax.imshow(images[i, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    plt.suptitle("DDPM MNIST — Final Samples (64)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("notebooks/final_samples_mnist.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved notebooks/final_samples_mnist.png")


if __name__ == "__main__":
    train_mnist_gpu()