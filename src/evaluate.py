"""
Evaluation & Analysis — Measuring Generative Quality.

FID (Fréchet Inception Distance):
    The standard metric for generative image quality. Measures how similar
    the DISTRIBUTION of generated images is to the DISTRIBUTION of real images.

    How it works:
    1. Pass real images through Inception-v3 → extract features (2048-dim vectors)
    2. Pass generated images through Inception-v3 → extract features
    3. Fit a Gaussian (mean + covariance) to each set of features
    4. Compute the Fréchet distance between the two Gaussians:
       FID = ||μ_real - μ_gen||² + Tr(Σ_real + Σ_gen - 2·(Σ_real·Σ_gen)^½)

    Lower FID = better quality.
    FID = 0:  generated images are statistically identical to real images
    FID < 20: excellent (SOTA diffusion models)
    FID < 50: good (recognizable, diverse)
    FID < 100: decent (our models will likely land here)
    FID > 100: poor (blurry, wrong structure)

    WHY FID, NOT JUST "LOOKING AT IMAGES":
    Human judgment is subjective and doesn't scale. FID captures two things:
    1. Quality (are individual images good?) — bad images have unusual features
    2. Diversity (do we cover the full distribution?) — mode collapse shows as
       low variance in the feature space

    LIMITATION: FID uses Inception-v3 trained on ImageNet. For CIFAR-10 (32×32),
    images must be upscaled to 299×299, which introduces artifacts. Our FID
    numbers won't be directly comparable to papers that compute FID differently.
    The relative comparison between our models is what matters.

Usage:
    python -m src.evaluate                # run all evaluations
    python -m src.evaluate --mnist-only   # just MNIST evaluation
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms, models

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.diffusion import DiffusionSchedule
from src.unet import UNet
from src.sample import ddpm_sample, ddim_sample, load_model
from src.train import EMA


# -----------------------------------------------------------------------
# FID Computation
# -----------------------------------------------------------------------

class InceptionFeatureExtractor:
    """
    Extract features from Inception-v3 for FID computation.

    We use the pool3 layer (2048-dimensional) as the feature representation.
    This is the standard choice for FID — deep enough to capture semantics,
    but not so deep that it only captures class labels.
    """

    def __init__(self, device="cpu"):
        self.device = device
        # Load pretrained Inception-v3
        self.model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        self.model.eval()
        self.model.to(device)

        # Hook to capture features before the final FC layer
        self.features = None
        # The avgpool layer outputs (batch, 2048, 1, 1)
        self.model.avgpool.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        """Capture the output of avgpool."""
        self.features = output

    @torch.no_grad()
    def extract(self, images, batch_size=50):
        """
        Extract 2048-dim features from a batch of images.

        Args:
            images: tensor (N, C, H, W) in [0, 1]
            batch_size: process in chunks to save memory

        Returns:
            features: numpy array (N, 2048)
        """
        # Inception expects 299×299 RGB images
        # If grayscale (MNIST), convert to 3-channel
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)

        all_features = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size].to(self.device)

            # Resize to 299×299 (Inception's expected input size)
            batch = F.interpolate(batch, size=(299, 299),
                                  mode="bilinear", align_corners=False)

            # Inception normalization (mean and std from ImageNet)
            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
            batch = (batch - mean) / std

            # Forward pass (we only care about the hooked features)
            _ = self.model(batch)

            # features shape: (batch, 2048, 1, 1) → (batch, 2048)
            feat = self.features.squeeze(-1).squeeze(-1).cpu().numpy()
            all_features.append(feat)

        return np.concatenate(all_features, axis=0)


def compute_fid(real_features, gen_features):
    """
    Compute Fréchet Inception Distance between two sets of features.

    FID = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2·(Σ₁·Σ₂)^½)

    Where:
        μ₁, Σ₁ = mean and covariance of real image features
        μ₂, Σ₂ = mean and covariance of generated image features

    The first term measures how different the "average" images are.
    The second term measures how different the "spread" of images is.
    """
    # Compute mean and covariance for real images
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)

    # Compute mean and covariance for generated images
    mu_gen = np.mean(gen_features, axis=0)
    sigma_gen = np.cov(gen_features, rowvar=False)

    # ||μ₁ - μ₂||² — difference in means
    diff = mu_real - mu_gen
    mean_term = np.dot(diff, diff)

    # Tr(Σ₁ + Σ₂ - 2·(Σ₁·Σ₂)^½) — difference in covariances
    # We need the matrix square root of Σ₁·Σ₂
    # Use eigenvalue decomposition for numerical stability
    from scipy import linalg
    covmean, _ = linalg.sqrtm(sigma_real @ sigma_gen, disp=False)

    # sqrtm can return complex numbers due to numerical errors
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    cov_term = np.trace(sigma_real + sigma_gen - 2 * covmean)

    return mean_term + cov_term


# -----------------------------------------------------------------------
# Generate samples for evaluation
# -----------------------------------------------------------------------

@torch.no_grad()
def generate_mnist_samples(model, schedule, n=1000, device="cpu"):
    """Generate n MNIST images using DDIM."""
    all_images = []
    batch_size = 64

    for i in range(0, n, batch_size):
        current_batch = min(batch_size, n - i)
        images = ddim_sample(model, schedule, num_images=current_batch,
                             ddim_steps=50, eta=0.0, device=device)
        all_images.append(images.cpu())
        if (i + current_batch) % 256 == 0 or i + current_batch == n:
            print(f"  Generated {i + current_batch}/{n}")

    return torch.cat(all_images, dim=0)


@torch.no_grad()
def generate_cifar_samples(model, schedule, n=1000, device="cpu", is_conditional=False):
    """Generate n CIFAR images using DDIM."""
    from src.conditional import guided_sample

    all_images = []
    batch_size = 64

    for i in range(0, n, batch_size):
        current_batch = min(batch_size, n - i)

        if is_conditional:
            # Generate balanced across classes
            labels = torch.arange(10).repeat(current_batch // 10 + 1)[:current_batch]
            labels = labels.to(device)
            images = guided_sample(model, schedule, labels,
                                   guidance_scale=5.0, ddim_steps=50, device=device)
        else:
            # Unconditional DDIM
            from src.train_cifar import sample_cifar
            images = sample_cifar(model, schedule, num_images=current_batch,
                                  ddim_steps=50, device=device)

        all_images.append(images.cpu())
        if (i + current_batch) % 256 == 0 or i + current_batch == n:
            print(f"  Generated {i + current_batch}/{n}")

    return torch.cat(all_images, dim=0)


# -----------------------------------------------------------------------
# Load real dataset samples
# -----------------------------------------------------------------------

def load_real_mnist(n=1000):
    """Load n real MNIST images as tensors in [0, 1]."""
    dataset = datasets.MNIST(root="./data", train=True, download=True,
                             transform=transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=n, shuffle=True)
    images, _ = next(iter(loader))
    return images[:n]


def load_real_cifar(n=1000):
    """Load n real CIFAR-10 images as tensors in [0, 1]."""
    dataset = datasets.CIFAR10(root="./data", train=True, download=True,
                               transform=transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=n, shuffle=True)
    images, _ = next(iter(loader))
    return images[:n]


# -----------------------------------------------------------------------
# Side-by-side comparison
# -----------------------------------------------------------------------

def compare_real_vs_generated(real_images, gen_images, title, save_path, n_show=8):
    """Show real and generated images side by side."""
    is_grayscale = real_images.shape[1] == 1

    fig, axes = plt.subplots(2, n_show, figsize=(2 * n_show, 4.5))

    for i in range(n_show):
        # Real
        ax = axes[0, i]
        if is_grayscale:
            ax.imshow(real_images[i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(real_images[i].permute(1, 2, 0).numpy())
        ax.axis("off")
        if i == 0:
            ax.set_ylabel("Real", fontsize=12, fontweight="bold",
                          rotation=0, labelpad=40, va="center")

        # Generated
        ax = axes[1, i]
        if is_grayscale:
            ax.imshow(gen_images[i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(gen_images[i].permute(1, 2, 0).numpy())
        ax.axis("off")
        if i == 0:
            ax.set_ylabel("Generated", fontsize=12, fontweight="bold",
                          rotation=0, labelpad=40, va="center")

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


# -----------------------------------------------------------------------
# DDIM step count vs quality
# -----------------------------------------------------------------------

def step_count_analysis(model, schedule, device="cpu"):
    """Compare DDIM sample quality at different step counts."""
    step_counts = [10, 20, 50, 100, 200]

    fig, axes = plt.subplots(len(step_counts), 8, figsize=(16, 2.2 * len(step_counts)))

    torch.manual_seed(42)
    for row, steps in enumerate(step_counts):
        torch.manual_seed(42)
        images = ddim_sample(model, schedule, num_images=8,
                             ddim_steps=steps, eta=0.0, device=device)
        for col in range(8):
            ax = axes[row, col]
            ax.imshow(images[col, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(f"{steps} steps", fontsize=10, rotation=0,
                              labelpad=50, va="center")

    plt.suptitle("DDIM Sampling: Step Count vs Quality (same starting noise)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("notebooks/step_count_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved notebooks/step_count_analysis.png")


# -----------------------------------------------------------------------
# Main evaluation
# -----------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    os.makedirs("notebooks", exist_ok=True)

    results = {}
    num_fid_samples = 1000  # standard is 10K-50K, we use 1K for speed

    # ══════════════════════════════════════════════════════════════
    # MNIST Evaluation
    # ══════════════════════════════════════════════════════════════

    mnist_ckpt = "checkpoints/ddpm_mnist_final.pt"
    if os.path.exists(mnist_ckpt):
        print("\n" + "=" * 60)
        print("MNIST EVALUATION")
        print("=" * 60)

        model, schedule = load_model(mnist_ckpt, device=device)

        # Generate samples
        print(f"\nGenerating {num_fid_samples} MNIST samples...")
        t0 = time.time()
        gen_images = generate_mnist_samples(model, schedule, n=num_fid_samples,
                                            device=device)
        gen_time = time.time() - t0
        print(f"Generation time: {gen_time:.1f}s")

        # Load real images
        real_images = load_real_mnist(n=num_fid_samples)
        print(f"Real images: {real_images.shape}, Generated: {gen_images.shape}")

        # Visual comparison
        compare_real_vs_generated(real_images, gen_images,
                                  "MNIST: Real vs Generated",
                                  "notebooks/mnist_real_vs_gen.png")

        # FID
        print("\nComputing MNIST FID...")
        extractor = InceptionFeatureExtractor(device=device)
        real_feat = extractor.extract(real_images)
        gen_feat = extractor.extract(gen_images)
        fid = compute_fid(real_feat, gen_feat)
        print(f"MNIST FID: {fid:.2f}")
        results["mnist_fid"] = fid

        # Step count analysis
        print("\nStep count analysis...")
        step_count_analysis(model, schedule, device=device)

    # ══════════════════════════════════════════════════════════════
    # CIFAR-10 Unconditional Evaluation
    # ══════════════════════════════════════════════════════════════

    cifar_ckpt = "checkpoints/ddpm_cifar_final.pt"
    if os.path.exists(cifar_ckpt):
        print("\n" + "=" * 60)
        print("CIFAR-10 UNCONDITIONAL EVALUATION")
        print("=" * 60)

        ckpt = torch.load(cifar_ckpt, map_location=device, weights_only=False)
        cfg = ckpt["config"]

        model = UNet(
            in_channels=3, out_channels=3,
            channel_list=cfg["channel_list"],
            time_dim=cfg["time_dim"],
            num_res_blocks=cfg["num_res_blocks"],
            attention_levels=cfg.get("attention_levels"),
            num_heads=cfg.get("num_heads", 4),
        ).to(device)

        # Load EMA weights
        for name, param in model.named_parameters():
            if name in ckpt["ema_shadow"]:
                param.data.copy_(ckpt["ema_shadow"][name].to(device))
        model.eval()

        schedule = DiffusionSchedule(
            num_timesteps=cfg["num_timesteps"],
            schedule_type=cfg["schedule_type"],
        )

        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model: {n_params:,} params")

        # Generate samples
        print(f"\nGenerating {num_fid_samples} CIFAR samples...")
        t0 = time.time()
        gen_images = generate_cifar_samples(model, schedule, n=num_fid_samples,
                                            device=device, is_conditional=False)
        gen_time = time.time() - t0
        print(f"Generation time: {gen_time:.1f}s")

        # Load real images
        real_images = load_real_cifar(n=num_fid_samples)

        # Visual comparison
        compare_real_vs_generated(real_images, gen_images,
                                  "CIFAR-10 Unconditional: Real vs Generated",
                                  "notebooks/cifar_uncond_real_vs_gen.png")

        # FID
        print("\nComputing CIFAR unconditional FID...")
        extractor = InceptionFeatureExtractor(device=device)
        real_feat = extractor.extract(real_images)
        gen_feat = extractor.extract(gen_images)
        fid = compute_fid(real_feat, gen_feat)
        print(f"CIFAR-10 Unconditional FID: {fid:.2f}")
        results["cifar_uncond_fid"] = fid

    # ══════════════════════════════════════════════════════════════
    # CIFAR-10 Conditional Evaluation
    # ══════════════════════════════════════════════════════════════

    cond_ckpt = "checkpoints/ddpm_conditional_final.pt"
    if os.path.exists(cond_ckpt):
        print("\n" + "=" * 60)
        print("CIFAR-10 CONDITIONAL EVALUATION")
        print("=" * 60)

        from src.conditional import ConditionalUNet, guided_sample, CIFAR_CLASSES

        ckpt = torch.load(cond_ckpt, map_location=device, weights_only=False)
        cfg = ckpt["config"]

        model = ConditionalUNet(
            in_channels=3, out_channels=3,
            channel_list=cfg["channel_list"],
            time_dim=cfg["time_dim"],
            num_res_blocks=cfg["num_res_blocks"],
            attention_levels=cfg.get("attention_levels"),
            num_heads=cfg.get("num_heads", 4),
            num_classes=cfg.get("num_classes", 10),
        ).to(device)

        for name, param in model.named_parameters():
            if name in ckpt["ema_shadow"]:
                param.data.copy_(ckpt["ema_shadow"][name].to(device))
        model.eval()

        schedule = DiffusionSchedule(
            num_timesteps=cfg["num_timesteps"],
            schedule_type=cfg["schedule_type"],
        )

        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model: {n_params:,} params")

        # Generate samples (balanced across classes)
        print(f"\nGenerating {num_fid_samples} conditional CIFAR samples (w=5.0)...")
        t0 = time.time()
        gen_images = generate_cifar_samples(model, schedule, n=num_fid_samples,
                                            device=device, is_conditional=True)
        gen_time = time.time() - t0
        print(f"Generation time: {gen_time:.1f}s")

        # Load real images
        real_images = load_real_cifar(n=num_fid_samples)

        # Visual comparison
        compare_real_vs_generated(real_images, gen_images,
                                  "CIFAR-10 Conditional (w=5.0): Real vs Generated",
                                  "notebooks/cifar_cond_real_vs_gen.png")

        # FID
        print("\nComputing CIFAR conditional FID...")
        extractor = InceptionFeatureExtractor(device=device)
        real_feat = extractor.extract(real_images)
        gen_feat = extractor.extract(gen_images)
        fid = compute_fid(real_feat, gen_feat)
        print(f"CIFAR-10 Conditional FID (w=5.0): {fid:.2f}")
        results["cifar_cond_fid"] = fid

    # ══════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    # Results table
    print(f"\n{'Model':<35s} {'FID':>8s}  {'Params':>12s}")
    print("-" * 60)

    if "mnist_fid" in results:
        print(f"{'MNIST (DDIM 50 steps)':<35s} {results['mnist_fid']:>8.1f}  {'~4.5M':>12s}")
    if "cifar_uncond_fid" in results:
        print(f"{'CIFAR-10 Unconditional':<35s} {results['cifar_uncond_fid']:>8.1f}  {'~28M':>12s}")
    if "cifar_cond_fid" in results:
        print(f"{'CIFAR-10 Conditional (w=5.0)':<35s} {results['cifar_cond_fid']:>8.1f}  {'~28M':>12s}")

    print(f"\n{'Reference FID scores (from papers):'}")
    print(f"  DDPM (Ho et al. 2020, CIFAR-10):     3.17  (35M params, 800 epochs)")
    print(f"  Improved DDPM (Nichol 2021):          2.94")
    print(f"  Our model trains for 100 epochs with fewer params — higher FID is expected.")

    # Save results
    print(f"\nImages saved to notebooks/:")
    for f in sorted(os.listdir("notebooks")):
        if f.endswith(".png"):
            print(f"  {f}")

    print(f"\n✓ Evaluation complete!")

    return results


if __name__ == "__main__":
    # Check if running with --mnist-only flag
    mnist_only = "--mnist-only" in sys.argv
    if mnist_only:
        print("Running MNIST evaluation only...")

    main()