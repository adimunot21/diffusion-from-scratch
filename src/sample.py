"""
Sampling Algorithms — Generating Images from Noise.

Two algorithms for reversing the forward diffusion process:

1. DDPM (Denoising Diffusion Probabilistic Models):
   - The original algorithm from Ho et al. 2020
   - 1000 steps: x_T → x_{T-1} → ... → x_1 → x_0
   - Each step removes a tiny bit of noise and adds a tiny bit of fresh noise
   - Stochastic: same starting noise → different images each run
   - Highest quality but slow

2. DDIM (Denoising Diffusion Implicit Models):
   - Song et al. 2020 — the key insight: the forward process can be made
     NON-MARKOVIAN, meaning we can skip steps
   - 50 steps instead of 1000 → 20× faster
   - η parameter controls stochasticity: η=0 is fully deterministic,
     η=1 recovers DDPM
   - Deterministic mode (η=0): same starting noise → same image every time
     This enables meaningful interpolation in noise space

WHY DDIM WORKS (the key insight):
DDPM defines q(x_t | x_{t-1}) — each step depends on the previous step.
This means you MUST go through all 1000 steps.

DDIM redefines the forward process as q(x_t | x_0, x_{t-1}) in a way
where the MARGINALS q(x_t | x_0) are IDENTICAL to DDPM. The distribution
at each timestep is the same, but the path between timesteps is different.
This means a network trained with DDPM's objective works with DDIM's
sampling — no retraining needed.

The DDIM update rule can be interpreted as:
    1. Predict x_0 from the current x_t and predicted noise
    2. "Re-noise" x_0 to the next timestep (which can be many steps back)
This "predict then re-noise" formulation lets us skip arbitrary numbers
of steps, because we're always going through x_0 as an intermediate.
"""

import os
import time
import torch
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.diffusion import DiffusionSchedule
from src.unet import UNet
from src.train import EMA


# -----------------------------------------------------------------------
# DDPM Sampling
# -----------------------------------------------------------------------

@torch.no_grad()
def ddpm_sample(model, schedule, num_images=16, device="cpu"):
    """
    DDPM sampling: the full 1000-step reverse process.

    For t = T-1, T-2, ..., 1, 0:
        1. Predict noise: ε_θ = model(x_t, t)
        2. Compute posterior mean:
           μ_θ = (1/√α_t) · (x_t - β_t/√(1-ᾱ_t) · ε_θ)
        3. Sample: x_{t-1} = μ_θ + σ_t · z  where z ~ N(0,I)
           (no noise added at t=0)

    The posterior mean formula comes from Bayes' rule applied to:
        p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ_t² I)

    where σ_t² = β̃_t = β_t · (1-ᾱ_{t-1}) / (1-ᾱ_t)

    Args:
        model: trained U-Net
        schedule: DiffusionSchedule
        num_images: how many to generate
        device: "cpu" or "cuda"

    Returns:
        images: (num_images, C, H, W) in [0, 1]
    """
    model.eval()

    # Preload schedule tensors to device
    betas = schedule.betas.to(device)
    sqrt_recip_alphas = schedule.sqrt_recip_alphas.to(device)
    sqrt_one_minus_alphas_cumprod = schedule.sqrt_one_minus_alphas_cumprod.to(device)
    posterior_variance = schedule.posterior_variance.to(device)

    # Start from pure Gaussian noise
    x = torch.randn(num_images, 1, 28, 28, device=device)

    for t_val in reversed(range(schedule.num_timesteps)):
        t = torch.full((num_images,), t_val, device=device, dtype=torch.long)

        # 1. Predict noise
        noise_pred = model(x, t)

        # 2. Compute posterior mean
        # μ_θ(x_t, t) = (1/√α_t) · (x_t - β_t/√(1-ᾱ_t) · ε_θ)
        mean = sqrt_recip_alphas[t_val] * (
            x - betas[t_val] / sqrt_one_minus_alphas_cumprod[t_val] * noise_pred
        )

        # 3. Add noise (except at final step)
        if t_val > 0:
            noise = torch.randn_like(x)
            sigma = torch.sqrt(posterior_variance[t_val])
            x = mean + sigma * noise
        else:
            x = mean

    # Rescale [-1, 1] → [0, 1]
    x = (x + 1) / 2
    x = x.clamp(0, 1)
    return x


# -----------------------------------------------------------------------
# DDIM Sampling
# -----------------------------------------------------------------------

@torch.no_grad()
def ddim_sample(model, schedule, num_images=16, ddim_steps=50,
                eta=0.0, device="cpu"):
    """
    DDIM sampling: skip steps for fast generation.

    Instead of going through all 1000 timesteps, we select a subsequence
    of `ddim_steps` timesteps (e.g., [0, 20, 40, ..., 980]) and jump
    between them.

    The DDIM update rule (for going from timestep τ_i to τ_{i-1}):

        1. Predict x_0 from current x_t and predicted noise:
           x̂_0 = (x_t - √(1-ᾱ_t) · ε_θ) / √ᾱ_t

        2. Compute the "direction pointing to x_t":
           dir = √(1 - ᾱ_{t-1} - σ²) · ε_θ

        3. Add noise (controlled by η):
           σ = η · √((1-ᾱ_{t-1})/(1-ᾱ_t)) · √(1-ᾱ_t/ᾱ_{t-1})

        4. Update:
           x_{t-1} = √ᾱ_{t-1} · x̂_0 + dir + σ · z

    When η=0: σ=0, no random noise → fully deterministic
    When η=1: recovers the DDPM noise level → stochastic like DDPM

    WHY THIS WORKS:
    The key insight is step 1: we can always estimate x_0 from any x_t.
    Then we "re-noise" this x_0 estimate to a DIFFERENT timestep (τ_{i-1}),
    which can be many steps earlier than t. This bypasses all intermediate steps.

    The quality is nearly identical to DDPM because the noise prediction
    is just as accurate — the model was trained on all timesteps, so it
    can predict noise at t=500 just as well whether we arrived there from
    t=501 (DDPM) or t=550 (DDIM with 50 steps).

    Args:
        model: trained U-Net
        schedule: DiffusionSchedule
        num_images: how many to generate
        ddim_steps: number of denoising steps (50 is standard, 20 is fast)
        eta: stochasticity parameter (0=deterministic, 1=DDPM-like)
        device: "cpu" or "cuda"

    Returns:
        images: (num_images, C, H, W) in [0, 1]
    """
    model.eval()

    # Create the subsequence of timesteps
    # e.g., for ddim_steps=50, T=1000: [0, 20, 40, ..., 960, 980]
    # We reverse it for sampling: [980, 960, ..., 40, 20, 0]
    step_size = schedule.num_timesteps // ddim_steps
    timestep_seq = list(range(0, schedule.num_timesteps, step_size))

    # Load schedule values
    alphas_cumprod = schedule.alphas_cumprod.to(device)

    # Start from pure noise
    x = torch.randn(num_images, 1, 28, 28, device=device)

    # Walk backwards through the subsequence
    # At each step, we go from timestep_seq[i] to timestep_seq[i-1]
    for i in reversed(range(len(timestep_seq))):
        t_val = timestep_seq[i]
        t = torch.full((num_images,), t_val, device=device, dtype=torch.long)

        # Current and previous ᾱ values
        alpha_bar_t = alphas_cumprod[t_val]

        if i > 0:
            t_prev = timestep_seq[i - 1]
            alpha_bar_t_prev = alphas_cumprod[t_prev]
        else:
            # Last step: previous is t=0, ᾱ₀ = 1 (clean image)
            alpha_bar_t_prev = torch.tensor(1.0, device=device)

        # 1. Predict noise
        noise_pred = model(x, t)

        # 2. Predict x_0 from x_t and predicted noise
        # x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε  →  x_0 = (x_t - √(1-ᾱ_t)·ε) / √ᾱ_t
        x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
        # NOTE: We do NOT clamp x0_pred at intermediate steps.
        # At high noise timesteps, x0_pred can legitimately be outside [-1, 1].
        # Clamping introduces bias that accumulates over many steps,
        # causing blob artifacts with high step counts (e.g., 200 steps).

        # 3. Compute σ (noise level for this step)
        # σ = η · √((1-ᾱ_{t-1})/(1-ᾱ_t)) · √(1 - ᾱ_t/ᾱ_{t-1})
        if eta > 0 and i > 0:
            sigma = (eta *
                     torch.sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t)) *
                     torch.sqrt(1 - alpha_bar_t / alpha_bar_t_prev))
        else:
            sigma = torch.tensor(0.0, device=device)

        # 4. Compute "direction pointing to x_t"
        # This is the deterministic component of the update
        dir_xt = torch.sqrt(1 - alpha_bar_t_prev - sigma ** 2) * noise_pred

        # 5. DDIM update
        # x_{t-1} = √ᾱ_{t-1} · x̂_0 + dir + σ · z
        x = torch.sqrt(alpha_bar_t_prev) * x0_pred + dir_xt

        if sigma > 0:
            noise = torch.randn_like(x)
            x = x + sigma * noise

    # Rescale [-1, 1] → [0, 1]
    x = (x + 1) / 2
    x = x.clamp(0, 1)
    return x


# -----------------------------------------------------------------------
# Noise Space Interpolation
# -----------------------------------------------------------------------

@torch.no_grad()
def interpolate(model, schedule, num_interp=8, ddim_steps=50, device="cpu"):
    """
    Interpolate between two random noise vectors using DDIM.

    Because DDIM (η=0) is deterministic, each noise vector maps to exactly
    one image. Linearly interpolating between two noise vectors produces
    a smooth transition between two images — morphing digit shapes,
    blending stroke styles.

    This only works with deterministic sampling (η=0). With stochastic
    sampling, the same noise vector gives different images each time,
    so interpolation would be meaningless.

    Args:
        model: trained U-Net
        schedule: DiffusionSchedule
        num_interp: number of interpolation steps (including endpoints)
        ddim_steps: DDIM sampling steps
        device: "cpu" or "cuda"

    Returns:
        images: (num_interp, 1, 28, 28) in [0, 1]
    """
    model.eval()

    # Two random noise vectors (the "endpoints")
    z1 = torch.randn(1, 1, 28, 28, device=device)
    z2 = torch.randn(1, 1, 28, 28, device=device)

    # Spherical linear interpolation (slerp) is more principled for
    # high-dimensional Gaussians, but linear interpolation works fine
    # for visualization purposes
    alphas = torch.linspace(0, 1, num_interp, device=device)
    all_images = []

    for alpha in alphas:
        # Linear interpolation: z = (1-α)·z1 + α·z2
        z = (1 - alpha) * z1 + alpha * z2

        # Decode this noise vector using DDIM
        # We need to feed this specific noise through the reverse process
        # DDIM with η=0 is deterministic, so same z → same image
        images = _ddim_sample_from_noise(model, schedule, z, ddim_steps, device)
        all_images.append(images)

    return torch.cat(all_images, dim=0)


@torch.no_grad()
def _ddim_sample_from_noise(model, schedule, x, ddim_steps, device):
    """DDIM sampling starting from a specific noise tensor x."""
    model.eval()

    step_size = schedule.num_timesteps // ddim_steps
    timestep_seq = list(range(0, schedule.num_timesteps, step_size))
    alphas_cumprod = schedule.alphas_cumprod.to(device)

    num_images = x.shape[0]

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

    x = (x + 1) / 2
    x = x.clamp(0, 1)
    return x


# -----------------------------------------------------------------------
# Denoising Trajectory Visualization
# -----------------------------------------------------------------------

@torch.no_grad()
def visualize_denoising(model, schedule, ddim_steps=50, device="cpu"):
    """
    Show the step-by-step denoising process.

    Generates one image and captures intermediate x_t at regular intervals
    to show the progression from pure noise → recognizable digit → clean digit.

    Returns:
        trajectory: list of (timestep, image) tuples
    """
    model.eval()

    step_size = schedule.num_timesteps // ddim_steps
    timestep_seq = list(range(0, schedule.num_timesteps, step_size))
    alphas_cumprod = schedule.alphas_cumprod.to(device)

    x = torch.randn(1, 1, 28, 28, device=device)
    trajectory = [(schedule.num_timesteps, ((x + 1) / 2).clamp(0, 1))]

    for i in reversed(range(len(timestep_seq))):
        t_val = timestep_seq[i]
        t = torch.full((1,), t_val, device=device, dtype=torch.long)

        alpha_bar_t = alphas_cumprod[t_val]
        alpha_bar_t_prev = (alphas_cumprod[timestep_seq[i - 1]]
                            if i > 0 else torch.tensor(1.0, device=device))

        noise_pred = model(x, t)
        x0_pred = ((x - torch.sqrt(1 - alpha_bar_t) * noise_pred) /
                    torch.sqrt(alpha_bar_t))

        dir_xt = torch.sqrt(1 - alpha_bar_t_prev) * noise_pred
        x = torch.sqrt(alpha_bar_t_prev) * x0_pred + dir_xt

        # Save snapshot at regular intervals
        step_in_seq = len(timestep_seq) - i
        if step_in_seq % (ddim_steps // 10) == 0 or i == 0:
            img = ((x + 1) / 2).clamp(0, 1)
            trajectory.append((t_val, img))

    return trajectory


# -----------------------------------------------------------------------
# Model Loading
# -----------------------------------------------------------------------

def load_model(checkpoint_path="checkpoints/ddpm_mnist_final.pt", device="cpu"):
    """Load a trained model and its EMA weights."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    model = UNet(
        in_channels=1, out_channels=1,
        channel_list=cfg["channel_list"],
        time_dim=cfg["time_dim"],
        num_res_blocks=cfg["num_res_blocks"],
    ).to(device)

    # Load EMA weights (these produce better samples than raw training weights)
    for name, param in model.named_parameters():
        if name in ckpt["ema_shadow"]:
            param.data.copy_(ckpt["ema_shadow"][name].to(device))

    model.eval()

    schedule = DiffusionSchedule(
        num_timesteps=cfg["num_timesteps"],
        schedule_type=cfg["schedule_type"],
        beta_start=cfg["beta_start"],
        beta_end=cfg["beta_end"],
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded model ({n_params:,} params) from {checkpoint_path}")
    print(f"Trained for {ckpt['epoch']} epochs, {ckpt['global_step']} steps")

    return model, schedule


# -----------------------------------------------------------------------
# Visualization Helpers
# -----------------------------------------------------------------------

def save_image_grid(images, path, nrow=8, title=None):
    """Save a batch of images as a grid."""
    n = images.shape[0]
    ncol = min(nrow, n)
    nrow_actual = (n + ncol - 1) // ncol

    fig, axes = plt.subplots(nrow_actual, ncol,
                             figsize=(1.5 * ncol, 1.5 * nrow_actual))
    if nrow_actual == 1:
        axes = axes[np.newaxis, :]
    if ncol == 1:
        axes = axes[:, np.newaxis]

    for i in range(nrow_actual):
        for j in range(ncol):
            idx = i * ncol + j
            ax = axes[i, j]
            if idx < n:
                ax.imshow(images[idx, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")

    if title:
        plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# -----------------------------------------------------------------------
# Main — Run all sampling experiments
# -----------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    os.makedirs("notebooks", exist_ok=True)

    # ── Load model ──
    model, schedule = load_model(device=device)

    # ══════════════════════════════════════════════════════════════
    # Test 1: DDPM Sampling (1000 steps)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("TEST 1: DDPM Sampling (1000 steps)")
    print("=" * 60)

    t0 = time.time()
    ddpm_images = ddpm_sample(model, schedule, num_images=16, device=device)
    ddpm_time = time.time() - t0
    print(f"Generated 16 images in {ddpm_time:.1f}s ({ddpm_time/16:.1f}s per image)")
    print(f"Image shape: {ddpm_images.shape}")
    print(f"Pixel range: [{ddpm_images.min():.3f}, {ddpm_images.max():.3f}]")

    save_image_grid(ddpm_images, "notebooks/ddpm_samples.png",
                    title="DDPM Samples (1000 steps)")

    # ══════════════════════════════════════════════════════════════
    # Test 2: DDIM Sampling — varying step counts
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("TEST 2: DDIM Sampling (varying steps)")
    print("=" * 60)

    # Use the SAME starting noise so we can compare quality across step counts
    torch.manual_seed(42)
    fixed_noise = torch.randn(16, 1, 28, 28, device=device)

    for steps in [200, 100, 50, 20, 10]:
        torch.manual_seed(42)  # reset so we get the same noise

        t0 = time.time()
        ddim_images = ddim_sample(model, schedule, num_images=16,
                                  ddim_steps=steps, eta=0.0, device=device)
        elapsed = time.time() - t0
        print(f"DDIM {steps:4d} steps: {elapsed:.1f}s ({elapsed/16:.2f}s per image)")

        save_image_grid(ddim_images, f"notebooks/ddim_{steps}steps.png",
                        title=f"DDIM Samples ({steps} steps, η=0)")

    # ══════════════════════════════════════════════════════════════
    # Test 3: DDIM determinism — same noise → same image
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("TEST 3: DDIM Determinism (η=0)")
    print("=" * 60)

    torch.manual_seed(123)
    run1 = ddim_sample(model, schedule, num_images=4, ddim_steps=50,
                       eta=0.0, device=device)
    torch.manual_seed(123)
    run2 = ddim_sample(model, schedule, num_images=4, ddim_steps=50,
                       eta=0.0, device=device)

    max_diff = (run1 - run2).abs().max().item()
    print(f"Max pixel difference between two runs: {max_diff:.8f}")
    assert max_diff < 1e-5, f"DDIM η=0 should be deterministic, got diff={max_diff}"
    print("✓ DDIM (η=0) is deterministic — same noise produces identical images")

    # ══════════════════════════════════════════════════════════════
    # Test 4: DDIM stochastic (η=1) — should differ between runs
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("TEST 4: DDIM Stochastic (η=1)")
    print("=" * 60)

    torch.manual_seed(123)
    stoch1 = ddim_sample(model, schedule, num_images=4, ddim_steps=50,
                         eta=1.0, device=device)
    torch.manual_seed(456)  # different seed
    stoch2 = ddim_sample(model, schedule, num_images=4, ddim_steps=50,
                         eta=1.0, device=device)

    max_diff_stoch = (stoch1 - stoch2).abs().max().item()
    print(f"Max pixel difference (η=1, different seeds): {max_diff_stoch:.4f}")
    assert max_diff_stoch > 0.1, "η=1 should produce different images with different seeds"
    print("✓ DDIM (η=1) is stochastic — different runs produce different images")

    # ══════════════════════════════════════════════════════════════
    # Test 5: Noise interpolation
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("TEST 5: Noise Space Interpolation")
    print("=" * 60)

    torch.manual_seed(42)
    interp_images = interpolate(model, schedule, num_interp=10,
                                ddim_steps=50, device=device)
    print(f"Interpolation images shape: {interp_images.shape}")

    # Save as a single row
    fig, axes = plt.subplots(1, 10, figsize=(20, 2.5))
    for i in range(10):
        axes[i].imshow(interp_images[i, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        axes[i].axis("off")
        if i == 0:
            axes[i].set_title("Start", fontsize=10)
        elif i == 9:
            axes[i].set_title("End", fontsize=10)
    plt.suptitle("Noise Space Interpolation (DDIM, η=0)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("notebooks/interpolation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved notebooks/interpolation.png")

    # ══════════════════════════════════════════════════════════════
    # Test 6: Denoising trajectory
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("TEST 6: Denoising Trajectory")
    print("=" * 60)

    torch.manual_seed(77)
    trajectory = visualize_denoising(model, schedule, ddim_steps=50, device=device)
    print(f"Captured {len(trajectory)} snapshots")

    fig, axes = plt.subplots(1, len(trajectory), figsize=(2 * len(trajectory), 2.5))
    for i, (t_val, img) in enumerate(trajectory):
        axes[i].imshow(img[0, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        axes[i].set_title(f"t={t_val}", fontsize=9)
        axes[i].axis("off")
    plt.suptitle("Denoising Trajectory (DDIM, 50 steps)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("notebooks/denoising_trajectory.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved notebooks/denoising_trajectory.png")

    # ══════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"DDPM (1000 steps): {ddpm_time:.1f}s for 16 images")
    print(f"DDIM (50 steps):   ~20× faster")
    print(f"DDIM (η=0):        deterministic — enables interpolation")
    print(f"DDIM (η=1):        stochastic — like DDPM but fewer steps")
    print(f"\nSaved to notebooks/:")
    print(f"  ddpm_samples.png          — DDPM 1000-step samples")
    print(f"  ddim_Xsteps.png           — DDIM at various step counts")
    print(f"  interpolation.png         — smooth transitions between digits")
    print(f"  denoising_trajectory.png  — noise → digit step by step")
    print(f"\n✓ All Phase 4 tests passed!")