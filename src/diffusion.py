"""
Forward Diffusion Process — The Mathematical Foundation.

The forward process destroys images by adding Gaussian noise over T timesteps.
Starting from a clean image x₀, we produce progressively noisier versions
x₁, x₂, ..., x_T, where x_T is approximately pure Gaussian noise.

THE KEY INSIGHT: We don't need to add noise step-by-step. We can jump
directly from x₀ to ANY timestep t in one operation:

    x_t = √(ᾱ_t) · x₀ + √(1 - ᾱ_t) · ε    where ε ~ N(0, I)

This is just a weighted average of the original image and random noise.
As t increases, the image weight (√ᾱ_t) shrinks toward 0 and the noise
weight (√(1-ᾱ_t)) grows toward 1. At t=T, it's ~100% noise.

WHY THIS WORKS FOR GENERATION:
If we can train a network to predict the noise ε that was added at each step,
we can reverse the process — start from pure noise and iteratively subtract
the predicted noise to recover a clean image. The forward process gives us
unlimited training data: take any real image, noise it to a random timestep t,
and train the network to predict the noise we added.

TWO NOISE SCHEDULES:
- Linear: β increases linearly from 1e-4 to 0.02. Simple, used in original DDPM.
- Cosine: β follows a cosine curve. Preserves image structure longer in early steps.
  Better for higher resolutions. Introduced by Improved DDPM (Nichol & Dhariwal, 2021).
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class DiffusionSchedule:
    """
    Precomputes all the constants needed for the forward diffusion process.

    The noise schedule defines β_t (how much noise to add at each step).
    From β_t, we derive everything else:
        α_t = 1 - β_t                    (signal retention per step)
        ᾱ_t = ∏(s=1 to t) α_s           (cumulative signal retention)
        √ᾱ_t                             (coefficient for x₀ in q(x_t|x₀))
        √(1 - ᾱ_t)                       (coefficient for noise in q(x_t|x₀))

    All values are precomputed as tensors for fast indexing during training.
    """

    def __init__(self, num_timesteps=1000, schedule_type="linear",
                 beta_start=1e-4, beta_end=0.02):
        """
        Args:
            num_timesteps: T — total number of diffusion steps
            schedule_type: "linear" or "cosine"
            beta_start: β₁ (only used for linear schedule)
            beta_end: β_T (only used for linear schedule)
        """
        self.num_timesteps = num_timesteps

        # Compute betas based on schedule type
        if schedule_type == "linear":
            # β increases linearly from beta_start to beta_end
            # At step 1: very little noise (β=0.0001)
            # At step 1000: more noise per step (β=0.02)
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)

        elif schedule_type == "cosine":
            # Cosine schedule: ᾱ_t follows a cosine curve from ~1 to ~0
            # This means noise is added more gradually in early steps,
            # preserving image structure longer than the linear schedule.
            #
            # Formula: ᾱ_t = f(t) / f(0)  where f(t) = cos((t/T + s)/(1+s) · π/2)²
            # s = 0.008 is a small offset to prevent β_t from being too small at t=0
            s = 0.008
            steps = torch.arange(num_timesteps + 1, dtype=torch.float64)
            # f(t) = cos²((t/T + s) / (1 + s) · π/2)
            f_t = torch.cos(((steps / num_timesteps) + s) / (1 + s) * (torch.pi / 2)) ** 2
            # ᾱ_t = f(t) / f(0)
            alphas_cumprod = f_t / f_t[0]
            # β_t = 1 - ᾱ_t / ᾱ_{t-1}  (derived from ᾱ_t = ∏ α_s = ∏ (1-β_s))
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            # Clip to prevent numerical issues (β shouldn't exceed 0.999)
            self.betas = torch.clamp(betas, min=1e-5, max=0.999).float()

        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        # ─── Derive all other constants from betas ───

        # α_t = 1 - β_t  (how much signal is RETAINED at each step)
        self.alphas = 1.0 - self.betas

        # ᾱ_t = α₁ · α₂ · ... · α_t  (CUMULATIVE signal retention)
        # This is the key quantity: it tells us how much of x₀ survives at step t.
        # ᾱ₁ ≈ 0.9999  (almost all signal)
        # ᾱ₅₀₀ ≈ 0.05  (mostly noise)
        # ᾱ₁₀₀₀ ≈ 0.0001  (essentially pure noise)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # ᾱ_{t-1}: shifted by one step, with ᾱ₀ = 1 (no noise at t=0)
        # Needed for the reverse process posterior calculation (Phase 4)
        self.alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0]),           # ᾱ₀ = 1 (clean image)
            self.alphas_cumprod[:-1]       # ᾱ₁, ᾱ₂, ..., ᾱ_{T-1}
        ])

        # Coefficients for the forward process q(x_t | x₀):
        #   x_t = √ᾱ_t · x₀  +  √(1-ᾱ_t) · ε
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # ─── Constants needed for the reverse process (precomputed for Phase 4) ───

        # 1/√α_t  (used in reverse step mean calculation)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # Posterior variance: β̃_t = β_t · (1-ᾱ_{t-1}) / (1-ᾱ_t)
        # This is the variance of the reverse process q(x_{t-1} | x_t, x₀)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x_0, t, noise=None):
        """
        Forward process: sample x_t from q(x_t | x_0).

        Instead of applying noise t times sequentially, we jump directly
        to timestep t using the closed-form formula:

            x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε

        This is a REPARAMETERIZATION: x_t is a deterministic function of
        x_0 and ε, which means we can backpropagate through this operation.

        Args:
            x_0: clean images, shape (batch, channels, height, width)
            t: timesteps, shape (batch,) — each image can have a different t
            noise: optional pre-generated noise (if None, sample fresh)

        Returns:
            x_t: noised images at timesteps t
            noise: the noise that was added (this becomes the training target)
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # Gather the coefficients for each sample's timestep
        # sqrt_alphas_cumprod[t] has shape (batch,) — we need (batch, 1, 1, 1)
        # for broadcasting with images of shape (batch, C, H, W)
        sqrt_alpha = self._gather(self.sqrt_alphas_cumprod, t)
        sqrt_one_minus_alpha = self._gather(self.sqrt_one_minus_alphas_cumprod, t)

        # The forward process formula
        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

        return x_t, noise

    def _gather(self, values, t):
        """
        Gather values at timestep indices t, then reshape for broadcasting.

        values has shape (T,) — one value per timestep.
        t has shape (batch,) — one timestep per sample in the batch.
        Output has shape (batch, 1, 1, 1) — ready for broadcasting with images.

        Example:
            values = [0.99, 0.98, 0.97, ...]  (1000 values)
            t = [0, 500, 999]                   (3 samples at different timesteps)
            output = [[0.99], [~0.22], [~0.01]] reshaped to (3, 1, 1, 1)
        """
        gathered = values.gather(0, t)
        # Reshape from (batch,) to (batch, 1, 1, 1) for image broadcasting
        return gathered.view(-1, 1, 1, 1)


def visualize_forward_process(schedule, dataset, save_path="notebooks/forward_process.png"):
    """
    Show how images get progressively destroyed by the forward process.

    Picks 4 random MNIST digits and shows them at timesteps
    [0, 100, 250, 500, 750, 999]. At t=0 the image is clean,
    at t=999 it's indistinguishable from random noise.
    """
    timesteps_to_show = [0, 100, 250, 500, 750, 999]
    num_images = 4

    # Get some random images from the dataset
    indices = torch.randint(0, len(dataset), (num_images,))
    images = torch.stack([dataset[i][0] for i in indices])  # (4, 1, 28, 28)

    fig, axes = plt.subplots(num_images, len(timesteps_to_show),
                             figsize=(2.5 * len(timesteps_to_show), 2.5 * num_images))

    # Use the SAME noise for each image across timesteps
    # This way you can see the SAME noise being amplified, not different noise each time
    noise = torch.randn_like(images)

    for col, t_val in enumerate(timesteps_to_show):
        t = torch.full((num_images,), t_val, dtype=torch.long)
        x_t, _ = schedule.q_sample(images, t, noise=noise)

        for row in range(num_images):
            ax = axes[row, col]
            # Clamp to [0, 1] for display (noised images can go outside this range)
            img = x_t[row, 0].clamp(0, 1).numpy()
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if row == 0:
                # Show ᾱ_t value — how much original signal remains
                alpha_bar = schedule.alphas_cumprod[t_val].item()
                ax.set_title(f"t={t_val}\nᾱ={alpha_bar:.4f}", fontsize=10)

    plt.suptitle("Forward Diffusion Process: Clean → Pure Noise",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved forward process visualization to {save_path}")


def visualize_schedules(save_path="notebooks/noise_schedules.png"):
    """
    Compare linear and cosine noise schedules.

    Plots ᾱ_t (cumulative signal retention) and β_t (per-step noise)
    for both schedules. The cosine schedule preserves signal longer in
    early steps and destroys it faster at the end.
    """
    linear = DiffusionSchedule(1000, schedule_type="linear")
    cosine = DiffusionSchedule(1000, schedule_type="cosine")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    timesteps = np.arange(1000)

    # ── Plot 1: ᾱ_t (cumulative signal retention) ──
    ax = axes[0]
    ax.plot(timesteps, linear.alphas_cumprod.numpy(), "b-", linewidth=2, label="Linear")
    ax.plot(timesteps, cosine.alphas_cumprod.numpy(), "r-", linewidth=2, label="Cosine")
    ax.set_title("ᾱ_t (Cumulative Signal Retention)", fontweight="bold")
    ax.set_xlabel("Timestep t")
    ax.set_ylabel("ᾱ_t")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.annotate("Image mostly intact", xy=(100, 0.9), fontsize=9, color="gray")
    ax.annotate("Mostly noise", xy=(700, 0.1), fontsize=9, color="gray")

    # ── Plot 2: β_t (per-step noise amount) ──
    ax = axes[1]
    ax.plot(timesteps, linear.betas.numpy(), "b-", linewidth=2, label="Linear")
    ax.plot(timesteps, cosine.betas.numpy(), "r-", linewidth=2, label="Cosine")
    ax.set_title("β_t (Noise Added Per Step)", fontweight="bold")
    ax.set_xlabel("Timestep t")
    ax.set_ylabel("β_t")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Plot 3: SNR in dB ──
    # Signal-to-Noise Ratio = ᾱ_t / (1 - ᾱ_t)
    # In dB: 10 · log10(SNR)
    ax = axes[2]
    snr_linear = linear.alphas_cumprod / (1 - linear.alphas_cumprod)
    snr_cosine = cosine.alphas_cumprod / (1 - cosine.alphas_cumprod)
    ax.plot(timesteps, 10 * torch.log10(snr_linear).numpy(), "b-", linewidth=2, label="Linear")
    ax.plot(timesteps, 10 * torch.log10(snr_cosine).numpy(), "r-", linewidth=2, label="Cosine")
    ax.set_title("Signal-to-Noise Ratio (dB)", fontweight="bold")
    ax.set_xlabel("Timestep t")
    ax.set_ylabel("SNR (dB)")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, label="SNR = 1 (equal)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("Noise Schedule Comparison: Linear vs Cosine",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved noise schedule comparison to {save_path}")


# -----------------------------------------------------------------------
# Tests & Visualization
# -----------------------------------------------------------------------

if __name__ == "__main__":
    from torchvision import datasets, transforms

    print("=" * 60)
    print("PHASE 1: Forward Diffusion Process")
    print("=" * 60)

    # ── Test 1: Linear schedule values ──
    print("\n── Test 1: Linear Schedule ──")
    schedule = DiffusionSchedule(num_timesteps=1000, schedule_type="linear")

    print(f"β range: [{schedule.betas[0]:.6f}, {schedule.betas[-1]:.6f}]")
    print(f"  Expected: [0.000100, 0.020000]")
    assert abs(schedule.betas[0] - 1e-4) < 1e-7, "β₁ should be 1e-4"
    assert abs(schedule.betas[-1] - 0.02) < 1e-7, "β_T should be 0.02"

    print(f"\nᾱ_t at key timesteps:")
    for t in [0, 100, 250, 500, 750, 999]:
        print(f"  t={t:4d}: ᾱ={schedule.alphas_cumprod[t]:.6f}"
              f"  √ᾱ={schedule.sqrt_alphas_cumprod[t]:.4f}"
              f"  √(1-ᾱ)={schedule.sqrt_one_minus_alphas_cumprod[t]:.4f}")

    # At t=0: ᾱ should be close to 1 (almost no noise)
    assert schedule.alphas_cumprod[0] > 0.99, "ᾱ₁ should be ~1 (barely any noise)"
    # At t=999: ᾱ should be close to 0 (almost pure noise)
    assert schedule.alphas_cumprod[999] < 0.01, "ᾱ_T should be ~0 (pure noise)"
    print("✓ Schedule values correct\n")

    # ── Test 2: Cosine schedule ──
    print("── Test 2: Cosine Schedule ──")
    cosine = DiffusionSchedule(num_timesteps=1000, schedule_type="cosine")
    print(f"ᾱ₁ (cosine):    {cosine.alphas_cumprod[0]:.6f}")
    print(f"ᾱ₅₀₀ (cosine):  {cosine.alphas_cumprod[500]:.6f}")
    print(f"ᾱ₁₀₀₀ (cosine): {cosine.alphas_cumprod[999]:.6f}")
    assert cosine.alphas_cumprod[0] > 0.99, "Cosine ᾱ₁ should be ~1"
    assert cosine.alphas_cumprod[999] < 0.01, "Cosine ᾱ_T should be ~0"
    # Cosine schedule preserves signal longer than linear in early steps
    assert cosine.alphas_cumprod[250] > schedule.alphas_cumprod[250], \
        "Cosine should retain more signal than linear at t=250"
    print("✓ Cosine schedule correct\n")

    # ── Test 3: q_sample shapes and properties ──
    print("── Test 3: Forward Sampling (q_sample) ──")
    batch_size = 8
    x_0 = torch.randn(batch_size, 1, 28, 28)  # fake "images"
    t = torch.randint(0, 1000, (batch_size,))

    x_t, noise = schedule.q_sample(x_0, t)
    print(f"x_0 shape:   {x_0.shape}")
    print(f"t shape:     {t.shape}")
    print(f"x_t shape:   {x_t.shape}")
    print(f"noise shape: {noise.shape}")
    assert x_t.shape == x_0.shape, "x_t should have same shape as x_0"
    assert noise.shape == x_0.shape, "noise should have same shape as x_0"

    # Verify the formula: x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
    sqrt_a = schedule.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_1ma = schedule.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
    x_t_manual = sqrt_a * x_0 + sqrt_1ma * noise
    assert torch.allclose(x_t, x_t_manual, atol=1e-6), "q_sample doesn't match manual formula"
    print("✓ q_sample formula verified\n")

    # ── Test 4: Noise properties at different timesteps ──
    print("── Test 4: Noise Properties ──")
    # At t=0: x_t should be very close to x_0
    t_zero = torch.zeros(batch_size, dtype=torch.long)
    x_t_early, _ = schedule.q_sample(x_0, t_zero)
    diff_early = (x_t_early - x_0).abs().mean().item()
    print(f"Mean difference at t=0:    {diff_early:.6f}  (should be ~0)")
    assert diff_early < 0.02, "At t=0, x_t should be very close to x_0"

    # At t=999: x_t should be very close to pure noise
    t_final = torch.full((batch_size,), 999, dtype=torch.long)
    x_t_late, noise_late = schedule.q_sample(x_0, t_final)
    # At t=999, √ᾱ ≈ 0, so x_t ≈ √(1-ᾱ) · noise ≈ noise
    diff_from_noise = (x_t_late - noise_late).abs().mean().item()
    print(f"Diff from pure noise at t=999: {diff_from_noise:.6f}  (should be ~0)")
    assert diff_from_noise < 0.05, "At t=999, x_t should be approximately pure noise"
    print("✓ Noise properties correct\n")

    # ── Test 5: Visualize with real MNIST data ──
    print("── Test 5: Visualization ──")
    mnist = datasets.MNIST(root="./data", train=True, download=True,
                           transform=transforms.ToTensor())

    visualize_forward_process(schedule, mnist, save_path="notebooks/forward_process.png")
    visualize_schedules(save_path="notebooks/noise_schedules.png")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Timesteps (T):    {schedule.num_timesteps}")
    print(f"β range:          [{schedule.betas[0]:.6f}, {schedule.betas[-1]:.6f}]")
    print(f"ᾱ_1:              {schedule.alphas_cumprod[0]:.6f}  (clean)")
    print(f"ᾱ_500:            {schedule.alphas_cumprod[500]:.6f}  (half-noised)")
    print(f"ᾱ_1000:           {schedule.alphas_cumprod[999]:.6f}  (pure noise)")
    print(f"\nPrecomputed tensors:")
    print(f"  betas:                         shape {schedule.betas.shape}")
    print(f"  alphas_cumprod:                shape {schedule.alphas_cumprod.shape}")
    print(f"  sqrt_alphas_cumprod:           shape {schedule.sqrt_alphas_cumprod.shape}")
    print(f"  sqrt_one_minus_alphas_cumprod: shape {schedule.sqrt_one_minus_alphas_cumprod.shape}")
    print(f"  posterior_variance:            shape {schedule.posterior_variance.shape}")
    print("\n✓ All Phase 1 tests passed!")