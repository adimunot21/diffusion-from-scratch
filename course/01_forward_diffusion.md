# Chapter 1: The Forward Diffusion Process — Destroying Images with Math

## The Problem

We need a mathematically precise way to gradually destroy an image by adding noise. "Gradually" is key — if we added all the noise at once, we'd lose all information immediately and the reverse process would have nothing to work with. Instead, we add a tiny bit of noise at each of 1000 timesteps, creating a smooth path from clean image to pure noise.

This path is what the neural network will learn to reverse. The smoother and more predictable the forward path, the easier the reverse path is to learn.

## Part 1: One Step of Noise Addition

### The Forward Step

At each timestep, we add a small amount of Gaussian noise:

```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t) · x_{t-1}, β_t · I)
```

In plain English: to get `x_t` from `x_{t-1}`, we:
1. Shrink the image slightly: multiply by `√(1-β_t)` (a number just below 1)
2. Add Gaussian noise with variance `β_t`

β_t (beta) is the **noise schedule** — it controls how much noise to add at each step. It's small: β₁ = 0.0001, β₁₀₀₀ = 0.02.

### A Numerical Example (Single Pixel)

Say one pixel has value 0.8 at step t-1, and β_t = 0.01:

```
x_t = √(1 - 0.01) · 0.8 + √0.01 · ε      where ε ~ N(0, 1)
    = √0.99 · 0.8 + 0.1 · ε
    = 0.9950 · 0.8 + 0.1 · ε
    = 0.7960 + 0.1 · ε

If ε = 0.5:   x_t = 0.7960 + 0.05 = 0.846
If ε = -1.2:  x_t = 0.7960 - 0.12 = 0.676
```

The pixel value barely changed — it went from 0.8 to somewhere around 0.7-0.85. One step of noise is almost imperceptible. But over 1000 steps, these tiny perturbations accumulate and eventually overwhelm the signal completely.

### Why Shrink Before Adding Noise?

The factor `√(1-β_t)` slightly shrinks the image toward zero at each step. Without this shrinkage, the variance of x_t would grow without bound — each step adds noise, so the total noise keeps accumulating. The shrinkage ensures the process converges to a standard normal distribution N(0, I) at the end, regardless of what the original image looked like.

Think of it like this: at each step, we're mixing the current image with a bit of fresh noise, with the noise getting a slightly larger share each time. After enough steps, the mixture is almost entirely noise.

## Part 2: The Key Insight — Jumping to Any Timestep

### The Slow Way

To get x₅₀₀ from x₀, we'd need to apply the noise step 500 times:

```
x₁ = √(1-β₁) · x₀ + √β₁ · ε₁
x₂ = √(1-β₂) · x₁ + √β₂ · ε₂
...
x₅₀₀ = √(1-β₅₀₀) · x₄₉₉ + √β₅₀₀ · ε₅₀₀
```

500 sequential operations. During training, we need x_t for a random t every single batch — this would be impossibly slow.

### The Fast Way — Closed-Form Solution

Because each step is a Gaussian operation applied to the previous step (which is also Gaussian), the entire chain collapses into a single operation:

```
q(x_t | x₀) = N(x_t; √ᾱ_t · x₀, (1-ᾱ_t) · I)
```

Which means:

```
x_t = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε      where ε ~ N(0, I)
```

One multiplication, one addition, one random sample. Instant, regardless of t.

### Deriving ᾱ_t

Define:

```
α_t = 1 - β_t           (signal retention per step)
ᾱ_t = α₁ · α₂ · ... · α_t   (cumulative signal retention)
     = ∏(s=1 to t) α_s
```

ᾱ_t is the cumulative product of all the α values up to step t. It tells us what fraction of the original signal survives at step t.

### Why the Cumulative Product Works

Consider two steps. After step 1:

```
x₁ = √α₁ · x₀ + √(1-α₁) · ε₁
```

After step 2, substitute x₁:

```
x₂ = √α₂ · x₁ + √(1-α₂) · ε₂
   = √α₂ · (√α₁ · x₀ + √(1-α₁) · ε₁) + √(1-α₂) · ε₂
   = √(α₁·α₂) · x₀ + √α₂·√(1-α₁) · ε₁ + √(1-α₂) · ε₂
```

The x₀ coefficient is `√(α₁·α₂) = √ᾱ₂`. The noise terms are two independent Gaussians. When you add independent Gaussians, their variances add:

```
Var(noise) = α₂·(1-α₁) + (1-α₂)
           = α₂ - α₁·α₂ + 1 - α₂
           = 1 - α₁·α₂
           = 1 - ᾱ₂
```

So `x₂ = √ᾱ₂ · x₀ + √(1-ᾱ₂) · ε` where ε is a single standard Gaussian. This pattern extends to any number of steps by induction.

## Part 3: Noise Schedules — How Fast to Destroy

### Linear Schedule

The simplest approach: β increases linearly from β₁ = 0.0001 to β_T = 0.02.

```python
betas = torch.linspace(0.0001, 0.02, 1000)
```

This creates a smooth, predictable noise progression:

```
t=1:    β=0.0001  α=0.9999  ᾱ≈0.9999  → 99.99% signal
t=100:  β=0.0021  α=0.9979  ᾱ≈0.9811  → 98% signal  (barely noisy)
t=250:  β=0.0052  α=0.9948  ᾱ≈0.8910  → 89% signal  (slightly noisy)
t=500:  β=0.0101  α=0.9899  ᾱ≈0.4996  → 50% signal  (half noise)
t=750:  β=0.0151  α=0.9849  ᾱ≈0.1104  → 11% signal  (mostly noise)
t=999:  β=0.0200  α=0.9800  ᾱ≈0.0061  → 0.6% signal (pure noise)
```

The signal-to-noise ratio crosses 50% around step 500 — the midpoint of the schedule.

### Cosine Schedule

Introduced by Nichol & Dhariwal (2021), the cosine schedule preserves signal longer in early steps:

```
ᾱ_t = f(t) / f(0)    where f(t) = cos²((t/T + s)/(1+s) · π/2)
```

The cosine curve drops slowly at first, then faster near the end. This means early timesteps make barely any change (good — the network has an easy target), while late timesteps still reach pure noise.

```
Linear ᾱ at t=250:  0.8910
Cosine ᾱ at t=250:  0.9431  ← more signal preserved

Linear ᾱ at t=750:  0.1104
Cosine ᾱ at t=750:  0.0455  ← destroys faster at the end
```

The cosine schedule works better for higher-resolution images because it spends more of the timestep budget on the "slightly noisy" range where the network can learn fine details. We use linear for MNIST, cosine for CIFAR-10.

### Signal-to-Noise Ratio (SNR)

The SNR at timestep t is:

```
SNR(t) = ᾱ_t / (1 - ᾱ_t)
```

This is the ratio of signal power to noise power. In decibels: `SNR_dB = 10 · log₁₀(SNR)`.

```
t=0:    SNR → ∞    (+40 dB)   pure signal
t=500:  SNR ≈ 1    (0 dB)     equal signal and noise
t=999:  SNR → 0    (-22 dB)   pure noise
```

The SNR monotonically decreases — the image gets progressively noisier. At SNR = 0 dB (around t=500 for linear schedule), signal and noise have equal power. A human can barely recognize the image at this point.

## Part 4: The Reparameterization Trick

The formula `x_t = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε` is a **reparameterization**. Instead of saying "x_t is sampled from a Gaussian distribution," we write x_t as a deterministic function of x₀ and ε, where ε is the only random part.

This matters for training: we need to compute gradients through x_t (since the network processes x_t). If x_t were defined as "sample from a distribution," we couldn't backpropagate through it. The reparameterization makes x_t a differentiable function of x₀ — the randomness is "factored out" into ε, which doesn't need gradients.

### Implementation

```python
def q_sample(self, x_0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_0)

    sqrt_alpha = self._gather(self.sqrt_alphas_cumprod, t)
    sqrt_one_minus_alpha = self._gather(self.sqrt_one_minus_alphas_cumprod, t)

    x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
    return x_t, noise
```

The `_gather` function handles batching — each image in the batch can be at a different timestep:

```python
def _gather(self, values, t):
    gathered = values.gather(0, t)
    return gathered.view(-1, 1, 1, 1)  # reshape for broadcasting with (B, C, H, W)
```

If the batch has 8 images with timesteps [100, 500, 250, 999, ...], we gather 8 different values of √ᾱ_t, reshape them to (8, 1, 1, 1), and broadcast across all pixels.

## Part 5: Precomputed Constants

During training, we need to compute `x_t = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε` millions of times. Computing ᾱ_t from scratch each time (multiplying t values together) would be wasteful. Instead, we precompute everything once:

```python
# From β_t, derive everything:
self.alphas = 1.0 - self.betas                              # α_t
self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)      # ᾱ_t
self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)    # √ᾱ_t
self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)  # √(1-ᾱ_t)
```

Each of these is a 1D tensor of length T=1000. Looking up the value for timestep t is a single array index — O(1).

We also precompute constants needed for the reverse process (Chapter 4):

```python
# 1/√α_t — used in the reverse step mean calculation
self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

# ᾱ_{t-1} with ᾱ₀ = 1
self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])

# Posterior variance: β̃_t = β_t · (1-ᾱ_{t-1}) / (1-ᾱ_t)
self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
```

These posterior values come from Bayes' rule applied to the reverse process — we'll derive them in Chapter 4.

## Part 6: What the Forward Process Looks Like

At key timesteps with the linear schedule:

```
t=0:     ᾱ=1.0000  The original image, untouched
t=100:   ᾱ=0.9811  Barely noisy — a human can't tell the difference
t=250:   ᾱ=0.8910  Slightly grainy, like a low-light photo
t=500:   ᾱ=0.4996  Clearly noisy, but shapes are visible
t=750:   ᾱ=0.1104  Very noisy — can barely make out the outline
t=999:   ᾱ=0.0061  Indistinguishable from random noise
```

A crucial property: at t=999, ᾱ_t ≈ 0, so x_t ≈ ε — the original image is completely buried in noise. This means our starting point for generation (pure Gaussian noise) is a valid sample from q(x_T), and the reverse process can begin.

## Part 7: Why Pixel Range [-1, 1]?

We normalize images to [-1, 1] before applying the forward process (not [0, 1]):

```python
transforms.Normalize((0.5,), (0.5,))  # maps [0,1] → [-1,1]
```

The forward process adds Gaussian noise centered at 0. If images were in [0, 1] (mean ~0.5), then at t=T the distribution of x_T would be centered around 0.5 × √ᾱ_T ≈ 0 but with a slight bias. Centering images around 0 makes x_T a clean N(0, I) distribution — no bias, no asymmetry.

This also means the neural network's output (predicted noise) is symmetric around 0, which is natural for Gaussian noise prediction.

## Summary

| Concept | Formula | What It Means |
|---------|---------|---------------|
| One step | x_t = √(1-β_t)·x_{t-1} + √β_t·ε | Shrink signal, add noise |
| Jump to any t | x_t = √ᾱ_t·x₀ + √(1-ᾱ_t)·ε | Direct from clean to any noise level |
| Signal retention | ᾱ_t = ∏α_s | Fraction of original image at step t |
| Linear schedule | β from 1e-4 to 0.02 | Simple, gradual noise increase |
| Cosine schedule | ᾱ_t = cos²(adjusted) | Preserves signal longer, better for RGB |
| SNR | ᾱ_t/(1-ᾱ_t) | Signal power vs noise power |
| Reparameterization | x_t = f(x₀, ε) | Differentiable — enables backprop |

## What's Next

In [Chapter 2](02_unet.md), we build the U-Net — the neural network that takes a noised image and a timestep, and predicts what noise was added. The architecture is an encoder-decoder with skip connections, processing at multiple resolutions to capture both global structure and fine detail.