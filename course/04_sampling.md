# Chapter 4: Sampling — Generating Images from Pure Noise

## The Problem

We have a network trained to predict noise at any timestep. Now we need to run the reverse process: start from pure noise x_T ~ N(0, I) and iteratively denoise until we get a clean image x₀.

This chapter covers two algorithms: DDPM (the original, 1000 steps) and DDIM (the fast version, 50 steps). Understanding both reveals why DDIM was such an important breakthrough.

## Part 1: DDPM Sampling — The Full Reverse Process

### The Reverse Step

At each step, we want to compute p(x_{t-1} | x_t) — the distribution of the slightly-less-noisy image given the current noisy image. Bayes' rule gives us the posterior:

```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ_t² I)
```

The mean μ_θ is computed using the network's noise prediction:

```
μ_θ(x_t, t) = (1/√α_t) · (x_t - β_t/√(1-ᾱ_t) · ε_θ(x_t, t))
```

The variance σ_t² is fixed (not learned):

```
σ_t² = β̃_t = β_t · (1-ᾱ_{t-1}) / (1-ᾱ_t)
```

### Deriving the Mean Formula

This comes from the posterior q(x_{t-1} | x_t, x₀), which we can compute in closed form:

Since x_t = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε, we can solve for x₀:

```
x₀ = (x_t - √(1-ᾱ_t) · ε) / √ᾱ_t
```

But we don't know the true ε — that's what the network predicts. Substituting ε_θ (the network's prediction):

```
x̂₀ = (x_t - √(1-ᾱ_t) · ε_θ) / √ᾱ_t
```

Plugging this estimated x₀ into the posterior mean formula and simplifying gives us the expression above. The key insight: the network never directly predicts x_{t-1}. It predicts the noise, from which we compute x₀, from which we compute the posterior mean of x_{t-1}.

### A Worked Example (Single Pixel)

Current state: x_t = 0.35 at timestep t=500.

```
Noise schedule at t=500:
  α₅₀₀ = 0.9899,  β₅₀₀ = 0.0101
  ᾱ₅₀₀ = 0.4996,  √ᾱ₅₀₀ = 0.7068
  √(1-ᾱ₅₀₀) = 0.7074
  β̃₅₀₀ = 0.0101 · (1-ᾱ₄₉₉)/(1-ᾱ₅₀₀) ≈ 0.0101

Network predicts: ε_θ = 0.82

Posterior mean:
  μ = (1/√0.9899) · (0.35 - 0.0101/0.7074 · 0.82)
    = 1.0051 · (0.35 - 0.01428 · 0.82)
    = 1.0051 · (0.35 - 0.01171)
    = 1.0051 · 0.3383
    = 0.340

Sample x_{t-1}:
  x₄₉₉ = 0.340 + √0.0101 · z    where z ~ N(0,1)
        = 0.340 + 0.1005 · z

If z = 0.3:  x₄₉₉ = 0.370
If z = -0.5: x₄₉₉ = 0.290
```

The pixel moved slightly — from 0.35 to ~0.34 (mean), with small random variation. One step barely changes anything. But after 1000 such steps, starting from random noise, a coherent image emerges.

### The Complete DDPM Sampling Algorithm

```python
x = torch.randn(num_images, 1, 28, 28)  # Start from pure noise

for t_val in reversed(range(1000)):      # t = 999, 998, ..., 1, 0
    t = torch.full((num_images,), t_val)
    
    noise_pred = model(x, t)             # Predict noise
    
    # Compute posterior mean
    mean = sqrt_recip_alphas[t_val] * (
        x - betas[t_val] / sqrt_one_minus_alphas_cumprod[t_val] * noise_pred
    )
    
    if t_val > 0:
        x = mean + sqrt(posterior_variance[t_val]) * torch.randn_like(x)
    else:
        x = mean  # No noise at the final step
```

The `if t_val > 0` check is crucial: at the very last step (t=0 → clean image), we don't add noise. We want the final output to be deterministic given the mean.

### Why 1000 Steps Is Necessary for DDPM

Each step removes a tiny amount of noise — the same tiny amount that was added in the corresponding forward step. The noise schedule determines the step size: β_t ranges from 0.0001 to 0.02, so each reverse step makes only a 0.01-2% change.

Skipping steps would mean jumping from, say, t=500 to t=400 in one step — but the network was trained to go from t to t-1, not t to t-100. The noise prediction at t=500 tells you what to do for the tiny step from 500 to 499, not the large step from 500 to 400. Using it for a large step produces artifacts.

This is DDIM's key insight: redefine the process so large steps ARE valid.

## Part 2: DDIM — The Fast Sampler

### The Key Insight

DDPM defines the forward process as a Markov chain: each step depends only on the previous step. x_t depends on x_{t-1}, which depends on x_{t-2}, and so on. This forces 1000 sequential steps.

Song et al. (2020) observed: the neural network was trained on the MARGINAL distribution q(x_t | x₀) = N(√ᾱ_t · x₀, (1-ᾱ_t) · I), not the transition distribution q(x_t | x_{t-1}). The marginals don't care about the path between timesteps — only about the endpoint.

This means we can define a DIFFERENT forward process that has the SAME marginals but a DIFFERENT path. Specifically, a non-Markovian process where x_t depends on both x_{t-1} AND x₀. This different path allows larger steps.

### The DDIM Update Rule

Instead of the Markov chain "x_t → x_{t-1}", DDIM goes through x₀ as an intermediate:

```
Step 1: Predict x₀ from x_t
    x̂₀ = (x_t - √(1-ᾱ_t) · ε_θ) / √ᾱ_t

Step 2: "Re-noise" x̂₀ to a DIFFERENT timestep (possibly far away)
    x_{t-Δ} = √ᾱ_{t-Δ} · x̂₀ + √(1-ᾱ_{t-Δ}-σ²) · ε_θ + σ · z
```

The parameter σ controls stochasticity:
- σ = 0: fully deterministic (DDIM)
- σ = √(β̃_t): matches DDPM's noise level

### The Simplified DDIM Formula (η = 0)

When η = 0 (deterministic mode), σ = 0 and the formula simplifies:

```
x₀_pred = (x_t - √(1-ᾱ_t) · ε_θ) / √ᾱ_t         "predict clean image"
dir_xt = √(1-ᾱ_{t_prev}) · ε_θ                      "direction toward x_t"
x_{t_prev} = √ᾱ_{t_prev} · x₀_pred + dir_xt         "go to previous timestep"
```

The crucial insight: **t_prev can be ANY earlier timestep**, not just t-1. The network's noise prediction at timestep t is valid for computing x₀_pred regardless of where we go next. We can jump from t=900 to t=880 (small step) or t=900 to t=700 (large step) with the same prediction.

### Timestep Subsequence

For 50 DDIM steps across 1000 timesteps:

```python
step_size = 1000 // 50  # = 20
timestep_seq = [0, 20, 40, 60, ..., 960, 980]
```

We walk backward: 980 → 960 → 940 → ... → 40 → 20 → 0. Each jump skips 20 timesteps, using only 50 network evaluations instead of 1000.

### A Worked Example

At t=500, with t_prev=480 (DDIM step size 20):

```
x_t = 0.35    (current noisy value)

ᾱ₅₀₀ = 0.4996,  √ᾱ₅₀₀ = 0.7068,  √(1-ᾱ₅₀₀) = 0.7074
ᾱ₄₈₀ = 0.5398,  √ᾱ₄₈₀ = 0.7347,  √(1-ᾱ₄₈₀) = 0.6784

Network predicts: ε_θ = 0.82

Step 1 — predict x₀:
  x̂₀ = (0.35 - 0.7074 · 0.82) / 0.7068
      = (0.35 - 0.5801) / 0.7068
      = -0.2301 / 0.7068
      = -0.3256

Step 2 — compute direction:
  dir = √(1 - 0.5398) · 0.82 = 0.6784 · 0.82 = 0.5563

Step 3 — go to t=480:
  x₄₈₀ = 0.7347 · (-0.3256) + 0.5563
        = -0.2392 + 0.5563
        = 0.3171
```

The pixel moved from 0.35 to 0.317 — a single DDIM step handling what would take 20 DDPM steps.

## Part 3: DDIM Determinism and Interpolation

### Deterministic Generation (η = 0)

With η = 0, DDIM has no random noise injection. The same starting noise z always produces the same image:

```
z = torch.randn(1, 1, 28, 28)  →  always produces the same digit
```

We verified this in our tests: two runs with the same seed produced images with maximum pixel difference < 1e-5 — essentially identical.

### Noise Space Interpolation

Because DDIM is deterministic, each point in noise space maps to exactly one image. We can smoothly traverse between two images by interpolating their noise vectors:

```
z₁ → image A (e.g., a "4")
z₂ → image B (e.g., a "7")

For α from 0 to 1:
  z_interp = (1-α) · z₁ + α · z₂
  z_interp → smoothly morphing image from A to B
```

Our interpolation results showed smooth transitions — the 4 gradually transforms its structure through intermediate shapes into a different digit. This only works because the mapping from noise to image is continuous and well-behaved.

With DDPM (stochastic), the same noise vector produces different images each run, making interpolation meaningless.

## Part 4: DDIM vs DDPM — Quality Comparison

From our experiments:

```
DDPM (1000 steps): Clean, sharp digits. ~30-60s for 16 images on CPU.
DDIM (200 steps):  Nearly identical quality. ~6-10s.
DDIM (50 steps):   Very similar quality, tiny quality loss. ~2-3s.
DDIM (20 steps):   Slight quality degradation. ~1s.
DDIM (10 steps):   Noticeable quality loss but still recognizable. ~0.5s.
```

The sweet spot is DDIM with 50 steps: 20× faster than DDPM with negligible quality difference. This is why DDIM is the standard sampler in practice.

### Why Quality Barely Degrades

The network's noise prediction at timestep t is based on the marginal distribution q(x_t | x₀), which is the same regardless of how we got to x_t. Whether we arrived at t=500 from t=501 (DDPM) or from t=520 (DDIM with step size 20), the noise prediction is equally valid. The network doesn't care about the path — it only cares about the current state.

The small quality difference at very low step counts (10 steps) comes from the x₀ prediction becoming less accurate at high noise levels. With fewer steps, each step covers a larger range, and the x₀ prediction from near-pure noise (high t) is inherently noisier.

### The x₀ Clamp Bug

In our initial implementation, we clamped x₀_pred to [-1, 1] at every step:

```python
x0_pred = x0_pred.clamp(-1, 1)  # THIS CAUSED BUGS
```

At high noise timesteps, x₀_pred can legitimately be outside [-1, 1] — the prediction is uncertain, and clamping introduces a systematic bias toward zero. With 10 steps, 10 small biases are negligible. With 200 steps, 200 biases accumulated and produced blob artifacts — every image collapsed to a similar amorphous shape.

The fix was simple: remove the clamp from intermediate steps. The final output still gets clamped to [0, 1] for display, but intermediate x₀_pred values are left unclamped.

This is a good lesson in debugging generative models: a bug can produce plausible-looking but wrong outputs at one setting (10 steps) and completely wrong outputs at another (200 steps). Always test across a range of parameters.

## Part 5: The Denoising Trajectory

Our trajectory visualization shows what happens at each step during DDIM sampling:

```
t=1000: Pure random noise. No structure visible.
t=800:  Still noise. Maybe very faint large-scale patterns.
t=600:  Noise, but with a hint of brightness distribution.
t=400:  Something emerging — a bright region where the digit will be.
t=300:  Recognizable digit shape, but noisy and rough.
t=200:  Clear digit with some noise on the edges.
t=100:  Clean digit, minor artifacts.
t=0:    Final clean digit.
```

The interesting thing: the digit's identity is "decided" early (around t=400-300), but the fine details (stroke thickness, exact curves) are refined in the last steps (t=100-0). This mirrors the hierarchical structure of image generation — large-scale decisions first, details last.

## Summary

| Aspect | DDPM | DDIM (η=0) |
|--------|------|------------|
| Steps | 1000 (all timesteps) | 50 (every 20th timestep) |
| Speed | ~30-60s per batch | ~2-3s per batch |
| Quality | Highest | Nearly identical |
| Stochastic? | Yes (different each run) | No (deterministic) |
| Interpolation | Not possible | Smooth, meaningful |
| Core formula | μ_θ + σ·z (Gaussian step) | √ᾱ·x̂₀ + dir (predict-then-re-noise) |

| Concept | What | Why |
|---------|------|-----|
| DDPM reverse step | Posterior mean + Gaussian noise | Follows the original Markov chain backward |
| DDIM skip steps | Predict x₀, re-noise to earlier t | Marginals are identical, path doesn't matter |
| Deterministic DDIM | σ = 0, no noise injection | Same noise → same image, enables interpolation |
| Timestep subsequence | [0, 20, 40, ..., 980] | 50 steps instead of 1000, 20× faster |

## What's Next

In [Chapter 5](05_scaling_cifar.md), we scale from grayscale 28×28 MNIST to color 32×32 CIFAR-10. The key addition is self-attention in the U-Net — letting distant pixels communicate directly, which is essential for coherent structure in natural images.