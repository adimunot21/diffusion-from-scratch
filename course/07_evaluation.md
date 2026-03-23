# Chapter 7: Evaluation — Measuring What We Built

## The Problem

How do you know if a generative model is good? You can look at samples and say "those look like digits" or "that's clearly a horse," but human judgment is subjective, doesn't scale, and can't capture subtle quality differences. We need a quantitative metric.

## Part 1: FID — Fréchet Inception Distance

### The Intuition

FID compares two DISTRIBUTIONS of images, not individual images. It asks: "if I looked at the statistical properties of 1000 real images and 1000 generated images, how different are they?"

The key idea: don't compare raw pixels (two images of different cats have very different pixels but are both "good cats"). Instead, compare in a learned feature space where semantic similarity matters.

### Step by Step

**Step 1: Extract features.**
Pass both real and generated images through Inception-v3 (a pretrained ImageNet classifier). Extract features from the second-to-last layer (2048-dimensional). These features capture semantic content — "this looks like an animal" rather than "pixel (5,7) has value 0.3."

**Step 2: Fit Gaussians.**
Compute the mean vector μ and covariance matrix Σ for both sets of features:

```
Real images:      μ_real (2048-dim vector), Σ_real (2048×2048 matrix)
Generated images: μ_gen  (2048-dim vector), Σ_gen  (2048×2048 matrix)
```

**Step 3: Compute Fréchet distance.**
The Fréchet distance between two Gaussians N(μ₁, Σ₁) and N(μ₂, Σ₂) is:

```
FID = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2·(Σ₁·Σ₂)^½)
      \_________/   \________________________________/
       mean diff            covariance diff
```

The first term measures: are the average generated images similar to the average real images? (Do generated horses look like real horses on average?)

The second term measures: is the diversity of generated images similar to the diversity of real images? (Do we get the same variety of horse poses, colors, and backgrounds?)

### A Simplified Example

Imagine a 2D feature space where real images cluster in a circle centered at (5, 3) with radius 2, and generated images cluster at (5.5, 3.2) with radius 1.8.

```
Mean difference: ||(5,3) - (5.5,3.2)||² = 0.25 + 0.04 = 0.29
    Small — the clusters are centered in roughly the same place

Covariance difference: real images spread more (radius 2 vs 1.8)
    Small — diversity is similar

FID ≈ 0.29 + (small covariance term) ≈ low
    This is a good generative model!
```

Now imagine generated images cluster at (8, 1) with radius 0.5:

```
Mean difference: ||(5,3) - (8,1)||² = 9 + 4 = 13
    Large — generated images look systematically different

Covariance difference: radius 0.5 vs 2.0 — much less diverse
    Large — mode collapse, generating only one type of image

FID ≈ 13 + (large covariance term) ≈ high
    This is a bad generative model!
```

### FID Score Interpretation

```
FID ≈ 0:     Generated images are statistically identical to real images
FID < 10:    Excellent — SOTA diffusion models achieve this
FID < 30:    Good — high-quality, diverse generation
FID < 70:    Decent — recognizable but with visible artifacts or limited diversity
FID < 150:   Poor — blurry, wrong structure, or limited diversity
FID > 150:   Very poor — barely resembles real data
```

### Limitations

FID has known issues:
- **Inception bias:** Features come from a model trained on ImageNet. For domains far from ImageNet (medical images, satellite imagery), the features may not capture the right semantics.
- **Sample size:** FID is unreliable with fewer than ~1000 samples. We used 1000; papers use 10,000-50,000.
- **Resolution mismatch:** Inception expects 299×299 input. Our 32×32 CIFAR images get upscaled, introducing interpolation artifacts. Our FID numbers aren't directly comparable to papers that compute FID on higher-resolution outputs.
- **No per-sample quality:** FID is a distributional metric. A model could generate 990 perfect images and 10 garbage images and still get a good FID. It doesn't catch individual failures.

Despite these limitations, FID is the standard metric because it captures both quality AND diversity in a single number, and relative comparisons (model A vs model B on the same data) are reliable.

## Part 2: Our Results

### FID Scores

```
Model                              FID    Parameters
─────────────────────────────────────────────────────
MNIST (DDIM 50 steps)             34.1    ~4.5M
CIFAR-10 Unconditional            71.2    ~28M
CIFAR-10 Conditional (w=5.0)      65.3    ~28M
─────────────────────────────────────────────────────
Reference: DDPM paper (CIFAR-10)   3.17   ~35M (800 epochs)
Reference: Improved DDPM           2.94   ~35M
```

### What These Numbers Mean

**MNIST FID 34.1:** Good for a learning project. Our generated digits are clean and diverse, covering all 10 digit classes. The gap from SOTA (~1-5) comes from limited training time and model size. The Inception features also aren't ideal for MNIST (they were trained on natural images, not handwritten digits).

**CIFAR-10 Unconditional FID 71.2:** Decent. Objects are recognizable with correct colors and compositions, but blurry at 32×32 and lacking fine texture detail. The gap from SOTA (3.17) comes from training for 100 epochs instead of 800, and using 28M params instead of 35M.

**CIFAR-10 Conditional FID 65.3:** Lower (better) than unconditional, confirming that classifier-free guidance improves quality. The class condition gives the network additional information, letting it commit to class-specific features rather than hedging. This is the key finding: guidance helps even with a relatively small model.

### Why Our FID Is Higher Than Papers

The primary factors, in order of impact:

1. **Training duration.** We trained for 100 epochs (~39K steps). The DDPM paper trained for ~800K steps — roughly 20× more. More training means the network sees more diverse noise examples and learns subtler patterns.

2. **Model size.** Our 28M params vs their 35M. More parameters means more capacity for complex patterns like textures and fine edges.

3. **FID computation.** We used 1000 samples; papers use 50,000. More samples give a more reliable FID estimate. We also upscaled 32×32 to 299×299 for Inception, which papers handle more carefully.

4. **Hyperparameter tuning.** Papers typically tune learning rate, EMA decay, noise schedule, and other hyperparameters extensively. We used standard defaults.

## Part 3: Visual Analysis

### DDIM Step Count vs Quality

Our step count analysis showed:

```
10 steps:   Recognizable digits but noticeably rough edges, some artifacts
20 steps:   Good quality, minor imperfections
50 steps:   Near-identical to DDPM quality
100 steps:  Indistinguishable from 1000-step DDPM
200 steps:  Identical to 100 steps (no further improvement)
```

The practical sweet spot is 50 steps — 20× faster than DDPM with negligible quality loss. Below 20 steps, quality degrades visibly. Above 100, there's no improvement.

### Real vs Generated Comparison

Side-by-side comparisons of real and generated images reveal:

**What the model does well:**
- Overall structure (digit shapes, object silhouettes)
- Color palette (blue skies, green grass, correct car colors)
- Composition (objects centered, backgrounds consistent)
- Class diversity (generates all 10 CIFAR classes with variety)

**What the model struggles with:**
- Fine textures (fur, feathers, metal reflections are blurry)
- Sharp edges (boundaries between objects and backgrounds are soft)
- Small details (eyes, wheels, windows lack definition)
- Complex scenes (images with multiple objects or complex backgrounds)

These limitations are primarily resolution and training time constraints, not fundamental algorithmic issues. The same architecture trained longer at higher resolution would produce dramatically sharper images.

## Part 4: What We Built — The Complete System

```
┌──────────────────────────────────────────────────────────────┐
│                    DIFFUSION FROM SCRATCH                     │
│                                                              │
│  Forward Process (Chapter 1)                                 │
│  ├── Linear + cosine noise schedules                        │
│  ├── Closed-form q(x_t|x₀) = N(√ᾱ_t·x₀, (1-ᾱ_t)I)       │
│  └── Precomputed ᾱ_t, √ᾱ_t, √(1-ᾱ_t) for all T steps     │
│                                                              │
│  U-Net Denoiser (Chapter 2)                                  │
│  ├── Sinusoidal time embedding → MLP                         │
│  ├── ResBlocks with time conditioning                        │
│  ├── Encoder (downsample) + Decoder (upsample) + skips       │
│  ├── Self-attention at low resolutions (CIFAR)               │
│  └── GroupNorm, SiLU, strided conv down, nearest-neighbor up │
│                                                              │
│  Training (Chapter 3)                                        │
│  ├── Loss = MSE(ε_pred, ε) — predict the noise              │
│  ├── Random timesteps per image                              │
│  ├── EMA for stable sampling weights                         │
│  └── Gradient clipping for training stability                │
│                                                              │
│  Sampling (Chapter 4)                                        │
│  ├── DDPM: 1000-step reverse process (highest quality)       │
│  ├── DDIM: 50-step skip sampling (20× faster)               │
│  ├── Deterministic DDIM (η=0) enables interpolation          │
│  └── Denoising trajectory visualization                      │
│                                                              │
│  Scaling (Chapter 5)                                         │
│  ├── MNIST → CIFAR-10 (grayscale → RGB, 28×28 → 32×32)     │
│  ├── Self-attention for long-range dependencies              │
│  └── Cosine schedule for better fine-detail preservation     │
│                                                              │
│  Conditional Generation (Chapter 6)                          │
│  ├── Class embedding added to time embedding                 │
│  ├── 10% label dropout during training                       │
│  ├── Classifier-free guidance: ε_uncond + w·(ε_cond - ε_uncond)│
│  └── Guidance scale w controls quality vs diversity          │
│                                                              │
│  Evaluation (Chapter 7)                                      │
│  ├── FID: MNIST 34.1, CIFAR uncond 71.2, CIFAR cond 65.3   │
│  ├── Guidance improves FID (65.3 < 71.2)                    │
│  └── 50 DDIM steps ≈ 1000 DDPM steps in quality             │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## Part 5: What Could Be Improved

### More Training

The single largest improvement would come from training longer. Our CIFAR models trained for 100 epochs (~39K steps). The DDPM paper used ~800K steps. More training would sharpen textures, improve fine details, and significantly lower FID.

### Larger Model

Going from 28M to 35M+ params would add capacity for complex patterns. More residual blocks (3 instead of 2 per level) and wider channels (192, 384, 384, 768) would capture finer details.

### Better Sampling

Newer samplers like DPM-Solver++ can match DDPM quality in 10-20 steps instead of 50, by using higher-order ODE solvers on the diffusion process.

### Higher Resolution

Training at 64×64 or 128×128 would dramatically improve visual quality. This requires a larger model and longer training but no algorithmic changes. Modern systems like Stable Diffusion generate at 512×512 or 1024×1024 using the same core algorithm at much larger scale.

## Course Complete

You've built the complete diffusion pipeline from scratch:

1. **A forward process** that mathematically destroys images through controlled Gaussian noise addition, with the key insight that any timestep is reachable in one step via the cumulative noise schedule.

2. **A U-Net denoiser** that processes images at multiple scales with skip connections, conditioned on the noise level through sinusoidal time embeddings, with optional self-attention for long-range spatial relationships.

3. **A training loop** that teaches the network to predict noise using simple MSE loss — no labels, no rewards, no adversarial dynamics. EMA smooths the sampling weights for stable generation.

4. **Two sampling algorithms** — DDPM (1000 steps, highest quality) and DDIM (50 steps, 20× faster) — that reverse the forward process to generate new images from pure Gaussian noise.

5. **Conditional generation** using classifier-free guidance — the same technique that powers Stable Diffusion, DALL-E, and Diffusion Policy for robotics. Random label dropout during training enables amplified conditioning at inference.

6. **Quantitative evaluation** using FID to measure the statistical similarity between generated and real image distributions.

The same mathematical framework — forward diffusion, noise prediction, iterative denoising — is what powers the most capable generative AI systems today. The only differences between our implementation and Stable Diffusion are scale (model size, resolution, dataset, compute) and the type of conditioning (class labels vs text embeddings). The principles are identical.