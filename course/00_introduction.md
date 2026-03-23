# Chapter 0: Introduction — Generating Images from Noise

## The Problem

You want a computer to create new images — images that don't exist yet, but look like they came from a real dataset. Not copying or blending existing images, but genuinely generating new ones from scratch.

This is fundamentally different from everything else in machine learning. Classification asks "what is this?" Reinforcement learning asks "what should I do?" Generation asks "what could exist?" The output isn't a label or an action — it's a high-dimensional, continuous object (an image) that must look natural to a human observer.

## The Idea That Makes Diffusion Work

Imagine taking a photograph and gradually adding TV static to it. At first, you can still see the photo clearly. A bit more static, and it gets grainy. More, and you can barely make out shapes. Eventually, it's pure static — the original image is completely gone.

Now imagine you could train a neural network to reverse this process. Show it a slightly noisy image, and it learns to remove the noise. Show it a very noisy image, and it learns to reconstruct what the original might have looked like. If this network gets good enough, you could start from **pure random static** and denoise it step by step, and something recognizable would emerge — not a copy of any training image, but a new image that follows the same patterns the network learned.

This is diffusion.

```
FORWARD PROCESS (destroy):
Clean image → slightly noisy → more noisy → ... → pure random noise
    x₀     →      x₁       →     x₂     → ... →      x_T

REVERSE PROCESS (generate):
Pure noise → slightly less noisy → ... → almost clean → clean image
    x_T    →       x_{T-1}      → ... →     x₁      →     x₀
```

The forward process is fixed — just adding Gaussian noise according to a schedule. No learning involved. The reverse process is where the neural network comes in: it learns to predict and remove the noise at each step.

## A Brief History

### 2014: GANs (Generative Adversarial Networks)

Two networks competing: a Generator creates fake images, a Discriminator tries to tell real from fake. They push each other to improve. GANs dominated image generation for years but suffered from training instability (mode collapse, vanishing gradients) and couldn't easily cover the full diversity of a dataset.

### 2013-2020: VAEs (Variational Autoencoders)

Encode images into a compressed latent space, then decode back. The latent space is forced to be smooth (Gaussian), so you can sample new points and decode them into new images. Easier to train than GANs but produced blurrier images.

### 2020: DDPM — Diffusion Takes Over

Ho, Jain, and Abbeel published "Denoising Diffusion Probabilistic Models" (DDPM). The core idea: model the data distribution by learning to reverse a gradual noising process. The key advantages:

- **Stable training.** The loss is simple MSE — "predict the noise I added." No adversarial dynamics, no mode collapse.
- **High quality.** Matched or exceeded GAN quality on image benchmarks.
- **Full diversity.** The stochastic sampling process naturally covers the entire data distribution — no mode collapse.

The downside: slow sampling. Generating one image required 1000 sequential neural network calls. This was addressed by DDIM (Song et al. 2020), which showed you could skip steps and sample in 50 steps with nearly identical quality.

### 2021-Present: Diffusion Everywhere

Diffusion became the dominant generative framework:

- **DALL-E 2, Imagen, Stable Diffusion** — text-to-image generation
- **Video generation** — Sora, Runway
- **Audio** — music and speech generation
- **Robotics** — Diffusion Policy (Chi et al. 2023) treats action generation as denoising, achieving state-of-the-art robot manipulation
- **Protein design** — RFdiffusion generates novel protein structures
- **3D** — generating 3D models and scenes

## The Three Components We'll Build

### 1. The Forward Process (Chapter 1)

The mathematical foundation. We define exactly how to add noise — how much at each step, how the noise accumulates, and the key insight that lets us jump to any noise level in one operation (instead of applying noise step by step).

```
x_t = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε     where ε ~ N(0, I)

At t=0:   √ᾱ ≈ 1.0,  √(1-ᾱ) ≈ 0.0  →  x_t ≈ x₀         (clean image)
At t=500: √ᾱ ≈ 0.7,  √(1-ᾱ) ≈ 0.7  →  x_t = 70% image + 70% noise
At t=999: √ᾱ ≈ 0.08, √(1-ᾱ) ≈ 1.0  →  x_t ≈ ε           (pure noise)
```

### 2. The U-Net Denoiser (Chapter 2)

The neural network that predicts noise. Unlike the ResNet from AlphaZero (which processed at a single resolution), the U-Net operates at multiple scales — downsampling to capture global structure ("what digit is this?"), then upsampling to reconstruct per-pixel detail ("what noise was added to this specific pixel?"). Skip connections carry fine details from encoder to decoder.

```
Input (noised image + timestep)
    │
    ▼
┌──────────┐     ┌──────────────────────┐
│ Encoder   │ ──→ │ Skip connections     │
│ 28→14→7   │     │ (preserve details)   │
└──────────┘     └────────┬─────────────┘
    │                      │
    ▼                      ▼
┌──────────┐     ┌──────────────────────┐
│ Bottleneck│     │ Decoder              │
│ (7×7)     │ ──→ │ 7→14→28             │
└──────────┘     └──────────────────────┘
    │
    ▼
Output (predicted noise, same size as input)
```

### 3. The Training Loop (Chapter 3)

Beautifully simple: take a clean image, add random noise at a random timestep, ask the network to predict what noise was added. Loss = MSE between predicted and actual noise. No labels needed, no reward signals, no adversarial training.

```python
# The entire training step in pseudocode:
x_0 = random_training_image()
t = random_timestep()
noise = random_gaussian_noise()
x_t = add_noise(x_0, t, noise)     # forward process
noise_pred = network(x_t, t)        # predict the noise
loss = MSE(noise_pred, noise)        # simple MSE loss
loss.backward()                      # update the network
```

### 4. Sampling (Chapter 4)

Two algorithms for generating images. DDPM walks through all 1000 steps — slow but highest quality. DDIM skips steps for 20× speed with negligible quality loss. DDIM also enables deterministic generation (same noise → same image), unlocking smooth interpolation between images.

### 5. Scaling to CIFAR-10 (Chapter 5)

Moving from grayscale 28×28 digits to color 32×32 natural images. The key addition: self-attention layers in the U-Net, letting distant pixels communicate directly instead of only through stacked convolutions.

### 6. Conditional Generation (Chapter 6)

Controlling **what** the model generates. Classifier-free guidance — the same technique behind Stable Diffusion's text conditioning — lets us say "generate a horse" or "generate an airplane" and the model obeys. The trick: randomly drop the condition during training, then amplify it during sampling.

### 7. Evaluation (Chapter 7)

How to measure generative quality. FID (Fréchet Inception Distance) — the standard metric — compares the statistical distribution of generated images to real images. Our results: MNIST FID 34.1, CIFAR-10 unconditional FID 71.2, CIFAR-10 conditional FID 65.3.

## What Makes Diffusion Different From Your Previous Projects

| Aspect | AlphaZero / RL | Diffusion |
|--------|---------------|-----------|
| Goal | Choose actions to maximize reward | Generate new data samples |
| Output | Discrete action or value | Continuous image (784-3072 dimensions) |
| Training signal | Game outcome, reward | "Predict the noise I added" (self-supervised) |
| Inference | Single forward pass | 50-1000 iterative denoising steps |
| Architecture | ResNet (single scale) | U-Net (multi-scale encoder-decoder) |
| Math foundation | Bellman equations, policy gradients | Gaussian distributions, variational inference |

The biggest conceptual shift: in RL, the network makes one decision per forward pass. In diffusion, generating a single image requires 50-1000 sequential forward passes, each removing a small amount of noise. The network is used as a building block in an iterative algorithm, not as a standalone decision-maker.

## Setup

### Create Environment

```bash
conda create -n diffusion python=3.11 -y
conda activate diffusion
```

### Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install "numpy<2" matplotlib tqdm scipy
```

### Project Structure

```
diffusion-from-scratch/
├── src/
│   ├── diffusion.py       ← Forward process, noise schedules
│   ├── unet.py            ← U-Net denoiser architecture
│   ├── train.py           ← MNIST training loop + EMA
│   ├── train_kaggle.py    ← MNIST GPU training
│   ├── sample.py          ← DDPM and DDIM sampling
│   ├── train_cifar.py     ← CIFAR-10 unconditional training
│   ├── conditional.py     ← Class conditioning + classifier-free guidance
│   └── evaluate.py        ← FID computation, analysis
├── checkpoints/           ← Saved model weights
├── notebooks/             ← Generated images, plots
└── course/                ← These course files
```

## What's Next

In [Chapter 1](01_forward_diffusion.md), we build the forward diffusion process — the mathematical foundation that defines exactly how images are destroyed by noise. This is pure math, no neural networks yet. Understanding it deeply is essential because the reverse process (generation) is derived directly from the forward process.