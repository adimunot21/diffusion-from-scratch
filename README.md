# Diffusion Models from Scratch

A from-scratch implementation of Denoising Diffusion Probabilistic Models (DDPM) — the same core algorithm behind Stable Diffusion, DALL-E, and Diffusion Policy for robotics. The model learns to generate images by reversing a gradual noising process, starting from pure Gaussian noise.

## How It Works

```
┌──────────────────────────────────────────────────────┐
│              THE DIFFUSION PROCESS                    │
│                                                       │
│  FORWARD (destroy — fixed, no learning):              │
│  Clean image → add noise → add noise → ... → pure noise │
│       x₀    →    x₁     →    x₂     → ... →   x_T   │
│                                                       │
│  REVERSE (generate — learned by neural network):      │
│  Pure noise → remove noise → remove noise → ... → clean │
│      x_T    →   x_{T-1}  →   x_{T-2}  → ... →   x₀  │
│                                                       │
│  Training: "I added this noise. Can you predict it?"  │
│  Loss = MSE(predicted_noise, actual_noise)            │
└──────────────────────────────────────────────────────┘
```

Three components, each built from first principles:

1. **Forward Diffusion Process**: Gradually adds Gaussian noise to images over 1000 timesteps using a predefined schedule. The key insight: we can jump to any noise level in one step via the cumulative noise schedule.

2. **U-Net Denoiser**: An encoder-decoder neural network with skip connections that takes a noised image and timestep, and predicts the noise that was added. Processes at multiple resolutions to capture both global structure and fine detail. Self-attention layers (for CIFAR-10) capture long-range spatial relationships.

3. **Sampling Algorithms**: DDPM (1000 steps, highest quality) and DDIM (50 steps, 20× faster). DDIM's key insight: redefine the forward process as non-Markovian so steps can be skipped without quality loss.

4. **Classifier-Free Guidance**: Controls what the model generates. Train with 10% random label dropout, then amplify the conditional signal at inference. The same technique powering Stable Diffusion's text-to-image generation.

## Results

### MNIST Samples (50 epochs, 4.5M params)
Clean, diverse handwritten digits generated from pure noise.

### CIFAR-10 Conditional Samples (100 epochs, 28M params)
Class-conditional generation with classifier-free guidance (w=5.0). Each class generates recognizable objects with correct colors and compositions.

### FID Scores
| Model | FID | Parameters |
|-------|-----|-----------|
| MNIST (DDIM 50 steps) | **34.1** | 4.5M |
| CIFAR-10 Unconditional | **71.2** | 28M |
| CIFAR-10 Conditional (w=5.0) | **65.3** | 28M |
| Reference: DDPM paper (800 epochs) | 3.17 | 35M |

Guidance improves FID (65.3 < 71.2), confirming it genuinely increases quality.

### Key Findings
- **DDIM matches DDPM quality at 20× speed**: 50 DDIM steps ≈ 1000 DDPM steps
- **Classifier-free guidance works**: conditional FID 65.3 vs unconditional 71.2
- **Deterministic DDIM enables interpolation**: smooth morphing between digits in noise space
- **Cosine schedule preserves fine detail**: better than linear for RGB images

## Architecture

### U-Net (MNIST — 4.5M params)
```
Input: 1×28×28 → ConvBlock → 64×28×28
  → Encoder: [64, 128, 256] with ResBlocks, downsample 28→14→7
  → Bottleneck: 256×7×7
  → Decoder: [256, 128, 64] with skip connections, upsample 7→14→28
  → Output: 1×28×28 (predicted noise)
Time conditioning: sinusoidal embedding → MLP → injected into every ResBlock
```

### U-Net (CIFAR-10 — 28M params)
```
Input: 3×32×32 → ConvBlock → 128×32×32
  → Encoder: [128, 256, 256, 512] with ResBlocks + attention at 16×16 and 8×8
  → Bottleneck: 512×4×4
  → Decoder: mirror of encoder with skip connections
  → Output: 3×32×32 (predicted noise)
Self-attention at low resolutions for long-range spatial dependencies
```

## Project Structure

```
diffusion-from-scratch/
├── src/
│   ├── diffusion.py       ← Forward process, noise schedules (linear + cosine)
│   ├── unet.py            ← U-Net with optional self-attention
│   ├── train.py           ← MNIST training loop + EMA
│   ├── train_kaggle.py    ← MNIST GPU training script
│   ├── sample.py          ← DDPM and DDIM sampling, interpolation
│   ├── train_cifar.py     ← CIFAR-10 unconditional training
│   ├── conditional.py     ← Class conditioning + classifier-free guidance
│   └── evaluate.py        ← FID computation, visual analysis
├── checkpoints/           ← Saved models (gitignored)
├── notebooks/             ← Generated images, plots, loss curves
├── course/                ← Detailed written course (8 chapters)
└── requirements.txt
```

## Setup

```bash
conda create -n diffusion python=3.11 -y
conda activate diffusion
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install "numpy<2" matplotlib tqdm scipy
```

## Usage

```bash
# Run forward diffusion visualization
python -m src.diffusion

# Test U-Net architecture (shapes, gradients)
python -m src.unet

# Train on MNIST (use train_kaggle.py for GPU)
python -m src.train

# Generate images with DDPM and DDIM sampling
python -m src.sample

# Train CIFAR-10 unconditional (GPU recommended)
python -m src.train_cifar

# Train CIFAR-10 conditional with classifier-free guidance (GPU)
python -m src.conditional

# Run FID evaluation
python -m src.evaluate
```

## Training

| Model | Hardware | Time | Epochs |
|-------|----------|------|--------|
| MNIST | Kaggle T4 | ~15 min | 50 |
| CIFAR-10 Unconditional | Kaggle T4 | ~3-4 hrs | 100 |
| CIFAR-10 Conditional | Kaggle T4 | ~3-4 hrs | 100 |

## Course: Learn Diffusion Models From Scratch

This project includes a detailed written course explaining every concept, every equation, and every design decision.

| Chapter | Topic | What You'll Learn |
|---------|-------|-------------------|
| [0: Introduction](course/00_introduction.md) | What are diffusion models? | History (GANs → VAEs → DDPM), the three components, how diffusion differs from RL |
| [1: Forward Diffusion](course/01_forward_diffusion.md) | Destroying images with math | Noise schedules, ᾱ_t derivation, closed-form q(x_t\|x₀), reparameterization trick |
| [2: U-Net](course/02_unet.md) | Multi-scale denoising architecture | Encoder-decoder, skip connections, time conditioning, ResBlocks, GroupNorm, self-attention |
| [3: Training](course/03_training.md) | Teaching noise prediction | MSE loss, EMA for stable sampling, gradient clipping, why predict ε not x₀ |
| [4: Sampling](course/04_sampling.md) | Generating from noise | DDPM reverse process, DDIM skip-step trick, deterministic generation, interpolation |
| [5: Scaling to CIFAR-10](course/05_scaling_cifar.md) | RGB images with attention | Self-attention in U-Net, cosine schedule, computational considerations |
| [6: Conditional Generation](course/06_conditional_generation.md) | Controlling what to generate | Class embedding, label dropout, classifier-free guidance, connection to Stable Diffusion |
| [7: Evaluation](course/07_evaluation.md) | Measuring quality | FID metric, our results, what works, what could improve |

## Built With
- PyTorch (nn.Conv2d, nn.Linear, nn.GroupNorm, nn.Embedding — no diffusion libraries)
- NumPy, matplotlib, scipy
- Trained on Kaggle T4 GPU