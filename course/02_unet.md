# Chapter 2: The U-Net — A Neural Network That Sees at Multiple Scales

## The Problem

The neural network receives two inputs — a noised image x_t and a timestep t — and must predict the noise ε that was added. The output has the **exact same shape** as the input (both are images), because noise exists at every pixel.

This is a "dense prediction" task — we need to produce a value for every pixel, not a single class label. Classification networks (like ResNets) reduce spatial dimensions to a single vector. We need the opposite: maintain spatial detail while also understanding global structure.

Consider what the network needs to do: at timestep t=100 (barely noisy), it needs to predict tiny, fine-grained noise — small pixel-level perturbations. At t=900 (very noisy), it needs to first figure out what the image might be (a 3? a 7?) and then predict the large-scale noise pattern. This requires understanding at multiple scales simultaneously.

## Part 1: Why U-Net, Not ResNet?

In AlphaZero, the ResNet processed the board at a single resolution (6×7) throughout. Every residual block operated on 64-channel feature maps of the same spatial size.

The U-Net operates differently:

```
ResNet (AlphaZero):
    (64, 6, 7) → ResBlock → (64, 6, 7) → ResBlock → (64, 6, 7) → ...
    Same resolution throughout — works for classification/value estimation

U-Net (Diffusion):
    (64, 28, 28) → Down → (128, 14, 14) → Down → (256, 7, 7)    [Encoder]
                                                      │
                                                      ▼
                                                  (256, 7, 7)      [Bottleneck]
                                                      │
    (64, 28, 28) ← Up ← (128, 14, 14) ← Up ← (256, 7, 7)       [Decoder]
    
    Multi-resolution — captures both global and local patterns
```

The encoder progressively downsamples, trading spatial resolution for more channels (more features per position). At the lowest resolution (7×7 for MNIST), each position has a wide receptive field — it "sees" a large portion of the original image. This is where the network understands global structure.

The decoder upsamples back to full resolution, reconstructing spatial detail. But upsampling alone would lose the fine details that were discarded during downsampling. Skip connections solve this.

### Skip Connections: The U-Net's Key Innovation

At each resolution level, the encoder's output is concatenated with the decoder's input:

```
Encoder output (14×14, 128ch) ──────── concatenate ────→ Decoder input (14×14, 256+128ch)
                                                          │
                                                    Decoder ResBlocks
                                                          │
                                                    (14×14, 128ch)
```

The encoder features carry fine spatial details ("there's a sharp edge at position (5, 12)"). The decoder features carry semantic understanding from the bottleneck ("this is probably a 3"). Concatenating them gives the decoder both — it can reconstruct detail-aware predictions.

Without skip connections, the decoder would need to hallucinate fine details from the low-resolution bottleneck — like trying to enlarge a tiny thumbnail while guessing what the sharp edges looked like.

## Part 2: Time Conditioning — Telling the Network the Noise Level

### Why the Network Needs the Timestep

The network must behave differently at different noise levels:

- **t=10 (barely noisy):** The image is almost clean. The noise is tiny, high-frequency perturbations. The network should predict small values.
- **t=500 (half noisy):** The image is recognizable but grainy. The network needs to separate structural patterns from noise.
- **t=990 (almost pure noise):** The image is mostly noise. The network must rely on global statistical patterns to guess what noise looks like in the context of "this might be a digit."

A single network handles all 1000 noise levels. The timestep tells it which "mode" to operate in.

### Sinusoidal Embedding

We encode the integer timestep using the same sinusoidal formula as Transformer positional encoding:

```
PE(t, 2i)   = sin(t / 10000^(2i/d))
PE(t, 2i+1) = cos(t / 10000^(2i/d))
```

For `time_dim = 128`, this creates a 128-dimensional vector from a single integer. Each dimension oscillates at a different frequency:

```
Dimension 0-1:   fast oscillation    (changes rapidly with t)
Dimension 62-63: slow oscillation    (changes slowly with t)
Dimension 126-127: very slow         (barely changes across 1000 steps)
```

This gives the network a rich representation where nearby timesteps (t=100 and t=101) have similar embeddings, but distant timesteps (t=100 and t=900) have very different embeddings.

### From Raw Embedding to Useful Features

The sinusoidal values are fixed (not learned). We pass them through a small MLP to create learnable time features:

```python
self.mlp = nn.Sequential(
    nn.Linear(time_dim, time_dim * 4),   # 128 → 512
    nn.SiLU(),                            # non-linearity
    nn.Linear(time_dim * 4, time_dim),   # 512 → 128
)
```

Shape trace for a batch of 4 timesteps:

```
Input timesteps:     (4,)         integers like [10, 250, 500, 990]
Sinusoidal encoding: (4, 128)     fixed sin/cos features
After MLP:           (4, 128)     learned time features
```

The MLP lets the network learn which aspects of the timestep matter most. Maybe the raw sinusoidal embedding overemphasizes low frequencies — the MLP can reweight them.

## Part 3: The ResBlock — Core Building Block

Each ResBlock processes features with two convolutions, a skip connection, and time injection:

```python
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels))

        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_conv = nn.Identity()
```

The data flow for input (batch=1, 64 channels, 14×14) with time_dim=128:

```
x: (1, 64, 14, 14) ───────────────────────────────────────┐
    │                                                       │ skip_conv
    ▼                                                       │ (1×1 conv if channels change)
  GroupNorm(64) → SiLU → Conv3×3(64→128)                   │
    │                                                       │
    ▼ (1, 128, 14, 14)                                     │
  + time_proj(t_emb): (1, 128) → reshape → (1, 128, 1, 1)  │
    │                    broadcasts across 14×14             │
    ▼                                                       │
  GroupNorm(128) → SiLU → Conv3×3(128→128)                  │
    │                                                       │
    ▼ (1, 128, 14, 14)                                     │
  + ────────────────────────────────────────────────────────┘
    │
    ▼
  Output: (1, 128, 14, 14)
```

### Why GroupNorm, Not BatchNorm?

BatchNorm computes mean and variance across the batch dimension. During sampling, we often generate one image at a time (batch size = 1). With batch size 1, BatchNorm's statistics are meaningless — the mean of one sample is just that sample, variance is zero.

GroupNorm splits channels into groups and normalizes within each group, independent of batch size:

```
64 channels, 8 groups → 8 groups of 8 channels each
Each group is normalized independently
Works identically whether batch size is 1 or 128
```

### Why SiLU (Swish), Not ReLU?

SiLU(x) = x · σ(x), where σ is the sigmoid function. Unlike ReLU which has a hard zero for negative inputs, SiLU is smooth everywhere — small negative inputs produce small negative outputs rather than exactly zero. This smoother gradient landscape helps optimization in generative models. All modern diffusion implementations use SiLU.

### Time Injection Between Convolutions

The time embedding is injected **between** the two convolutions, not before the first or after the last:

```python
h = self.conv1(F.silu(self.norm1(x)))     # first conv processes spatial features
h = h + self.time_proj(t_emb)[..., None, None]  # add time info
h = self.conv2(F.silu(self.norm2(h)))     # second conv processes time-aware features
```

The first convolution extracts spatial features without time information. The time embedding then modulates these features — "given that this is timestep 500, adjust these spatial features accordingly." The second convolution then processes the time-adjusted features. This two-stage approach lets the network cleanly separate spatial processing from temporal conditioning.

## Part 4: Downsampling and Upsampling

### Downsample: Strided Convolution

```python
class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
```

A 3×3 convolution with stride 2 halves both spatial dimensions:

```
Input:  (batch, 64, 28, 28)
Output: (batch, 64, 14, 14)
```

Stride 2 means the filter moves 2 pixels at a time instead of 1, producing an output half the size. This is a **learned** downsample — the conv weights determine how to combine 2×2 neighborhoods. This preserves more information than max pooling or average pooling.

### Upsample: Nearest-Neighbor + Convolution

```python
class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)
```

Two steps: first double the spatial size by duplicating each pixel into a 2×2 block (nearest-neighbor interpolation), then smooth with a convolution.

```
Input:         (batch, 128, 7, 7)
After interp:  (batch, 128, 14, 14)    ← each pixel duplicated to 2×2
After conv:    (batch, 128, 14, 14)    ← smoothed
```

Why not use transposed convolution (`nn.ConvTranspose2d`)? Transposed convolutions are known to produce **checkerboard artifacts** — a grid-like pattern visible in generated images. These artifacts arise because the transposed convolution's overlapping output regions create uneven contributions. Nearest-neighbor + conv avoids this entirely.

## Part 5: The Complete U-Net Architecture

### MNIST Configuration

```
Encoder:
    Input:  (1, 28, 28)
    conv_in → (64, 28, 28)
    
    Level 0: ResBlock(64→64) → ResBlock(64→64) → (64, 28, 28)     ─── skip₀
             Downsample → (64, 14, 14)
    
    Level 1: ResBlock(64→128) → ResBlock(128→128) → (128, 14, 14)  ─── skip₁
             Downsample → (128, 7, 7)
    
    Level 2: ResBlock(128→256) → ResBlock(256→256) → (256, 7, 7)   ─── skip₂

Bottleneck:
    ResBlock(256→256) → (256, 7, 7)

Decoder:
    Concat(bottleneck, skip₂) → (512, 7, 7)
    Level 0: ResBlock(512→256) → ResBlock(256→256) → (256, 7, 7)
             Upsample → (256, 14, 14)
    
    Concat(up, skip₁) → (384, 14, 14)     ← 256 + 128
    Level 1: ResBlock(384→128) → ResBlock(128→128) → (128, 14, 14)
             Upsample → (128, 28, 28)
    
    Concat(up, skip₀) → (192, 28, 28)     ← 128 + 64
    Level 2: ResBlock(192→64) → ResBlock(64→64) → (64, 28, 28)

Output:
    GroupNorm → SiLU → Conv1×1 → (1, 28, 28)    ← predicted noise
```

Total parameters: ~4.5M.

Notice how channels double when resolution halves (64→128→256) and how skip connections increase decoder input channels (256+256=512, 256+128=384, 128+64=192). The first ResBlock at each decoder level handles this channel mismatch with its 1×1 skip_conv.

### CIFAR-10 Configuration

Larger channels, one more resolution level, and self-attention at levels 1 and 2:

```
Encoder:                              Decoder:
(128, 32, 32)                         (128, 32, 32)
    ↓ Down                                ↑ Up
(256, 16, 16) + Attention             (256, 16, 16) + Attention
    ↓ Down                                ↑ Up
(256, 8, 8) + Attention               (256, 8, 8) + Attention
    ↓ Down                                ↑ Up
(512, 4, 4)                           (512, 4, 4)
    ↓                                     ↑
    └── Bottleneck (512, 4, 4) ──────────┘
```

Total parameters: ~28M. Attention is added at 16×16 and 8×8 resolutions — see Chapter 5 for why.

## Part 6: Input and Output

### Input: Noised Image + Timestep

The network receives:

```
x_t:  (batch, 1, 28, 28)   — the noised image (what does this look like?)
t:    (batch,)               — the timestep (how noisy is it?)
```

These are fundamentally different types of information. The image is spatial (pixels), the timestep is a scalar. The U-Net handles them differently: the image enters through `conv_in`, the timestep enters through the sinusoidal embedding + MLP, and they're combined inside every ResBlock via the time projection.

### Output: Predicted Noise

```
ε_pred: (batch, 1, 28, 28)  — same shape as input
```

This is the network's best guess of what Gaussian noise was added to create x_t from x₀. During training, we compare this to the actual noise ε that was used. During sampling, we use this prediction to take one denoising step.

### The Output Convolution

```python
self.out_norm = nn.GroupNorm(num_groups=8, num_channels=64)
self.out_conv = nn.Conv2d(64, 1, kernel_size=1)
```

The final 1×1 convolution projects from 64 channels down to 1 channel (or 3 for RGB). The kernel size is 1×1 because no spatial processing is needed at this point — the decoder has already done all the spatial reasoning. We just need to combine the 64 features at each pixel into a single noise prediction.

## Part 7: Parameter Count Analysis

For the MNIST U-Net with channels [64, 128, 256]:

```
Time embedding:     ~66K    (MLP: 128→512→128)
Input conv:         ~640    (1→64, 3×3)
Encoder ResBlocks:  ~1.1M   (6 blocks across 3 levels)
Downsamplers:       ~74K    (2 strided convs)
Bottleneck:         ~590K   (1 ResBlock at 256 channels)
Decoder ResBlocks:  ~1.9M   (6 blocks, larger due to concatenated skip channels)
Upsamplers:         ~330K   (2 conv layers after interpolation)
Output conv:        ~65     (64→1, 1×1)
─────────────────────────
Total:              ~4.5M
```

The decoder is larger than the encoder because the concatenated skip connections double the input channels for the first ResBlock at each level. This is by design — the decoder has the harder job (reconstructing detail), so it gets more capacity.

For comparison:
- AlphaZero ResNet: 377K parameters (6×7 board, simple patterns)
- MNIST U-Net: ~4.5M parameters (28×28 images, spatial patterns at multiple scales)
- CIFAR U-Net: ~28M parameters (32×32 RGB, complex natural images, self-attention)

## Summary

| Component | What | Why |
|-----------|------|-----|
| Encoder (downsample) | 28→14→7, channels increase | Capture global structure at low resolution |
| Decoder (upsample) | 7→14→28, channels decrease | Reconstruct spatial detail |
| Skip connections | Concatenate encoder→decoder | Preserve fine details lost during downsampling |
| Time embedding | Sinusoidal → MLP → inject into ResBlocks | Tell network the noise level at every layer |
| ResBlock | Conv → time → Conv + residual | Core processing with time conditioning |
| GroupNorm | Normalize within channel groups | Works at any batch size (crucial for sampling) |
| SiLU activation | x · sigmoid(x) | Smooth gradients, better than ReLU for generation |
| 1×1 output conv | 64ch → 1ch | Project features to noise prediction |

## What's Next

In [Chapter 3](03_training.md), we train this U-Net to predict noise. The training loop is remarkably simple — sample a random image, add random noise at a random timestep, predict the noise, compute MSE loss. But the details matter: EMA for stable sampling, gradient clipping for training stability, and the surprising elegance of the noise prediction objective.