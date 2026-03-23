# Chapter 5: Scaling to CIFAR-10 — Color Images and Self-Attention

## The Problem

MNIST was a warmup: grayscale, 28×28, 10 classes of similar shapes (digits). CIFAR-10 is a real challenge: RGB color, 32×32, 10 classes of fundamentally different objects (airplanes, cars, cats, deer, dogs, frogs, horses, ships, trucks). The network must learn color, texture, shape, and spatial relationships that are far more complex than digit strokes.

Two things change: the U-Net gets bigger (more channels, one more resolution level), and it gets self-attention layers. Everything else — the diffusion math, the training objective, the sampling algorithms — stays identical.

## Part 1: What Changes from MNIST

| Aspect | MNIST | CIFAR-10 |
|--------|-------|----------|
| Image size | 1×28×28 (grayscale) | 3×32×32 (RGB) |
| Channels | [64, 128, 256] | [128, 256, 256, 512] |
| Resolution levels | 3 (28→14→7) | 4 (32→16→8→4) |
| Attention | None | At 16×16 and 8×8 |
| Parameters | ~4.5M | ~28M |
| Noise schedule | Linear | Cosine |
| Training epochs | 50 | 100 |
| Time dimension | 128 | 256 |

### Why More Channels?

MNIST digits are binary patterns — bright strokes on dark backgrounds. You can represent them with relatively few feature channels. CIFAR-10 images have color gradients, textures (fur, metal, sky), and complex shapes. More channels give the network more capacity to represent these diverse features simultaneously.

### Why One More Resolution Level?

CIFAR images are 32×32 (vs 28×28 for MNIST). An extra downsampling step gives us a 4×4 bottleneck, where each position represents a large receptive field — roughly 1/4 of the image. This global view is essential for coherent object generation.

### Why Cosine Schedule?

The linear schedule adds noise too aggressively in early timesteps for higher-resolution color images. Early steps should barely change the image, giving the network easy training examples for fine details. The cosine schedule preserves signal longer in early steps, spending more of the 1000-step budget in the "slightly noisy" range where fine detail matters.

## Part 2: Self-Attention in the U-Net

### Why Convolutions Aren't Enough

A 3×3 convolution sees only a 3×3 local neighborhood. To connect distant pixels, information must flow through many layers: a 5-block ResNet has an effective receptive field of about 11×11 pixels. For a 32×32 image, this means a pixel in the top-left can't directly influence a pixel in the bottom-right.

For natural images, long-range dependencies matter:
- The sky should be a consistent color across the entire top of the image
- Two eyes should be roughly symmetric
- A car's front implies something about its rear
- Ships have both a hull and a mast separated by many pixels

Self-attention solves this by connecting ALL positions in one operation.

### How Self-Attention Works in Images

Self-attention treats each spatial position as a "token," exactly like word tokens in a Transformer:

```
Feature map: (batch, channels, H, W)
    │
    ▼ Reshape to sequence
Tokens: (batch, H×W, channels)     ← each pixel is a token
    │
    ▼ Standard attention: Q, K, V projections
Q = tokens @ W_q                    (batch, H×W, channels)
K = tokens @ W_k                    (batch, H×W, channels)
V = tokens @ W_v                    (batch, H×W, channels)
    │
    ▼ Attention scores
attn = softmax(Q @ K^T / √d_head)  (batch, H×W, H×W)
    │
    ▼ Weighted sum
out = attn @ V                      (batch, H×W, channels)
    │
    ▼ Reshape back to spatial
Output: (batch, channels, H, W)     ← same shape as input
```

The attention matrix (H×W, H×W) has one entry for every pair of positions. Position (3, 5) attending to position (12, 8) means "the feature at pixel (3,5) should incorporate information from pixel (12,8)." The network learns WHICH positions to attend to.

### Multi-Head Attention

We split the channels into 4 heads, each attending independently:

```
channels = 256, num_heads = 4 → head_dim = 64

Head 1: might learn to attend to same-color regions
Head 2: might learn to attend to symmetric positions
Head 3: might learn to attend to edges at similar orientations
Head 4: might learn to attend to nearby context
```

This is identical to multi-head attention in your Transformer project — same math, same motivation (different heads learn different attention patterns).

### Why Only at Low Resolutions?

Attention is O(n²) where n = number of positions:

```
At 32×32: n = 1024, attention matrix = 1024² = 1,048,576 entries  ← expensive
At 16×16: n = 256,  attention matrix = 256²  = 65,536 entries     ← manageable
At 8×8:   n = 64,   attention matrix = 64²   = 4,096 entries      ← cheap
At 4×4:   n = 16,   attention matrix = 16²   = 256 entries        ← trivial
```

We apply attention at 16×16 and 8×8 (the middle two resolutions). At 32×32, the cost would quadruple with minimal benefit — local patterns at full resolution are well-captured by convolutions. At 4×4, there are only 16 positions, so even convolutions have a near-global receptive field — attention adds little.

The design pattern: **convolutions for local patterns, attention for global relationships.** Each ResBlock does convolution first, then attention (if enabled) captures long-range dependencies on top of the local features.

### Implementation

```python
class SelfAttention(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        b, c, h, w = x.shape
        residual = x

        x = self.norm(x)
        x = x.view(b, c, h * w).permute(0, 2, 1)  # (B, H*W, C)

        q = self.q_proj(x).view(b, h*w, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).view(b, h*w, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(b, h*w, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        out = out.permute(0, 2, 1, 3).reshape(b, h*w, c)
        out = self.out_proj(out)
        out = out.permute(0, 2, 1).view(b, c, h, w)

        return out + residual  # residual connection
```

Shape trace for 256-channel features at 16×16 resolution:

```
Input:    (B, 256, 16, 16)
Reshape:  (B, 256, 16×16) → permute → (B, 256, 256)
Q, K, V:  each (B, 4, 256, 64)        ← 4 heads, 64 dim each
Attn:     (B, 4, 256, 256)             ← each position attends to all others
Out:      (B, 4, 256, 64) → (B, 256, 256) → (B, 256, 16, 16)
+ residual → (B, 256, 16, 16)
```

Note the residual connection: even if the attention learns nothing useful early in training, the block passes through the input unchanged. The attention can only help, never hurt (same principle as ResBlock skip connections).

## Part 3: Data Augmentation

For CIFAR-10, we add random horizontal flipping:

```python
transforms.RandomHorizontalFlip()
```

A car facing left is equally valid as a car facing right. A horse in profile can face either direction. This doubles the effective dataset size for free.

We didn't flip MNIST because many digits are NOT symmetric — a horizontally flipped 2 doesn't look like any real digit. But most natural objects and scenes are roughly symmetric.

## Part 4: Training Results

```
Epoch 1:   loss = 0.130    (random predictions)
Epoch 20:  loss = 0.058    (learning basic structure)
Epoch 60:  loss = 0.056    (refining details)
Epoch 100: loss = 0.055    (converged)
```

The loss is higher than MNIST (0.055 vs 0.021) because CIFAR-10 is fundamentally harder: 3 channels of diverse natural images vs 1 channel of simple digit strokes. The noise prediction task is harder when the underlying image has more complexity.

### Sample Quality Progression

```
Epoch 1:   Colored noise blobs — no recognizable objects
Epoch 20:  Vague shapes and appropriate colors
Epoch 60:  Recognizable objects — cars, planes, animals visible
Epoch 100: Clear objects with correct colors, blurry details
```

The samples at epoch 100 are recognizable but not photorealistic — expected for 32×32 resolution with 28M params trained for 100 epochs. The DDPM paper trained for ~800 epochs with 35M params to achieve FID 3.17. Our setup prioritized learning the concepts over SOTA results.

## Part 5: Computational Considerations

### Memory

The 28M parameter model with batch size 128 on CIFAR-10 uses roughly:
- Model weights: ~110MB (28M × 4 bytes)
- Optimizer states (Adam): ~220MB (two momentum buffers)
- Activations: ~500MB-1GB (depends on batch size)
- Total: ~1-1.5GB VRAM

Well within T4's 16GB VRAM. Batch size 128 is a good balance between training speed and memory.

### Training Time

100 epochs on T4 GPU took approximately 3-4 hours. The attention layers add ~30% overhead compared to a conv-only model, but the quality improvement justifies the cost.

## Summary

| Concept | What | Why |
|---------|------|-----|
| Larger U-Net | [128,256,256,512], 28M params | More capacity for diverse objects |
| Self-attention | At 16×16 and 8×8 resolutions | Long-range spatial relationships |
| Cosine schedule | ᾱ follows cosine curve | Preserves fine detail longer |
| Horizontal flip | Random mirror augmentation | 2× effective data for symmetric objects |
| 4 resolution levels | 32→16→8→4 | One more level for 32×32 images |

## What's Next

In [Chapter 6](06_conditional_generation.md), we add class conditioning — telling the model WHAT to generate. The technique we use, classifier-free guidance, is the same one that powers Stable Diffusion's text-to-image generation. The math is surprisingly simple: train with random label dropout, then amplify the conditional signal during sampling.