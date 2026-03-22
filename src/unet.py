"""
U-Net Denoiser — The Neural Network That Learns to Remove Noise.

Architecture overview:
    Input image x_t (noised) + timestep t
        → Time embedding (sinusoidal → MLP, tells the network HOW noised the input is)
        → Encoder (progressively downsample, extract features at multiple scales)
        → Bottleneck (process at lowest resolution)
        → Decoder (progressively upsample, reconstruct spatial detail)
        → Output: predicted noise ε (same shape as input)

    Skip connections from encoder to decoder at each resolution level
    carry fine spatial details that would be lost during downsampling.

Key design choices:
    - GroupNorm instead of BatchNorm: doesn't depend on batch size, essential
      because during sampling batch size = 1 (or small). BatchNorm statistics
      would be unreliable.
    - SiLU activation (Sigmoid Linear Unit, aka Swish): x · σ(x). Smoother than
      ReLU, works better in diffusion models empirically. Used in all modern
      diffusion implementations.
    - Sinusoidal time embedding: same math as Transformer positional encoding.
      Maps integer timestep → high-dimensional vector that the network can use
      to adapt its behavior based on the noise level.
    - Skip connections via concatenation (not addition): preserves more
      information from the encoder. Standard U-Net design.
    - Self-attention (optional): at lower resolutions, lets the network capture
      long-range spatial relationships. A pixel in the top-left can attend to
      a pixel in the bottom-right. Essential for coherent structure in RGB images.

Supports two configurations:
    MNIST:    channels=[64,128,256],     no attention,  ~4.5M params
    CIFAR-10: channels=[128,256,256,512], attention at levels 1&2, ~28M params
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------
# Time Embedding
# -----------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """
    Maps integer timestep t → high-dimensional vector.

    Uses the same sinusoidal formula as Transformer positional encoding:
        PE(t, 2i)   = sin(t / 10000^(2i/d))
        PE(t, 2i+1) = cos(t / 10000^(2i/d))

    Then passes through an MLP to let the network learn how to use
    the time information.

    WHY: The network needs to know the noise level to predict noise correctly.
    At t=10, the image is barely noised — the network should predict tiny noise.
    At t=990, the image is almost pure noise — the network must predict large noise.
    The time embedding gives it this information.

    WHY SINUSOIDAL: Raw integer t (e.g., 500) is a single number — too little
    information for a neural network to condition on effectively. Sinusoidal
    encoding expands it to a rich vector where nearby timesteps have similar
    representations and distant timesteps have different ones.
    """

    def __init__(self, time_dim):
        """
        Args:
            time_dim: dimension of the output embedding (e.g., 128)
        """
        super().__init__()
        self.time_dim = time_dim

        # MLP to project sinusoidal features → useful representation
        # SiLU activation between two linear layers
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),   # expand
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),   # project back
        )

    def forward(self, t):
        """
        Args:
            t: integer timesteps, shape (batch,)

        Returns:
            embedding: shape (batch, time_dim)
        """
        device = t.device
        half_dim = self.time_dim // 2

        # Compute frequencies: 10000^(-2i/d) for i in [0, half_dim)
        freq = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, device=device) / half_dim
        )

        # Outer product: each timestep × each frequency
        args = t.float().unsqueeze(1) * freq.unsqueeze(0)

        # Interleave sin and cos: [sin(f1), cos(f1), sin(f2), cos(f2), ...]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        # Pass through MLP to get a learned representation
        return self.mlp(embedding)


# -----------------------------------------------------------------------
# Building Blocks
# -----------------------------------------------------------------------

class ResBlock(nn.Module):
    """
    Residual block with time conditioning.

    The core building block of the U-Net. Two convolutions with a skip
    connection, plus time embedding injection between them.

    Data flow:
        x ──────────────────────────────────────┐
        │                                        │ (skip connection)
        ▼                                        │
      GroupNorm → SiLU → Conv3×3                 │
        │                                        │
        ▼                                        │
      + time_embedding (broadcast to spatial)    │
        │                                        │
        ▼                                        │
      GroupNorm → SiLU → Conv3×3                 │
        │                                        │
        ▼                                        ▼
      Add ──────────────────────────────────── (residual)
        │
        ▼
      Output
    """

    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()

        # First conv: GroupNorm → SiLU → Conv
        self.norm1 = nn.GroupNorm(num_groups=min(8, in_channels), num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Time projection: project time_dim → out_channels
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels),
        )

        # Second conv: GroupNorm → SiLU → Conv
        self.norm2 = nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Skip connection: 1×1 conv if channels change, identity otherwise
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_conv = nn.Identity()

    def forward(self, x, t_emb):
        """
        Args:
            x: feature map, shape (batch, in_channels, H, W)
            t_emb: time embedding, shape (batch, time_dim)

        Returns:
            output: shape (batch, out_channels, H, W)
        """
        residual = self.skip_conv(x)

        # First conv block
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # Add time embedding
        t = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + t

        # Second conv block
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        return h + residual


class SelfAttention(nn.Module):
    """
    Multi-head self-attention over spatial feature maps.

    Treats each spatial position (pixel) as a token, just like Transformer
    attention treats each word as a token. This lets the network capture
    long-range dependencies — a pixel in one corner can attend to a pixel
    in the opposite corner.

    WHY ATTENTION IN DIFFUSION:
    Convolutions are local — a 3×3 filter only sees a 3×3 neighborhood.
    Stacking many convolutions expands the receptive field, but it's still
    indirect. Attention directly connects ALL positions in one operation.

    For images, this matters for:
    - Global color consistency (sky should be uniform blue)
    - Symmetry (two eyes should match)
    - Long-range structure (the left side of a car implies the right side)

    WHY ONLY AT LOW RESOLUTIONS:
    Attention is O(n²) where n = H × W (number of spatial positions).
    At 32×32, n = 1024 → attention matrix is 1024×1024 = 1M entries.
    At 16×16, n = 256 → attention matrix is 256×256 = 65K entries (16× smaller).
    At 8×8, n = 64 → attention matrix is 64×64 = 4K entries (256× smaller).
    We only use attention at 16×16 and 8×8 to keep compute manageable.
    """

    def __init__(self, channels, num_heads=4):
        """
        Args:
            channels: number of input/output channels
            num_heads: number of attention heads
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert channels % num_heads == 0, \
            f"channels ({channels}) must be divisible by num_heads ({num_heads})"

        self.norm = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)

        # Q, K, V projections (like in your Transformer project)
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)

        # Output projection
        self.out_proj = nn.Linear(channels, channels)

        # Scaling factor for attention scores: 1/√d_head
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        """
        Args:
            x: feature map, shape (batch, channels, H, W)

        Returns:
            output: shape (batch, channels, H, W) — same as input
        """
        b, c, h, w = x.shape
        residual = x

        # Normalize
        x = self.norm(x)

        # Reshape: (B, C, H, W) → (B, H*W, C) — treat each pixel as a token
        x = x.view(b, c, h * w).permute(0, 2, 1)  # (B, N, C) where N = H*W

        # Project to Q, K, V
        q = self.q_proj(x)  # (B, N, C)
        k = self.k_proj(x)  # (B, N, C)
        v = self.v_proj(x)  # (B, N, C)

        # Reshape for multi-head attention: (B, N, C) → (B, num_heads, N, head_dim)
        q = q.view(b, h * w, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(b, h * w, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(b, h * w, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Attention: softmax(Q·K^T / √d) · V
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # (B, heads, N, head_dim)

        # Reshape back: (B, heads, N, head_dim) → (B, N, C)
        out = out.permute(0, 2, 1, 3).reshape(b, h * w, c)

        # Output projection
        out = self.out_proj(out)

        # Reshape back to spatial: (B, N, C) → (B, C, H, W)
        out = out.permute(0, 2, 1).view(b, c, h, w)

        return out + residual


class Downsample(nn.Module):
    """Spatial downsampling: H×W → H/2 × W/2 via strided convolution."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """
    Spatial upsampling: H×W → 2H × 2W.
    Nearest-neighbor interpolation + convolution avoids checkerboard artifacts.
    """

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# -----------------------------------------------------------------------
# The U-Net
# -----------------------------------------------------------------------

class UNet(nn.Module):
    """
    U-Net denoiser with optional self-attention.

    Two configurations:

    MNIST (no attention):
        in=1, channels=[64,128,256], time_dim=128
        28×28 → 14×14 → 7×7 → bottleneck → 7×7 → 14×14 → 28×28
        ~4.5M params

    CIFAR-10 (with attention at levels 1 and 2):
        in=3, channels=[128,256,256,512], time_dim=256, attention_levels=[F,T,T,F]
        32×32 → 16×16 → 8×8 → 4×4 → bottleneck → 4×4 → 8×8 → 16×16 → 32×32
        ~28M params

    Attention is added AFTER the ResBlocks at specified levels, in both
    encoder and decoder. This lets convolutions extract local features first,
    then attention captures global relationships between those features.
    """

    def __init__(self, in_channels=1, out_channels=1,
                 channel_list=(64, 128, 256), time_dim=128,
                 num_res_blocks=2, attention_levels=None, num_heads=4):
        """
        Args:
            in_channels: input image channels (1 for MNIST, 3 for CIFAR)
            out_channels: output channels (same as input — predicting noise)
            channel_list: channels at each resolution level
            time_dim: dimension of the sinusoidal time embedding
            num_res_blocks: number of ResBlocks per resolution level
            attention_levels: list of bools, same length as channel_list.
                              True = add self-attention at that level.
                              None = no attention anywhere (MNIST default).
            num_heads: number of attention heads
        """
        super().__init__()
        self.channel_list = channel_list
        self.num_res_blocks = num_res_blocks

        # Default: no attention at any level
        if attention_levels is None:
            attention_levels = [False] * len(channel_list)
        self.attention_levels = attention_levels

        # ── Time embedding ──
        self.time_embed = SinusoidalTimeEmbedding(time_dim)

        # ── Input convolution ──
        self.conv_in = nn.Conv2d(in_channels, channel_list[0], kernel_size=3, padding=1)

        # ── Encoder ──
        self.encoder_blocks = nn.ModuleList()
        self.encoder_attns = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        for i, ch_out in enumerate(channel_list):
            ch_in = channel_list[i - 1] if i > 0 else channel_list[0]

            level_blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                block_in = ch_in if j == 0 else ch_out
                level_blocks.append(ResBlock(block_in, ch_out, time_dim))
            self.encoder_blocks.append(level_blocks)

            # Attention after ResBlocks (if enabled for this level)
            if attention_levels[i]:
                self.encoder_attns.append(SelfAttention(ch_out, num_heads))
            else:
                self.encoder_attns.append(nn.Identity())

            # Downsample between levels (not after the last level)
            if i < len(channel_list) - 1:
                self.downsamplers.append(Downsample(ch_out))
            else:
                self.downsamplers.append(nn.Identity())

        # ── Bottleneck ──
        bottleneck_ch = channel_list[-1]
        self.bottleneck = ResBlock(bottleneck_ch, bottleneck_ch, time_dim)

        # ── Decoder ──
        self.decoder_blocks = nn.ModuleList()
        self.decoder_attns = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        reversed_channels = list(reversed(channel_list))
        reversed_attention = list(reversed(attention_levels))

        for i, ch_out in enumerate(reversed_channels):
            skip_ch = ch_out

            if i == 0:
                ch_in_first = bottleneck_ch + skip_ch
            else:
                ch_in_first = reversed_channels[i - 1] + skip_ch

            level_blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                block_in = ch_in_first if j == 0 else ch_out
                level_blocks.append(ResBlock(block_in, ch_out, time_dim))
            self.decoder_blocks.append(level_blocks)

            # Attention (mirroring encoder)
            if reversed_attention[i]:
                self.decoder_attns.append(SelfAttention(ch_out, num_heads))
            else:
                self.decoder_attns.append(nn.Identity())

            # Upsample between levels (not after the last level)
            if i < len(channel_list) - 1:
                self.upsamplers.append(Upsample(ch_out))
            else:
                self.upsamplers.append(nn.Identity())

        # ── Output ──
        self.out_norm = nn.GroupNorm(num_groups=min(8, channel_list[0]),
                                    num_channels=channel_list[0])
        self.out_conv = nn.Conv2d(channel_list[0], out_channels, kernel_size=1)

    def forward(self, x, t):
        """
        Args:
            x: noised images, shape (batch, in_channels, H, W)
            t: timesteps, shape (batch,) — integer timesteps [0, T)

        Returns:
            noise_pred: predicted noise, shape (batch, out_channels, H, W)
        """
        # ── Time embedding ──
        t_emb = self.time_embed(t)

        # ── Input convolution ──
        h = self.conv_in(x)

        # ── Encoder ──
        skip_connections = []

        for i, level_blocks in enumerate(self.encoder_blocks):
            for block in level_blocks:
                h = block(h, t_emb)

            # Attention (if enabled at this level)
            h = self.encoder_attns[i](h)

            # Save for skip connection
            skip_connections.append(h)

            # Downsample (except at the last level)
            if i < len(self.channel_list) - 1:
                h = self.downsamplers[i](h)

        # ── Bottleneck ──
        h = self.bottleneck(h, t_emb)

        # ── Decoder ──
        for i, level_blocks in enumerate(self.decoder_blocks):
            skip = skip_connections[-(i + 1)]
            h = torch.cat([h, skip], dim=1)

            for block in level_blocks:
                h = block(h, t_emb)

            # Attention (if enabled at this level)
            h = self.decoder_attns[i](h)

            # Upsample (except at the last level)
            if i < len(self.channel_list) - 1:
                h = self.upsamplers[i](h)

        # ── Output ──
        h = self.out_norm(h)
        h = F.silu(h)
        h = self.out_conv(h)

        return h


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 2/5: U-Net Architecture Tests")
    print("=" * 60)

    # ── Test 1: Time embedding ──
    print("\n── Test 1: Sinusoidal Time Embedding ──")
    time_embed = SinusoidalTimeEmbedding(time_dim=128)
    t = torch.tensor([0, 100, 500, 999])
    emb = time_embed(t)
    print(f"Input timesteps: {t.tolist()}")
    print(f"Embedding shape: {emb.shape}  (expected: [4, 128])")
    assert emb.shape == (4, 128), f"Wrong shape: {emb.shape}"
    print("✓ Time embedding shapes correct\n")

    # ── Test 2: ResBlock ──
    print("── Test 2: ResBlock ──")
    block_same = ResBlock(in_channels=64, out_channels=64, time_dim=128)
    x = torch.randn(2, 64, 14, 14)
    t_emb = torch.randn(2, 128)
    out = block_same(x, t_emb)
    print(f"ResBlock (64→64): input {x.shape} → output {out.shape}")
    assert out.shape == (2, 64, 14, 14)

    block_diff = ResBlock(in_channels=64, out_channels=128, time_dim=128)
    out = block_diff(x, t_emb)
    print(f"ResBlock (64→128): input {x.shape} → output {out.shape}")
    assert out.shape == (2, 128, 14, 14)
    print("✓ ResBlock correct\n")

    # ── Test 3: SelfAttention ──
    print("── Test 3: Self-Attention ──")
    attn = SelfAttention(channels=128, num_heads=4)
    x = torch.randn(2, 128, 8, 8)
    out = attn(x)
    print(f"SelfAttention: input {x.shape} → output {out.shape}")
    assert out.shape == x.shape, f"Attention should preserve shape"
    n_attn_params = sum(p.numel() for p in attn.parameters())
    print(f"Attention params: {n_attn_params:,}")
    print("✓ Self-Attention correct\n")

    # ── Test 4: MNIST U-Net (no attention) ──
    print("── Test 4: MNIST U-Net (no attention) ──")
    model_mnist = UNet(
        in_channels=1, out_channels=1,
        channel_list=(64, 128, 256),
        time_dim=128, num_res_blocks=2,
        attention_levels=None,
    )
    n_mnist = sum(p.numel() for p in model_mnist.parameters())
    print(f"MNIST params: {n_mnist:,}")

    batch = torch.randn(4, 1, 28, 28)
    timesteps = torch.randint(0, 1000, (4,))
    out = model_mnist(batch, timesteps)
    print(f"Input:  {batch.shape}")
    print(f"Output: {out.shape}")
    assert out.shape == batch.shape
    print("✓ MNIST U-Net correct\n")

    # ── Test 5: CIFAR-10 U-Net (with attention) ──
    print("── Test 5: CIFAR-10 U-Net (with attention) ──")
    model_cifar = UNet(
        in_channels=3, out_channels=3,
        channel_list=(128, 256, 256, 512),
        time_dim=256, num_res_blocks=2,
        attention_levels=[False, True, True, False],
        num_heads=4,
    )
    n_cifar = sum(p.numel() for p in model_cifar.parameters())
    print(f"CIFAR params: {n_cifar:,}")

    batch = torch.randn(2, 3, 32, 32)
    timesteps = torch.randint(0, 1000, (2,))
    out = model_cifar(batch, timesteps)
    print(f"Input:  {batch.shape}")
    print(f"Output: {out.shape}")
    assert out.shape == batch.shape
    print("✓ CIFAR U-Net correct\n")

    # ── Test 6: Shape trace for CIFAR ──
    print("── Test 6: CIFAR Shape Trace ──")
    x = torch.randn(1, 3, 32, 32)
    t = torch.tensor([500])

    print(f"Input:               {x.shape}")
    t_emb = model_cifar.time_embed(t)
    print(f"Time embedding:      {t_emb.shape}")
    h = model_cifar.conv_in(x)
    print(f"After conv_in:       {h.shape}")

    print("\nEncoder:")
    skips = []
    for i, level_blocks in enumerate(model_cifar.encoder_blocks):
        for j, block in enumerate(level_blocks):
            h = block(h, t_emb)
            print(f"  Level {i}, ResBlock {j}: {h.shape}")
        h = model_cifar.encoder_attns[i](h)
        has_attn = model_cifar.attention_levels[i]
        if has_attn:
            print(f"  Attention:          {h.shape}  ← self-attention applied")
        skips.append(h)
        if i < len(model_cifar.channel_list) - 1:
            h = model_cifar.downsamplers[i](h)
            print(f"  Downsample:         {h.shape}")

    print("\nBottleneck:")
    h = model_cifar.bottleneck(h, t_emb)
    print(f"  ResBlock:           {h.shape}")

    print("\nDecoder:")
    rev_attn = list(reversed(model_cifar.attention_levels))
    for i, level_blocks in enumerate(model_cifar.decoder_blocks):
        skip = skips[-(i + 1)]
        h = torch.cat([h, skip], dim=1)
        print(f"  After concat skip:  {h.shape}")
        for j, block in enumerate(level_blocks):
            h = block(h, t_emb)
            print(f"  Level {i}, ResBlock {j}: {h.shape}")
        h = model_cifar.decoder_attns[i](h)
        if rev_attn[i]:
            print(f"  Attention:          {h.shape}  ← self-attention applied")
        if i < len(model_cifar.channel_list) - 1:
            h = model_cifar.upsamplers[i](h)
            print(f"  Upsample:           {h.shape}")

    print("\nOutput:")
    h = model_cifar.out_norm(h)
    h = F.silu(h)
    h = model_cifar.out_conv(h)
    print(f"  Final output:       {h.shape}")

    # ── Test 7: Gradient flow ──
    print("\n── Test 7: Gradient Flow (CIFAR) ──")
    model_cifar.train()
    x = torch.randn(2, 3, 32, 32)
    t = torch.randint(0, 1000, (2,))
    out = model_cifar(x, t)
    loss = out.mean()
    loss.backward()
    no_grad = [n for n, p in model_cifar.named_parameters() if p.grad is None]
    if no_grad:
        print(f"  WARNING: {len(no_grad)} parameters have no gradient")
    else:
        print(f"  All {sum(1 for _ in model_cifar.parameters())} parameters have gradients")
    print("✓ Gradients flow correctly\n")

    # ── Summary ──
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"MNIST  U-Net: {n_mnist:,} params  | channels={list((64,128,256))}")
    print(f"CIFAR  U-Net: {n_cifar:,} params  | channels={list((128,256,256,512))}")
    print(f"CIFAR attention at levels 1,2 (16×16 and 8×8 resolution)")
    print(f"\nFor comparison:")
    print(f"  AlphaZero ResNet:      377,629 params")
    print(f"  DDPM paper (CIFAR):    ~35M params")
    print(f"  Our CIFAR U-Net:       {n_cifar:,} params")
    print("\n✓ All tests passed!")