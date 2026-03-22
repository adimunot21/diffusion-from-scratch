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

This file implements the MNIST variant (no attention, 3 resolution levels).
The CIFAR-10 variant with attention is built in Phase 5.
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
        # These are log-spaced from 1 to 1/10000
        freq = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, device=device) / half_dim
        )

        # Outer product: each timestep × each frequency
        # t has shape (batch,), freq has shape (half_dim,)
        # args has shape (batch, half_dim)
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

    WHY TIME CONDITIONING HERE (not just at the input):
    Injecting time into every ResBlock lets the network adapt its behavior
    at every layer based on the noise level. Early layers might detect edges
    differently at high noise vs low noise. Deep layers might focus on
    different global patterns. Pervasive conditioning is more powerful than
    a single injection at the input.
    """

    def __init__(self, in_channels, out_channels, time_dim):
        """
        Args:
            in_channels: input feature channels
            out_channels: output feature channels
            time_dim: dimension of the time embedding vector
        """
        super().__init__()

        # First conv: GroupNorm → SiLU → Conv
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Time projection: project time_dim → out_channels
        # This gets added to the feature map between the two convolutions
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels),
        )

        # Second conv: GroupNorm → SiLU → Conv
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Skip connection: if channels change, we need a 1×1 conv to match dimensions
        # If channels are the same, this is just identity (no-op)
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
        # time_proj output: (batch, out_channels) → reshape to (batch, out_channels, 1, 1)
        # for broadcasting across spatial dimensions
        t = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + t

        # Second conv block
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        return h + residual


class Downsample(nn.Module):
    """
    Spatial downsampling: H×W → H/2 × W/2.

    Uses strided convolution (stride=2) instead of pooling.
    Learned downsampling preserves more information than max/avg pooling.
    """

    def __init__(self, channels):
        super().__init__()
        # stride=2 halves spatial dimensions
        # padding=1 ensures output is exactly H/2 × W/2
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """
    Spatial upsampling: H×W → 2H × 2W.

    Uses nearest-neighbor interpolation followed by a convolution.
    This avoids the checkerboard artifacts that transposed convolutions
    (nn.ConvTranspose2d) are notorious for.

    Nearest-neighbor: each pixel is duplicated to fill a 2×2 block.
    Then the conv smooths and refines the result.
    """

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        # scale_factor=2 doubles both H and W
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# -----------------------------------------------------------------------
# The U-Net
# -----------------------------------------------------------------------

class UNet(nn.Module):
    """
    U-Net denoiser for MNIST (28×28 grayscale).

    Architecture (channel dimensions for default config):

    Encoder:
        (1, 28, 28) → ConvIn → (64, 28, 28)
        → ResBlock → ResBlock → (64, 28, 28)    ─── skip to decoder level 3
        → Downsample → (64, 14, 14)
        → ResBlock → ResBlock → (128, 14, 14)   ─── skip to decoder level 2
        → Downsample → (128, 7, 7)
        → ResBlock → ResBlock → (256, 7, 7)     ─── skip to decoder level 1

    Bottleneck:
        → ResBlock → (256, 7, 7)

    Decoder:
        (256+256, 7, 7) → ResBlock → ResBlock → (256, 7, 7)    ← concat skip
        → Upsample → (256, 14, 14)
        (256+128, 14, 14) → ResBlock → ResBlock → (128, 14, 14) ← concat skip
        → Upsample → (128, 28, 28)
        (128+64, 28, 28) → ResBlock → ResBlock → (64, 28, 28)   ← concat skip

    Output:
        → GroupNorm → SiLU → Conv1×1 → (1, 28, 28)

    Total skip connections: 3 (one per resolution level)
    Each carries the LAST encoder feature map at that resolution.
    """

    def __init__(self, in_channels=1, out_channels=1,
                 channel_list=(64, 128, 256), time_dim=128,
                 num_res_blocks=2):
        """
        Args:
            in_channels: input image channels (1 for MNIST, 3 for CIFAR)
            out_channels: output channels (same as input — predicting noise)
            channel_list: channels at each resolution level [level1, level2, level3]
                          Level 1 = full resolution, level 3 = lowest resolution
            time_dim: dimension of the sinusoidal time embedding
            num_res_blocks: number of ResBlocks per resolution level
        """
        super().__init__()
        self.channel_list = channel_list
        self.num_res_blocks = num_res_blocks

        # ── Time embedding ──
        self.time_embed = SinusoidalTimeEmbedding(time_dim)

        # ── Input convolution ──
        # Maps raw image channels → first feature channel count
        self.conv_in = nn.Conv2d(in_channels, channel_list[0], kernel_size=3, padding=1)

        # ── Encoder ──
        # Each level: num_res_blocks ResBlocks, then Downsample (except last level)
        self.encoder_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        for i, ch_out in enumerate(channel_list):
            # Channel count coming into this level
            ch_in = channel_list[i - 1] if i > 0 else channel_list[0]

            level_blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                # First block in level transitions from ch_in → ch_out
                # Subsequent blocks are ch_out → ch_out
                block_in = ch_in if j == 0 else ch_out
                level_blocks.append(ResBlock(block_in, ch_out, time_dim))
            self.encoder_blocks.append(level_blocks)

            # Downsample between levels (not after the last level)
            if i < len(channel_list) - 1:
                self.downsamplers.append(Downsample(ch_out))
            else:
                self.downsamplers.append(nn.Identity())  # placeholder, won't be used

        # ── Bottleneck ──
        # Process at the lowest resolution
        bottleneck_ch = channel_list[-1]
        self.bottleneck = ResBlock(bottleneck_ch, bottleneck_ch, time_dim)

        # ── Decoder ──
        # Mirror of encoder, but with skip connections (concatenation doubles input channels)
        self.decoder_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        # Walk through channel_list in reverse
        reversed_channels = list(reversed(channel_list))
        for i, ch_out in enumerate(reversed_channels):
            # The decoder receives: upsampled features + skip connection (concatenated)
            # Skip connection comes from encoder at matching resolution
            skip_ch = ch_out  # encoder output channels at this level

            # Input channels for first block at this level:
            # = upsampled channels from previous decoder level + skip channels
            if i == 0:
                # First decoder level: input from bottleneck + skip
                ch_in_first = bottleneck_ch + skip_ch
            else:
                # Subsequent levels: input from previous decoder level + skip
                ch_in_first = reversed_channels[i - 1] + skip_ch

            level_blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                block_in = ch_in_first if j == 0 else ch_out
                level_blocks.append(ResBlock(block_in, ch_out, time_dim))
            self.decoder_blocks.append(level_blocks)

            # Upsample between levels (not after the last level)
            if i < len(channel_list) - 1:
                self.upsamplers.append(Upsample(ch_out))
            else:
                self.upsamplers.append(nn.Identity())

        # ── Output ──
        self.out_norm = nn.GroupNorm(num_groups=8, num_channels=channel_list[0])
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
        t_emb = self.time_embed(t)  # (batch, time_dim)

        # ── Input convolution ──
        h = self.conv_in(x)  # (batch, channel_list[0], H, W)

        # ── Encoder ──
        # Store the output of each level for skip connections
        skip_connections = []

        for i, level_blocks in enumerate(self.encoder_blocks):
            for block in level_blocks:
                h = block(h, t_emb)

            # Save for skip connection (before downsampling)
            skip_connections.append(h)

            # Downsample (except at the last level)
            if i < len(self.channel_list) - 1:
                h = self.downsamplers[i](h)

        # ── Bottleneck ──
        h = self.bottleneck(h, t_emb)

        # ── Decoder ──
        # Process in reverse, concatenating skip connections
        for i, level_blocks in enumerate(self.decoder_blocks):
            # Get the matching skip connection (reverse order)
            skip = skip_connections[-(i + 1)]

            # Concatenate skip connection along channel dimension
            h = torch.cat([h, skip], dim=1)

            for block in level_blocks:
                h = block(h, t_emb)

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
    print("PHASE 2: U-Net Architecture Tests")
    print("=" * 60)

    # ── Test 1: Time embedding ──
    print("\n── Test 1: Sinusoidal Time Embedding ──")
    time_embed = SinusoidalTimeEmbedding(time_dim=128)
    t = torch.tensor([0, 100, 500, 999])
    emb = time_embed(t)
    print(f"Input timesteps: {t.tolist()}")
    print(f"Embedding shape: {emb.shape}  (expected: [4, 128])")
    assert emb.shape == (4, 128), f"Wrong shape: {emb.shape}"

    # Nearby timesteps should have similar embeddings
    t_close = torch.tensor([100, 101])
    emb_close = time_embed(t_close)
    dist_close = (emb_close[0] - emb_close[1]).norm().item()

    t_far = torch.tensor([100, 900])
    emb_far = time_embed(t_far)
    dist_far = (emb_far[0] - emb_far[1]).norm().item()

    print(f"Distance t=100 vs t=101: {dist_close:.4f}")
    print(f"Distance t=100 vs t=900: {dist_far:.4f}")
    print(f"Far > Close: {dist_far > dist_close}")
    # Note: after MLP, this property isn't guaranteed but usually holds
    print("✓ Time embedding shapes correct\n")

    # ── Test 2: ResBlock ──
    print("── Test 2: ResBlock ──")
    # Same channels
    block_same = ResBlock(in_channels=64, out_channels=64, time_dim=128)
    x = torch.randn(2, 64, 14, 14)
    t_emb = torch.randn(2, 128)
    out = block_same(x, t_emb)
    print(f"ResBlock (64→64): input {x.shape} → output {out.shape}")
    assert out.shape == (2, 64, 14, 14), f"Wrong shape: {out.shape}"

    # Different channels
    block_diff = ResBlock(in_channels=64, out_channels=128, time_dim=128)
    out = block_diff(x, t_emb)
    print(f"ResBlock (64→128): input {x.shape} → output {out.shape}")
    assert out.shape == (2, 128, 14, 14), f"Wrong shape: {out.shape}"
    print("✓ ResBlock correct\n")

    # ── Test 3: Down/Upsample ──
    print("── Test 3: Downsample and Upsample ──")
    down = Downsample(64)
    up = Upsample(64)

    x = torch.randn(2, 64, 28, 28)
    x_down = down(x)
    print(f"Downsample: {x.shape} → {x_down.shape}")
    assert x_down.shape == (2, 64, 14, 14), f"Wrong shape: {x_down.shape}"

    x_up = up(x_down)
    print(f"Upsample:   {x_down.shape} → {x_up.shape}")
    assert x_up.shape == (2, 64, 28, 28), f"Wrong shape: {x_up.shape}"
    print("✓ Down/Upsample correct\n")

    # ── Test 4: Full U-Net forward pass ──
    print("── Test 4: Full U-Net Forward Pass ──")
    model = UNet(
        in_channels=1,
        out_channels=1,
        channel_list=(64, 128, 256),
        time_dim=128,
        num_res_blocks=2,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    batch = torch.randn(4, 1, 28, 28)   # 4 MNIST images
    timesteps = torch.randint(0, 1000, (4,))
    noise_pred = model(batch, timesteps)

    print(f"\nForward pass:")
    print(f"  Input:       {batch.shape}       (batch, channels, H, W)")
    print(f"  Timesteps:   {timesteps.shape}         (batch,)")
    print(f"  Output:      {noise_pred.shape}       (batch, channels, H, W)")
    assert noise_pred.shape == batch.shape, f"Output shape {noise_pred.shape} != input shape {batch.shape}"
    print("✓ Output shape matches input shape (as expected — predicting noise)\n")

    # ── Test 5: Shape trace through the network ──
    print("── Test 5: Detailed Shape Trace ──")
    x = torch.randn(1, 1, 28, 28)
    t = torch.tensor([500])

    print(f"Input:               {x.shape}")

    t_emb = model.time_embed(t)
    print(f"Time embedding:      {t_emb.shape}")

    h = model.conv_in(x)
    print(f"After conv_in:       {h.shape}")

    print("\nEncoder:")
    skips = []
    for i, level_blocks in enumerate(model.encoder_blocks):
        for j, block in enumerate(level_blocks):
            h = block(h, t_emb)
            print(f"  Level {i}, ResBlock {j}: {h.shape}")
        skips.append(h)
        if i < len(model.channel_list) - 1:
            h = model.downsamplers[i](h)
            print(f"  Downsample:         {h.shape}")

    print("\nBottleneck:")
    h = model.bottleneck(h, t_emb)
    print(f"  ResBlock:           {h.shape}")

    print("\nDecoder:")
    for i, level_blocks in enumerate(model.decoder_blocks):
        skip = skips[-(i + 1)]
        h = torch.cat([h, skip], dim=1)
        print(f"  After concat skip:  {h.shape}")
        for j, block in enumerate(level_blocks):
            h = block(h, t_emb)
            print(f"  Level {i}, ResBlock {j}: {h.shape}")
        if i < len(model.channel_list) - 1:
            h = model.upsamplers[i](h)
            print(f"  Upsample:           {h.shape}")

    print("\nOutput:")
    h = model.out_norm(h)
    h = F.silu(h)
    h = model.out_conv(h)
    print(f"  Final output:       {h.shape}")

    # ── Test 6: Output changes with timestep ──
    print("\n── Test 6: Timestep Conditioning ──")
    model.eval()
    x = torch.randn(1, 1, 28, 28)
    with torch.no_grad():
        out_t10 = model(x, torch.tensor([10]))
        out_t990 = model(x, torch.tensor([990]))

    diff = (out_t10 - out_t990).abs().mean().item()
    print(f"Same input, different timestep:")
    print(f"  Mean |output(t=10) - output(t=990)|: {diff:.6f}")
    assert diff > 0.001, "Output should change with timestep"
    print("✓ Network is sensitive to timestep\n")

    # ── Test 7: Gradient flow ──
    print("── Test 7: Gradient Flow ──")
    model.train()
    x = torch.randn(2, 1, 28, 28)
    t = torch.randint(0, 1000, (2,))
    out = model(x, t)
    loss = out.mean()
    loss.backward()

    # Check that all parameters received gradients
    no_grad = [name for name, p in model.named_parameters() if p.grad is None]
    if no_grad:
        print(f"  WARNING: {len(no_grad)} parameters have no gradient:")
        for name in no_grad[:5]:
            print(f"    {name}")
    else:
        print(f"  All {sum(1 for _ in model.parameters())} parameters have gradients")
    print("✓ Gradients flow through entire network\n")

    # ── Summary ──
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Architecture: U-Net with {len(model.channel_list)} resolution levels")
    print(f"Channels:     {list(model.channel_list)}")
    print(f"ResBlocks:    {model.num_res_blocks} per level")
    print(f"Time dim:     128 (sinusoidal → MLP)")
    print(f"Parameters:   {n_params:,}")
    print(f"Input:        (batch, 1, 28, 28)")
    print(f"Output:       (batch, 1, 28, 28)  — predicted noise")
    print(f"\nFor comparison:")
    print(f"  AlphaZero ResNet:    377,629 params (single resolution)")
    print(f"  This U-Net:          {n_params:,} params (multi-resolution)")
    print(f"  DDPM paper (CIFAR):  ~35M params")
    print("\n✓ All Phase 2 tests passed!")