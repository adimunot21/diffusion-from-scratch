# Chapter 6: Conditional Generation — Controlling What the Model Creates

## The Problem

Our unconditional CIFAR-10 model generates random images — sometimes a car, sometimes a frog, you can't control it. We want to say "generate a horse" and get a horse.

More broadly, the problem of conditional generation is: how do you steer a generative model to produce specific outputs? For CIFAR-10, the condition is a class label. For Stable Diffusion, it's a text prompt. For Diffusion Policy in robotics, it's the current state of the world. The conditioning mechanism is the same in all cases.

## Part 1: Adding Class Information to the U-Net

### Class Embedding

A class label is an integer (0-9). Just like timesteps, raw integers aren't rich enough for a neural network to condition on. We use a learned embedding:

```python
self.class_embed = nn.Embedding(num_classes + 1, time_dim)  # 11 × 256
```

This is a lookup table with 11 rows (10 classes + 1 "null" class). Each row is a 256-dimensional vector that the network learns during training. It's the same concept as word embeddings in your Transformer project: each class gets its own learned representation.

The +1 is for the null class (index 10), used during unconditional training — explained in Part 2.

### Combining Time and Class

The class embedding is simply added to the time embedding:

```python
t_emb = self.time_embed(t)           # (batch, 256) — "how noisy?"
c_emb = self.class_embed(labels)     # (batch, 256) — "what class?"
cond = t_emb + c_emb                 # (batch, 256) — combined signal
```

This combined vector is injected into every ResBlock, exactly where the time embedding was before. The network receives both "it's timestep 500" and "it should be a horse" through the same conditioning pathway.

### Why Addition, Not Concatenation?

Concatenation would double the conditioning dimension (256 + 256 = 512), requiring changes to every ResBlock's time projection layer. Addition keeps the dimension at 256 and is simpler. It works because the time and class embeddings are learned in the same space — they're both 256-dimensional vectors, and the network learns to disentangle the time and class information through their different patterns of activation.

This is standard in diffusion implementations. Stable Diffusion also adds text embeddings to time embeddings.

## Part 2: Classifier-Free Guidance — The Key Technique

### The Problem with Naive Conditioning

If we always provide the class label during training, the model CAN use it to guide generation. But in practice, it often learns to weakly depend on the label — the condition helps a little but doesn't strongly steer the output. Generated images might be vaguely class-appropriate but not clearly recognizable.

Why? The training loss (MSE on noise prediction) doesn't explicitly reward class adherence. It just rewards accurate noise prediction. The model might learn that ignoring the class label and predicting "generic" noise is good enough for a low loss.

### The Solution: Random Label Dropout

During training, we randomly replace the real class label with a "null" label 10% of the time:

```python
drop_mask = torch.rand(batch_size) < 0.1           # 10% probability
labels = torch.where(drop_mask,
                     torch.full_like(labels, 10),   # null class (index 10)
                     labels)                         # real class
```

This teaches the model two capabilities:
1. **Conditional mode** (90% of training): "Given that this is a horse, predict the noise"
2. **Unconditional mode** (10% of training): "Without knowing the class, predict the noise"

The model stores both capabilities in the same weights. The class embedding for index 10 (null) effectively means "I don't know what class this is."

### Guidance at Inference Time

At sampling time, we run the model TWICE per denoising step:

```python
# 1. Conditional: "what noise if this is class c?"
noise_cond = model(x_t, t, class_labels)

# 2. Unconditional: "what noise without knowing the class?"
noise_uncond = model(x_t, t, null_labels)

# 3. Guided: amplify the difference
noise_guided = noise_uncond + w * (noise_cond - noise_uncond)
```

### What the Guidance Formula Does

Let's trace through with concrete numbers for a single pixel:

```
noise_uncond = 0.50    "generic noise prediction (no class info)"
noise_cond   = 0.65    "noise prediction knowing it's a horse"

The difference: 0.65 - 0.50 = 0.15
    This is "what the class label CHANGES about the prediction"
    It's the "horse direction" in noise space

With w = 1.0 (no guidance):
    noise = 0.50 + 1.0 × 0.15 = 0.65   (just the conditional prediction)

With w = 5.0 (moderate guidance):
    noise = 0.50 + 5.0 × 0.15 = 1.25   (strongly pushed toward "horse")

With w = 10.0 (strong guidance):
    noise = 0.50 + 10.0 × 0.15 = 2.00  (very strongly pushed toward "horse")
```

Higher `w` means: "be MORE like a horse, even at the cost of naturalness." The model overshoots what a real horse looks like, producing oversaturated but very class-consistent images.

### The Guidance Scale Tradeoff

```
w = 1.0:  No guidance. Conditional prediction is used as-is.
          Result: diverse but sometimes wrong class. Low class adherence.

w = 3.0:  Moderate guidance. Classes are recognizable.
          Result: good balance of quality and class adherence.

w = 5.0:  Standard guidance. Clear class identity.
          Result: strong class adherence, good diversity.

w = 7-10: Strong guidance. Very clear class, but less natural.
          Result: oversaturated colors, reduced diversity, but
          unmistakable class identity.
```

This is exactly the "CFG scale" slider in Stable Diffusion. When users adjust that slider from 7 to 15, they're doing the same thing — amplifying the text condition's influence on generation.

## Part 3: Training Implementation

The training loop is almost identical to the unconditional version. The only change is the label dropout:

```python
for images, labels in train_loader:
    # Random label dropout for classifier-free guidance
    drop_mask = torch.rand(batch_size) < 0.1
    labels = torch.where(drop_mask, torch.full_like(labels, null_label), labels)

    # Forward diffusion (identical)
    t = torch.randint(0, 1000, (batch_size,))
    noise = torch.randn_like(images)
    x_t = sqrt_alpha[t] * images + sqrt_one_minus_alpha[t] * noise

    # Predict noise (now with class label)
    noise_pred = model(x_t, t, labels)
    loss = F.mse_loss(noise_pred, noise)

    # Update (identical)
    loss.backward()
    optimizer.step()
```

The only new lines are the `drop_mask` and `torch.where`. Everything else — diffusion schedule, loss function, optimizer, EMA — is unchanged.

### Loss Curve

The conditional model's loss curve is virtually identical to the unconditional model's:

```
Unconditional: 0.130 → 0.055 over 100 epochs
Conditional:   0.127 → 0.055 over 100 epochs
```

The 10% label dropout adds negligible noise to training. The model simply learns to handle an additional input (the class embedding) without any change in difficulty.

## Part 4: Sampling with Guidance

```python
def guided_sample(model, schedule, class_labels, guidance_scale=5.0, ddim_steps=50):
    null_labels = torch.full_like(class_labels, model.num_classes)  # index 10
    x = torch.randn(num_images, 3, 32, 32)

    for i in reversed(range(len(timestep_seq))):
        # Two forward passes per step
        noise_cond = model(x, t, class_labels)
        noise_uncond = model(x, t, null_labels)

        # Classifier-free guidance
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

        # DDIM update (identical to unconditional)
        x0_pred = (x - sqrt_one_minus_alpha * noise_pred) / sqrt_alpha
        x = sqrt_alpha_prev * x0_pred + dir_xt

    return x
```

The cost: 2× forward passes per step (one conditional, one unconditional). For 50 DDIM steps, that's 100 forward passes instead of 50. Worth it for the dramatic improvement in class adherence.

## Part 5: Results

### Class Grid (10 Classes × 8 Samples, w=5.0)

Our final results showed clear class consistency:

```
Airplane row:    Blue skies, plane silhouettes, consistent across samples
Automobile row:  Car shapes on roads, correct proportions
Bird row:        Birds on branches, sky backgrounds
Cat row:         Face close-ups with ears and eyes
Deer row:        Brown animals in green/forest settings
Dog row:         Face close-ups, multiple breeds
Frog row:        Green/orange frogs with textured skin
Horse row:       Full body shots on grass, remarkably consistent
Ship row:        Blue water with boat shapes
Truck row:       Red/colored vehicles, boxy proportions
```

Without guidance (w=1.0), classes would be mixed — an "airplane" row would have some cars and birds mixed in. With w=5.0, the class identity is clear in nearly every sample.

### FID Comparison

```
Unconditional CIFAR-10: FID = 71.2
Conditional CIFAR-10 (w=5.0): FID = 65.3
```

The conditional model has a LOWER (better) FID, confirming that guidance genuinely improves quality. The condition gives the network additional information that helps it generate more realistic images — knowing "this should be a horse" lets it commit to horse-like features rather than hedging between multiple classes.

## Part 6: Connection to Modern Diffusion Systems

### Stable Diffusion

Replace our class embedding with a text encoder (CLIP):

```
Our system:        class_label (integer) → nn.Embedding → 256-dim vector
Stable Diffusion:  text_prompt (string)  → CLIP encoder → 768-dim vectors
```

The rest is identical: the text embedding is injected into the U-Net (via cross-attention instead of addition), training uses random text dropout (10%), and sampling uses classifier-free guidance with a guidance scale (typically w=7.5).

### Diffusion Policy (Robotics)

Replace our class embedding with a state encoder:

```
Our system:       class_label → embedding → U-Net → predicted image noise
Diffusion Policy: robot_state → encoder   → U-Net → predicted action noise
```

Instead of denoising noise into an image, Diffusion Policy denoises noise into an action trajectory — a sequence of robot joint positions or end-effector poses. The conditioning is the current state (camera image + robot joint positions), and guidance pushes the generated actions toward state-appropriate behavior.

The mathematical framework is identical. If you understand this chapter, you understand the core of Diffusion Policy.

## Summary

| Concept | What | Why |
|---------|------|-----|
| Class embedding | nn.Embedding(11, 256) | Learnable class representations |
| Time + class addition | cond = t_emb + c_emb | Single conditioning signal for ResBlocks |
| 10% label dropout | Replace label with "null" randomly | Teach unconditional mode for guidance |
| Null class (index 10) | "I don't know the class" | Unconditional prediction baseline |
| Guidance formula | ε_uncond + w·(ε_cond - ε_uncond) | Amplify class signal at inference |
| Guidance scale w | Controls class adherence vs diversity | Higher w = more class-consistent |
| 2× forward passes | Conditional + unconditional per step | Required for guidance computation |

## What's Next

In [Chapter 7](07_evaluation.md), we evaluate everything we built: FID scores for quantitative quality measurement, visual comparisons between real and generated images, analysis of what works and what could be improved, and connections to the broader field of generative modeling.