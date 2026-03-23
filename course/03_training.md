# Chapter 3: Training — Teaching the Network to Predict Noise

## The Problem

We have a forward process that destroys images (Chapter 1) and a U-Net that takes noised images as input (Chapter 2). Now we need to train the U-Net to predict what noise was added — so that later (Chapter 4) we can reverse the process and generate new images.

The training objective is the simplest part of the entire diffusion pipeline. After the mathematical machinery of the noise schedule and the architectural complexity of the U-Net, the training loop is almost anticlimactically straightforward.

## Part 1: The Training Objective

### What the Network Learns

At each training step:

1. Sample a clean image x₀ from the dataset
2. Sample a random timestep t ~ Uniform(0, 999)
3. Sample random noise ε ~ N(0, I)
4. Create the noised image: x_t = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε
5. Ask the network to predict the noise: ε_pred = network(x_t, t)
6. Loss = MSE(ε_pred, ε)

That's it. The network learns to answer: "Given this noised image and this noise level, what noise was added?"

### The Loss Function

```python
loss = F.mse_loss(noise_pred, noise)
```

Mean squared error between predicted noise and actual noise, averaged over all pixels and all samples in the batch. No weighting, no tricks.

### A Worked Example

Consider a tiny 2×2 grayscale image:

```
x₀ = [[0.8, 0.2],      (clean image)
       [0.5, 0.9]]

t = 500  →  √ᾱ₅₀₀ ≈ 0.707,  √(1-ᾱ₅₀₀) ≈ 0.707

ε = [[-0.3, 1.2],       (random noise, sampled from N(0,1))
     [ 0.8, -0.5]]

x_t = 0.707 · x₀ + 0.707 · ε
    = [[0.707·0.8 + 0.707·(-0.3),  0.707·0.2 + 0.707·1.2],
       [0.707·0.5 + 0.707·0.8,     0.707·0.9 + 0.707·(-0.5)]]
    = [[0.354,  0.990],
       [0.919,  0.283]]

Network sees x_t and t=500, predicts:
ε_pred = [[-0.25, 1.10],
          [ 0.75, -0.60]]

Loss = mean((ε_pred - ε)²)
     = mean((-0.25-(-0.3))², (1.10-1.2)², (0.75-0.8)², (-0.60-(-0.5))²)
     = mean(0.0025, 0.01, 0.0025, 0.01)
     = 0.00625
```

Small loss — the network's prediction was close to the actual noise.

### Why Predict Noise, Not the Clean Image?

Three equivalent formulations exist. Given `x_t = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε`:

1. **Predict ε** (noise): Given x_t, predict ε. Then x₀ = (x_t - √(1-ᾱ_t)·ε_pred) / √ᾱ_t
2. **Predict x₀** (clean image): Given x_t, predict x₀ directly. Then ε = (x_t - √ᾱ_t·x₀_pred) / √(1-ᾱ_t)
3. **Predict score** ∇log p(x_t): The gradient of the log probability density.

All three are mathematically equivalent — you can convert between them. But predicting ε is preferred because:

**Consistent target scale.** The noise ε is always drawn from N(0, I), so the target has mean 0 and variance 1 regardless of the timestep or image content. Predicting x₀ would have targets with wildly different scales — bright images have different pixel ranges than dark images, and the "correct" x₀ might be hard to pin down at high noise levels.

**Numerically stable.** At high noise (large t), √ᾱ_t is tiny. Predicting x₀ would require dividing by this tiny number, amplifying errors. Predicting ε avoids this division during training.

## Part 2: The Training Loop

```python
for images, _labels in train_loader:
    # 1. Random timesteps — each image gets a different t
    t = torch.randint(0, 1000, (batch_size,), device=device)

    # 2. Random noise
    noise = torch.randn_like(images)

    # 3. Forward process — create noised images
    x_t = sqrt_alpha[t].view(-1,1,1,1) * images + sqrt_one_minus_alpha[t].view(-1,1,1,1) * noise

    # 4. Network prediction
    noise_pred = model(x_t, t)

    # 5. Loss and update
    loss = F.mse_loss(noise_pred, noise)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
```

### Why Random Timesteps Per Image?

Each image in the batch gets a different random timestep. This means the network simultaneously practices:
- Predicting tiny noise at t=10 (image barely changed)
- Predicting moderate noise at t=500 (half signal, half noise)
- Predicting large noise at t=990 (image almost gone)

Over many batches, every timestep gets roughly equal representation. The network learns to handle all noise levels, not just one.

### Why We Ignore Labels

```python
for images, _labels in train_loader:
```

The underscore prefix on `_labels` signals "we don't use this." For unconditional generation, the network doesn't know what digit or object it's looking at. It just learns the general structure of images — edges, textures, shapes — and how to separate that structure from noise.

This is purely self-supervised learning. The training signal comes entirely from the noise we added, not from any human-provided labels.

## Part 3: EMA — Exponential Moving Average

### The Problem EMA Solves

SGD updates are noisy — each batch pushes the weights in a slightly different direction. The current weights are optimized for the most recent few batches, not the overall dataset. For classification, this noise doesn't matter much (accuracy is robust to small weight perturbations). For generation, it matters a lot — sampling involves 50-1000 sequential network calls, and small weight noise compounds across steps.

### The Solution

Maintain a shadow copy of the weights that's a running average:

```
θ_ema = decay · θ_ema + (1 - decay) · θ_current
```

With decay = 0.9999, each update moves θ_ema by only 0.01% toward the current weights. This effectively averages over the last ~10,000 updates, smoothing out training noise.

### A Numerical Example

Suppose one weight bounces between 0.5 and 0.7 across training steps:

```
Step 1: θ = 0.50  →  θ_ema = 0.9999 × 0 + 0.0001 × 0.50 = 0.00005
Step 2: θ = 0.70  →  θ_ema = 0.9999 × 0.00005 + 0.0001 × 0.70 ≈ 0.00012
...after 10,000 steps bouncing 0.5-0.7...
θ_ema ≈ 0.60     (the average, stable)
```

The EMA weights converge to the average behavior, filtering out the noise.

### Usage Pattern

```python
ema = EMA(model, decay=0.9999)

# During training:
optimizer.step()
ema.update(model)          # blend current weights into EMA

# During sampling:
ema.apply_shadow(model)    # swap in EMA weights
images = sample(model)     # generate with smooth weights
ema.restore(model)         # swap back to training weights
```

We train with the raw weights (they need to be responsive to gradients) but sample with the EMA weights (they need to be stable). The swap-in/swap-out pattern keeps both sets available.

### Connection to Your Previous Projects

In DQN (your RL project), you used a target network — a copy of the Q-network updated periodically. EMA is the continuous version of the same idea: instead of copying weights every N steps, we blend them continuously. The motivation is identical: stability. DQN needed stable targets for Q-learning; diffusion needs stable weights for sequential sampling.

## Part 4: Gradient Clipping

```python
nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

Occasionally, a batch produces unusually large gradients — maybe a very noisy image at a high timestep creates an extreme prediction. Without clipping, this one bad batch could move the weights far from their current position, undoing many previous updates.

Gradient clipping computes the global norm of all gradients and rescales them if the norm exceeds 1.0:

```
global_norm = √(Σ ||grad_i||²)    for all parameters i

If global_norm > 1.0:
    scale all gradients by (1.0 / global_norm)
```

This preserves the direction of the gradient update but caps its magnitude. The max norm of 1.0 is standard for diffusion models.

## Part 5: Training Dynamics — What Actually Happens

### Loss Progression

```
Epoch 1:   loss ≈ 0.053    Network outputs near-random noise predictions
Epoch 5:   loss ≈ 0.026    Learning basic image structure (edges, flat regions)
Epoch 10:  loss ≈ 0.023    Fine-tuning mid-frequency patterns
Epoch 20:  loss ≈ 0.022    Diminishing returns — most patterns learned
Epoch 50:  loss ≈ 0.021    Slow refinement of subtle details
```

The rapid drop in the first 5 epochs is the network learning obvious patterns: "backgrounds should be dark, digit strokes should be bright, noise is everywhere else." The slow decline afterward is learning subtler patterns: "the curve of a 2 transitions smoothly," "a 7's horizontal bar connects to its vertical stroke."

### What the Loss Number Means

MSE loss of 0.021 means the average squared error per pixel is 0.021. The average absolute error is √0.021 ≈ 0.145. Since the noise is drawn from N(0, 1), this means the network's prediction is typically off by about 0.15 standard deviations — quite good. It's predicting about 85% of the noise correctly.

### Why Sampling Quality ≠ Training Loss

A loss of 0.022 at epoch 10 and 0.021 at epoch 50 looks similar numerically, but the sample quality is dramatically different. Here's why:

During sampling, the network is called 50-1000 times sequentially. Each small prediction error compounds: if step 999 is slightly wrong, step 998 starts from a slightly wrong position, makes its own error, and so on. The errors accumulate over hundreds of steps.

A 5% improvement in per-step accuracy translates to a much larger improvement in final sample quality because the improvement compounds across all steps. This is why 50 epochs produces much cleaner samples than 10 epochs despite nearly identical loss numbers.

## Part 6: Data Normalization

```python
transforms.Normalize((0.5,), (0.5,))  # [0, 1] → [-1, 1]
```

Images from `ToTensor()` are in [0, 1]. We shift to [-1, 1] by subtracting 0.5 and dividing by 0.5.

Why: the forward process assumes images and noise share the same center (zero). Gaussian noise has mean 0. If images have mean 0.5, the forward process would be asymmetric — noise would push pixels toward 0 but images would be centered at 0.5. Centering both at 0 makes the math clean and the network's job symmetric.

## Part 7: Hardware Considerations

### MNIST on CPU

Our initial plan estimated 30-40 minutes for MNIST training on CPU. The actual time was ~35 minutes **per epoch** — far too slow for 50 epochs. The 4.5M parameter U-Net, while small by modern standards, requires substantial compute for each forward+backward pass at 28×28 resolution.

Lesson learned: even "small" generative models need GPU acceleration for reasonable training times. We moved MNIST training to Kaggle T4, where 50 epochs completed in ~15 minutes.

### GPU Acceleration

The T4 GPU accelerates training through:
- Batched matrix multiplications (128 images processed simultaneously)
- Fast memory access (HBM vs DDR4)
- Parallel convolution operations (thousands of CUDA cores)

The code changes are minimal — mostly `.to(device)` calls and `pin_memory=True` in the DataLoader. The algorithm is identical; only the hardware differs.

## Summary

| Concept | What | Why |
|---------|------|-----|
| Training objective | MSE(ε_pred, ε) | Simple, stable, well-scaled target |
| Random timesteps | t ~ Uniform(0, T-1) per image | Network learns all noise levels |
| Noise prediction | Predict ε, not x₀ | Consistent target scale, numerically stable |
| EMA (decay=0.9999) | Running average of weights | Stable weights for sampling |
| Gradient clipping (1.0) | Cap gradient norm | Prevent rare large batches from destabilizing |
| Data normalization | Images to [-1, 1] | Center data at 0, matching noise distribution |
| Adam optimizer (lr=2e-4) | Adaptive learning rates | Standard for diffusion, stable convergence |

## What's Next

In [Chapter 4](04_sampling.md), we reverse the forward process to generate images. Starting from pure Gaussian noise, we iteratively denoise using the trained network. Two algorithms: DDPM (1000 steps, highest quality) and DDIM (50 steps, 20× faster, nearly identical quality). DDIM also enables deterministic generation and smooth interpolation between images.