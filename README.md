# Diffusion Models from Scratch

A from-scratch implementation of Denoising Diffusion Probabilistic Models (DDPM) — the same core algorithm behind Stable Diffusion, DALL-E, and Diffusion Policy for robotics. The model learns to generate images by learning to reverse a gradual noising process, starting from pure Gaussian noise.

## Status

🚧 **In progress** — Phase 0 (Setup) complete.

## Project Structure

```
diffusion-from-scratch/
├── src/
│   ├── diffusion.py       ← Forward process, noise schedules
│   ├── unet.py            ← U-Net denoiser architecture
│   ├── train.py           ← Training loop + EMA
│   ├── sample.py          ← DDPM and DDIM sampling
│   ├── train_cifar.py     ← CIFAR-10 training (GPU)
│   ├── conditional.py     ← Class conditioning + classifier-free guidance
│   └── evaluate.py        ← FID computation, analysis
├── checkpoints/           ← Saved models (gitignored)
├── notebooks/             ← Generated images, plots
├── course/                ← Detailed written course
└── requirements.txt
```

## Setup

```bash
conda create -n diffusion python=3.11 -y
conda activate diffusion
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install "numpy<2" matplotlib tqdm
```

## Built With

- PyTorch (nn.Conv2d, nn.Linear, nn.GroupNorm — no diffusion libraries)
- NumPy, matplotlib
- Trained on Kaggle T4 GPU (CIFAR-10) and CPU (MNIST)