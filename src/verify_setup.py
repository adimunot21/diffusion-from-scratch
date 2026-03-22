"""
Phase 0: Verify that all dependencies are installed and working.
Downloads MNIST to confirm torchvision dataset access.
"""

import sys
print(f"Python: {sys.version}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

import torchvision
print(f"torchvision: {torchvision.__version__}")

import numpy as np
print(f"NumPy: {np.__version__}")

import matplotlib
print(f"matplotlib: {matplotlib.__version__}")

# Test basic tensor operations
x = torch.randn(2, 3, 28, 28)  # batch of 2, 1-channel 28x28 (like MNIST but 3-ch)
print(f"\nRandom tensor shape: {x.shape}")
print(f"Mean: {x.mean():.4f}, Std: {x.std():.4f}")

# Download MNIST (small, ~50MB) to verify dataset access
print("\nDownloading MNIST (first time only)...")
from torchvision import datasets, transforms
mnist = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.ToTensor(),
)
print(f"MNIST training set: {len(mnist)} images")
img, label = mnist[0]
print(f"Single image shape: {img.shape} (channels, height, width)")
print(f"Pixel range: [{img.min():.1f}, {img.max():.1f}]")
print(f"Label: {label}")

# Quick matplotlib test
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 5, figsize=(10, 2))
for i in range(5):
    img, label = mnist[i]
    axes[i].imshow(img.squeeze(0), cmap="gray")
    axes[i].set_title(f"Label: {label}")
    axes[i].axis("off")
plt.suptitle("MNIST Samples", fontweight="bold")
plt.tight_layout()
plt.savefig("notebooks/mnist_samples.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved MNIST samples to notebooks/mnist_samples.png")

print("\n✓ All checks passed! Ready for Phase 1.")