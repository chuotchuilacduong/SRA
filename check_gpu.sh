#!/bin/bash
# Check GPU availability and configuration

echo "=========================================="
echo "GPU Configuration Check"
echo "=========================================="
echo ""

/Users/hiro/miniconda3/envs/linearag311/bin/python3 << 'PY'
import torch
import sys

print("PyTorch Information:")
print(f"  Version: {torch.__version__}")
print()

print("Metal GPU (macOS):")
print(f"  Available: {torch.backends.mps.is_available()}")
print(f"  Built: {torch.backends.mps.is_built()}")
print()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Active Device: {device}")
print()

if torch.backends.mps.is_available():
    print("✅ Metal GPU is ready!")
    # Test GPU
    try:
        x = torch.randn(100, 100, device='mps')
        y = torch.randn(100, 100, device='mps')
        z = torch.matmul(x, y)
        print("✅ GPU tensor operations working")
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
else:
    print("⚠️  Metal GPU not available, using CPU")

print()
print("Other GPU/Accelerators:")
print(f"  CUDA available: {torch.cuda.is_available()}")

PY

echo "=========================================="
