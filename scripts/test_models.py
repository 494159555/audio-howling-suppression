"""
Simple test script to verify all models can be imported and initialized.

Usage:
    python -m scripts.test_models

Author: Research Team
Date: 2026-3-23
Version: 1.0.0
"""

import torch
from src.models import (
    AudioUNet3,
    AudioUNet5,
    AudioUNet5Attention,
    AudioUNet5Residual,
    AudioUNet5Dilated,
    AudioUNet5Optimized,
)

print("="*80)
print("Testing All U-Net Models")
print("="*80)

models = [
    ("AudioUNet3", AudioUNet3),
    ("AudioUNet5", AudioUNet5),
    ("AudioUNet5Attention", AudioUNet5Attention),
    ("AudioUNet5Residual", AudioUNet5Residual),
    ("AudioUNet5Dilated", AudioUNet5Dilated),
    ("AudioUNet5Optimized", AudioUNet5Optimized),
]

sample_input = torch.randn(2, 1, 256, 100)
print(f"\nSample input shape: {sample_input.shape}\n")

for name, model_class in models:
    try:
        model = model_class()
        output = model(sample_input)
        params = sum(p.numel() for p in model.parameters())
        print(f"✓ {name:<25} | Params: {params:>10,} | Output: {tuple(output.shape)}")
    except Exception as e:
        print(f"✗ {name:<25} | Error: {str(e)[:50]}")

print("\n" + "="*80)
print("All models tested!")
print("="*80)