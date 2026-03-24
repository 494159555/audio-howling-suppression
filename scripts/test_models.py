"""模型测试脚本

验证所有模型是否可以正确导入和初始化
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
print("测试所有U-Net模型")
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
print(f"\n样本输入形状: {sample_input.shape}\n")

for name, model_class in models:
    try:
        model = model_class()
        output = model(sample_input)
        params = sum(p.numel() for p in model.parameters())
        print(f"✓ {name:<25} | 参数量: {params:>10,} | 输出: {tuple(output.shape)}")
    except Exception as e:
        print(f"✗ {name:<25} | 错误: {str(e)[:50]}")

print("\n" + "="*80)
print("所有模型测试完成!")
print("="*80)