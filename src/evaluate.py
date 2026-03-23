"""
Model Evaluation Module - Audio Howling Suppression Model Validation and Testing

This module provides evaluation functionality for trained audio howling suppression models,
including model loading, validation set performance evaluation, and command-line interface
for specifying model checkpoint paths.

File Functions:
- Load trained models and evaluate performance on validation set
- Calculate average Log-L1 Loss on validation set
- Provide command-line interface for specifying model checkpoint paths

Main Components:
- evaluate_model function: Core evaluation function for model loading and data evaluation
- Command-line argument parsing: Supports --checkpoint parameter for model file specification

Important Parameters:
Function Parameters:
- checkpoint_path: Model checkpoint file path (.pth file)
- batch_size: Batch size (default: 4)

Evaluation Configuration:
- Uses validation set data for evaluation (cfg.VAL_CLEAN_DIR, cfg.VAL_NOISY_DIR)
- Loss function: L1Loss computed in Log domain
- Device: Automatically selects CUDA or CPU

Output Results:
- Validation set average Log-L1 Loss
- Detailed evaluation process logs
- Model filename and performance metrics

Usage:
Command line:
    python -m src.evaluate --checkpoint experiments\exp_20251212_032136_unet5\checkpoints\best_model.pth

Code call:
    from src.evaluate import evaluate_model
    avg_loss = evaluate_model("path/to/model.pth", batch_size=4)

Author: Research Team
Date: 2026-3-23
Version: 2.0.0
"""

# Standard library imports
import argparse
import os

# Third-party imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Local imports
from src.config import cfg
from src.dataset import HowlingDataset
from src.models.unet_v2 import AudioUNet5


def evaluate_model(checkpoint_path, batch_size=4):
    """Load model and evaluate on validation set.
    
    Loads a trained model checkpoint and evaluates its performance on the validation
    dataset, computing the average Log-L1 loss.
    
    Args:
        checkpoint_path (str): Path to the model checkpoint file (.pth)
        batch_size (int, optional): Batch size for evaluation. Defaults to 4.
        
    Returns:
        float: Average Log-L1 loss on validation set, or None if evaluation fails
    """
    device = cfg.DEVICE
    print(f"正在使用设备: {device}")

    # 1. 准备数据
    # 检查验证集路径是否存在，不存在则警告
    if not os.path.exists(cfg.VAL_CLEAN_DIR):
        print(f"警告：验证集路径 {cfg.VAL_CLEAN_DIR} 不存在。")
        print("请确保你已经在 data/processed/val 下放入了验证数据。")
        return

    val_dataset = HowlingDataset(
        clean_dir=cfg.VAL_CLEAN_DIR,
        howling_dir=cfg.VAL_NOISY_DIR,
        sample_rate=cfg.SAMPLE_RATE,
        chunk_len=cfg.CHUNK_LEN,
        n_fft=cfg.N_FFT,
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=cfg.NUM_WORKERS
    )

    # 2. 加载模型结构
    print("正在构建模型 AudioUNet5...")
    model = AudioUNet5().to(device)

    # 3. 加载权重
    if os.path.exists(checkpoint_path):
        print(f"正在加载权重: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
    else:
        print(f"错误：找不到权重文件 {checkpoint_path}")
        return

    # 4. 评估循环
    model.eval()
    criterion = nn.L1Loss()
    total_loss = 0.0

    print("开始评估...")
    with torch.no_grad():
        for batch_idx, (noisy_mag, clean_mag) in enumerate(val_loader):
            noisy_mag = noisy_mag.to(device)
            clean_mag = clean_mag.to(device)

            # 推理
            pred_mag = model(noisy_mag)

            # 计算 Log 域 Loss (保持和你训练逻辑一致)
            loss = criterion(
                torch.log10(pred_mag + 1e-8), torch.log10(clean_mag + 1e-8)
            )

            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    print("==========================================")
    print("验证集评估完成")
    print("模型文件: {}".format(os.path.basename(checkpoint_path)))
    print("平均 Log-L1 Loss: {:.4f}".format(avg_loss))
    print("==========================================")

    return avg_loss


if __name__ == "__main__":
    # 使用 argparse 允许命令行传参
    parser = argparse.ArgumentParser(description="Evaluate AudioUNet Model")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the .pth model file"
    )

    args = parser.parse_args()

    evaluate_model(args.checkpoint)
