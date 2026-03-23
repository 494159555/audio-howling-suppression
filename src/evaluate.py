'''
模型评估模块 - 音频啸叫抑制模型验证与测试

文件功能：
- 加载训练好的模型并在验证集上进行性能评估
- 计算模型在验证集上的平均Log-L1 Loss
- 提供命令行接口，支持指定模型检查点路径

主要组件：
- evaluate_model函数：核心评估函数，负责模型加载和数据评估
- 命令行参数解析：支持通过--checkpoint参数指定模型文件

重要参数：
函数参数：
- checkpoint_path: 模型检查点文件路径(.pth文件)
- batch_size: 批大小(默认4)

评估配置：
- 使用验证集数据进行评估(cfg.VAL_CLEAN_DIR, cfg.VAL_NOISY_DIR)
- 损失函数：L1Loss在Log域计算
- 设备：自动选择CUDA或CPU

输出结果：
- 验证集平均Log-L1 Loss
- 详细的评估过程日志
- 模型文件名和性能指标

使用方法：
命令行使用：
python -m src.evaluate --checkpoint experiments\exp_20251212_032136_unet5\checkpoints\best_model.pth

代码调用：
from src.evaluate import evaluate_model
avg_loss = evaluate_model("path/to/model.pth", batch_size=4)
'''

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from src.config import cfg
from src.models.unet_v2 import AudioUNet5
from src.dataset import HowlingDataset


def evaluate_model(checkpoint_path, batch_size=4):
    """
    加载模型并在验证集上计算平均 Loss
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
