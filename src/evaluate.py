"""模型评估模块

评估训练好的音频啸叫抑制模型
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import cfg
from src.dataset import HowlingDataset
from src.models.unet_v2 import AudioUNet5


def evaluate_model(checkpoint_path, batch_size=4):
    """在验证集上评估模型
    
    Args:
        checkpoint_path: 模型检查点路径
        batch_size: 批大小
        
    Returns:
        平均Log-L1损失
    """
    device = cfg.DEVICE
    print(f"正在使用设备: {device}")

    # 1. 准备数据
    if not os.path.exists(cfg.VAL_CLEAN_DIR):
        print(f"警告：验证集路径 {cfg.VAL_CLEAN_DIR} 不存在")
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

    # 2. 加载模型
    print("正在构建模型 AudioUNet5...")
    model = AudioUNet5().to(device)

    if os.path.exists(checkpoint_path):
        print(f"正在加载权重: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
    else:
        print(f"错误：找不到权重文件 {checkpoint_path}")
        return

    # 3. 评估
    model.eval()
    criterion = nn.L1Loss()
    total_loss = 0.0

    print("开始评估...")
    with torch.no_grad():
        for batch_idx, (noisy_mag, clean_mag) in enumerate(val_loader):
            noisy_mag = noisy_mag.to(device)
            clean_mag = clean_mag.to(device)

            pred_mag = model(noisy_mag)

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
    parser = argparse.ArgumentParser(description="评估AudioUNet模型")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="模型检查点路径"
    )

    args = parser.parse_args()

    evaluate_model(args.checkpoint)