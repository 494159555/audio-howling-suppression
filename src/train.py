"""训练模块

音频啸叫抑制模型训练脚本，支持实验管理、监控和可视化
"""

import argparse
from datetime import datetime
import os
import shutil
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from src.config import cfg
from src.dataset import HowlingDataset
from src.models import (
    AudioUNet3,
    AudioUNet5,
    AudioUNet5Attention,
    AudioUNet5Residual,
    AudioUNet5Dilated,
    AudioUNet5Optimized,
    AudioUNet5LSTM,
    AudioUNet5TemporalAttention,
    AudioUNet5ConvLSTM,
    AudioUNet5GAN,
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='音频啸叫抑制模型训练脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=cfg.DEFAULT_MODEL,
        choices=list(cfg.AVAILABLE_MODELS.keys()),
        help=f'选择模型 (默认: {cfg.DEFAULT_MODEL})'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='配置文件路径 (YAML格式)'
    )
    
    parser.add_argument('--lr', '--learning-rate', type=float, default=None,
                       help='学习率')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='批大小')
    parser.add_argument('--epochs', type=int, default=None,
                       help='训练轮数')
    parser.add_argument('--num-workers', type=int, default=None,
                       help='数据加载线程数')
    
    parser.add_argument('--exp-name', type=str, default=None,
                       help='实验名称 (默认自动生成)')
    
    return parser.parse_args()


def load_config_from_yaml(config_path: str) -> dict:
    """从YAML文件加载配置"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def train() -> None:
    """训练模型"""
    args = parse_args()
    
    model_name = args.model
    
    # 加载配置文件
    config_override = {}
    if args.config is not None:
        file_config = load_config_from_yaml(args.config)
        config_override.update(file_config)
    
    # 命令行参数覆盖
    if args.lr is not None:
        config_override['learning_rate'] = args.lr
    if args.batch_size is not None:
        config_override['batch_size'] = args.batch_size
    if args.epochs is not None:
        config_override['num_epochs'] = args.epochs
    if args.num_workers is not None:
        config_override['num_workers'] = args.num_workers
    
    # 应用配置覆盖
    if config_override:
        print("📝 使用自定义配置:")
        for key, value in config_override.items():
            print(f"  {key}: {value}")
        print()
    
    # 获取模型类
    model_class_name = cfg.AVAILABLE_MODELS[model_name]
    
    model_class = None
    for cls in [AudioUNet3, AudioUNet5, AudioUNet5Attention, 
                AudioUNet5Residual, AudioUNet5Dilated, AudioUNet5Optimized,
                AudioUNet5LSTM, AudioUNet5TemporalAttention, 
                AudioUNet5ConvLSTM, AudioUNet5GAN]:
        if cls.__name__ == model_class_name:
            model_class = cls
            break
    
    if model_class is None:
        raise ValueError(f"未找到模型类: {model_class_name}")
    
    # 应用配置到cfg
    if 'learning_rate' in config_override:
        cfg.LEARNING_RATE = config_override['learning_rate']
    if 'batch_size' in config_override:
        cfg.BATCH_SIZE = config_override['batch_size']
    if 'num_epochs' in config_override:
        cfg.NUM_EPOCHS = config_override['num_epochs']
    if 'num_workers' in config_override:
        cfg.NUM_WORKERS = config_override['num_workers']
    
    # 1. 实验环境初始化
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.exp_name is not None:
        experiment_name = f"exp_{timestamp}_{args.exp_name}"
    else:
        experiment_name = f"exp_{timestamp}_{model_name}"

    exp_dir = cfg.EXP_DIR / experiment_name
    checkpoint_dir = exp_dir / "checkpoints"
    log_dir = exp_dir / "logs"

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 备份配置文件
    config_source = os.path.join("src", "config.py")
    config_target = exp_dir / "config_backup.py"

    if os.path.exists(config_source):
        shutil.copy(config_source, config_target)
        print(f"✅ 配置已备份至: {config_target}")
    else:
        print("⚠️ 警告: 未找到 src/config.py")

    print(f"\n{'='*60}")
    print(f"🚀 开始新实验: {experiment_name}")
    print(f"📁 输出目录: {exp_dir}")
    print(f"{'='*60}\n")

    writer = SummaryWriter(log_dir=str(log_dir))

    # 2. 数据准备
    print("📦 正在加载数据...")

    train_dataset = HowlingDataset(
        clean_dir=cfg.TRAIN_CLEAN_DIR,
        howling_dir=cfg.TRAIN_NOISY_DIR,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
    )

    val_dataset = HowlingDataset(
        clean_dir=cfg.VAL_CLEAN_DIR, 
        howling_dir=cfg.VAL_NOISY_DIR
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
    )

    print(f"✅ 训练样本数: {len(train_dataset)}, 验证样本数: {len(val_dataset)}\n")

    # 3. 模型与优化器
    device = cfg.DEVICE
    print(f"💻 使用设备: {device}")

    model = model_class().to(device)
    print(f"🔧 使用模型: {model.__class__.__name__}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 模型参数量: {total_params:,}\n")

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.SCHEDULER_FACTOR,
        patience=cfg.SCHEDULER_PATIENCE, verbose=True,
    )

    # 记录超参数到TensorBoard
    writer.add_text("Experiment/Name", experiment_name, 0)
    writer.add_text(
        "Hyperparameters",
        f"""
    **Model**: {model.__class__.__name__}
    - Parameters: {total_params:,}
    
    **Training**:
    - LR: {cfg.LEARNING_RATE}
    - Batch: {cfg.BATCH_SIZE}
    - Epochs: {cfg.NUM_EPOCHS}
    
    **Data**:
    - Sample Rate: {cfg.SAMPLE_RATE} Hz
    - Train: {len(train_dataset)}
    - Val: {len(val_dataset)}
    """, 0,
    )

    # 保存实验配置
    import json
    config_dict = {
        "experiment_name": experiment_name,
        "model": model.__class__.__name__,
        "total_params": total_params,
        "learning_rate": cfg.LEARNING_RATE,
        "batch_size": cfg.BATCH_SIZE,
        "num_epochs": cfg.NUM_EPOCHS,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
    }
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=4)

    # 4. 训练循环
    best_val_loss = float("inf")
    best_epoch = 0

    train_losses = []
    val_losses = []
    learning_rates = []

    print(f"\n{'='*60}")
    print(f"🏋️ 开始训练...")
    print(f"{'='*60}\n")

    for epoch in range(cfg.NUM_EPOCHS):
        # 训练阶段
        model.train()
        train_loss_accum = 0.0
        start_time = time.time()

        for batch_idx, (noisy_mag, clean_mag) in enumerate(train_loader):
            noisy_mag = noisy_mag.to(device)
            clean_mag = clean_mag.to(device)

            optimizer.zero_grad()

            pred_mag = model(noisy_mag)

            loss = criterion(
                torch.log10(pred_mag + 1e-8), 
                torch.log10(clean_mag + 1e-8)
            )

            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float("inf")
            )

            optimizer.step()

            train_loss_accum += loss.item()

            # 记录到TensorBoard
            global_step = epoch * len(train_loader) + batch_idx
            if batch_idx % 10 == 0:
                writer.add_scalar("Loss/Train_Step", loss.item(), global_step)
                writer.add_scalar("Gradients/Norm", grad_norm, global_step)

        avg_train_loss = train_loss_accum / len(train_loader)
        train_losses.append(avg_train_loss)

        # 验证阶段
        model.eval()
        val_loss_accum = 0.0

        with torch.no_grad():
            for noisy_mag, clean_mag in val_loader:
                noisy_mag = noisy_mag.to(device)
                clean_mag = clean_mag.to(device)

                pred_mag = model(noisy_mag)

                val_loss = criterion(
                    torch.log10(pred_mag + 1e-8), 
                    torch.log10(clean_mag + 1e-8)
                )
                val_loss_accum += val_loss.item()

        avg_val_loss = val_loss_accum / len(val_loader)
        val_losses.append(avg_val_loss)

        # Epoch结算
        duration = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]
        learning_rates.append(current_lr)

        print(
            f"Epoch [{epoch+1:3d}/{cfg.NUM_EPOCHS}] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {duration:.1f}s"
        )

        # 记录到TensorBoard
        writer.add_scalar("Loss/Train_Epoch", avg_train_loss, epoch)
        writer.add_scalar("Loss/Val_Epoch", avg_val_loss, epoch)
        writer.add_scalar("Time/Epoch_Duration", duration, epoch)
        writer.add_scalar("Training/Learning_Rate", current_lr, epoch)

        # 监控过拟合
        overfitting_ratio = avg_val_loss / avg_train_loss if avg_train_loss > 0 else 1.0
        writer.add_scalar("Training/Overfitting_Ratio", overfitting_ratio, epoch)

        # 可视化频谱图
        if epoch % 5 == 0:
            with torch.no_grad():
                sample_noisy, sample_clean = next(iter(val_loader))
                sample_noisy = sample_noisy[0:1].to(device)
                sample_clean = sample_clean[0:1].to(device)
                sample_pred = model(sample_noisy)

                writer.add_image(f"Spectrogram/Noisy", sample_noisy[0], epoch, dataformats="CHW")
                writer.add_image(f"Spectrogram/Clean", sample_clean[0], epoch, dataformats="CHW")
                writer.add_image(f"Spectrogram/Predicted", sample_pred[0], epoch, dataformats="CHW")

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "best_val_loss": best_val_loss,
            }
            torch.save(checkpoint, checkpoint_dir / "best_model.pth")
            print(f"🎉 验证集Loss下降，保存最佳模型")

        scheduler.step(avg_val_loss)


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"训练发生错误: {e}")
        import traceback
        traceback.print_exc()