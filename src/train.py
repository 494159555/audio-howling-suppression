"""
增强训练模块 - 音频啸叫抑制模型高级训练脚本

本模块实现了U-Net模型的增强训练流水线，包含综合的实验管理、
监控和可视化功能，包括梯度监控、过拟合检测和频谱图可视化。

文件功能:
- 实现U-Net模型的增强训练流水线
- 提供详细的实验管理、监控和可视化功能
- 支持梯度监控、过拟合检测、频谱图可视化等

新增功能:
- 模型参数量统计和日志记录
- 超参数和实验信息记录到TensorBoard
- JSON格式的实验配置保存
- 梯度范数监控（训练稳定性）
- 过拟合比率计算（val_loss/train_loss比率）
- 频谱图可视化（每5个epoch）
- 完整的检查点保存（包含优化器状态）
- 支持命令行参数和配置文件

使用方法:
直接运行:
    python src/train.py

指定模型:
    python src/train.py --model unet_v3_attention

使用配置文件:
    python src/train.py --config configs/unet_attention.yaml

作者: 研究团队
日期: 2026-3-23
版本: 3.0.0
"""

# Standard library imports
import argparse
from datetime import datetime
import os
import shutil
import time

# Third-party imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

# Local imports
from src.config import cfg
from src.dataset import HowlingDataset
from src.models import (
    AudioUNet3,
    AudioUNet5,
    AudioUNet5Attention,
    AudioUNet5Residual,
    AudioUNet5Dilated,
    AudioUNet5Optimized,
)


def parse_args():
    """解析命令行参数。
    
    Returns:
        argparse.Namespace: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(
        description='音频啸叫抑制模型训练脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 使用默认模型训练
  python src/train.py
  
  # 指定模型
  python src/train.py --model unet_v3_attention
  python src/train.py --model unet_v6_optimized
  
  # 使用配置文件
  python src/train.py --config configs/unet_attention.yaml
  
  # 覆盖配置文件中的参数
  python src/train.py --config configs/unet_attention.yaml --lr 2e-4 --epochs 100
  
  # 修改多个参数
  python src/train.py --model unet_v6_optimized --lr 2e-4 --batch-size 4 --epochs 80
        """
    )
    
    # 模型选择
    parser.add_argument(
        '--model',
        type=str,
        default=cfg.DEFAULT_MODEL,
        choices=list(cfg.AVAILABLE_MODELS.keys()),
        help=f'选择模型 (默认: {cfg.DEFAULT_MODEL})'
    )
    
    # 配置文件
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='配置文件路径 (YAML格式)'
    )
    
    # 训练参数（可以覆盖config.py或配置文件中的值）
    parser.add_argument(
        '--lr',
        '--learning-rate',
        type=float,
        default=None,
        help='学习率 (覆盖配置文件)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='批大小 (覆盖配置文件)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='训练轮数 (覆盖配置文件)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help='数据加载线程数 (覆盖配置文件)'
    )
    
    # 实验名称
    parser.add_argument(
        '--exp-name',
        type=str,
        default=None,
        help='实验名称 (默认自动生成)'
    )
    
    return parser.parse_args()


def load_config_from_yaml(config_path: str) -> dict:
    """从YAML文件加载配置。
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        dict: 配置字典
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def train() -> None:
    """执行模型的增强训练流水线。"""
    # 解析命令行参数
    args = parse_args()
    
    # 确定使用的模型名称
    model_name = args.model
    
    # 加载配置文件（如果指定）
    config_override = {}
    if args.config is not None:
        file_config = load_config_from_yaml(args.config)
        config_override.update(file_config)
    
    # 命令行参数覆盖配置
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
    
    # 动态导入模型类
    model_class = None
    for cls in [AudioUNet3, AudioUNet5, AudioUNet5Attention, 
                AudioUNet5Residual, AudioUNet5Dilated, AudioUNet5Optimized]:
        if cls.__name__ == model_class_name:
            model_class = cls
            break
    
    if model_class is None:
        raise ValueError(f"未找到模型类: {model_class_name}")
    
    # 应用配置覆盖到cfg
    if 'learning_rate' in config_override:
        cfg.LEARNING_RATE = config_override['learning_rate']
    if 'batch_size' in config_override:
        cfg.BATCH_SIZE = config_override['batch_size']
    if 'num_epochs' in config_override:
        cfg.NUM_EPOCHS = config_override['num_epochs']
    if 'num_workers' in config_override:
        cfg.NUM_WORKERS = config_override['num_workers']
    
    # ==========================================
    # 1. 实验环境初始化 (Experiment Setup)
    # ==========================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 使用自定义实验名称或自动生成
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
        print("⚠️ 警告: 未找到 src/config.py，无法备份配置！")

    print(f"\n{'='*60}")
    print(f"🚀 开始新实验: {experiment_name}")
    print(f"📁 输出目录: {exp_dir}")
    print(f"{'='*60}\n")

    # 初始化 TensorBoard
    writer = SummaryWriter(log_dir=str(log_dir))

    # ==========================================
    # 2. 数据准备 (Data Preparation)
    # ==========================================
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

    # ==========================================
    # 3. 模型与优化器 (Model & Optimizer)
    # ==========================================
    device = cfg.DEVICE
    print(f"💻 使用设备: {device}")

    # 使用动态选择的模型类
    model = model_class().to(device)
    print(f"🔧 使用模型: {model.__class__.__name__}")

    # 统计模型参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 模型参数量: {total_params:,}\n")

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.SCHEDULER_FACTOR,
        patience=cfg.SCHEDULER_PATIENCE,
        verbose=True,
    )

    # 记录超参数和模型信息到 TensorBoard
    writer.add_text("Experiment/Name", experiment_name, 0)
    writer.add_text(
        "Hyperparameters",
        f"""
    **Model Architecture**: {model.__class__.__name__}
    - Total Parameters: {total_params:,}
    
    **Training Config**:
    - Learning Rate: {cfg.LEARNING_RATE}
    - Batch Size: {cfg.BATCH_SIZE}
    - Epochs: {cfg.NUM_EPOCHS}
    - Optimizer: Adam
    - Loss Function: L1Loss (Log domain)
    - Scheduler: ReduceLROnPlateau (factor={cfg.SCHEDULER_FACTOR}, patience={cfg.SCHEDULER_PATIENCE})
    
    **Data Config**:
    - Sample Rate: {cfg.SAMPLE_RATE} Hz
    - Chunk Length: {cfg.CHUNK_LEN}s
    - N_FFT: {cfg.N_FFT}
    - Hop Length: {cfg.HOP_LENGTH}
    - Train Samples: {len(train_dataset)}
    - Val Samples: {len(val_dataset)}
    """,
        0,
    )

    # 保存实验配置为JSON（方便后续分析）
    import json

    config_dict = {
        "experiment_name": experiment_name,
        "model": model.__class__.__name__,
        "total_params": total_params,
        "learning_rate": cfg.LEARNING_RATE,
        "batch_size": cfg.BATCH_SIZE,
        "num_epochs": cfg.NUM_EPOCHS,
        "sample_rate": cfg.SAMPLE_RATE,
        "n_fft": cfg.N_FFT,
        "hop_length": cfg.HOP_LENGTH,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
    }
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=4)

    # ==========================================
    # 4. 训练循环 (Training Loop)
    # ==========================================
    best_val_loss = float("inf")
    best_epoch = 0

    # 用于记录每个epoch的指标
    train_losses = []
    val_losses = []
    learning_rates = []

    print(f"\n{'='*60}")
    print(f"🏋️ 开始训练...")
    print(f"{'='*60}\n")

    for epoch in range(cfg.NUM_EPOCHS):
        # --- 训练阶段 ---
        model.train()
        train_loss_accum = 0.0
        start_time = time.time()

        for batch_idx, (noisy_mag, clean_mag) in enumerate(train_loader):
            noisy_mag = noisy_mag.to(device)
            clean_mag = clean_mag.to(device)

            optimizer.zero_grad()

            # 前向传播
            pred_mag = model(noisy_mag)

            # Log 域 Loss
            loss = criterion(
                torch.log10(pred_mag + 1e-8), 
                torch.log10(clean_mag + 1e-8)
            )

            loss.backward()

            # 记录梯度范数（监控训练稳定性）
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float("inf")
            )

            optimizer.step()

            train_loss_accum += loss.item()

            # 每10个batch记录一次
            global_step = epoch * len(train_loader) + batch_idx
            if batch_idx % 10 == 0:
                writer.add_scalar("Loss/Train_Step", loss.item(), global_step)
                writer.add_scalar("Gradients/Norm", grad_norm, global_step)

        avg_train_loss = train_loss_accum / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- 验证阶段 ---
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

        # --- Epoch 结算 ---
        duration = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]
        learning_rates.append(current_lr)

        # 更详细的打印信息
        print(
            f"Epoch [{epoch+1:3d}/{cfg.NUM_EPOCHS}] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {duration:.1f}s"
        )

        # 记录到 TensorBoard
        writer.add_scalar("Loss/Train_Epoch", avg_train_loss, epoch)
        writer.add_scalar("Loss/Val_Epoch", avg_val_loss, epoch)
        writer.add_scalar("Time/Epoch_Duration", duration, epoch)
        writer.add_scalar("Training/Learning_Rate", current_lr, epoch)

        # 记录训练/验证loss的比值（监控过拟合）
        overfitting_ratio = avg_val_loss / avg_train_loss if avg_train_loss > 0 else 1.0
        writer.add_scalar("Training/Overfitting_Ratio", overfitting_ratio, epoch)

        # 可视化频谱图（每5个epoch保存一次）
        if epoch % 5 == 0:
            with torch.no_grad():
                # 取验证集第一个batch的第一个样本
                sample_noisy, sample_clean = next(iter(val_loader))
                sample_noisy = sample_noisy[0:1].to(device)  # [1, 1, 256, T]
                sample_clean = sample_clean[0:1].to(device)
                sample_pred = model(sample_noisy)

                # 记录到TensorBoard (使用 CHW 格式)
                writer.add_image(
                    f"Spectrogram/Noisy", sample_noisy[0], epoch, dataformats="CHW"
                )
                writer.add_image(
                    f"Spectrogram/Clean", sample_clean[0], epoch, dataformats="CHW"
                )
                writer.add_image(
                    f"Spectrogram/Predicted", sample_pred[0], epoch, dataformats="CHW"
                )

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1

            # 保存完整的checkpoint（包含优化器状态）
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
            print(
                f"🎉 验证集 Loss 下降，保存最佳模型: {checkpoint_dir / 'best_model.pth'}"
            )

        # 学习率调度器更新
        scheduler.step(avg_val_loss)


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        # 如果训练中途报错，打印错误信息
        print(f"训练发生错误: {e}")
        import traceback
        traceback.print_exc()