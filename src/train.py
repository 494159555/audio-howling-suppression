"""
Enhanced Training Module - Audio Howling Suppression Model Advanced Training Script

This module implements an enhanced training pipeline for AudioUNet5 model with
comprehensive experiment management, monitoring, and visualization features including
gradient monitoring, overfitting detection, and spectrogram visualization.

File Functions:
- Implement enhanced training pipeline for AudioUNet5 model
- Provide detailed experiment management, monitoring, and visualization features
- Support gradient monitoring, overfitting detection, spectrogram visualization, etc.

Main Components:
- train function: Enhanced main training function
- Experiment environment initialization: Complete experiment directory management and config backup
- Data preparation: Training and validation dataset loading
- Model and optimizer: Model construction, parameter statistics, optimizer configuration
- Enhanced training loop: Includes detailed monitoring and visualization features

New Features:
- Model parameter count statistics and logging
- Hyperparameters and experiment info logging to TensorBoard
- JSON format experiment configuration saving
- Gradient norm monitoring (training stability)
- Overfitting ratio calculation (val_loss/train_loss ratio)
- Spectrogram visualization (every 5 epochs)
- Complete checkpoint saving (includes optimizer state)

Important Parameters:
Training Configuration:
- NUM_EPOCHS: Number of training epochs (50)
- BATCH_SIZE: Batch size (8)
- LEARNING_RATE: Learning rate (1e-4)
- NUM_WORKERS: Data loading threads (2)

Monitoring Parameters:
- Gradient norm monitoring: Detects gradient explosion/vanishing
- Overfitting ratio: val_loss/train_loss, >1.0 indicates overfitting
- Spectrogram visualization interval: Save every 5 epochs

Output Files:
- best_model.pth: Complete checkpoint (model + optimizer + scheduler state)
- config.json: Experiment configuration JSON file
- config_backup.py: Configuration file backup
- TensorBoard logs: Contains training metrics, gradients, spectrograms, etc.

Usage:
Direct run:
    python src/train_v2.py

Code call:
    from src.train_v2 import train
    train()

Author: Research Team
Date: 2026-3-23
Version: 2.0.0
"""

# Standard library imports
import datetime
import os
import shutil
import time

# Third-party imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Local imports
from src.config import cfg
from src.dataset import HowlingDataset
from src.models import AudioUNet5


def train() -> None:
    """Execute enhanced training pipeline for AudioUNet5 model.
    
    This function implements a comprehensive training pipeline with experiment management,
    monitoring, and visualization features. It handles experiment directory setup,
    data loading, model initialization, training loop with monitoring,
    and checkpoint saving.
    
    The training includes:
    - Experiment environment initialization with config backup
    - Data preparation with train/validation splits
    - Model and optimizer configuration
    - Enhanced training loop with TensorBoard logging
    - Gradient norm monitoring for training stability
    - Overfitting ratio calculation
    - Spectrogram visualization every 5 epochs
    - Complete checkpoint saving (model + optimizer + scheduler state)
    """
    # ==========================================
    # 1. 实验环境初始化 (Experiment Setup)
    # ==========================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"exp_{timestamp}_unet5"

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
        clean_dir=cfg.VAL_CLEAN_DIR, howling_dir=cfg.VAL_NOISY_DIR
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

    model = AudioUNet5().to(device)

    # ⭐ 新增：统计模型参数量
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

    # ⭐ 新增：记录超参数和模型信息到 TensorBoard
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

    # ⭐ 新增：保存实验配置为JSON（方便后续分析）
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

    # ⭐ 新增：用于记录每个epoch的指标
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
                torch.log10(pred_mag + 1e-8), torch.log10(clean_mag + 1e-8)
            )

            loss.backward()

            # ⭐ 新增：记录梯度范数（监控训练稳定性）
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float("inf")
            )

            optimizer.step()

            train_loss_accum += loss.item()

            # 每10个batch记录一次
            global_step = epoch * len(train_loader) + batch_idx
            if batch_idx % 10 == 0:
                writer.add_scalar("Loss/Train_Step", loss.item(), global_step)
                writer.add_scalar("Gradients/Norm", grad_norm, global_step)  # ⭐ 新增

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
                    torch.log10(pred_mag + 1e-8), torch.log10(clean_mag + 1e-8)
                )
                val_loss_accum += val_loss.item()

        avg_val_loss = val_loss_accum / len(val_loader)
        val_losses.append(avg_val_loss)

        # --- Epoch 结算 ---
        duration = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]
        learning_rates.append(current_lr)

        # ⭐ 改进：更详细的打印信息
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

        # ⭐ 新增：记录训练/验证loss的比值（监控过拟合）
        overfitting_ratio = avg_val_loss / avg_train_loss if avg_train_loss > 0 else 1.0
        writer.add_scalar("Training/Overfitting_Ratio", overfitting_ratio, epoch)

        # ⭐ 新增：可视化频谱图（每5个epoch保存一次）
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

            # ⭐ 改进：保存完整的checkpoint（包含优化器状态）
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
