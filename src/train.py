'''
增强训练模块 - 音频啸叫抑制模型高级训练脚本

文件功能：
- 实现AudioUNet5模型的增强版训练流程
- 提供更详细的实验管理、监控和可视化功能
- 支持梯度监控、过拟合检测、频谱可视化等高级特性

主要组件：
- train函数：增强版主训练函数
- 实验环境初始化：完整的实验目录管理和配置备份
- 数据准备：训练和验证数据集加载
- 模型与优化器：模型构建、参数统计、优化器配置
- 增强训练循环：包含详细监控和可视化功能

新增特性：
- 模型参数量统计和记录
- 超参数和实验信息记录到TensorBoard
- JSON格式实验配置保存
- 梯度范数监控(训练稳定性)
- 过拟合比率计算(验证集/训练集loss比值)
- 频谱图可视化(每5个epoch)
- 完整checkpoint保存(包含优化器状态)

重要参数：
训练配置：
- NUM_EPOCHS: 训练轮数(50)
- BATCH_SIZE: 批大小(8)
- LEARNING_RATE: 学习率(1e-4)
- NUM_WORKERS: 数据加载线程数(2)

监控参数：
- 梯度范数监控：检测梯度爆炸/消失
- 过拟合比率：val_loss/train_loss，>1.0表示过拟合
- 频谱可视化间隔：每5个epoch保存一次

输出文件：
- best_model.pth: 完整checkpoint(模型+优化器+调度器状态)
- config.json: 实验配置JSON文件
- config_backup.py: 配置文件备份
- TensorBoard日志：包含训练指标、梯度、频谱图等

使用方法：
直接运行：
python src/train_v2.py

代码调用：
from src.train_v2 import train
train()
'''

import os
import time
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from src.config import cfg
from src.dataset import HowlingDataset
from src.models import AudioUNet5


def train():
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
