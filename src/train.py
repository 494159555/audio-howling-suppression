"""
Basic Training Module for Audio Howling Suppression

This module implements a comprehensive training pipeline for the AudioUNet5 model,
including experiment management, data loading, model training, and validation.
Supports TensorBoard logging and automatic best model saving.

Author: Research Team
Date: 2024-12-14
Version: 2.0.0
"""

# Standard library imports
import os
import shutil
import time
from datetime import datetime

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
    """Main training function for AudioUNet5 model.
    
    Implements complete training pipeline including experiment setup, data preparation,
    model initialization, training loop, validation, and model saving.
    
    The function automatically:
        - Creates experiment directory with timestamp
        - Backs up configuration file
        - Sets up TensorBoard logging
        - Loads training and validation datasets
        - Initializes model, loss function, and optimizer
        - Runs training with validation
        - Saves best model based on validation loss
        - Implements learning rate scheduling
        
    Raises:
        FileNotFoundError: If data directories don't exist
        RuntimeError: If training fails due to GPU memory or other issues
    """
    # ==========================
    # 1. Experiment Environment Setup
    # ==========================
    # Generate unique experiment ID using timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"exp_{timestamp}_unet5"

    # Create experiment directory structure
    exp_dir = cfg.EXP_DIR / experiment_name
    checkpoint_dir = exp_dir / "checkpoints"
    log_dir = exp_dir / "logs"

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Backup configuration file
    config_source = os.path.join("src", "config.py")
    config_target = exp_dir / "config_backup.py"

    if os.path.exists(config_source):
        shutil.copy(config_source, config_target)
        print(f"✓ Configuration backed up to: {config_target}")
    else:
        print("⚠ Warning: src/config.py not found, configuration backup skipped!")

    print(f"🚀 Starting new experiment: {experiment_name}")
    print(f"📁 Output directory: {exp_dir}")

    # Initialize TensorBoard for logging
    writer = SummaryWriter(log_dir=str(log_dir))

    # ==========================
    # 2. Data Preparation
    # ==========================
    print("📊 Loading datasets...")

    # Training dataset
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

    # Validation dataset
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

    print(f"📈 Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # ==========================
    # 3. Model and Optimizer Setup
    # ==========================
    device = cfg.DEVICE
    print(f"🔧 Using device: {device}")

    # Initialize model
    model = AudioUNet5().to(device)

    # Loss function - L1 loss in log domain for better audio quality
    criterion = nn.L1Loss()

    # Optimizer - Adam with default parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    # Learning rate scheduler - ReduceLROnPlateau for adaptive learning
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.SCHEDULER_FACTOR,
        patience=cfg.SCHEDULER_PATIENCE,
        verbose=True,
    )

    # ==========================
    # 4. Training Loop
    # ==========================
    best_val_loss = float("inf")

    for epoch in range(cfg.NUM_EPOCHS):
        # --------------------------
        # Training Phase
        # --------------------------
        model.train()
        train_loss_accum = 0.0

        epoch_start_time = time.time()

        for batch_idx, (noisy_mag, clean_mag) in enumerate(train_loader):
            # Move data to device
            noisy_mag = noisy_mag.to(device)
            clean_mag = clean_mag.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            pred_mag = model(noisy_mag)

            # [ALGORITHM] Compute loss in log domain
            # Reason: Log domain provides better perceptual quality and numerical stability
            loss = criterion(
                torch.log10(pred_mag + 1e-8), 
                torch.log10(clean_mag + 1e-8)
            )

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item()

            # Log training steps periodically
            global_step = epoch * len(train_loader) + batch_idx
            if batch_idx % 10 == 0:
                writer.add_scalar("Loss/Train_Step", loss.item(), global_step)

        avg_train_loss = train_loss_accum / len(train_loader)

        # --------------------------
        # Validation Phase
        # --------------------------
        model.eval()
        val_loss_accum = 0.0

        with torch.no_grad():
            for noisy_mag, clean_mag in val_loader:
                # Move data to device
                noisy_mag = noisy_mag.to(device)
                clean_mag = clean_mag.to(device)

                # Forward pass
                pred_mag = model(noisy_mag)

                # Compute validation loss in log domain
                val_loss = criterion(
                    torch.log10(pred_mag + 1e-8), 
                    torch.log10(clean_mag + 1e-8)
                )
                val_loss_accum += val_loss.item()

        avg_val_loss = val_loss_accum / len(val_loader)

        # --------------------------
        # Epoch Summary
        # --------------------------
        epoch_duration = time.time() - epoch_start_time
        print(
            f"Epoch [{epoch+1}/{cfg.NUM_EPOCHS}] "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Time: {epoch_duration:.1f}s"
        )

        # Log to TensorBoard
        writer.add_scalar("Loss/Train_Epoch", avg_train_loss, epoch)
        writer.add_scalar("Loss/Val_Epoch", avg_val_loss, epoch)
        writer.add_scalar("Time/Epoch", epoch_duration, epoch)

        # Log learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("Training/Learning_Rate", current_lr, epoch)

        # --------------------------
        # Model Saving
        # --------------------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pth")
            print(
                f"🎉 Validation loss improved! Best model saved: {checkpoint_dir / 'best_model.pth'}"
            )

        # --------------------------
        # Learning Rate Scheduling
        # --------------------------
        scheduler.step(avg_val_loss)

    # Close TensorBoard writer
    writer.close()
    print("✅ Training completed successfully!")


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
