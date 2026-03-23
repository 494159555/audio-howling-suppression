"""
Configuration Module

This module defines global configuration parameters for the audio howling suppression
project, including paths, audio processing parameters, training hyperparameters,
and device settings.

Author: Research Team
Version: 2.0.0
"""

# Standard library imports
from pathlib import Path

# Third-party imports
# None

# Local imports
# None


class Config:
    """Global configuration class for audio howling suppression project.
    
    This class centralizes all configuration parameters including file paths,
    audio processing settings, training hyperparameters, and device configuration.
    
    Attributes:
        PROJECT_ROOT (Path): Root directory of the project
        DATA_ROOT (Path): Root directory for data files
        TRAIN_CLEAN_DIR (Path): Directory for training clean audio files
        TRAIN_NOISY_DIR (Path): Directory for training howling audio files
        VAL_CLEAN_DIR (Path): Directory for validation clean audio files
        VAL_NOISY_DIR (Path): Directory for validation howling audio files
        EXP_DIR (Path): Directory for experiment outputs
        SAMPLE_RATE (int): Audio sample rate in Hz
        CHUNK_LEN (float): Audio chunk length in seconds
        CHUNK_SIZE (int): Number of samples per chunk
        N_FFT (int): FFT window size
        HOP_LENGTH (int): Hop length for STFT
        BATCH_SIZE (int): Batch size for training
        LEARNING_RATE (float): Learning rate for optimizer
        NUM_EPOCHS (int): Number of training epochs
        NUM_WORKERS (int): Number of data loader workers
        SCHEDULER_FACTOR (float): Learning rate reduction factor
        SCHEDULER_PATIENCE (int): Patience for learning rate scheduler
    """
    
    # ==========================
    # 1. Path Configuration
    # ==========================
    PROJECT_ROOT = Path(__file__).parent.parent
    
    DATA_ROOT = PROJECT_ROOT / "data"
    
    TRAIN_CLEAN_DIR = DATA_ROOT / "train" / "clean"
    TRAIN_NOISY_DIR = DATA_ROOT / "train" / "howling"
    
    VAL_CLEAN_DIR = DATA_ROOT / "dev" / "clean"
    VAL_NOISY_DIR = DATA_ROOT / "dev" / "howling"
    
    # Experiment output directory
    EXP_DIR = PROJECT_ROOT / "experiments"
    
    # ==========================
    # 2. Audio Processing Parameters
    # ==========================
    SAMPLE_RATE = 16000      # 采样率：CD质量的一半，适合语音处理
    CHUNK_LEN = 3.0          # 音频片段长度：3秒包含足够上下文信息
    CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_LEN)  # 计算chunk大小（采样点数）
    N_FFT = 512              # FFT窗口：2的9次方，适合频谱分析
    HOP_LENGTH = 128         # 跳跃长度：N_FFT的1/4，保证时频分辨率平衡
    
    # ==========================
    # 3. Training Hyperparameters
    # ==========================
    BATCH_SIZE = 8           # 批大小：根据GPU内存调整
    LEARNING_RATE = 1e-4     # 学习率：Adam优化器的默认学习率
    NUM_EPOCHS = 50          # 训练轮数：足够的训练轮数
    NUM_WORKERS = 2          # DataLoader线程数：根据CPU核心数调整
    
    # Learning rate scheduler parameters
    SCHEDULER_FACTOR = 0.5   # 学习率衰减因子：每次减半
    SCHEDULER_PATIENCE = 3   # 学习率衰减等待轮数：3轮无改善则衰减
    
    # ==========================
    # 4. Device Configuration
    # ==========================
    
    @property
    def DEVICE(self):
        """Get the appropriate device for computation.
        
        Returns:
            torch.device: CUDA device if available, otherwise CPU
        """
        import torch
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Global configuration instance
cfg = Config()
