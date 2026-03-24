"""配置模块

定义音频啸叫抑制项目的全局配置参数
"""

from pathlib import Path


class Config:
    """全局配置类"""
    
    # ============ 路径配置 ============
    PROJECT_ROOT = Path(__file__).parent.parent
    
    DATA_ROOT = PROJECT_ROOT / "data"
    
    TRAIN_CLEAN_DIR = DATA_ROOT / "train" / "clean"
    TRAIN_NOISY_DIR = DATA_ROOT / "train" / "howling"
    
    VAL_CLEAN_DIR = DATA_ROOT / "dev" / "clean"
    VAL_NOISY_DIR = DATA_ROOT / "dev" / "howling"
    
    EXP_DIR = PROJECT_ROOT / "experiments"
    
    # ============ 音频处理参数 ============
    SAMPLE_RATE = 16000      # 采样率
    CHUNK_LEN = 3.0          # 音频片段长度(秒)
    CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_LEN)
    N_FFT = 512              # FFT窗口大小
    HOP_LENGTH = 128         # STFT跳跃长度
    
    # ============ 训练超参数 ============
    BATCH_SIZE = 8           # 批大小
    LEARNING_RATE = 1e-4     # 学习率
    NUM_EPOCHS = 50          # 训练轮数
    NUM_WORKERS = 2          # 数据加载线程数
    
    # 学习率调度器参数
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 3
    
    # ============ 模型配置 ============
    AVAILABLE_MODELS = {
        'unet_v1': 'AudioUNet3',
        'unet_v2': 'AudioUNet5',
        'unet_v3_attention': 'AudioUNet5Attention',
        'unet_v4_residual': 'AudioUNet5Residual',
        'unet_v5_dilated': 'AudioUNet5Dilated',
        'unet_v6_optimized': 'AudioUNet5Optimized',
        'unet_v7_lstm': 'AudioUNet5LSTM',
        'unet_v8_temporal_attention': 'AudioUNet5TemporalAttention',
        'unet_v9_convlstm': 'AudioUNet5ConvLSTM',
        'unet_v10_gan': 'AudioUNet5GAN',
    }
    
    DEFAULT_MODEL = 'unet_v2'
    
    MODEL_DESCRIPTIONS = {
        'unet_v1': '3层U-Net (基线)',
        'unet_v2': '5层U-Net (基线)',
        'unet_v3_attention': '5层U-Net + 注意力门',
        'unet_v4_residual': '5层U-Net + 残差块',
        'unet_v5_dilated': '5层U-Net + 空洞卷积',
        'unet_v6_optimized': '5层U-Net + 注意力+残差+空洞',
        'unet_v7_lstm': '5层U-Net + 双向LSTM',
        'unet_v8_temporal_attention': '5层U-Net + 时间注意力',
        'unet_v9_convlstm': '5层U-Net + ConvLSTM',
        'unet_v10_gan': '5层U-Net + GAN框架',
    }
    
    LOSS_FUNCTIONS = {
        'l1': 'L1损失',
        'mse': 'MSE损失',
        'spectral': '频谱损失',
        'multitask': '多任务损失',
        'multitask_consistency': '多任务一致性损失',
        'adversarial': '对抗损失',
    }
    
    DEFAULT_LOSS = 'multitask'

    # ============ 设备配置 ============
    @property
    def DEVICE(self):
        """获取计算设备"""
        import torch
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


cfg = Config()