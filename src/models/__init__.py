'''
模型模块 - 音频啸叫抑制模型集合

本模块包含所有可用的音频啸叫抑制模型：

深度学习模型：
- AudioUNet3: 3层U-Net架构
- AudioUNet5: 5层U-Net架构  
- AudioUNet5Attention: 5层U-Net + 注意力机制
- AudioUNet5LSTM: 5层U-Net + LSTM时序建模
- AudioUNet5TemporalAttention: 5层U-Net + 时间注意力
- AudioUNet5ConvLSTM: 5层U-Net + ConvLSTM
- AudioUNet5GAN: 5层U-Net + GAN框架
- AudioCNN: 卷积神经网络
- AudioRNN: 循环神经网络

传统方法：
- FrequencyShiftMethod: 移频移向法
- GainSuppressionMethod: 增益抑制法
- AdaptiveFeedbackMethod: 自适应反馈抵消法

损失函数：
- SpectralLoss: 频谱损失
- MultiTaskLoss: 多任务损失
- SpectralConsistencyLoss: 频谱一致性损失
- AdversarialLoss: 对抗损失
- Discriminator: GAN判别器

使用方法：
from src.models import AudioUNet5LSTM, MultiTaskLoss
model = AudioUNet5LSTM()
loss_fn = MultiTaskLoss()
'''

from .unet_v1 import AudioUNet3
from .unet_v2 import AudioUNet5
from .unet_v3_attention import AudioUNet5Attention
from .unet_v4_residual import AudioUNet5Residual
from .unet_v5_dilated import AudioUNet5Dilated
from .unet_v6_optimized import AudioUNet5Optimized
from .unet_v7_lstm import AudioUNet5LSTM
from .unet_v8_temporal_attention import AudioUNet5TemporalAttention
from .unet_v9_convlstm import AudioUNet5ConvLSTM
from .unet_v10_gan import AudioUNet5GAN
from .CNN import AudioCNN
from .RNN import AudioRNN

# 导入损失函数
from .loss_functions import (
    SpectralLoss,
    MultiTaskLoss,
    SpectralConsistencyLoss,
    AdversarialLoss,
    Discriminator
)

# 导入传统方法
from ..traditional.frequency_shift import FrequencyShiftMethod
from ..traditional.gain_suppression import GainSuppressionMethod
from ..traditional.adaptive_feedback import AdaptiveFeedbackMethod

__all__ = [
    # 深度学习模型 - 基础版本
    'AudioUNet3',
    'AudioUNet5',
    
    # 深度学习模型 - 改进版本
    'AudioUNet5Attention',      # 注意力机制
    'AudioUNet5Residual',      # 残差连接
    'AudioUNet5Dilated',       # 空洞卷积
    'AudioUNet5Optimized',     # 综合改进
    
    # 深度学习模型 - 时序建模
    'AudioUNet5LSTM',         # LSTM集成
    'AudioUNet5TemporalAttention',  # 时间注意力
    'AudioUNet5ConvLSTM',     # ConvLSTM
    
    # 深度学习模型 - GAN框架
    'AudioUNet5GAN',          # GAN框架
    
    # 其他深度学习模型
    'AudioCNN', 
    'AudioRNN',
    
    # 损失函数
    'SpectralLoss',
    'MultiTaskLoss',
    'SpectralConsistencyLoss',
    'AdversarialLoss',
    'Discriminator',
    
    # 传统方法
    'FrequencyShiftMethod',
    'GainSuppressionMethod',
    'AdaptiveFeedbackMethod'
]