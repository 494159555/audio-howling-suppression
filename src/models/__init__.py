"""模型模块

音频啸叫抑制模型集合
"""

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
from .unet_v11_multiscale import AudioUNet5MultiScale
from .unet_v12_pyramid import AudioUNet5Pyramid
from .unet_v13_fpn import AudioUNet5FPN
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

# 导入数据增强
from .augmentation import (
    AudioAugmentation,
    SpecAugment,
    MixupAugmentation,
    AdversarialAugmentation,
    CombinedAugmentation
)

# 导入训练策略
from .training_strategies import (
    MixedPrecisionTrainer,
    CosineAnnealingWarmupScheduler,
    OneCycleScheduler,
    CurriculumLearning,
    create_lr_scheduler
)

# 导入后处理
from .post_processing import (
    AdaptivePostProcessing,
    MultiFrameSmoother,
    AdaptiveGainControl,
    PostProcessingPipeline
)

# 导入传统方法
from ..traditional.frequency_shift import FrequencyShiftMethod
from ..traditional.gain_suppression import GainSuppressionMethod
from ..traditional.adaptive_feedback import AdaptiveFeedbackMethod

__all__ = [
    # 深度学习模型
    'AudioUNet3',
    'AudioUNet5',
    'AudioUNet5Attention',
    'AudioUNet5Residual',
    'AudioUNet5Dilated',
    'AudioUNet5Optimized',
    'AudioUNet5LSTM',
    'AudioUNet5TemporalAttention',
    'AudioUNet5ConvLSTM',
    'AudioUNet5GAN',
    'AudioUNet5MultiScale',
    'AudioUNet5Pyramid',
    'AudioUNet5FPN',
    'AudioCNN', 
    'AudioRNN',
    
    # 损失函数
    'SpectralLoss',
    'MultiTaskLoss',
    'SpectralConsistencyLoss',
    'AdversarialLoss',
    'Discriminator',
    
    # 数据增强
    'AudioAugmentation',
    'SpecAugment',
    'MixupAugmentation',
    'AdversarialAugmentation',
    'CombinedAugmentation',
    
    # 训练策略
    'MixedPrecisionTrainer',
    'CosineAnnealingWarmupScheduler',
    'OneCycleScheduler',
    'CurriculumLearning',
    'create_lr_scheduler',
    
    # 后处理
    'AdaptivePostProcessing',
    'MultiFrameSmoother',
    'AdaptiveGainControl',
    'PostProcessingPipeline',
    
    # 传统方法
    'FrequencyShiftMethod',
    'GainSuppressionMethod',
    'AdaptiveFeedbackMethod'
]