'''
模型模块 - 音频啸叫抑制模型集合

本模块包含所有可用的音频啸叫抑制模型：

深度学习模型：
- AudioUNet3: 3层U-Net架构
- AudioUNet5: 5层U-Net架构  
- AudioCNN: 卷积神经网络
- AudioRNN: 循环神经网络

传统方法：
- FrequencyShiftMethod: 移频移向法
- GainSuppressionMethod: 增益抑制法
- AdaptiveFeedbackMethod: 自适应反馈抵消法

使用方法：
from src.models import AudioUNet5, FrequencyShiftMethod
model = AudioUNet5()
traditional_method = FrequencyShiftMethod()
'''

from .unet_v1 import AudioUNet3
from .unet_v2 import AudioUNet5
from .CNN import AudioCNN
from .RNN import AudioRNN

# 导入传统方法
from ..traditional.frequency_shift import FrequencyShiftMethod
from ..traditional.gain_suppression import GainSuppressionMethod
from ..traditional.adaptive_feedback import AdaptiveFeedbackMethod

__all__ = [
    # 深度学习模型
    'AudioUNet3',
    'AudioUNet5',
    'AudioCNN', 
    'AudioRNN',
    
    # 传统方法
    'FrequencyShiftMethod',
    'GainSuppressionMethod',
    'AdaptiveFeedbackMethod'
]
