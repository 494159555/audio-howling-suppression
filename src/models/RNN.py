'''
RNN模型 - 循环神经网络音频啸叫抑制

文件功能：
- 实现基于RNN(LSTM)的音频啸叫抑制模型
- 结合CNN特征提取和LSTM时序建模
- 适合捕捉音频信号的时序依赖关系

主要组件：
- AudioRNN类：继承自nn.Module的RNN模型
- 前置卷积层：提取局部频谱特征
- 双向LSTM：处理时序信息，捕捉长期依赖
- 后置卷积层：恢复频谱维度并生成掩膜

网络架构：
前置卷积(特征提取)：
- pre_conv: [B,1,256,T] -> [B,64,256,T]
- 使用两个3x3卷积层，保持空间维度不变

LSTM层(时序建模)：
- 输入维度重组：[B,64,256,T] -> [B*256,T,64]
- 双向LSTM：处理每个频率bin的时序特征
- 输出：[B*256,T,hidden_size*2]

后置卷积(恢复维度)：
- 维度恢复：[B*256,T,hidden_size*2] -> [B,hidden_size*2,256,T]
- post_conv: [B,hidden_size*2,256,T] -> [B,1,256,T]
- 使用三个卷积层逐步减少通道数

重要参数：
模型参数：
- freq_bins: 频率bin数量(默认256)
- hidden_size: LSTM隐藏层大小(默认128)
- num_layers: LSTM层数(默认2)
- 双向LSTM：同时考虑前后文信息

网络层配置：
- 卷积核：3x3，padding=1
- LSTM输入：64维特征向量
- LSTM输出：hidden_size*2维(双向)
- Dropout：0.2(多层LSTM时)
- 输出激活：Sigmoid，生成[0,1]掩膜

数据处理流程：
1. 输入：线性幅度谱 [B,1,256,T]
2. Log变换：torch.log10(x + 1e-8)
3. CNN特征提取：局部空间特征
4. 维度重组：适配LSTM输入格式
5. LSTM处理：时序特征建模
6. 维度恢复：恢复频谱图格式
7. CNN后处理：生成最终掩膜
8. 输出：线性域掩膜与输入相乘

特点：
- 时序建模：LSTM捕捉音频的时间依赖性
- 双向处理：同时考虑过去和未来信息
- 混合架构：CNN提取空间特征，RNN建模时序关系
- 频率并行：每个频率bin独立进行时序处理

使用方法：
from src.models.RNN import AudioRNN
model = AudioRNN(freq_bins=256, hidden_size=128, num_layers=2)
output = model(input_spectrogram)  # input_spectrogram: [B,1,256,T]
'''

import torch
import torch.nn as nn


class AudioRNN(nn.Module):
    """
    RNN模型用于啸叫抑制
    适合捕捉时序依赖关系
    使用双向LSTM处理每个频率bin的时序特征
    """

    def __init__(self, freq_bins=256, hidden_size=128, num_layers=2):
        super(AudioRNN, self).__init__()

        self.freq_bins = freq_bins
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 前置卷积层：提取局部特征
        # [B, 1, 256, T] -> [B, 64, 256, T]
        self.pre_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 双向LSTM：处理时序信息
        # 输入: [B*256, T, 64] -> 输出: [B*256, T, hidden_size*2]
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0,
        )

        # 后置卷积层：恢复频谱维度
        # [B, hidden_size*2, 256, T] -> [B, 1, 256, T]
        self.post_conv = nn.Sequential(
            nn.Conv2d(hidden_size * 2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),  # 输出掩膜
        )

    def forward(self, x):
        # 内部Log变换
        x_log = torch.log10(x + 1e-8)

        B, C, F, T = x_log.shape  # [B, 1, 256, T]

        # 1. 前置卷积提取特征
        conv_out = self.pre_conv(x_log)  # [B, 64, 256, T]

        # 2. 重排维度以便LSTM处理
        # [B, 64, 256, T] -> [B, 256, T, 64] -> [B*256, T, 64]
        conv_out = conv_out.permute(0, 2, 3, 1)  # [B, 256, T, 64]
        conv_out = conv_out.reshape(B * F, T, -1)  # [B*256, T, 64]

        # 3. LSTM处理时序
        lstm_out, _ = self.lstm(conv_out)  # [B*256, T, hidden_size*2]

        # 4. 恢复空间维度
        # [B*256, T, hidden_size*2] -> [B, 256, T, hidden_size*2] -> [B, hidden_size*2, 256, T]
        lstm_out = lstm_out.reshape(B, F, T, -1)  # [B, 256, T, hidden_size*2]
        lstm_out = lstm_out.permute(0, 3, 1, 2)  # [B, hidden_size*2, 256, T]

        # 5. 后置卷积生成掩膜
        mask = self.post_conv(lstm_out)  # [B, 1, 256, T]

        # 线性域掩膜
        output = x * mask
        return output
