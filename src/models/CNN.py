'''
CNN模型 - 纯卷积神经网络音频啸叫抑制

文件功能：
- 实现基于纯CNN架构的音频啸叫抑制模型
- 使用编码器-解码器结构进行频谱图处理
- 适合捕捉频谱图的局部特征和空间模式

主要组件：
- AudioCNN类：继承自nn.Module的CNN模型
- 编码器：4层卷积下采样，逐步提取特征
- 解码器：4层转置卷积上采样，恢复分辨率

网络架构：
编码器(下采样)：
- conv1: [B,1,256,T] -> [B,32,128,T]
- conv2: [B,32,128,T] -> [B,64,64,T]  
- conv3: [B,64,64,T] -> [B,128,32,T]
- conv4: [B,128,32,T] -> [B,256,16,T]

解码器(上采样)：
- upconv4: [B,256,16,T] -> [B,128,32,T]
- upconv3: [B,128,32,T] -> [B,64,64,T]
- upconv2: [B,64,64,T] -> [B,32,128,T]
- upconv1: [B,32,128,T] -> [B,1,256,T]

重要参数：
网络层参数：
- 卷积核大小：3x3
- 下采样步长：(2,1) - 频率方向2倍，时间方向不变
- 激活函数：编码器使用LeakyReLU(0.2)，解码器使用ReLU
- 输出激活：Sigmoid，生成[0,1]范围的掩膜

数据处理：
- 输入：线性幅度谱 [B,1,256,T]
- 内部处理：Log域特征提取 torch.log10(x + 1e-8)
- 输出：线性域掩膜，与输入相乘得到最终结果

特点：
- 无跳跃连接：纯编码器-解码器结构
- 局部特征提取：适合处理频谱图的局部模式
- 掩膜机制：输出掩膜与原始输入相乘

使用方法：
from src.models.CNN import AudioCNN
model = AudioCNN()
output = model(input_spectrogram)  # input_spectrogram: [B,1,256,T]
'''

import torch
import torch.nn as nn


class AudioCNN(nn.Module):
    """
    纯CNN模型用于啸叫抑制
    适合捕捉频谱图的局部特征
    """

    def __init__(self):
        super(AudioCNN, self).__init__()

        # Encoder: 逐层下采样提取特征
        # [B, 1, 256, T] -> [B, 32, 128, T]
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # [B, 32, 128, T] -> [B, 64, 64, T]
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # [B, 64, 64, T] -> [B, 128, 32, T]
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # [B, 128, 32, T] -> [B, 256, 16, T]
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Decoder: 逐层上采样恢复分辨率
        # [B, 256, 16, T] -> [B, 128, 32, T]
        self.upconv4 = nn.Sequential(
            nn.ConvTranspose2d(
                256, 128, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # [B, 128, 32, T] -> [B, 64, 64, T]
        self.upconv3 = nn.Sequential(
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # [B, 64, 64, T] -> [B, 32, 128, T]
        self.upconv2 = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # [B, 32, 128, T] -> [B, 1, 256, T]
        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(
                32, 1, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)
            ),
            nn.Sigmoid(),  # 输出掩膜，范围[0,1]
        )

    def forward(self, x):
        # 内部Log变换（与UNet保持一致）
        x_log = torch.log10(x + 1e-8)

        # Encoder
        e1 = self.conv1(x_log)
        e2 = self.conv2(e1)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)

        # Decoder（注意：没有skip connection）
        d4 = self.upconv4(e4)
        d3 = self.upconv3(d4)
        d2 = self.upconv2(d3)
        mask = self.upconv1(d2)

        # 线性域掩膜
        output = x * mask
        return output
