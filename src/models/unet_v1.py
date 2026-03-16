'''
U-Net v1模型 - 3层U-Net音频啸叫抑制

文件功能：
- 实现基于U-Net架构的音频啸叫抑制模型(3层版本)
- 使用编码器-解码器结构带跳跃连接
- 适合频谱图的端到端修复任务

主要组件：
- AudioUNet3类：继承自nn.Module的3层U-Net模型
- 编码器：3层卷积下采样，逐步提取特征
- 解码器：3层转置卷积上采样，恢复分辨率
- 跳跃连接：连接编码器和解码器对应层

网络架构：
编码器(下采样)：
- enc1: [B,1,256,T] -> [B,16,128,T]
- enc2: [B,16,128,T] -> [B,32,64,T]
- enc3: [B,32,64,T] -> [B,64,32,T] (bottleneck)

解码器(上采样)：
- dec3: [B,64,32,T] -> [B,32,64,T] + enc2跳跃连接
- dec2: [B,64,64,T] -> [B,16,128,T] + enc1跳跃连接
- dec1: [B,32,128,T] -> [B,1,256,T] + enc1跳跃连接

重要参数：
网络层配置：
- 卷积核大小：3x3
- 下采样步长：(2,1) - 频率方向2倍，时间方向不变
- 激活函数：编码器使用LeakyReLU(0.2)，解码器使用ReLU
- 输出激活：Sigmoid，生成[0,1]范围的掩膜

跳跃连接：
- dec3输入：enc3输出 + enc2特征
- dec2输入：dec3输出 + enc1特征  
- dec1输入：dec2输出 + enc1特征

数据处理：
- 输入：线性幅度谱 [B,1,256,T]
- 内部处理：Log域特征提取 torch.log10(x + 1e-8)
- 输出：线性域掩膜，与输入相乘得到最终结果

特点：
- U-Net架构：经典的编码器-解码器结构
- 跳跃连接：保留低级特征，防止信息丢失
- 浅层网络：仅3层，参数量较少
- 掩膜机制：输出掩膜与原始输入相乘

与v2版本区别：
- 更浅的网络：3层 vs 5层
- 更少的参数：适合快速实验和原型开发
- 基础功能：包含U-Net的核心特性

使用方法：
from src.models.unet_v1 import AudioUNet3
model = AudioUNet3()
output = model(input_spectrogram)  # input_spectrogram: [B,1,256,T]
'''

import torch
import torch.nn as nn


class AudioUNet3(nn.Module):
    def __init__(self):
        super(AudioUNet3, self).__init__()

        # --- Encoder ---
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )

        # --- Decoder ---
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(
                64, 16, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(
                32, 1, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)
            ),
            nn.Sigmoid(),  # 输出 0-1 的 Mask
        )

    def forward(self, x):
        # x: [Batch, 1, 256, Time] (线性幅度 Linear Magnitude)

        # [关键修改]：提取 Log 特征喂给网络
        x_log = torch.log10(x + 1e-8)

        # Encoder (使用 Log 特征)
        e1 = self.enc1(x_log)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        # Decoder + Skip Connections
        d3 = self.dec3(e3)
        d3_cat = torch.cat([d3, e2], dim=1)

        d2 = self.dec2(d3_cat)
        d2_cat = torch.cat([d2, e1], dim=1)

        mask = self.dec1(d2_cat)

        # [关键修改]：Mask 作用于线性幅度 x
        # 1.0 = 保留，0.0 = 消除
        output = x * mask
        return output
