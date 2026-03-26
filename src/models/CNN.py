"""
============================================================
CNN 模型 - 纯卷积神经网络音频啸叫抑制（基线模型）
============================================================

【文件功能】
这个文件实现了一个纯CNN模型，用于音频啸叫抑制任务。
这是一个基线模型，没有跳跃连接，仅通过编码器-解码器结构处理频谱图。

【主要组件】
- AudioCNN 类：纯CNN模型
  - 编码器：4层卷积下采样，提取局部特征
  - 解码器：4层转置卷积上采样，恢复分辨率
  - 特点：没有跳跃连接，纯编码器-解码器结构

【网络架构】
编码器（下采样过程）：
  输入: [B, 1, 256, T]
    ↓
  conv1: [B, 1, 256, T] → [B, 32, 128, T]
    ↓
  conv2: [B, 32, 128, T] → [B, 64, 64, T]
    ↓
  conv3: [B, 64, 64, T] → [B, 128, 32, T]
    ↓
  conv4: [B, 128, 32, T] → [B, 256, 16, T]

解码器（上采样过程）：
  conv4: [B, 256, 16, T]
    ↓
  upconv4: [B, 256, 16, T] → [B, 128, 32, T]
    ↓
  upconv3: [B, 128, 32, T] → [B, 64, 64, T]
    ↓
  upconv2: [B, 64, 64, T] → [B, 32, 128, T]
    ↓
  upconv1: [B, 32, 128, T] → [B, 1, 256, T]

【关键参数说明】
- 卷积核大小：3×3（局部特征提取）
- 下采样步长：(2, 1)（频率方向下采样，时间方向保持）
- 激活函数：
  * 编码器：LeakyReLU(0.2, inplace=True)
  * 解码器：ReLU(inplace=True)
  * 输出层：Sigmoid（生成[0,1]掩膜）
- 批归一化：每层使用，加速训练

【数据处理流程】
1. 输入：线性幅度谱 [B, 1, 256, T]
2. Log变换：log10(x + 1e-8)
3. 编码器：提取局部特征
4. 解码器：重建频谱（无跳跃连接）
5. 输出：[0,1]掩膜
6. 最终结果：输入 × 掩膜

【模型特点】
✓ 简单结构：纯编码器-解码器，没有跳跃连接
✓ 局部特征：适合捕捉频谱图的局部模式
✓ 参数较少：约3.8M参数
✓ 训练快速：结构简单，收敛快

【与U-Net区别】
- U-Net：有跳跃连接，保留多尺度信息
- CNN（本模型）：无跳跃连接，仅依赖编码器的抽象特征

【使用场景】
- 作为基线模型对比
- 快速原型验证
- 研究跳跃连接的作用

【使用示例】
```python
from src.models.CNN import AudioCNN
import torch

# 创建模型
model = AudioCNN()

# 准备输入
input_spec = torch.randn(4, 1, 256, 376)

# 前向传播
output_spec = model(input_spec)
```
"""

import torch
import torch.nn as nn


class AudioCNN(nn.Module):
    """纯CNN模型用于音频啸叫抑制（基线模型）

    这是一个简单的编码器-解码器结构，没有跳跃连接。
    适合作为基线模型来对比U-Net的效果。

    【工作原理】
    1. 编码器：逐层下采样提取局部特征，共4层
    2. 解码器：逐层上采样重建频谱，共4层
    3. 注意：没有跳跃连接，信息仅通过编码器-解码器传递

    【输入输出】
    输入: [batch, 1, 256, time] - 含啸叫的幅度谱
    输出: [batch, 1, 256, time] - 抑制啸叫后的幅度谱

    【网络层】
    conv1-conv4: 4层编码器，逐层提取特征
    upconv1-upconv4: 4层解码器，逐层重建频谱
    """

    def __init__(self):
        """初始化纯CNN模型"""
        super(AudioCNN, self).__init__()

        # =================== 编码器部分（4层）===================
        # 编码器的作用：提取局部特征，同时降低分辨率

        # 编码器第1层：输入 [B,1,256,T] -> 输出 [B,32,128,T]
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 编码器第2层：输入 [B,32,128,T] -> 输出 [B,64,64,T]
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 编码器第3层：输入 [B,64,64,T] -> 输出 [B,128,32,T]
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 编码器第4层：输入 [B,128,32,T] -> 输出 [B,256,16,T]
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # =================== 解码器部分（4层）===================
        # 解码器的作用：恢复分辨率，重建频谱

        # 解码器第4层：输入 [B,256,16,T] -> 输出 [B,128,32,T]
        self.upconv4 = nn.Sequential(
            nn.ConvTranspose2d(
                256, 128,
                kernel_size=3,
                stride=(2, 1),
                padding=1,
                output_padding=(1, 0)
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # 解码器第3层：输入 [B,128,32,T] -> 输出 [B,64,64,T]
        self.upconv3 = nn.Sequential(
            nn.ConvTranspose2d(
                128, 64,
                kernel_size=3,
                stride=(2, 1),
                padding=1,
                output_padding=(1, 0)
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # 解码器第2层：输入 [B,64,64,T] -> 输出 [B,32,128,T]
        self.upconv2 = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32,
                kernel_size=3,
                stride=(2, 1),
                padding=1,
                output_padding=(1, 0)
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # 解码器第1层（输出层）：输入 [B,32,128,T] -> 输出 [B,1,256,T]
        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(
                32, 1,
                kernel_size=3,
                stride=(2, 1),
                padding=1,
                output_padding=(1, 0)
            ),
            nn.Sigmoid(),  # 生成[0,1]范围的掩膜
        )

    def forward(self, x):
        """前向传播函数

        Args:
            x: 输入频谱，格式为 [batch, 1, 256, time]

        Returns:
            output: 处理后的频谱，格式为 [batch, 1, 256, time]
        """
        # =================== 步骤1：输入预处理 ===================
        # 将线性幅度谱转换为对数域
        x_log = torch.log10(x + 1e-8)

        # =================== 步骤2：编码器前向传播 ===================
        # 逐层下采样，提取局部特征
        e1 = self.conv1(x_log)  # [B,1,256,T] -> [B,32,128,T]
        e2 = self.conv2(e1)     # [B,32,128,T] -> [B,64,64,T]
        e3 = self.conv3(e2)     # [B,64,64,T] -> [B,128,32,T]
        e4 = self.conv4(e3)     # [B,128,32,T] -> [B,256,16,T]

        # =================== 步骤3：解码器前向传播 ===================
        # 注意：这里没有跳跃连接，直接使用编码器的输出
        d4 = self.upconv4(e4)   # [B,256,16,T] -> [B,128,32,T]
        d3 = self.upconv3(d4)   # [B,128,32,T] -> [B,64,64,T]
        d2 = self.upconv2(d3)   # [B,64,64,T] -> [B,32,128,T]
        mask = self.upconv1(d2) # [B,32,128,T] -> [B,1,256,T]

        # =================== 步骤4：应用掩膜 ===================
        output = x * mask

        return output
