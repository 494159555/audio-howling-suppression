"""
============================================================
U-Net v2 模型 - 5层U-Net音频啸叫抑制（默认模型）
============================================================

【文件功能】
这个文件实现了一个5层U-Net模型，用于音频啸叫抑制任务。
这是本项目的默认模型，提供了强大的特征提取和频谱重建能力。

【主要组件】
- AudioUNet5 类：5层U-Net模型
  - 编码器：5层卷积下采样，提取深层抽象特征
  - 解码器：5层转置卷积上采样，重建高质量频谱
  - 跳跃连接：连接每一层对应的编码器和解码器，保留多尺度信息

【网络架构】
编码器（下采样过程）：
  输入: [B, 1, 256, T]
    ↓
  enc1: [B, 1, 256, T] → [B, 16, 128, T]  (下采样2倍)
    ↓
  enc2: [B, 16, 128, T] → [B, 32, 64, T]   (下采样2倍)
    ↓
  enc3: [B, 32, 64, T] → [B, 64, 32, T]    (下采样2倍)
    ↓
  enc4: [B, 64, 32, T] → [B, 128, 16, T]   (下采样2倍)
    ↓
  enc5: [B, 128, 16, T] → [B, 256, 8, T]   (下采样2倍，瓶颈层)

解码器（上采样过程）：
  enc5: [B, 256, 8, T]
    ↓ + enc4跳跃连接
  dec5: [B, 256, 8, T] → [B, 128, 16, T]
    ↓ 拼接 enc4: [B, 128+128, 16, T]
    ↓
  dec4: [B, 256, 16, T] → [B, 64, 32, T]
    ↓ 拼接 enc3: [B, 64+64, 32, T]
    ↓
  dec3: [B, 128, 32, T] → [B, 32, 64, T]
    ↓ 拼接 enc2: [B, 32+32, 64, T]
    ↓
  dec2: [B, 64, 64, T] → [B, 16, 128, T]
    ↓ 拼接 enc1: [B, 16+16, 128, T]
    ↓
  dec1: [B, 32, 128, T] → [B, 1, 256, T]  (输出掩膜)

【关键参数说明】
- 卷积核大小：3×3（局部特征提取）
- 下采样步长：(2, 1)（频率方向下采样，时间方向保持）
- 激活函数：
  * 编码器：LeakyReLU(0.2, inplace=True)（允许负梯度，节省内存）
  * 解码器：ReLU(inplace=True)（标准激活，节省内存）
  * 输出层：Sigmoid（生成[0,1]掩膜）
- 批归一化：每层使用，加速训练
- inplace=True：直接在原内存上操作，节省显存

【数据处理流程】
1. 输入：线性幅度谱 [B, 1, 256, T]
2. Log变换：log10(x + 1e-8)（对数域处理）
3. 编码器：提取5个尺度的特征
4. 解码器：通过跳跃连接重建频谱
5. 输出：[0,1]掩膜
6. 最终结果：输入 × 掩膜

【模型特点】
✓ 深层架构：5层编码器和5层解码器，强大的特征提取能力（约4.5M参数）
✓ 跳跃连接：每一层都有跳跃连接，保留多尺度信息
✓ Log域处理：提高数值稳定性
✓ 掩膜机制：输出掩膜与输入相乘，保留原始信息
✓ 批归一化：提高训练稳定性

【与v1版本区别】
- v1：3层U-Net，参数少（约1.2M），速度快
- v2（本模型）：5层U-Net，参数多（约4.5M），性能更强，是默认模型

【使用示例】
```python
from src.models.unet_v2 import AudioUNet5
import torch

# 创建模型
model = AudioUNet5()

# 准备输入：[批次=4, 通道=1, 频率=256, 时间=376]
input_spec = torch.randn(4, 1, 256, 376)

# 前向传播
output_spec = model(input_spec)  # 输出: [4, 1, 256, 376]

# 输出是输入与掩膜的乘积
```
"""

import torch
import torch.nn as nn


class AudioUNet5(nn.Module):
    """5层U-Net模型用于音频啸叫抑制（默认模型）

    这是项目的主力模型，通过深层的编码器-解码器结构和多层跳跃连接
    来实现高质量的音频频谱修复。

    【工作原理】
    1. 编码器：逐层下采样提取特征，共5层，特征越来越抽象
    2. 瓶颈层：最深层，特征最抽象，分辨率最小
    3. 解码器：逐层上采样重建频谱，共5层
    4. 跳跃连接：每一层编码器的特征都传递给对应的解码器层

    【输入输出】
    输入: [batch, 1, 256, time] - 含啸叫的幅度谱
    输出: [batch, 1, 256, time] - 抑制啸叫后的幅度谱

    【网络层】
    enc1-enc5: 5层编码器，逐层提取特征
    dec1-de5: 5层解码器，逐层重建频谱
    """

    def __init__(self):
        """初始化5层U-Net模型"""
        super(AudioUNet5, self).__init__()

        # =================== 编码器部分（5层）===================
        # 编码器的作用：提取特征，同时降低分辨率（下采样）

        # 编码器第1层：输入 [B,1,256,T] -> 输出 [B,16,128,T]
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=(2, 1), padding=1),
            # stride=(2,1): 频率方向步长2（下采样），时间方向步长1

            nn.BatchNorm2d(16),  # 批归一化
            nn.LeakyReLU(0.2, inplace=True),
            # inplace=True: 直接修改输入张量，节省内存
        )

        # 编码器第2层：输入 [B,16,128,T] -> 输出 [B,32,64,T]
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 编码器第3层：输入 [B,32,64,T] -> 输出 [B,64,32,T]
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 编码器第4层：输入 [B,64,32,T] -> 输出 [B,128,16,T]
        self.enc4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 编码器第5层（瓶颈层）：输入 [B,128,16,T] -> 输出 [B,256,8,T]
        # 这是最深层，特征最抽象，分辨率最小
        self.enc5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # =================== 解码器部分（5层）===================
        # 解码器的作用：恢复分辨率，重建频谱（上采样）

        # 解码器第5层：输入 [B,256,8,T] -> 输出 [B,128,16,T]
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(
                256, 128,  # 通道数从256降到128
                kernel_size=3,
                stride=(2, 1),  # 频率方向上采样2倍
                padding=1,
                output_padding=(1, 0)  # 补偿上采样时的尺寸损失
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),  # 解码器使用ReLU
        )

        # 解码器第4层：输入 [B,256,16,T] (拼接后) -> 输出 [B,64,32,T]
        # 注意：输入256通道是因为 dec5的128 + enc4的128 = 256
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(
                256, 64,  # 通道数从256降到64
                kernel_size=3,
                stride=(2, 1),
                padding=1,
                output_padding=(1, 0)
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # 解码器第3层：输入 [B,128,32,T] (拼接后) -> 输出 [B,32,64,T]
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(
                128, 32,  # 通道数从128降到32
                kernel_size=3,
                stride=(2, 1),
                padding=1,
                output_padding=(1, 0)
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # 解码器第2层：输入 [B,64,64,T] (拼接后) -> 输出 [B,16,128,T]
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(
                64, 16,  # 通道数从64降到16
                kernel_size=3,
                stride=(2, 1),
                padding=1,
                output_padding=(1, 0)
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        # 解码器第1层（输出层）：输入 [B,32,128,T] (拼接后) -> 输出 [B,1,256,T]
        # 输出Sigmoid激活，生成[0,1]范围的掩膜
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(
                32, 1,  # 最终输出单通道
                kernel_size=3,
                stride=(2, 1),
                padding=1,
                output_padding=(1, 0)
            ),
            nn.Sigmoid(),  # Sigmoid将输出限制在[0,1]，生成掩膜
        )

    def forward(self, x):
        """前向传播函数

        Args:
            x: 输入频谱，格式为 [batch, 1, 256, time]
               - batch: 批次大小
               - 1: 通道数（单通道频谱）
               - 256: 频率bin数
               - time: 时间帧数

        Returns:
            output: 处理后的频谱，格式为 [batch, 1, 256, time]
                    是输入频谱与预测掩膜的乘积
        """
        # =================== 步骤1：输入预处理 ===================
        # 将线性幅度谱转换为对数域，提高数值稳定性
        x_log = torch.log10(x + 1e-8)

        # =================== 步骤2：编码器前向传播 ===================
        # 逐层下采样，提取5个尺度的特征
        e1 = self.enc1(x_log)  # [B,1,256,T] -> [B,16,128,T]
        e2 = self.enc2(e1)     # [B,16,128,T] -> [B,32,64,T]
        e3 = self.enc3(e2)     # [B,32,64,T] -> [B,64,32,T]
        e4 = self.enc4(e3)     # [B,64,32,T] -> [B,128,16,T]
        e5 = self.enc5(e4)     # [B,128,16,T] -> [B,256,8,T] (瓶颈层)

        # =================== 步骤3：解码器前向传播 + 跳跃连接 ===================
        # 每一层解码器都会接收对应的编码器特征（跳跃连接）

        # dec5 + enc4跳跃连接
        d5 = self.dec5(e5)                      # [B,256,8,T] -> [B,128,16,T]
        d5_cat = torch.cat([d5, e4], dim=1)     # 拼接: [B,128+128,16,T] = [B,256,16,T]

        # dec4 + enc3跳跃连接
        d4 = self.dec4(d5_cat)                  # [B,256,16,T] -> [B,64,32,T]
        d4_cat = torch.cat([d4, e3], dim=1)     # 拼接: [B,64+64,32,T] = [B,128,32,T]

        # dec3 + enc2跳跃连接
        d3 = self.dec3(d4_cat)                  # [B,128,32,T] -> [B,32,64,T]
        d3_cat = torch.cat([d3, e2], dim=1)     # 拼接: [B,32+32,64,T] = [B,64,64,T]

        # dec2 + enc1跳跃连接
        d2 = self.dec2(d3_cat)                  # [B,64,64,T] -> [B,16,128,T]
        d2_cat = torch.cat([d2, e1], dim=1)     # 拼接: [B,16+16,128,T] = [B,32,128,T]

        # dec1：生成最终的掩膜
        mask = self.dec1(d2_cat)                # [B,32,128,T] -> [B,1,256,T]

        # =================== 步骤4：应用掩膜 ===================
        # 将预测的掩膜与原始输入相乘
        # mask值接近1.0 = 保留该频率成分（干净信号）
        # mask值接近0.0 = 抑制该频率成分（啸叫噪声）
        output = x * mask

        return output
