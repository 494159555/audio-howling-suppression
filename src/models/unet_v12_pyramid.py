"""
============================================================
U-Net v12 模型 - 5层U-Net + 金字塔池化模块
============================================================

【文件功能】
这个文件实现了一个在瓶颈层使用金字塔池化模块（PPM）的U-Net架构，
用于捕获多尺度上下文信息，提高音频啸叫抑制效果。

【主要组件】
- PyramidPoolingModule 类：金字塔池化模块
  - 多尺度自适应平均池化
  - 特征上采样和拼接
  - 通道投影
- AudioUNet5Pyramid 类：带PPM的5层U-Net

【网络架构】
编码器（下采样）：
  输入: [B, 1, 256, T]
    ↓
  enc1: [B, 1, 256, T] → [B, 16, 128, T]
    ↓
  enc2: [B, 16, 128, T] → [B, 32, 64, T]
    ↓
  enc3: [B, 32, 64, T] → [B, 64, 32, T]
    ↓
  enc4: [B, 64, 32, T] → [B, 128, 16, T]
    ↓
  enc5: [B, 128, 16, T] → [B, 256, 8, T]（瓶颈层）

金字塔池化模块（瓶颈层）：
  输入: [B, 256, 8, T]
    ↓ 4个不同尺度的池化
  池化1: [B, 256, 8, T] → [B, 64, 1, 1] → 上采样 → [B, 64, 8, T]
  池化2: [B, 256, 8, T] → [B, 64, 2, 2] → 上采样 → [B, 64, 8, T]
  池化3: [B, 256, 8, T] → [B, 64, 3, 3] → 上采样 → [B, 64, 8, T]
  池化4: [B, 256, 8, T] → [B, 64, 6, 6] → 上采样 → [B, 64, 8, T]
    ↓ 拼接 + 投影
  输出: [B, 256, 8, T]

解码器（上采样）：
  dec5: [B, 256, 8, T] → [B, 128, 16, T] + enc4跳跃连接
    ↓
  dec4: [B, 256, 16, T] → [B, 64, 32, T] + enc3跳跃连接
    ↓
  dec3: [B, 128, 32, T] → [B, 32, 64, T] + enc2跳跃连接
    ↓
  dec2: [B, 64, 64, T] → [B, 16, 128, T] + enc1跳跃连接
    ↓
  dec1: [B, 32, 128, T] → [B, 1, 256, T]

【关键参数说明】
- pyramid_levels: 金字塔层级（默认：(1, 2, 3, 6)）
- reduction_ratio: 通道降维比率（默认: 4）
- 池化尺度：1×1, 2×2, 3×3, 6×6
- 激活函数：LeakyReLU(0.2)，ReLU，Sigmoid

【数据处理流程】
1. 输入：线性幅度谱 [B, 1, 256, T]
2. Log变换：log10(x + 1e-8)
3. 编码器：提取5个尺度的特征
4. PPM：在瓶颈层捕获多尺度上下文
5. 解码器：通过跳跃连接重建频谱
6. 输出：[0,1]掩膜
7. 最终结果：输入 × 掩膜

【模型特点】
✓ 多尺度上下文：在瓶颈层聚合不同尺度的信息
✓ 全局感知：通过大尺度池化捕获全局信息
✓ 局部细节：保留小尺度池化的局部信息
✓ 参数高效：使用通道降维减少参数量
✓ 性能提升：最小参数增加带来性能提升

【与其他版本区别】
- v2：标准5层U-Net
- v12（本模型）：添加金字塔池化模块，捕获多尺度上下文

【使用示例】
```python
from src.models.unet_v12_pyramid import AudioUNet5Pyramid
import torch

# 创建模型
model = AudioUNet5Pyramid(pyramid_levels=(1, 2, 3, 6))

# 准备输入
input_spec = torch.randn(4, 1, 256, 376)

# 前向传播
output_spec = model(input_spec)  # 输出: [4, 1, 256, 376]
```
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PyramidPoolingModule(nn.Module):
    """金字塔池化模块（PPM）用于多尺度上下文聚合

    该模块通过在不同尺度上应用自适应平均池化来捕获
    多尺度上下文信息，然后上采样并拼接特征。

    基于： "Pyramid Scene Parsing Network" (Zhao et al., 2017)

    【工作原理】
    1. 对输入特征应用4个不同尺度的自适应平均池化
    2. 每个池化分支通过1×1卷积降维
    3. 上采样回原始尺寸
    4. 拼接所有分支并投影回原始通道数

    Args:
        in_channels: 输入通道数
        pool_sizes: 池化输出尺寸列表（默认: (1, 2, 3, 6)）
        reduction_ratio: 通道降维比率
    """
    
    def __init__(self, in_channels, pool_sizes=(1, 2, 3, 6), reduction_ratio=4):
        super(PyramidPoolingModule, self).__init__()

        # 金字塔层数
        self.pool_sizes = pool_sizes
        self.num_levels = len(pool_sizes)

        # 计算每层的降维通道数
        reduced_channels = in_channels // reduction_ratio

        # 创建金字塔分支
        self.pyramid_branches = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=size),
                nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
                nn.BatchNorm2d(reduced_channels),
                nn.ReLU(inplace=True)
            )
            for size in pool_sizes
        ])

        # 输出投影以恢复原始通道维度
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels + reduced_channels * self.num_levels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """应用金字塔池化模块

        Args:
            x: 输入特征图 [B, C, H, W]

        Returns:
            output: 带有多尺度上下文的输出特征图 [B, C, H, W]
        """
        input_size = x.size()[2:]

        # 应用每个金字塔层级
        pyramid_features = []
        for branch in self.pyramid_branches:
            # 池化和投影
            pooled = branch(x)
            # 上采样回原始尺寸
            upsampled = F.interpolate(
                pooled,
                size=input_size,
                mode='bilinear',
                align_corners=True
            )
            pyramid_features.append(upsampled)

        # 拼接原始特征和金字塔特征
        pyramid_concat = torch.cat([x] + pyramid_features, dim=1)

        # 投影回原始通道数
        output = self.out_conv(pyramid_concat)

        return output


class AudioUNet5Pyramid(nn.Module):
    """5层U-Net + 金字塔池化用于音频啸叫抑制

    这个模型通过在瓶颈层添加金字塔池化模块来扩展
    标准5层U-Net，使模型能够捕获多尺度上下文信息。

    【工作原理】
    1. 编码器：提取5个尺度的空间特征
    2. PPM：在瓶颈层聚合多尺度上下文信息
    3. 解码器：通过跳跃连接重建频谱
    4. 多尺度特征：增强特征表示能力

    【输入输出】
    输入: [batch, 1, 256, time] - 含啸叫的幅度谱
    输出: [batch, 1, 256, time] - 抑制啸叫后的幅度谱

    【网络层】
    enc1-enc5: 5层编码器
    ppm: 金字塔池化模块
    dec1-dec5: 5层解码器

    Args:
        pyramid_levels: PPM的池化尺寸（默认: (1, 2, 3, 6)）
        reduction_ratio: PPM的通道降维比率（默认: 4）
    """
    
    def __init__(self, pyramid_levels=(1, 2, 3, 6), reduction_ratio=4):
        """初始化带金字塔池化的U-Net

        Args:
            pyramid_levels: 金字塔池化模块的池化尺寸
            reduction_ratio: 通道降维比率
        """
        super(AudioUNet5Pyramid, self).__init__()

        # =================== 编码器部分（5层）===================
        
        # 编码器第1层：输入 [B,1,256,T] -> 输出 [B,16,128,T]
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
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
        self.enc5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # =================== 金字塔池化模块（瓶颈层）===================
        self.ppm = PyramidPoolingModule(
            in_channels=256,
            pool_sizes=pyramid_levels,
            reduction_ratio=reduction_ratio
        )

        # =================== 解码器部分（5层）===================
        
        # 解码器第5层：输入 [B,256,8,T] -> 输出 [B,128,16,T]
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(
                256, 128, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # 解码器第4层：输入 [B,256,16,T] (拼接后) -> 输出 [B,64,32,T]
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(
                256, 64, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # 解码器第3层：输入 [B,128,32,T] (拼接后) -> 输出 [B,32,64,T]
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(
                128, 32, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # 解码器第2层：输入 [B,64,64,T] (拼接后) -> 输出 [B,16,128,T]
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(
                64, 16, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        # 解码器第1层（输出层）：输入 [B,32,128,T] (拼接后) -> 输出 [B,1,256,T]
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(
                32, 1, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)
            ),
            nn.Sigmoid(),  # Sigmoid将输出限制在[0,1]，生成掩膜
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入频谱 [B, 1, 256, T]

        Returns:
            output: 输出频谱 [B, 1, 256, T]
        """
        # =================== 步骤1：输入预处理 ===================
        x_log = torch.log10(x + 1e-8)

        # =================== 步骤2：编码器前向传播 ===================
        e1 = self.enc1(x_log)    # [B, 16, 128, T]
        e2 = self.enc2(e1)        # [B, 32, 64, T]
        e3 = self.enc3(e2)        # [B, 64, 32, T]
        e4 = self.enc4(e3)        # [B, 128, 16, T]
        e5 = self.enc5(e4)        # [B, 256, 8, T] - 瓶颈层

        # =================== 步骤3：应用金字塔池化模块（瓶颈层）===================
        # 捕获多尺度上下文信息
        e5_ppm = self.ppm(e5)     # [B, 256, 8, T]

        # =================== 步骤4：解码器前向传播 + 跳跃连接 ===================
        # 解码器第5层 + enc4跳跃连接
        d5 = self.dec5(e5_ppm)    # [B, 128, 16, T]
        d5_cat = torch.cat([d5, e4], dim=1)  # [B, 256, 16, T]

        # 解码器第4层 + enc3跳跃连接
        d4 = self.dec4(d5_cat)    # [B, 64, 32, T]
        d4_cat = torch.cat([d4, e3], dim=1)  # [B, 128, 32, T]

        # 解码器第3层 + enc2跳跃连接
        d3 = self.dec3(d4_cat)    # [B, 32, 64, T]
        d3_cat = torch.cat([d3, e2], dim=1)  # [B, 64, 64, T]

        # 解码器第2层 + enc1跳跃连接
        d2 = self.dec2(d3_cat)    # [B, 16, 128, T]
        d2_cat = torch.cat([d2, e1], dim=1)  # [B, 32, 128, T]

        # 解码器第1层：生成最终的掩膜
        mask = self.dec1(d2_cat)  # [B, 1, 256, T]

        # =================== 步骤5：应用掩膜 ===================
        output = x * mask
        return output