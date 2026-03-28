"""
============================================================
U-Net v13 模型 - 5层U-Net + 特征金字塔网络(FPN)
============================================================

【文件功能】
这个文件实现了一个基于特征金字塔网络（FPN）的U-Net架构，
通过自顶向下路径和横向连接实现强大的多尺度特征融合。

【主要组件】
- AudioUNet5FPN 类：带FPN的5层U-Net
  - 自底向上路径（编码器）：提取多尺度特征
  - 横向连接（1×1卷积）：统一通道数
  - 自顶向下路径（上采样）：传播高层语义信息
  - 解码器：使用FPN特征重建（相加融合）

【网络架构】
编码器（自底向上路径）：
  C1: [B,1,256,T] → [B,16,128,T]
  C2: [B,16,128,T] → [B,32,64,T]
  C3: [B,32,64,T] → [B,64,32,T]
  C4: [B,64,32,T] → [B,128,16,T]
  C5: [B,128,16,T] → [B,256,8,T]（瓶颈层）

横向连接（1×1卷积）：
  L5: C5 → [B,64,8,T]
  L4: C4 → [B,64,16,T]
  L3: C3 → [B,64,32,T]
  L2: C2 → [B,64,64,T]
  L1: C1 → [B,64,128,T]

自顶向下路径（上采样 + 相加）：
  P5: L5（最顶层）
  P4: 上采样(P5) + L4
  P3: 上采样(P4) + L3
  P2: 上采样(P3) + L2
  P1: 上采样(P2) + L1（最底层）

解码器（使用FPN特征，相加融合）：
  dec5: P5 → 与P4相加
  dec4: 结果 → 与P3相加
  dec3: 结果 → 与P2相加
  dec2: 结果 → 与P1相加
  dec1: 结果 → 最终掩膜

【关键参数说明】
- fpn_channels: FPN特征图的通道数（默认: 64）
- use_fpn_fusion: 是否使用FPN融合（默认: True）
- 平滑层：3×3卷积平滑FPN输出
- 横向连接：1×1卷积统一通道数
- 融合方式：特征相加（element-wise addition）而非拼接

【数据处理流程】
1. 输入：线性幅度谱 [B, 1, 256, T]
2. 编码器：自底向上提取特征（C1-C5）
3. 横向连接：1×1卷积降维（L1-L5）
4. FPN：自顶向下路径融合（P1-P5）
5. 解码器：逐步上采样并相加FPN特征
6. 输出：[0,1]掩膜
7. 最终结果：输入 × 掩膜

【模型特点】
✓ 强大的多尺度特征融合
✓ 高层语义信息传播到所有尺度
✓ 改进的特征表示
✓ 更好地检测不同尺度的啸叫
✓ 标准FPN解码方式（相加而非拼接）

【与其他版本区别】
- v2：标准5层U-Net，简单跳跃连接（拼接）
- v11：多尺度U-Net，处理不同频段
- v12：使用金字塔池化模块（PPM）
- v13（本模型）：使用特征金字塔网络（FPN），相加融合

【修复说明】
- 修复了解码器通道数不匹配的问题
- 使用标准FPN解码方式：逐步上采样并相加，而非拼接
- 所有FPN特征图通道数统一为fpn_channels（64）

【使用示例】
```python
from src.models.unet_v13_fpn import AudioUNet5FPN
import torch

# 创建模型
model = AudioUNet5FPN(fpn_channels=64)

# 准备输入
input_spec = torch.randn(4, 1, 256, 376)

# 前向传播
output_spec = model(input_spec)  # 输出: [4, 1, 256, 376]
```
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioUNet5FPN(nn.Module):
    """5层U-Net + 特征金字塔网络用于音频啸叫抑制

    这个模型通过结合特征金字塔网络（FPN）来增强标准U-Net，
    实现强大的多尺度特征融合：
    1. 自底向上路径（编码器）：提取多尺度特征
    2. 自顶向下路径（横向上采样）：传播高层语义信息
    3. 横向连接：结合编码器的空间信息

    基于： "Feature Pyramid Networks for Object Detection" (Lin et al., 2017)

    【工作原理】
    1. 编码器：自底向上提取5个尺度的特征
    2. 横向连接：1×1卷积将所有特征映射到相同通道数
    3. 自顶向下：从顶层开始，上采样并与横向特征相加
    4. 平滑：3×3卷积平滑FPN输出
    5. 解码器：使用融合后的多尺度特征重建

    【输入输出】
    输入: [batch, 1, 256, time] - 含啸叫的幅度谱
    输出: [batch, 1, 256, time] - 抑制啸叫后的幅度谱

    【网络层】
    enc1-enc5: 编码器层（C1-C5）
    lateral1-lateral5: 横向连接（L1-L5）
    smooth1-smooth5: 平滑层
    dec1-dec5: 解码器层

    Args:
        fpn_channels: FPN特征图的通道数（默认: 64）
        use_fpn_fusion: 是否使用FPN融合（默认: True）
    """
    
    def __init__(self, fpn_channels=64, use_fpn_fusion=True):
        """初始化带特征金字塔网络的U-Net

        Args:
            fpn_channels: FPN特征图的通道数
            use_fpn_fusion: 是否使用FPN融合或标准U-Net
        """
        super(AudioUNet5FPN, self).__init__()

        self.use_fpn_fusion = use_fpn_fusion

        # =================== 编码器（自底向上路径）===================
        
        # C1: [B, 1, 256, T] -> [B, 16, 128, T]
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # C2: [B, 16, 128, T] -> [B, 32, 64, T]
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # C3: [B, 32, 64, T] -> [B, 64, 32, T]
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # C4: [B, 64, 32, T] -> [B, 128, 16, T]
        self.enc4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # C5: [B, 128, 16, T] -> [B, 256, 8, T]（瓶颈层）
        self.enc5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # =================== 横向连接（1×1卷积）===================
        # 降维到fpn_channels以高效融合
        self.lateral5 = nn.Conv2d(256, fpn_channels, kernel_size=1)
        self.lateral4 = nn.Conv2d(128, fpn_channels, kernel_size=1)
        self.lateral3 = nn.Conv2d(64, fpn_channels, kernel_size=1)
        self.lateral2 = nn.Conv2d(32, fpn_channels, kernel_size=1)
        self.lateral1 = nn.Conv2d(16, fpn_channels, kernel_size=1)

        # =================== 平滑层（3×3卷积用于FPN输出）===================
        self.smooth5 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        self.smooth4 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        self.smooth3 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        self.smooth2 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        self.smooth1 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)

        # =================== 解码器（使用FPN特征）===================
        # 标准FPN解码器：逐步上采样，每一层处理对应尺度的FPN特征

        # Decoder Layer 5: P5 [B, 64, 8, T] -> [B, 64, 16, T]
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(
                fpn_channels, fpn_channels, kernel_size=3, stride=(2, 1),
                padding=1, output_padding=(1, 0)
            ),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True),
        )

        # Decoder Layer 4: [B, 64, 16, T] -> [B, 64, 32, T]
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(
                fpn_channels, fpn_channels, kernel_size=3, stride=(2, 1),
                padding=1, output_padding=(1, 0)
            ),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True),
        )

        # Decoder Layer 3: [B, 64, 32, T] -> [B, 64, 64, T]
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(
                fpn_channels, fpn_channels, kernel_size=3, stride=(2, 1),
                padding=1, output_padding=(1, 0)
            ),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True),
        )

        # Decoder Layer 2: [B, 64, 64, T] -> [B, 64, 128, T]
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(
                fpn_channels, fpn_channels, kernel_size=3, stride=(2, 1),
                padding=1, output_padding=(1, 0)
            ),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True),
        )

        # Final Layer: [B, 64, 128, T] -> [B, 1, 256, T]
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(
                fpn_channels, 1, kernel_size=3, stride=(2, 1),
                padding=1, output_padding=(1, 0)
            ),
            nn.Sigmoid(),
        )

    def _build_fpn(self, c5, c4, c3, c2, c1):
        """构建特征金字塔网络特征

        Args:
            c5, c4, c3, c2, c1: 编码器特征图

        Returns:
            (p5, p4, p3, p2, p1): FPN特征图元组
        """
        # 应用横向连接
        l5 = self.lateral5(c5)  # [B, 64, 8, T]
        l4 = self.lateral4(c4)  # [B, 64, 16, T]
        l3 = self.lateral3(c3)  # [B, 64, 32, T]
        l2 = self.lateral2(c2)  # [B, 64, 64, T]
        l1 = self.lateral1(c1)  # [B, 64, 128, T]

        # 自顶向下路径（上采样和相加）
        # 从顶层开始（P5）
        p5 = self.smooth5(l5)  # [B, 64, 8, T]

        # P4：上采样P5并与L4相加
        p5_up = F.interpolate(p5, size=l4.shape[2:], mode='bilinear', align_corners=True)
        p4 = self.smooth4(p5_up + l4)  # [B, 64, 16, T]

        # P3：上采样P4并与L3相加
        p4_up = F.interpolate(p4, size=l3.shape[2:], mode='bilinear', align_corners=True)
        p3 = self.smooth3(p4_up + l3)  # [B, 64, 32, T]

        # P2：上采样P3并与L2相加
        p3_up = F.interpolate(p3, size=l2.shape[2:], mode='bilinear', align_corners=True)
        p2 = self.smooth2(p3_up + l2)  # [B, 64, 64, T]

        # P1：上采样P2并与L1相加
        p2_up = F.interpolate(p2, size=l1.shape[2:], mode='bilinear', align_corners=True)
        p1 = self.smooth1(p2_up + l1)  # [B, 64, 128, T]

        return p5, p4, p3, p2, p1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入频谱 [B, 1, 256, T]

        Returns:
            output: 输出频谱 [B, 1, 256, T]
        """
        # =================== 步骤1：输入预处理 ===================
        x_log = torch.log10(x + 1e-8)

        # =================== 步骤2：编码器前向传播（自底向上）===================
        c1 = self.enc1(x_log)    # [B, 16, 128, T]
        c2 = self.enc2(c1)        # [B, 32, 64, T]
        c3 = self.enc3(c2)        # [B, 64, 32, T]
        c4 = self.enc4(c3)        # [B, 128, 16, T]
        c5 = self.enc5(c4)        # [B, 256, 8, T] - 瓶颈层

        # =================== 步骤3：构建特征金字塔网络 ===================
        # FPN融合是这个模型的核心特性
        p5, p4, p3, p2, p1 = self._build_fpn(c5, c4, c3, c2, c1)

        # =================== 步骤4：FPN解码器前向传播 ===================
        # 标准FPN解码方式：从最顶层开始逐步上采样并融合

        # 从P5开始，逐步上采样到原始分辨率
        # P5: [B, 64, 8, T] -> [B, 64, 16, T]
        d5 = self.dec5(p5)

        # 确保尺寸匹配
        if d5.shape[2] != p4.shape[2] or d5.shape[3] != p4.shape[3]:
            d5 = F.interpolate(d5, size=p4.shape[2:], mode='bilinear', align_corners=False)

        # 与P4融合（相加而不是拼接）
        d5_fused = d5 + p4  # [B, 64, 16, T]

        # 上采样到P3的尺寸
        d4 = self.dec4(d5_fused)  # [B, 64, 32, T]

        # 确保尺寸匹配
        if d4.shape[2] != p3.shape[2] or d4.shape[3] != p3.shape[3]:
            d4 = F.interpolate(d4, size=p3.shape[2:], mode='bilinear', align_corners=False)

        # 与P3融合
        d4_fused = d4 + p3  # [B, 64, 32, T]

        # 上采样到P2的尺寸
        d3 = self.dec3(d4_fused)  # [B, 64, 64, T]

        # 确保尺寸匹配
        if d3.shape[2] != p2.shape[2] or d3.shape[3] != p2.shape[3]:
            d3 = F.interpolate(d3, size=p2.shape[2:], mode='bilinear', align_corners=False)

        # 与P2融合
        d3_fused = d3 + p2  # [B, 64, 64, T]

        # 上采样到P1的尺寸
        d2 = self.dec2(d3_fused)  # [B, 64, 128, T]

        # 确保尺寸匹配
        if d2.shape[2] != p1.shape[2] or d2.shape[3] != p1.shape[3]:
            d2 = F.interpolate(d2, size=p1.shape[2:], mode='bilinear', align_corners=False)

        # 与P1融合
        d2_fused = d2 + p1  # [B, 64, 128, T]

        # 最终层：生成掩膜，上采样到原始输入尺寸
        mask = self.dec1(d2_fused)  # [B, 1, 256, T]

        # 确保最终输出尺寸匹配输入
        if mask.shape[2] != x.shape[2] or mask.shape[3] != x.shape[3]:
            mask = F.interpolate(mask, size=x.shape[2:], mode='bilinear', align_corners=False)

        # =================== 步骤5：应用掩膜 ===================
        output = x * mask
        return output