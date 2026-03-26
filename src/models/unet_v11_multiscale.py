"""
============================================================
U-Net v11 模型 - 多尺度U-Net并行处理
============================================================

【文件功能】
这个文件实现了一个多尺度U-Net架构，使用不同深度的U-Net
处理不同的频率范围，为低频、中频和高频提供专门的处理。

【主要组件】
- AudioUNet3 类：3层U-Net，处理低频和高频
- AudioUNet5 类：5层U-Net，处理中频
- AudioUNet5MultiScale 类：多尺度融合模型

【网络架构】
整体架构：
  输入: [B, 1, 256, T]
    ↓ 分成3个频段
  低频 [0:64]: [B, 1, 64, T] → AudioUNet3（浅层，快速）
  中频 [64:192]: [B, 1, 128, T] → AudioUNet5（平衡）
  高频 [192:256]: [B, 1, 64, T] → AudioUNet3（浅层，避免维度问题）
    ↓ 拼接输出
  融合: [B, 3, 256, T] → 1×1卷积 → [B, 1, 256, T]

AudioUNet3（3层）：
  编码器: [B,1,64,T] → [B,16,32,T] → [B,32,16,T] → [B,64,8,T]
  解码器: [B,64,8,T] → [B,32,16,T] → [B,16,32,T] → [B,1,64,T]

AudioUNet5（5层）：
  编码器: [B,1,128,T] → ... → [B,256,4,T]
  解码器: [B,256,4,T] → ... → [B,1,128,T]

【关键参数说明】
- 低频处理：3层U-Net，参数少，速度快
- 中频处理：5层U-Net，平衡性能和速度
- 高频处理：3层U-Net（避免过度下采样导致的维度不匹配）
- 融合方式：1×1卷积融合3个尺度的输出

【数据处理流程】
1. 输入：线性幅度谱 [B, 1, 256, T]
2. 分频段：分成低、中、高三个频段
3. 并行处理：每个频段用专门的U-Net处理
4. 融合：拼接三个输出并通过1×1卷积
5. 输出：[0,1]掩膜
6. 最终结果：输入 × 掩膜

【模型特点】
✓ 分频处理：不同频段用不同深度的网络
✓ 高效计算：低频用浅层网络，节省计算
✓ 稳定架构：避免过度下采样导致的维度不匹配
✓ 灵活架构：可根据频率特性调整网络深度
✓ 多尺度融合：融合不同尺度的特征

【与其他版本区别】
- v2：单一5层U-Net处理所有频段
- v11（本模型）：多个不同深度U-Net处理不同频段

【使用示例】
```python
from src.models.unet_v11_multiscale import AudioUNet5MultiScale
import torch

# 创建模型
model = AudioUNet5MultiScale()

# 准备输入
input_spec = torch.randn(4, 1, 256, 376)

# 前向传播
output_spec = model(input_spec)  # 输出: [4, 1, 256, 376]
```
"""

import torch
import torch.nn as nn


class AudioUNet3(nn.Module):
    """3层U-Net用于低频处理"""
    
    def __init__(self):
        super(AudioUNet3, self).__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Decoder
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=(2, 1), padding=1, output_padding=(0, 0)
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(
                64, 16, kernel_size=3, stride=(2, 1), padding=1, output_padding=(0, 0)
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(
                32, 1, kernel_size=3, stride=(2, 1), padding=1, output_padding=(0, 0)
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_log = torch.log10(x + 1e-8)

        e1 = self.enc1(x_log)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        d3 = self.dec3(e3)

        # 确保尺寸匹配后再拼接
        if d3.shape[2] != e2.shape[2] or d3.shape[3] != e2.shape[3]:
            # 调整 d3 的尺寸以匹配 e2
            d3 = nn.functional.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)

        d3_cat = torch.cat([d3, e2], dim=1)

        d2 = self.dec2(d3_cat)

        # 确保尺寸匹配后再拼接
        if d2.shape[2] != e1.shape[2] or d2.shape[3] != e1.shape[3]:
            # 调整 d2 的尺寸以匹配 e1
            d2 = nn.functional.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)

        d2_cat = torch.cat([d2, e1], dim=1)

        mask = self.dec1(d2_cat)

        # 确保最终输出尺寸匹配输入
        if mask.shape[2] != x.shape[2] or mask.shape[3] != x.shape[3]:
            mask = nn.functional.interpolate(mask, size=x.shape[2:], mode='bilinear', align_corners=False)

        output = x * mask
        return output


class AudioUNet7(nn.Module):
    """7层U-Net用于高频处理

    深层网络，参数多，处理精细的高频段处理单元。
    """
    
    def __init__(self):
        super(AudioUNet7, self).__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.enc4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.enc5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.enc6 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.enc7 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Decoder
        self.dec7 = nn.Sequential(
            nn.ConvTranspose2d(
                1024, 512, kernel_size=3, stride=(2, 1), padding=1, output_padding=(0, 0)
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.dec6 = nn.Sequential(
            nn.ConvTranspose2d(
                1024, 256, kernel_size=3, stride=(2, 1), padding=1, output_padding=(0, 0)
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(
                512, 128, kernel_size=3, stride=(2, 1), padding=1, output_padding=(0, 0)
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(
                256, 64, kernel_size=3, stride=(2, 1), padding=1, output_padding=(0, 0)
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(
                128, 32, kernel_size=3, stride=(2, 1), padding=1, output_padding=(0, 0)
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(
                64, 16, kernel_size=3, stride=(2, 1), padding=1, output_padding=(0, 0)
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(
                32, 1, kernel_size=3, stride=(2, 1), padding=1, output_padding=(0, 0)
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_log = torch.log10(x + 1e-8)

        e1 = self.enc1(x_log)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)

        d7 = self.dec7(e7)

        # 确保尺寸匹配后再拼接
        if d7.shape[2] != e6.shape[2] or d7.shape[3] != e6.shape[3]:
            d7 = nn.functional.interpolate(d7, size=e6.shape[2:], mode='bilinear', align_corners=False)

        d7_cat = torch.cat([d7, e6], dim=1)

        d6 = self.dec6(d7_cat)

        # 确保尺寸匹配后再拼接
        if d6.shape[2] != e5.shape[2] or d6.shape[3] != e5.shape[3]:
            d6 = nn.functional.interpolate(d6, size=e5.shape[2:], mode='bilinear', align_corners=False)

        d6_cat = torch.cat([d6, e5], dim=1)

        d5 = self.dec5(d6_cat)

        # 确保尺寸匹配后再拼接
        if d5.shape[2] != e4.shape[2] or d5.shape[3] != e4.shape[3]:
            d5 = nn.functional.interpolate(d5, size=e4.shape[2:], mode='bilinear', align_corners=False)

        d5_cat = torch.cat([d5, e4], dim=1)

        d4 = self.dec4(d5_cat)

        # 确保尺寸匹配后再拼接
        if d4.shape[2] != e3.shape[2] or d4.shape[3] != e3.shape[3]:
            d4 = nn.functional.interpolate(d4, size=e3.shape[2:], mode='bilinear', align_corners=False)

        d4_cat = torch.cat([d4, e3], dim=1)

        d3 = self.dec3(d4_cat)

        # 确保尺寸匹配后再拼接
        if d3.shape[2] != e2.shape[2] or d3.shape[3] != e2.shape[3]:
            d3 = nn.functional.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)

        d3_cat = torch.cat([d3, e2], dim=1)

        d2 = self.dec2(d3_cat)

        # 确保尺寸匹配后再拼接
        if d2.shape[2] != e1.shape[2] or d2.shape[3] != e1.shape[3]:
            d2 = nn.functional.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)

        d2_cat = torch.cat([d2, e1], dim=1)

        mask = self.dec1(d2_cat)

        # 确保最终输出尺寸匹配输入
        if mask.shape[2] != x.shape[2] or mask.shape[3] != x.shape[3]:
            mask = nn.functional.interpolate(mask, size=x.shape[2:], mode='bilinear', align_corners=False)

        output = x * mask
        return output


class AudioUNet5(nn.Module):
    """5层U-Net用于中频处理

    平衡的网络深度，兼顾性能和速度的中频段处理单元。
    """
    
    def __init__(self):
        super(AudioUNet5, self).__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.enc4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.enc5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Decoder
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(
                256, 128, kernel_size=3, stride=(2, 1), padding=1, output_padding=(0, 0)
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(
                256, 64, kernel_size=3, stride=(2, 1), padding=1, output_padding=(0, 0)
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(
                128, 32, kernel_size=3, stride=(2, 1), padding=1, output_padding=(0, 0)
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(
                64, 16, kernel_size=3, stride=(2, 1), padding=1, output_padding=(0, 0)
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(
                32, 1, kernel_size=3, stride=(2, 1), padding=1, output_padding=(0, 0)
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_log = torch.log10(x + 1e-8)

        e1 = self.enc1(x_log)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        d5 = self.dec5(e5)

        # 确保尺寸匹配后再拼接
        if d5.shape[2] != e4.shape[2] or d5.shape[3] != e4.shape[3]:
            d5 = nn.functional.interpolate(d5, size=e4.shape[2:], mode='bilinear', align_corners=False)

        d5_cat = torch.cat([d5, e4], dim=1)

        d4 = self.dec4(d5_cat)

        # 确保尺寸匹配后再拼接
        if d4.shape[2] != e3.shape[2] or d4.shape[3] != e3.shape[3]:
            d4 = nn.functional.interpolate(d4, size=e3.shape[2:], mode='bilinear', align_corners=False)

        d4_cat = torch.cat([d4, e3], dim=1)

        d3 = self.dec3(d4_cat)

        # 确保尺寸匹配后再拼接
        if d3.shape[2] != e2.shape[2] or d3.shape[3] != e2.shape[3]:
            d3 = nn.functional.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)

        d3_cat = torch.cat([d3, e2], dim=1)

        d2 = self.dec2(d3_cat)

        # 确保尺寸匹配后再拼接
        if d2.shape[2] != e1.shape[2] or d2.shape[3] != e1.shape[3]:
            d2 = nn.functional.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)

        d2_cat = torch.cat([d2, e1], dim=1)

        mask = self.dec1(d2_cat)

        # 确保最终输出尺寸匹配输入
        if mask.shape[2] != x.shape[2] or mask.shape[3] != x.shape[3]:
            mask = nn.functional.interpolate(mask, size=x.shape[2:], mode='bilinear', align_corners=False)

        output = x * mask
        return output


class AudioUNet5MultiScale(nn.Module):
    """多尺度U-Net用于音频啸叫抑制

    这个模型使用不同深度的多个U-Net处理不同的频率范围：
    - 低频（0-64 bins）：3层U-Net（浅层，快速）
    - 中频（64-192 bins）：5层U-Net（平衡）
    - 高频（192-256 bins）：3层U-Net（避免过度下采样）

    输出通过1×1卷积融合，生成最终的乘性掩膜。

    【工作原理】
    1. 分频：将输入频谱分成低、中、高三个频段
    2. 并行处理：每个频段用专门的U-Net处理
    3. 特殊处理：低频用浅层网络（快速），高频也用浅层网络避免过采样
    4. 融合：拼接三个尺度的输出并通过1×1卷积
    5. 重建：生成乘性掩膜应用于输入

    【输入输出】
    输入: [batch, 1, 256, time] - 含啸叫的幅度谱
    输出: [batch, 1, 256, time] - 抑制啸叫后的幅度谱

    【网络层】
    unet_low: 3层U-Net，处理低频 [0:64]
    unet_mid: 5层U-Net，处理中频 [64:192]
    unet_high: 3层U-Net，处理高频 [192:256]（使用AudioUNet3避免维度问题）
    fusion: 融合层，组合多尺度输出
    """

    def __init__(self):
        """初始化多尺度U-Net模型"""
        super(AudioUNet5MultiScale, self).__init__()

        # 初始化三个不同深度的U-Net
        # 注意：高频段使用AudioUNet3而不是AudioUNet7
        # 因为AudioUNet7的7层下采样会导致频率维度太小（64→32→16→8→4→2→1→1）
        # 在解码阶段拼接时会出现维度不匹配的问题
        self.unet_low = AudioUNet3()      # 低频处理（0-64 bins）
        self.unet_mid = AudioUNet5()      # 中频处理（64-192 bins）
        self.unet_high = AudioUNet3()     # 高频处理（192-256 bins），使用3层避免维度问题

        # 融合层：组合所有尺度的输出
        self.fusion = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """多尺度U-Net前向传播

        Args:
            x: 输入频谱 [B, 1, 256, T]

        Returns:
            output: 输出频谱 [B, 1, 256, T]
        """
        # 分割输入为三个频段
        # 低频: [0:64], 中频: [64:192], 高频: [192:256]
        x_low = x[:, :, 0:64, :]       # [B, 1, 64, T]
        x_mid = x[:, :, 64:192, :]     # [B, 1, 128, T]
        x_high = x[:, :, 192:256, :]   # [B, 1, 64, T]

        # 用专门的U-Net处理每个频段
        out_low = self.unet_low(x_low)     # [B, 1, 64, T]
        out_mid = self.unet_mid(x_mid)     # [B, 1, 128, T]
        out_high = self.unet_high(x_high)  # [B, 1, 64, T]

        # 沿通道维度拼接输出
        out_concat = torch.cat([out_low, out_mid, out_high], dim=1)  # [B, 3, 256, T]

        # 应用融合层
        mask = self.fusion(out_concat)  # [B, 1, 256, T]

        # 应用乘性掩膜
        output = x * mask

        return output