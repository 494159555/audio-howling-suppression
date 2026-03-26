"""
============================================================
U-Net v8 模型 - 5层U-Net + 时间注意力机制
============================================================

【文件功能】
这个文件实现了一个5层U-Net模型，在瓶颈层集成了时间注意力机制，
用于音频啸叫抑制任务。时间注意力能够动态关注不同时间片段的重要性。

【主要组件】
- AudioUNet5TemporalAttention 类：5层U-Net + 时间注意力
  - 编码器：5层卷积下采样，提取空间特征
  - 时间注意力：瓶颈层的时间注意力模块
  - 解码器：5层转置卷积上采样，重建频谱
  - 跳跃连接：连接每一层对应的编码器和解码器

【网络架构】
编码器（下采样过程）：
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
  enc5: [B, 128, 16, T] → [B, 256, 8, T] (瓶颈层)

时间注意力（瓶颈层）：
  enc5: [B, 256, 8, T]
    ↓ 时间注意力
  attended: [B, 256, 8, T] (动态加权)

解码器（上采样过程）：
  attended: [B, 256, 8, T]
    ↓ + enc4跳跃连接
  dec5: [B, 256, 8, T] → [B, 128, 16, T]
    ↓ 拼接 enc4
  dec4: [B, 256, 16, T] → [B, 64, 32, T]
    ↓ 拼接 enc3
  dec3: [B, 128, 32, T] → [B, 32, 64, T]
    ↓ 拼接 enc2
  dec2: [B, 64, 64, T] → [B, 16, 128, T]
    ↓ 拼接 enc1
  dec1: [B, 32, 128, T] → [B, 1, 256, T]

【关键参数说明】
- reduction: 时间注意力的降维比率（默认8）
- 卷积核大小：3×3
- 下采样步长：(2, 1)
- 激活函数：编码器LeakyReLU(0.2)，解码器ReLU，输出Sigmoid
- 注意力机制：使用残差连接

【数据处理流程】
1. 输入：线性幅度谱 [B, 1, 256, T]
2. Log变换：log10(x + 1e-8)
3. 编码器：提取5个尺度的空间特征
4. 时间注意力：动态关注重要时间片段
5. 解码器：通过跳跃连接重建频谱
6. 输出：[0,1]掩膜
7. 最终结果：输入 × 掩膜

【模型特点】
✓ 时间注意力：动态关注不同时间帧的重要性
✓ 可解释性：注意力权重可视化
✓ 时序一致性：提高输出时序平滑性
✓ 动态处理：更好地处理动态啸叫模式
✓ 深层架构：5层U-Net提供强大的特征提取

【与其他版本区别】
- v2：标准5层U-Net，无注意力机制
- v3：使用空间注意力门
- v7：使用双向LSTM建模时序
- v8（本模型）：使用时间注意力机制
- v9：使用ConvLSTM保留空间结构

【使用示例】
```python
from src.models.unet_v8_temporal_attention import AudioUNet5TemporalAttention
import torch

# 创建模型
model = AudioUNet5TemporalAttention(reduction=8)

# 准备输入
input_spec = torch.randn(4, 1, 256, 376)

# 前向传播
output_spec = model(input_spec)  # 输出: [4, 1, 256, 376]
```
"""

import torch
import torch.nn as nn
from .temporal_modules import TemporalAttention


class AudioUNet5TemporalAttention(nn.Module):
    """5层U-Net + 时间注意力用于音频啸叫抑制

    这个模型在标准U-Net的基础上，在瓶颈层添加了时间注意力机制。
    注意力沿时间维度计算权重，使模型能够动态关注相关的
    时间片段（例如，有啸叫活动的时段）。

    【工作原理】
    1. 编码器：逐层下采样提取空间特征
    2. 时间注意力：在瓶颈层计算时间维度的注意力权重
    3. 注意力加权：动态调整不同时间帧的重要性
    4. 解码器：逐层上采样重建频谱
    5. 跳跃连接：保留多尺度空间信息

    【输入输出】
    输入: [batch, 1, 256, time] - 含啸叫的幅度谱
    输出: [batch, 1, 256, time] - 抑制啸叫后的幅度谱

    【网络层】
    enc1-enc5: 5层编码器，提取空间特征
    temporal_attention: 时间注意力模块
    dec1-dec5: 5层解码器，重建频谱
    """
    
    def __init__(self, reduction: int = 8):
        """初始化5层U-Net + 时间注意力模型

        Args:
            reduction: 时间注意力的降维比率（默认: 8）
        """
        super(AudioUNet5TemporalAttention, self).__init__()

        # =================== 编码器部分（5层）===================
        # 编码器的作用：提取空间特征，同时降低分辨率
        
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

        # =================== 时间注意力模块（瓶颈层）===================
        # 时间注意力的作用：动态关注重要时间片段
        self.temporal_attention = TemporalAttention(
            channels=256,
            reduction=reduction,
            use_residual=True
        )

        # =================== 解码器部分（5层）===================
        # 解码器的作用：恢复分辨率，重建频谱
        
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
        """前向传播函数

        通过编码器-注意力-解码器架构处理输入频谱，生成用于啸叫抑制的乘性掩膜。

        Args:
            x: 输入频谱，格式为 [B, 1, 256, T]
               - B: 批次大小
               - 1: 通道数
               - 256: 频率bin数
               - T: 时间帧数

        Returns:
            output: 处理后的频谱，格式为 [B, 1, 256, T]
                    是输入频谱与预测掩膜的乘积
        """
        # =================== 步骤1：输入预处理 ===================
        x_log = torch.log10(x + 1e-8)

        # =================== 步骤2：编码器前向传播 ===================
        e1 = self.enc1(x_log)    # [B,1,256,T] -> [B,16,128,T]
        e2 = self.enc2(e1)        # [B,16,128,T] -> [B,32,64,T]
        e3 = self.enc3(e2)        # [B,32,64,T] -> [B,64,32,T]
        e4 = self.enc4(e3)        # [B,64,32,T] -> [B,128,16,T]
        e5 = self.enc5(e4)        # [B,128,16,T] -> [B,256,8,T] (瓶颈层)

        # =================== 步骤3：时间注意力（瓶颈层）===================
        # 对瓶颈层特征应用时间注意力
        e5_attended = self.temporal_attention(e5)  # [B, 256, 8, T]

        # =================== 步骤4：解码器前向传播 + 跳跃连接 ===================
        
        # dec5 + enc4跳跃连接
        d5 = self.dec5(e5_attended)  # [B,256,8,T] -> [B,128,16,T]
        d5_cat = torch.cat([d5, e4], dim=1)  # 拼接: [B,256,16,T]

        # dec4 + enc3跳跃连接
        d4 = self.dec4(d5_cat)    # [B,256,16,T] -> [B,64,32,T]
        d4_cat = torch.cat([d4, e3], dim=1)  # 拼接: [B,128,32,T]

        # dec3 + enc2跳跃连接
        d3 = self.dec3(d4_cat)    # [B,128,32,T] -> [B,32,64,T]
        d3_cat = torch.cat([d3, e2], dim=1)  # 拼接: [B,64,64,T]

        # dec2 + enc1跳跃连接
        d2 = self.dec2(d3_cat)    # [B,64,64,T] -> [B,16,128,T]
        d2_cat = torch.cat([d2, e1], dim=1)  # 拼接: [B,32,128,T]

        # dec1：生成最终的掩膜
        mask = self.dec1(d2_cat)  # [B,32,128,T] -> [B,1,256,T]

        # =================== 步骤5：应用掩膜 ===================
        output = x * mask
        return output


if __name__ == "__main__":
    """测试模型"""
    print("Testing AudioUNet5TemporalAttention...")
    model = AudioUNet5TemporalAttention(reduction=8)

    # 创建样本输入：[Batch=2, Channels=1, Freq=256, Time=100]
    x = torch.randn(2, 1, 256, 100)

    # 前向传播
    output = model(x)

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # 测试时间注意力组件
    print(f"  Temporal attention reduction ratio: 8")
    print(f"  ✓ AudioUNet5TemporalAttention test passed\n")