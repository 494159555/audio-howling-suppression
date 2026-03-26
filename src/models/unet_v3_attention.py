"""
============================================================
U-Net v3 模型 - 5层U-Net + 注意力门机制
============================================================

【文件功能】
这个文件实现了一个带注意力机制的5层U-Net模型，用于音频啸叫抑制任务。
通过注意力门机制，模型可以自动聚焦于与啸叫相关的特征，提高抑制精度。

【主要组件】
- AudioUNet5Attention 类：带注意力门的5层U-Net模型
  - 编码器：5层卷积下采样，提取特征
  - 解码器：5层转置卷积上采样，重建频谱
  - 注意力门：4个注意力模块，动态选择跳跃连接的特征

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

解码器（上采样过程 + 注意力门）：
  enc5: [B, 256, 8, T]
    ↓ dec5上采样
  dec5: [B, 256, 8, T] → [B, 128, 16, T]
    ↓ + att4注意力门
  e4_att = att4(dec5, enc4)  # 对enc4特征加权
    ↓ 拼接
  [B, 128+128, 16, T]
    ↓ dec4上采样
  dec4: [B, 256, 16, T] → [B, 64, 32, T]
    ↓ + att3注意力门
  e3_att = att3(dec4, enc3)  # 对enc3特征加权
    ↓ 拼接
  [B, 64+64, 32, T]
    ↓ dec3上采样
  dec3: [B, 128, 32, T] → [B, 32, 64, T]
    ↓ + att2注意力门
  e2_att = att2(dec3, enc2)  # 对enc2特征加权
    ↓ 拼接
  [B, 32+32, 64, T]
    ↓ dec2上采样
  dec2: [B, 64, 64, T] → [B, 16, 128, T]
    ↓ + att1注意力门
  e1_att = att1(dec2, enc1)  # 对enc1特征加权
    ↓ 拼接
  [B, 16+16, 128, T]
    ↓ dec1上采样
  dec1: [B, 32, 128, T] → [B, 1, 256, T] (输出掩膜)

【关键参数说明】
网络层参数：
- 卷积核大小：3×3（局部特征提取）
- 下采样步长：(2, 1)（频率方向下采样，时间方向保持）
- 激活函数：
  * 编码器：LeakyReLU(0.2, inplace=True)
  * 解码器：ReLU(inplace=True)
  * 输出层：Sigmoid（生成[0,1]掩膜）

注意力门参数：
- att4: F_g=128, F_l=128, F_int=64（第4层注意力）
- att3: F_g=64, F_l=64, F_int=32（第3层注意力）
- att2: F_g=32, F_l=32, F_int=16（第2层注意力）
- att1: F_g=16, F_l=16, F_int=8（第1层注意力）
  * F_g: 解码器特征维度（门控信号）
  * F_l: 编码器特征维度（需要加权的特征）
  * F_int: 中间层维度（计算注意力权重）

【数据处理流程】
1. 输入：线性幅度谱 [B, 1, 256, T]
2. Log变换：log10(x + 1e-8)
3. 编码器：提取5个尺度的特征
4. 解码器：通过注意力门选择性融合编码器特征
5. 输出：[0,1]掩膜
6. 最终结果：输入 × 掩膜

【注意力门工作原理】
1. 输入：解码器特征（门控信号）+ 编码器特征（待加权特征）
2. 计算：通过1×1卷积和激活函数计算注意力权重
3. 输出：加权后的编码器特征（值范围0-1）
4. 作用：让模型学会"关注"重要特征，"忽略"无关特征

【模型特点】
✓ 注意力机制：4个注意力门，动态选择特征
✓ 自动聚焦：自动识别并聚焦于啸叫频段
✓ 精准抑制：减少对正常音频的误伤
✓ 深层架构：5层编码器和5层解码器
✓ Log域处理：提高数值稳定性

【与v2版本区别】
- v2（标准U-Net）：简单拼接编码器和解码器特征
- v3（本模型）：使用注意力门动态选择编码器特征
- 优势：更智能的特征融合，更好的抑制效果

【使用示例】
```python
from src.models.unet_v3_attention import AudioUNet5Attention
import torch

# 创建模型
model = AudioUNet5Attention()

# 准备输入
input_spec = torch.randn(4, 1, 256, 376)

# 前向传播
output_spec = model(input_spec)

# 注意力门会自动聚焦于啸叫相关频段
```
"""

import torch
import torch.nn as nn
from .attention_modules import AttentionBlock


class AudioUNet5Attention(nn.Module):
    """带注意力门的5层U-Net模型用于音频啸叫抑制

    这个模型在标准U-Net的基础上，在所有跳跃连接上添加了注意力门。
    注意力门让模型能够自动聚焦于与啸叫相关的特征，提高抑制精度。

    【工作原理】
    1. 编码器：逐层下采样提取特征，共5层
    2. 注意力门：在解码器时，对编码器特征进行加权筛选
    3. 解码器：逐层上采样重建频谱，共5层
    4. 智能融合：不是简单拼接，而是通过注意力权重选择特征

    【注意力门的作用】
    想象注意力门就像一个"智能过滤器"：
    - 计算每个编码器特征的重要性（0-1的权重）
    - 重要的特征（如啸叫频段）权重接近1
    - 不重要的特征（如背景噪声）权重接近0
    - 只让重要特征通过，提高抑制精度

    【输入输出】
    输入: [batch, 1, 256, time] - 含啸叫的幅度谱
    输出: [batch, 1, 256, time] - 抑制啸叫后的幅度谱

    【网络层】
    enc1-enc5: 5层编码器
    dec1-de5: 5层解码器
    att1-att4: 4个注意力门（分别对应4个跳跃连接）
    """

    def __init__(self):
        """初始化带注意力门的5层U-Net模型"""
        super(AudioUNet5Attention, self).__init__()

        # =================== 编码器部分（5层）===================
        # 编码器的作用：提取特征，同时降低分辨率（下采样）

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

        # =================== 解码器部分（5层）===================
        # 解码器的作用：恢复分辨率，重建频谱（上采样）

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
            nn.Sigmoid(),  # 生成[0,1]范围的掩膜
        )

        # =================== 注意力门部分（4个）===================
        # 注意力门的作用：智能选择编码器特征中的重要部分

        # 注意力门4：用于第4层跳跃连接（dec5 + enc4）
        # 参数说明：
        # - F_g=128: 解码器第5层输出的通道数（门控信号）
        # - F_l=128: 编码器第4层输出的通道数（需要加权的特征）
        # - F_int=64: 中间层通道数（用于计算注意力权重）
        self.att4 = AttentionBlock(F_g=128, F_l=128, F_int=64)

        # 注意力门3：用于第3层跳跃连接（dec4 + enc3）
        self.att3 = AttentionBlock(F_g=64, F_l=64, F_int=32)

        # 注意力门2：用于第2层跳跃连接（dec3 + enc2）
        self.att2 = AttentionBlock(F_g=32, F_l=32, F_int=16)

        # 注意力门1：用于第1层跳跃连接（dec2 + enc1）
        self.att1 = AttentionBlock(F_g=16, F_l=16, F_int=8)

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
        # 逐层下采样，提取5个尺度的特征
        e1 = self.enc1(x_log)  # [B,1,256,T] -> [B,16,128,T]
        e2 = self.enc2(e1)     # [B,16,128,T] -> [B,32,64,T]
        e3 = self.enc3(e2)     # [B,32,64,T] -> [B,64,32,T]
        e4 = self.enc4(e3)     # [B,64,32,T] -> [B,128,16,T]
        e5 = self.enc5(e4)     # [B,128,16,T] -> [B,256,8,T] (瓶颈层)

        # =================== 步骤3：解码器前向传播 + 注意力门 ===================
        # 每一层解码器都会通过注意力门智能选择对应的编码器特征

        # dec5 + att4注意力门
        d5 = self.dec5(e5)                      # [B,256,8,T] -> [B,128,16,T]
        e4_att = self.att4(d5, e4)              # 对e4特征进行加权筛选
        # e4_att是加权后的e4特征，重要特征权重接近1，不重要特征权重接近0
        d5_cat = torch.cat([d5, e4_att], dim=1) # 拼接: [B,128+128,16,T] = [B,256,16,T]

        # dec4 + att3注意力门
        d4 = self.dec4(d5_cat)                  # [B,256,16,T] -> [B,64,32,T]
        e3_att = self.att3(d4, e3)              # 对e3特征进行加权筛选
        d4_cat = torch.cat([d4, e3_att], dim=1) # 拼接: [B,64+64,32,T] = [B,128,32,T]

        # dec3 + att2注意力门
        d3 = self.dec3(d4_cat)                  # [B,128,32,T] -> [B,32,64,T]
        e2_att = self.att2(d3, e2)              # 对e2特征进行加权筛选
        d3_cat = torch.cat([d3, e2_att], dim=1) # 拼接: [B,32+32,64,T] = [B,64,64,T]

        # dec2 + att1注意力门
        d2 = self.dec2(d3_cat)                  # [B,64,64,T] -> [B,16,128,T]
        e1_att = self.att1(d2, e1)              # 对e1特征进行加权筛选
        d2_cat = torch.cat([d2, e1_att], dim=1) # 拼接: [B,16+16,128,T] = [B,32,128,T]

        # dec1：生成最终的掩膜
        mask = self.dec1(d2_cat)                # [B,32,128,T] -> [B,1,256,T]

        # =================== 步骤4：应用掩膜 ===================
        # 将预测的掩膜与原始输入相乘
        output = x * mask

        return output


if __name__ == "__main__":
    """测试模型"""
    print("正在测试 AudioUNet5Attention...")
    model = AudioUNet5Attention()

    # 创建测试输入
    x = torch.randn(2, 1, 256, 100)

    # 前向传播
    output = model(x)

    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {output.shape}")

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")
    print(f"  ✓ AudioUNet5Attention 测试通过\n")
