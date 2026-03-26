"""
============================================================
U-Net v4 模型 - 5层U-Net + 残差连接
============================================================

【文件功能】
这个文件实现了一个带残差连接的5层U-Net模型，用于音频啸叫抑制任务。
通过残差块，模型能够更深层地训练，缓解梯度消失问题。

【主要组件】
- AudioUNet5Residual 类：带残差连接的5层U-Net模型
  - 编码器：5层，每层包含下采样 + 残差块
  - 解码器：5层转置卷积上采样，重建频谱
  - 残差块：5个残差模块，帮助深层网络训练

【网络架构】
编码器（下采样 + 残差块）：
  输入: [B, 1, 256, T]
    ↓ 下采样
  enc1_down: [B, 1, 256, T] → [B, 16, 128, T]
    ↓ 残差块
  enc1: [B, 16, 128, T] (残差块处理后的特征)
    ↓ 下采样
  enc2_down: [B, 16, 128, T] → [B, 32, 64, T]
    ↓ 残差块
  enc2: [B, 32, 64, T]
    ↓ 下采样
  enc3_down: [B, 32, 64, T] → [B, 64, 32, T]
    ↓ 残差块
  enc3: [B, 64, 32, T]
    ↓ 下采样
  enc4_down: [B, 64, 32, T] → [B, 128, 16, T]
    ↓ 残差块
  enc4: [B, 128, 16, T]
    ↓ 下采样
  enc5_down: [B, 128, 16, T] → [B, 256, 8, T]
    ↓ 残差块
  enc5: [B, 256, 8, T] (瓶颈层)

解码器（上采样）：
  enc5: [B, 256, 8, T]
    ↓ dec5上采样
  dec5: [B, 256, 8, T] → [B, 128, 16, T]
    ↓ 拼接 enc4
  [B, 128+128, 16, T] = [B, 256, 16, T]
    ↓ dec4上采样
  dec4: [B, 256, 16, T] → [B, 64, 32, T]
    ↓ 拼接 enc3
  [B, 64+64, 32, T] = [B, 128, 32, T]
    ↓ dec3上采样
  dec3: [B, 128, 32, T] → [B, 32, 64, T]
    ↓ 拼接 enc2
  [B, 32+32, 64, T] = [B, 64, 64, T]
    ↓ dec2上采样
  dec2: [B, 64, 64, T] → [B, 16, 128, T]
    ↓ 拼接 enc1
  [B, 16+16, 128, T] = [B, 32, 128, T]
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

残差块参数：
- res1-res5: 分别对应5个编码器层
  * channels: 每个残差块的通道数（16, 32, 64, 128, 256）
  * 作用：在编码器内部添加恒等映射，帮助梯度反向传播

【数据处理流程】
1. 输入：线性幅度谱 [B, 1, 256, T]
2. Log变换：log10(x + 1e-8)
3. 编码器：下采样 + 残差块处理
4. 解码器：上采样 + 跳跃连接
5. 输出：[0,1]掩膜
6. 最终结果：输入 × 掩膜

【残差块工作原理】
残差块允许数据"跳过"某些层，直接向前传递：
1. 输入特征 x
2. 主路径：卷积层 → 激活 → 卷积层 → 输出 F(x)
3. 捷径：直接传递输入 x
4. 输出：F(x) + x（残差连接）
5. 好处：梯度可以更容易地反向传播，缓解梯度消失

【模型特点】
✓ 残差连接：5个残差块，帮助深层网络训练
✓ 缓解梯度消失：让梯度更容易反向传播
✓ 更深网络：可以训练更深的模型
✓ 训练稳定：收敛更稳定，不容易发散
✓ Log域处理：提高数值稳定性

【与v2版本区别】
- v2（标准U-Net）：普通卷积层，可能遇到梯度消失
- v4（本模型）：使用残差块，训练更稳定
- 优势：适合训练非常深的网络

【使用示例】
```python
from src.models.unet_v4_residual import AudioUNet5Residual
import torch

# 创建模型
model = AudioUNet5Residual()

# 准备输入
input_spec = torch.randn(4, 1, 256, 376)

# 前向传播
output_spec = model(input_spec)

# 残差连接让训练更稳定
```
"""

import torch
import torch.nn as nn
from .attention_modules import ResidualBlock


class AudioUNet5Residual(nn.Module):
    """带残差连接的5层U-Net模型用于音频啸叫抑制

    这个模型在标准U-Net的基础上，在所有编码器层添加了残差块。
    残差连接让梯度更容易反向传播，帮助训练更深的网络。

    【工作原理】
    1. 编码器：每层先下采样，然后通过残差块处理
    2. 残差块：在块内部添加恒等映射（输入直接加到输出）
    3. 解码器：逐层上采样重建频谱
    4. 跳跃连接：拼接编码器和解码器特征

    【残差连接的好处】
    想象残差连接就像一条"捷径"：
    - 数据可以直接跳过某些层，不会损失信息
    - 梯度反向传播时也有捷径，不容易消失
    - 让我们可以训练更深的网络，获得更好的性能

    【输入输出】
    输入: [batch, 1, 256, time] - 含啸叫的幅度谱
    输出: [batch, 1, 256, time] - 抑制啸叫后的幅度谱

    【网络层】
    enc1_down-enc5_down: 5个下采样层
    res1-res5: 5个残差块
    dec1-dec5: 5个上采样层（解码器）
    """

    def __init__(self):
        """初始化带残差连接的5层U-Net模型"""
        super(AudioUNet5Residual, self).__init__()

        # =================== 编码器部分（5层，每层包含下采样+残差块）===================
        # 编码器的作用：提取特征，同时降低分辨率（下采样）
        # 每层编码器包含：下采样层 + 残差块

        # 编码器第1层
        # 下采样层：输入 [B,1,256,T] -> 输出 [B,16,128,T]
        self.enc1_down = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # 残差块：输入 [B,16,128,T] -> 输出 [B,16,128,T]
        # 在残差块内部，输入会直接加到输出上：output = F(x) + x
        self.res1 = ResidualBlock(channels=16)

        # 编码器第2层
        # 下采样层：输入 [B,16,128,T] -> 输出 [B,32,64,T]
        self.enc2_down = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # 残差块：32通道
        self.res2 = ResidualBlock(channels=32)

        # 编码器第3层
        # 下采样层：输入 [B,32,64,T] -> 输出 [B,64,32,T]
        self.enc3_down = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # 残差块：64通道
        self.res3 = ResidualBlock(channels=64)

        # 编码器第4层
        # 下采样层：输入 [B,64,32,T] -> 输出 [B,128,16,T]
        self.enc4_down = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # 残差块：128通道
        self.res4 = ResidualBlock(channels=128)

        # 编码器第5层（瓶颈层）
        # 下采样层：输入 [B,128,16,T] -> 输出 [B,256,8,T]
        self.enc5_down = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # 残差块：256通道
        self.res5 = ResidualBlock(channels=256)

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

        # =================== 步骤2：编码器前向传播（下采样 + 残差块）===================
        # 每一层编码器都包含：下采样层 + 残差块

        # 编码器第1层：先下采样，再通过残差块
        e1_down = self.enc1_down(x_log)    # [B,1,256,T] -> [B,16,128,T]
        e1 = self.res1(e1_down)            # [B,16,128,T] -> [B,16,128,T] (残差块处理)
        # 残差块内部：output = F(x) + x，输入直接加到输出上

        # 编码器第2层
        e2_down = self.enc2_down(e1)       # [B,16,128,T] -> [B,32,64,T]
        e2 = self.res2(e2_down)            # [B,32,64,T] -> [B,32,64,T]

        # 编码器第3层
        e3_down = self.enc3_down(e2)       # [B,32,64,T] -> [B,64,32,T]
        e3 = self.res3(e3_down)            # [B,64,32,T] -> [B,64,32,T]

        # 编码器第4层
        e4_down = self.enc4_down(e3)       # [B,64,32,T] -> [B,128,16,T]
        e4 = self.res4(e4_down)            # [B,128,16,T] -> [B,128,16,T]

        # 编码器第5层（瓶颈层）
        e5_down = self.enc5_down(e4)       # [B,128,16,T] -> [B,256,8,T]
        e5 = self.res5(e5_down)            # [B,256,8,T] -> [B,256,8,T] (最深层特征)

        # =================== 步骤3：解码器前向传播 + 跳跃连接 ===================
        # 每一层解码器都会拼接对应编码器层的特征（跳跃连接）

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
        output = x * mask

        return output


if __name__ == "__main__":
    """测试模型"""
    print("正在测试 AudioUNet5Residual...")
    model = AudioUNet5Residual()

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
    print(f"  ✓ AudioUNet5Residual 测试通过\n")
