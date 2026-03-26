"""
============================================================
U-Net v6 模型 - 5层U-Net 综合优化版（注意力+残差+空洞）
============================================================

【文件功能】
这个文件实现了最强版本的5层U-Net模型，结合了前三个版本的所有改进：
注意力门机制 + 残差连接 + 空洞卷积，用于音频啸叫抑制任务。

【主要组件】
- AudioUNet5Optimized 类：综合优化的5层U-Net模型
  - 编码器：5层，每层包含下采样 + 残差块，瓶颈层使用空洞卷积
  - 解码器：5层转置卷积上采样，重建频谱
  - 注意力门：4个注意力模块，智能选择跳跃连接特征
  - 空洞卷积：在瓶颈层使用多膨胀率卷积，扩大感受野

【网络架构】
编码器（下采样 + 残差块）：
  输入: [B, 1, 256, T]
    ↓ 下采样 + 残差块
  enc1: [B, 1, 256, T] → [B, 16, 128, T]
    ↓ 下采样 + 残差块
  enc2: [B, 16, 128, T] → [B, 32, 64, T]
    ↓ 下采样 + 残差块
  enc3: [B, 32, 64, T] → [B, 64, 32, T]
    ↓ 下采样 + 残差块
  enc4: [B, 64, 32, T] → [B, 128, 16, T]
    ↓ 下采样 + 残差块 + 空洞卷积
  enc5: [B, 128, 16, T] → [B, 256, 8, T] (瓶颈层，多尺度特征)

解码器（上采样 + 注意力门）：
  enc5: [B, 256, 8, T]
    ↓ dec5上采样
  dec5: [B, 256, 8, T] → [B, 128, 16, T]
    ↓ + att4注意力门
  e4_att = att4(dec5, enc4)  # 智能选择enc4特征
    ↓ 拼接
  [B, 256, 16, T]
    ↓ dec4上采样
  dec4: [B, 256, 16, T] → [B, 64, 32, T]
    ↓ + att3注意力门
  e3_att = att3(dec4, enc3)
    ↓ 拼接
  [B, 128, 32, T]
    ↓ dec3上采样
  dec3: [B, 128, 32, T] → [B, 32, 64, T]
    ↓ + att2注意力门
  e2_att = att2(dec3, enc2)
    ↓ 拼接
  [B, 64, 64, T]
    ↓ dec2上采样
  dec2: [B, 64, 64, T] → [B, 16, 128, T]
    ↓ + att1注意力门
  e1_att = att1(dec2, enc1)
    ↓ 拼接
  [B, 32, 128, T]
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
- res1-res5: 5个残差块，缓解梯度消失

空洞卷积参数：
- dilation_rates: [2, 4, 8]（膨胀率列表）
- 作用：扩大感受野，捕捉长距离时序依赖

注意力门参数：
- att1-att4: 4个注意力门，智能选择特征
- 作用：自动聚焦于啸叫相关频段

【三大改进协同作用】
1. 残差连接：让深层网络训练更稳定
2. 注意力门：让模型聚焦于重要特征
3. 空洞卷积：让模型看到更远的时序信息
4. 综合效果：1+1+1 > 3，全面性能提升

【模型特点】
✓ 三重优化：注意力 + 残差 + 空洞卷积
✓ 自动聚焦：智能选择重要特征
✓ 训练稳定：残差连接缓解梯度消失
✓ 长距离依赖：空洞卷积扩大感受野
✓ 最佳性能：结合所有改进的优势

【与其他版本区别】
- v2：标准U-Net，基线模型
- v3：+ 注意力门
- v4：+ 残差连接
- v5：+ 空洞卷积
- v6（本模型）：= v3 + v4 + v5，全面优化

【使用示例】
```python
from src.models.unet_v6_optimized import AudioUNet5Optimized
import torch

# 创建模型（综合优化版）
model = AudioUNet5Optimized(dilation_rates=[2, 4, 8])

# 准备输入
input_spec = torch.randn(4, 1, 256, 376)

# 前向传播
output_spec = model(input_spec)

# 三重改进带来最佳性能
```
"""

import torch
import torch.nn as nn
from .attention_modules import AttentionBlock, ResidualBlock, AtrousConvBlock


class AudioUNet5Optimized(nn.Module):
    """综合优化的5层U-Net模型用于音频啸叫抑制

    这是最强版本的U-Net，结合了三大改进：
    1. 注意力门：智能选择跳跃连接的特征
    2. 残差连接：让深层网络训练更稳定
    3. 空洞卷积：扩大感受野，捕捉长距离依赖

    【工作原理】
    1. 编码器：每层下采样 + 残差块，瓶颈层使用空洞卷积
    2. 注意力门：在解码器时智能选择编码器特征
    3. 解码器：逐层上采样重建频谱

    【三大改进协同作用】
    - 残差连接提供"训练捷径"，梯度更容易传播
    - 注意力门提供"智能过滤"，只让重要特征通过
    - 空洞卷积提供"望远镜"，看到更远的时序信息
    - 三者结合，全面性能提升

    【输入输出】
    输入: [batch, 1, 256, time] - 含啸叫的幅度谱
    输出: [batch, 1, 256, time] - 抑制啸叫后的幅度谱

    【网络层】
    enc1_down-enc5_down: 5个下采样层
    res1-res5: 5个残差块
    atrous_block: 空洞卷积块
    dec1-dec5: 5个上采样层
    att1-att4: 4个注意力门
    """

    def __init__(self, dilation_rates: list = [2, 4, 8]):
        """初始化综合优化的5层U-Net模型

        Args:
            dilation_rates: 空洞卷积的膨胀率列表，默认为 [2, 4, 8]
        """
        super(AudioUNet5Optimized, self).__init__()

        # =================== 编码器部分（5层，下采样 + 残差块）===================
        # 编码器的作用：提取特征，同时降低分辨率（下采样）

        # 编码器第1层
        self.enc1_down = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.res1 = ResidualBlock(channels=16)

        # 编码器第2层
        self.enc2_down = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.res2 = ResidualBlock(channels=32)

        # 编码器第3层
        self.enc3_down = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.res3 = ResidualBlock(channels=64)

        # 编码器第4层
        self.enc4_down = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.res4 = ResidualBlock(channels=128)

        # 编码器第5层（瓶颈层）
        self.enc5_down = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.res5 = ResidualBlock(channels=256)

        # 空洞卷积块：在瓶颈层扩大感受野
        self.atrous_block = AtrousConvBlock(
            in_channels=256,
            out_channels=256,
            dilation_rates=dilation_rates,
            kernel_size=3
        )

        # =================== 解码器部分（5层）===================
        # 解码器的作用：恢复分辨率，重建频谱（上采样）

        # 解码器第5层
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(
                256, 128, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # 解码器第4层
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(
                256, 64, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # 解码器第3层
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(
                128, 32, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # 解码器第2层
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(
                64, 16, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        # 解码器第1层（输出层）
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(
                32, 1, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)
            ),
            nn.Sigmoid(),
        )

        # =================== 注意力门部分（4个）===================
        # 注意力门的作用：智能选择编码器特征中的重要部分

        # 注意力门4：用于第4层跳跃连接
        self.att4 = AttentionBlock(F_g=128, F_l=128, F_int=64)

        # 注意力门3：用于第3层跳跃连接
        self.att3 = AttentionBlock(F_g=64, F_l=64, F_int=32)

        # 注意力门2：用于第2层跳跃连接
        self.att2 = AttentionBlock(F_g=32, F_l=32, F_int=16)

        # 注意力门1：用于第1层跳跃连接
        self.att1 = AttentionBlock(F_g=16, F_l=16, F_int=8)

    def forward(self, x):
        """前向传播函数

        Args:
            x: 输入频谱，格式为 [batch, 1, 256, time]

        Returns:
            output: 处理后的频谱，格式为 [batch, 1, 256, time]
        """
        # =================== 步骤1：输入预处理 ===================
        x_log = torch.log10(x + 1e-8)

        # =================== 步骤2：编码器前向传播（下采样 + 残差块）===================
        e1_down = self.enc1_down(x_log)
        e1 = self.res1(e1_down)

        e2_down = self.enc2_down(e1)
        e2 = self.res2(e2_down)

        e3_down = self.enc3_down(e2)
        e3 = self.res3(e3_down)

        e4_down = self.enc4_down(e3)
        e4 = self.res4(e4_down)

        # 瓶颈层：下采样 + 残差块 + 空洞卷积
        e5_down = self.enc5_down(e4)
        e5_res = self.res5(e5_down)
        e5 = self.atrous_block(e5_res)  # 多尺度特征

        # =================== 步骤3：解码器前向传播 + 注意力门 ===================
        # 每一层解码器都通过注意力门智能选择编码器特征

        # dec5 + att4注意力门
        d5 = self.dec5(e5)
        e4_att = self.att4(d5, e4)  # 智能选择e4特征
        d5_cat = torch.cat([d5, e4_att], dim=1)

        # dec4 + att3注意力门
        d4 = self.dec4(d5_cat)
        e3_att = self.att3(d4, e3)
        d4_cat = torch.cat([d4, e3_att], dim=1)

        # dec3 + att2注意力门
        d3 = self.dec3(d4_cat)
        e2_att = self.att2(d3, e2)
        d3_cat = torch.cat([d3, e2_att], dim=1)

        # dec2 + att1注意力门
        d2 = self.dec2(d3_cat)
        e1_att = self.att1(d2, e1)
        d2_cat = torch.cat([d2, e1_att], dim=1)

        # dec1：生成最终的掩膜
        mask = self.dec1(d2_cat)

        # =================== 步骤4：应用掩膜 ===================
        output = x * mask

        return output


if __name__ == "__main__":
    """测试模型"""
    print("正在测试 AudioUNet5Optimized...")
    model = AudioUNet5Optimized(dilation_rates=[2, 4, 8])

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
    print(f"  ✓ AudioUNet5Optimized 测试通过\n")
