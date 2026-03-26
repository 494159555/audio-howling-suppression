"""
============================================================
U-Net v5 模型 - 5层U-Net + 空洞卷积（多尺度感受野）
============================================================

【文件功能】
这个文件实现了一个带空洞卷积的5层U-Net模型，用于音频啸叫抑制任务。
通过空洞卷积，模型能够以更少的参数获得更大的感受野，捕捉长距离时序依赖。

【主要组件】
- AudioUNet5Dilated 类：带空洞卷积的5层U-Net模型
  - 编码器：4层普通卷积 + 1层空洞卷积（瓶颈层）
  - 解码器：5层转置卷积上采样，重建频谱
  - 空洞卷积块：在瓶颈层使用多膨胀率卷积，扩大感受野

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
  enc5_down: [B, 128, 16, T] → [B, 256, 8, T] (下采样)
    ↓ 空洞卷积块（3个并行卷积）
  atrous_block:
    - 膨胀率=2: 感受野扩大2倍
    - 膨胀率=4: 感受野扩大4倍
    - 膨胀率=8: 感受野扩大8倍
    ↓ 拼接
  enc5: [B, 256, 8, T] (多尺度特征)

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

空洞卷积参数：
- dilation_rates: [2, 4, 8]（膨胀率列表）
  * 膨胀率=2：卷积核元素之间间隔1个0
  * 膨胀率=4：卷积核元素之间间隔3个0
  * 膨胀率=8：卷积核元素之间间隔7个0
- 作用：在不增加参数的情况下扩大感受野
- 效果：能够看到更大范围的时序信息

【数据处理流程】
1. 输入：线性幅度谱 [B, 1, 256, T]
2. Log变换：log10(x + 1e-8)
3. 编码器：提取特征，瓶颈层使用空洞卷积
4. 解码器：上采样 + 跳跃连接
5. 输出：[0,1]掩膜
6. 最终结果：输入 × 掩膜

【空洞卷积工作原理】
空洞卷积通过在卷积核元素之间插入"空洞"来扩大感受野：
1. 普通卷积：3×3卷积核，感受野3×3
2. 空洞卷积（膨胀率=2）：3×3卷积核，间隔1个0，感受野5×5
3. 空洞卷积（膨胀率=4）：3×3卷积核，间隔3个0，感受野9×9
4. 好处：不用增加参数量就能看到更大范围的信息
5. 应用：在瓶颈层使用多膨胀率卷积，捕捉不同尺度的时序特征

【感受野对比】
- 普通卷积：只能看到相邻的几个时间点
- 空洞卷积：可以看到更远的时间点
- 优势：能够捕捉长距离的时序依赖关系，如持续的啸叫

【模型特点】
✓ 空洞卷积：在瓶颈层使用多膨胀率卷积
✓ 扩大感受野：不增加参数量，看到更大范围
✓ 多尺度特征：不同膨胀率捕捉不同尺度的特征
✓ 长距离依赖：能够捕捉长距离时序关系
✓ 参数高效：用更少的参数获得更大的感受野

【与v2版本区别】
- v2（标准U-Net）：瓶颈层使用普通卷积，感受野有限
- v5（本模型）：瓶颈层使用空洞卷积，感受野更大
- 优势：能够捕捉更长距离的时序依赖，适合处理持续的啸叫

【使用示例】
```python
from src.models.unet_v5_dilated import AudioUNet5Dilated
import torch

# 创建模型（可以自定义膨胀率）
model = AudioUNet5Dilated(dilation_rates=[2, 4, 8])

# 准备输入
input_spec = torch.randn(4, 1, 256, 376)

# 前向传播
output_spec = model(input_spec)

# 空洞卷积让模型能看到更大范围的时序信息
```
"""

import torch
import torch.nn as nn
from .attention_modules import AtrousConvBlock


class AudioUNet5Dilated(nn.Module):
    """带空洞卷积的5层U-Net模型用于音频啸叫抑制

    这个模型在标准U-Net的基础上，在瓶颈层添加了空洞卷积块。
    空洞卷积能够在不增加参数的情况下扩大感受野，捕捉长距离时序依赖。

    【工作原理】
    1. 编码器：前4层正常下采样，第5层使用空洞卷积
    2. 空洞卷积块：使用多个不同膨胀率的卷积并行处理
    3. 多尺度融合：将不同膨胀率的特征拼接，获得多尺度信息
    4. 解码器：逐层上采样重建频谱

    【空洞卷积的好处】
    想象空洞卷积就像"望远镜"：
    - 普通卷积：只能看邻近的地方（视野窄）
    - 空洞卷积：可以看更远的地方（视野宽）
    - 不需要增加参数量就能扩大视野
    - 能够捕捉长距离的时序依赖（如持续的啸叫）

    【输入输出】
    输入: [batch, 1, 256, time] - 含啸叫的幅度谱
    输出: [batch, 1, 256, time] - 抑制啸叫后的幅度谱

    【网络层】
    enc1-enc4: 4个普通编码器层
    enc5_down: 瓶颈层下采样
    atrous_block: 空洞卷积块（多膨胀率并行）
    dec1-dec5: 5个上采样层（解码器）
    """

    def __init__(self, dilation_rates: list = [2, 4, 8]):
        """初始化带空洞卷积的5层U-Net模型

        Args:
            dilation_rates: 空洞卷积的膨胀率列表
                默认为 [2, 4, 8]，表示使用3个不同膨胀率的卷积
                膨胀率越大，感受野越大，能看到越远的时序信息
        """
        super(AudioUNet5Dilated, self).__init__()

        # 保存膨胀率配置
        self.dilation_rates = dilation_rates

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

        # 编码器第5层（瓶颈层下采样）：输入 [B,128,16,T] -> 输出 [B,256,8,T]
        self.enc5_down = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # =================== 空洞卷积块（瓶颈层）===================
        # 空洞卷积的作用：在不增加参数的情况下扩大感受野
        # 使用多个不同膨胀率的卷积并行处理，获得多尺度特征

        self.atrous_block = AtrousConvBlock(
            in_channels=256,          # 输入通道数
            out_channels=256,         # 输出通道数
            dilation_rates=dilation_rates,  # 膨胀率列表，如 [2, 4, 8]
            kernel_size=3             # 卷积核大小
        )
        # 空洞卷积块内部会：
        # 1. 对每个膨胀率创建一个独立的卷积分支
        # 2. 每个分支以不同的"视野"处理输入
        # 3. 将所有分支的输出拼接，获得多尺度特征

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

        # =================== 步骤2：编码器前向传播 ===================
        # 前4层正常下采样
        e1 = self.enc1(x_log)    # [B,1,256,T] -> [B,16,128,T]
        e2 = self.enc2(e1)       # [B,16,128,T] -> [B,32,64,T]
        e3 = self.enc3(e2)       # [B,32,64,T] -> [B,64,32,T]
        e4 = self.enc4(e3)       # [B,64,32,T] -> [B,128,16,T]

        # 第5层（瓶颈层）：先下采样，再通过空洞卷积块
        e5_down = self.enc5_down(e4)    # [B,128,16,T] -> [B,256,8,T]
        e5 = self.atrous_block(e5_down)  # [B,256,8,T] -> [B,256,8,T]
        # 空洞卷积块内部使用多个膨胀率并行处理：
        # - 膨胀率=2：看到较远范围
        # - 膨胀率=4：看到更远范围
        # - 膨胀率=8：看到最远范围
        # 最终将这些不同尺度的特征融合

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
    print("正在测试 AudioUNet5Dilated...")
    model = AudioUNet5Dilated(dilation_rates=[2, 4, 8])

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
    print(f"  空洞卷积膨胀率: {model.dilation_rates}")
    print(f"  ✓ AudioUNet5Dilated 测试通过\n")
