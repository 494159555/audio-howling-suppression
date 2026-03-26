"""
============================================================
U-Net v10 模型 - 5层U-Net + GAN生成对抗网络
============================================================

【文件功能】
这个文件实现了一个基于GAN（生成对抗网络）框架的音频啸叫抑制模型。
生成器是5层U-Net，判别器是CNN网络，通过对抗训练提高生成质量。

【主要组件】
- AudioUNet5GAN 类：GAN框架模型
  - 生成器：5层U-Net，生成抑制啸叫后的频谱
  - 判别器：CNN网络，区分真实/生成频谱
  - 对抗训练：联合重建损失和对抗损失

【网络架构】
生成器（U-Net）：
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
    enc5: [B, 128, 16, T] → [B, 256, 8, T]

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

判别器（CNN）：
  输入: [B, 1, 256, T]
    ↓
  conv1: [B, 1, 256, T] → [B, 64, 128, T]
    ↓
  conv2: [B, 64, 128, T] → [B, 128, 64, T]
    ↓
  conv3: [B, 128, 64, T] → [B, 256, 32, T]
    ↓
  输出: [B, 1] (真实/生成概率)

【关键参数说明】
- 生成器：5层U-Net，重建损失 + 对抗损失
- 判别器：3层CNN，二分类（真实/生成）
- 卷积核大小：生成器3×3，判别器4×4
- 激活函数：LeakyReLU(0.2)，输出Sigmoid
- 损失函数：L1/L2 + 对抗损失

【数据处理流程】
1. 生成器：输入含啸叫频谱 → 输出干净频谱
2. 判别器：区分真实干净频谱和生成频谱
3. 对抗训练：生成器欺骗判别器，判别器识别生成频谱
4. 联合损失：重建损失 + 对抗损失

【模型特点】
✓ GAN框架：通过对抗训练提高生成质量
✓ 感知质量：生成更自然的音频
✓ 联合损失：重建损失 + 对抗损失
✓ 深层架构：5层U-Net生成器
✓ 稳定训练：使用批归一化

【与其他版本区别】
- v2：标准5层U-Net，仅重建损失
- v10（本模型）：添加GAN框架，对抗训练提高质量

【使用示例】
```python
from src.models.unet_v10_gan import AudioUNet5GAN
import torch

# 创建模型
model = AudioUNet5GAN()

# 准备输入
noisy_spec = torch.randn(4, 1, 256, 376).abs()
clean_spec = torch.randn(4, 1, 256, 376).abs()

# 生成器前向传播
pred_spec = model.generator(noisy_spec)  # 输出: [4, 1, 256, 376]

# 判别器前向传播
real_score = model.discriminator(clean_spec)  # 真实频谱分数
fake_score = model.discriminator(pred_spec)   # 生成频谱分数
```
"""

import torch
import torch.nn as nn
from .loss_functions import AdversarialLoss


class AudioUNet5GAN(nn.Module):
    """5层U-Net + GAN框架用于音频啸叫抑制

    这个模型实现了GAN框架，其中：
    - 生成器：5层U-Net，生成抑制啸叫后的频谱
    - 判别器：CNN网络，分类真实/生成频谱

    生成器使用联合重建损失和对抗损失训练，
    判别器训练以区分真实（干净）和生成（伪造）频谱。

    【工作原理】
    1. 生成器：将含啸叫频谱映射到干净频谱
    2. 判别器：判断频谱是真实的还是生成的
    3. 对抗训练：
       - 生成器试图欺骗判别器
       - 判别器试图正确识别生成频谱
    4. 联合优化：重建损失 + 对抗损失

    【输入输出】
    生成器输入: [batch, 1, 256, time] - 含啸叫的幅度谱
    生成器输出: [batch, 1, 256, time] - 生成的干净幅度谱
    判别器输入: [batch, 1, 256, time] - 频谱
    判别器输出: [batch, 1] - 真实概率

    【网络层】
    generator: U-Net生成器
    discriminator: CNN判别器
    """
    
    def __init__(self):
        """初始化GAN模型"""
        super(AudioUNet5GAN, self).__init__()

        # =================== 生成器（U-Net）===================
        self.generator = self._build_generator()

        # =================== 判别器（CNN）===================
        self.discriminator = self._build_discriminator()

    def _build_generator(self):
        """构建U-Net生成器"""
        class Generator(nn.Module):
            """U-Net生成器"""
            def __init__(self):
                super(Generator, self).__init__()

                # 编码器（下采样）- 5层
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

                # 解码器（上采样）- 5层
                self.dec5 = nn.Sequential(
                    nn.ConvTranspose2d(
                        256, 128, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)
                    ),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                )

                self.dec4 = nn.Sequential(
                    nn.ConvTranspose2d(
                        256, 64, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)
                    ),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                )

                self.dec3 = nn.Sequential(
                    nn.ConvTranspose2d(
                        128, 32, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)
                    ),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                )

                self.dec2 = nn.Sequential(
                    nn.ConvTranspose2d(
                        64, 16, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)
                    ),
                    nn.BatchNorm2d(16),
                    nn.ReLU(inplace=True),
                )

                self.dec1 = nn.Sequential(
                    nn.ConvTranspose2d(
                        32, 1, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)
                    ),
                    nn.Sigmoid(),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """生成器前向传播"""
                x_log = torch.log10(x + 1e-8)

                # 编码器
                e1 = self.enc1(x_log)    # [B, 16, 128, T]
                e2 = self.enc2(e1)        # [B, 32, 64, T]
                e3 = self.enc3(e2)        # [B, 64, 32, T]
                e4 = self.enc4(e3)        # [B, 128, 16, T]
                e5 = self.enc5(e4)        # [B, 256, 8, T]

                # 解码器 + 跳跃连接
                d5 = self.dec5(e5)        # [B, 128, 16, T]
                d5_cat = torch.cat([d5, e4], dim=1)

                d4 = self.dec4(d5_cat)    # [B, 64, 32, T]
                d4_cat = torch.cat([d4, e3], dim=1)

                d3 = self.dec3(d4_cat)    # [B, 32, 64, T]
                d3_cat = torch.cat([d3, e2], dim=1)

                d2 = self.dec2(d3_cat)    # [B, 16, 128, T]
                d2_cat = torch.cat([d2, e1], dim=1)

                mask = self.dec1(d2_cat)    # [B, 1, 256, T]
                output = x * mask
                return output

        return Generator()

    def _build_discriminator(self):
        """构建CNN判别器"""
        class Discriminator(nn.Module):
            """CNN判别器"""
            def __init__(self):
                super(Discriminator, self).__init__()

                # 卷积层
                self.conv_layers = nn.Sequential(
                    # 第1层
                    nn.Conv2d(1, 64, kernel_size=4, stride=(2, 1), padding=1),
                    nn.BatchNorm2d(64),
                    nn.LeakyReLU(0.2, inplace=True),

                    # 第2层
                    nn.Conv2d(64, 128, kernel_size=4, stride=(2, 1), padding=1),
                    nn.BatchNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True),

                    # 第3层
                    nn.Conv2d(128, 256, kernel_size=4, stride=(2, 1), padding=1),
                    nn.BatchNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True),
                )

                # 最终分类层
                self.final_layer = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(256, 1),
                    nn.Sigmoid()
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """判别器前向传播"""
                features = self.conv_layers(x)
                output = self.final_layer(features)
                return output

        return Discriminator()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """仅生成器的前向传播

        Args:
            x: 输入含噪频谱 [B, 1, 256, T]

        Returns:
            生成的干净频谱 [B, 1, 256, T]
        """
        return self.generator(x)


if __name__ == "__main__":
    """测试模型"""
    print("Testing AudioUNet5GAN...")
    model = AudioUNet5GAN()

    # 创建样本输入
    noisy_spec = torch.randn(2, 1, 256, 100).abs()
    clean_spec = torch.randn(2, 1, 256, 100).abs()

    # 生成器前向传播
    pred_spec = model.generator(noisy_spec)

    # 判别器前向传播
    real_score = model.discriminator(clean_spec)
    fake_score = model.discriminator(pred_spec)

    print(f"  Input noisy shape: {noisy_spec.shape}")
    print(f"  Generated clean shape: {pred_spec.shape}")
    print(f"  Real score shape: {real_score.shape}")
    print(f"  Fake score shape: {fake_score.shape}")
    print(f"  Real score mean: {real_score.mean().item():.4f}")
    print(f"  Fake score mean: {fake_score.mean().item():.4f}")

    # 统计参数
    gen_params = sum(p.numel() for p in model.generator.parameters())
    disc_params = sum(p.numel() for p in model.discriminator.parameters())
    total_params = gen_params + disc_params
    print(f"  Generator parameters: {gen_params:,}")
    print(f"  Discriminator parameters: {disc_params:,}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  ✓ AudioUNet5GAN test passed\n")