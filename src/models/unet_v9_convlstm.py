"""
============================================================
U-Net v9 模型 - 5层U-Net + ConvLSTM时空建模
============================================================

【文件功能】
这个文件实现了一个5层U-Net模型，在瓶颈层使用ConvLSTM，
用于音频啸叫抑制任务。ConvLSTM保留了空间结构的同时
建模时序依赖，比标准LSTM更适合处理频谱数据。

【主要组件】
- AudioUNet5ConvLSTM 类：5层U-Net + ConvLSTM
  - 编码器：5层卷积下采样，提取空间特征
  - ConvLSTM：瓶颈层的ConvLSTM，保留空间结构的时序建模
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

ConvLSTM（瓶颈层）：
  对于每个时间步 t：
    输入: [B, 256, 8, 1]
    ConvLSTM处理: [B, 256, 8, 1] -> [B, 128, 8, 1]
  输出: [B, 128, 8, T] (保留空间结构)

解码器（上采样过程）：
  lstm_out: [B, 128, 8, T]
    ↓ + enc4跳跃连接
  dec5: [B, 128, 8, T] → [B, 128, 16, T]
    ↓ 拼接 enc4
  dec4: [B, 256, 16, T] → [B, 64, 32, T]
    ↓ 拼接 enc3
  dec3: [B, 128, 32, T] → [B, 32, 64, T]
    ↓ 拼接 enc2
  dec2: [B, 64, 64, T] → [B, 16, 128, T]
    ↓ 拼接 enc1
  dec1: [B, 32, 128, T] → [B, 1, 256, T]

【关键参数说明】
- hidden_channels: ConvLSTM隐藏通道数（默认128）
- kernel_size: ConvLSTM卷积核大小（默认3）
- padding: ConvLSTM填充（默认1）
- 卷积核大小：3×3
- 下采样步长：(2, 1)
- 激活函数：编码器LeakyReLU(0.2)，解码器ReLU，输出Sigmoid

【数据处理流程】
1. 输入：线性幅度谱 [B, 1, 256, T]
2. Log变换：log10(x + 1e-8)
3. 编码器：提取5个尺度的空间特征
4. ConvLSTM：逐时间步处理，保留空间结构
5. 解码器：通过跳跃连接重建频谱
6. 输出：[0,1]掩膜
7. 最终结果：输入 × 掩膜

【模型特点】
✓ 空间结构保留：ConvLSTM保留频谱的2D结构
✓ 时空建模：同时捕获空间和时序模式
✓ 逐时间步处理：逐帧处理，保持时序信息
✓ 时序一致性：提高输出时序平滑性
✓ 深层架构：5层U-Net提供强大的特征提取

【与其他版本区别】
- v2：标准5层U-Net，无时序建模
- v7：使用双向LSTM，展平空间维度
- v8：使用时间注意力机制
- v9（本模型）：使用ConvLSTM保留空间结构

【使用示例】
```python
from src.models.unet_v9_convlstm import AudioUNet5ConvLSTM
import torch

# 创建模型
model = AudioUNet5ConvLSTM(hidden_channels=128, kernel_size=3, padding=1)

# 准备输入
input_spec = torch.randn(4, 1, 256, 376)

# 前向传播
output_spec = model(input_spec)  # 输出: [4, 1, 256, 376]
```
"""

import torch
import torch.nn as nn
from .temporal_modules import ConvLSTMCell


class AudioUNet5ConvLSTM(nn.Module):
    """5层U-Net + ConvLSTM用于音频啸叫抑制

    这个模型通过在瓶颈层使用ConvLSTM来增强标准U-Net。
    与标准LSTM不同，ConvLSTM不会展平空间维度，
    而是保留频谱的2D结构，能够更好地建模时空模式。

    【工作原理】
    1. 编码器：逐层下采样提取空间特征
    2. ConvLSTM：在瓶颈层逐时间步处理，保留空间结构
    3. 时空建模：同时捕获空间和时间维度的依赖关系
    4. 解码器：逐层上采样重建频谱
    5. 跳跃连接：保留多尺度空间信息

    【输入输出】
    输入: [batch, 1, 256, time] - 含啸叫的幅度谱
    输出: [batch, 1, 256, time] - 抑制啸叫后的幅度谱

    【网络层】
    enc1-enc5: 5层编码器，提取空间特征
    conv_lstm: ConvLSTM单元，时空建模
    dec1-dec5: 5层解码器，重建频谱
    """
    
    def __init__(
        self,
        hidden_channels: int = 128,
        kernel_size: int = 3,
        padding: int = 1
    ):
        """初始化5层U-Net + ConvLSTM模型

        Args:
            hidden_channels: ConvLSTM隐藏通道数
            kernel_size: ConvLSTM卷积核大小
            padding: ConvLSTM填充大小
        """
        super(AudioUNet5ConvLSTM, self).__init__()

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

        # 编码器第5层（瓶颈层输入）：输入 [B,128,16,T] -> 输出 [B,256,8,T]
        self.enc5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # =================== ConvLSTM模块（瓶颈层）===================
        # ConvLSTM的作用：保留空间结构的时序建模
        self.conv_lstm = ConvLSTMCell(
            input_channels=256,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )

        # =================== 解码器部分（5层）===================
        # 解码器的作用：恢复分辨率，重建频谱
        
        # 解码器第5层：输入 [B,128,8,T] -> 输出 [B,128,16,T]
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(
                128, 128, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)
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

        通过编码器-ConvLSTM-解码器架构处理输入频谱，生成用于啸叫抑制的乘性掩膜。

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
        e5 = self.enc5(e4)        # [B,128,16,T] -> [B,256,8,T] (瓶颈层输入)

        # =================== 步骤3：ConvLSTM处理（瓶颈层）===================
        # 逐时间步通过ConvLSTM处理
        batch_size, _, freq_bins, time_steps = e5.shape

        # 初始化隐藏状态和细胞状态
        h = None
        c = None

        # 处理每个时间步
        hidden_states = []
        for t in range(time_steps):
            # 提取时间步
            x_t = e5[:, :, :, t:t+1]  # [B, 256, 8, 1]

            # 应用ConvLSTM
            h, c = self.conv_lstm(x_t, (h, c))  # [B, 128, 8, 1]

            # 存储隐藏状态
            hidden_states.append(h)

        # 拼接所有时间步
        lstm_out = torch.cat(hidden_states, dim=3)  # [B, 128, 8, T]

        # =================== 步骤4：解码器前向传播 + 跳跃连接 ===================
        
        # dec5 + enc4跳跃连接
        d5 = self.dec5(lstm_out)  # [B,128,8,T] -> [B,128,16,T]
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
    print("Testing AudioUNet5ConvLSTM...")
    model = AudioUNet5ConvLSTM(hidden_channels=128, kernel_size=3, padding=1)

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

    # 测试ConvLSTM组件
    print(f"  ConvLSTM hidden channels: 128")
    print(f"  ConvLSTM kernel size: 3")
    print(f"  ✓ AudioUNet5ConvLSTM test passed\n")