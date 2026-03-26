"""
============================================================
RNN 模型 - 循环神经网络音频啸叫抑制（时序建模）
============================================================

【文件功能】
这个文件实现了一个基于RNN(LSTM)的混合模型，用于音频啸叫抑制任务。
结合CNN的空间特征提取能力和LSTM的时序建模能力。

【主要组件】
- AudioRNN 类：CNN+LSTM混合模型
  - 前置CNN：提取频谱的局部空间特征
  - 双向LSTM：捕捉时间序列的依赖关系
  - 后置CNN：恢复频谱维度并生成掩膜

【网络架构】
前置卷积（特征提取）：
  输入: [B, 1, 256, T]
    ↓
  pre_conv: [B, 1, 256, T] → [B, 64, 256, T]
  使用两个3×3卷积层，保持空间维度不变

LSTM层（时序建模）：
  输入: [B, 64, 256, T]
    ↓ 维度重组
  [B, 64, 256, T] → [B*256, T, 64]
    ↓ 双向LSTM
  [B*256, T, 64] → [B*256, T, 256]  (hidden_size*2 = 128*2)
    ↓ 维度恢复
  [B*256, T, 256] → [B, 256, 256, T]

后置卷积（恢复维度）：
  [B, 256, 256, T]
    ↓
  post_conv: [B, 256, 256, T] → [B, 1, 256, T]
  使用三个卷积层逐步减少通道数

【关键参数说明】
- freq_bins: 频率bin数量（默认256）
- hidden_size: LSTM隐藏层大小（默认128）
- num_layers: LSTM层数（默认2）
- bidirectional: 双向LSTM（同时考虑前后文）
- dropout: 0.2（多层LSTM时使用）

【数据处理流程】
1. 输入：线性幅度谱 [B, 1, 256, T]
2. Log变换：log10(x + 1e-8)
3. 前置CNN：提取局部空间特征
4. 维度重组：[B,64,256,T] → [B*256,T,64]
5. 双向LSTM：对每个频率bin进行时序建模
6. 维度恢复：[B*256,T,256] → [B,256,256,T]
7. 后置CNN：生成最终掩膜
8. 最终结果：输入 × 掩膜

【模型特点】
✓ 时序建模：LSTM捕捉音频的时间依赖性
✓ 双向处理：同时考虑过去和未来信息
✓ 混合架构：CNN提取空间特征 + RNN建模时序关系
✓ 频率并行：每个频率bin独立进行时序处理
✓ 长期依赖：LSTM能够捕捉长距离的时序关系

【与CNN/U-Net区别】
- CNN/U-Net：主要处理空间特征，时序信息有限
- RNN（本模型）：专门处理时序依赖，适合有时间相关性的音频

【使用示例】
```python
from src.models.RNN import AudioRNN
import torch

# 创建模型
model = AudioRNN(freq_bins=256, hidden_size=128, num_layers=2)

# 准备输入
input_spec = torch.randn(4, 1, 256, 376)

# 前向传播
output_spec = model(input_spec)
```
"""

import torch
import torch.nn as nn


class AudioRNN(nn.Module):
    """RNN模型用于音频啸叫抑制（时序建模）

    这是一个混合架构模型，结合了CNN的空间特征提取能力和
    LSTM的时序建模能力，特别适合处理具有时间相关性的音频信号。

    【工作原理】
    1. 前置CNN：提取频谱的局部空间特征
    2. 维度重组：将频谱图转换为LSTM可以处理的序列格式
    3. 双向LSTM：对每个频率bin独立进行时序建模
    4. 维度恢复：将LSTM输出转换回频谱图格式
    5. 后置CNN：生成最终的掩膜

    【输入输出】
    输入: [batch, 1, 256, time] - 含啸叫的幅度谱
    输出: [batch, 1, 256, time] - 抑制啸叫后的幅度谱

    【网络层】
    pre_conv: 前置卷积层，提取空间特征
    lstm: 双向LSTM层，时序建模
    post_conv: 后置卷积层，生成掩膜
    """

    def __init__(self, freq_bins=256, hidden_size=128, num_layers=2):
        """初始化RNN模型

        Args:
            freq_bins: 频率bin数量，默认256
            hidden_size: LSTM隐藏层大小，默认128
            num_layers: LSTM层数，默认2
        """
        super(AudioRNN, self).__init__()

        self.freq_bins = freq_bins
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # =================== 前置卷积层 ===================
        # 作用：提取频谱的局部空间特征
        # 输入: [B, 1, 256, T] -> 输出: [B, 64, 256, T]
        self.pre_conv = nn.Sequential(
            # 第1层卷积：1通道 -> 32通道
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # 第2层卷积：32通道 -> 64通道
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # =================== 双向LSTM层 ===================
        # 作用：对每个频率bin进行时序建模
        # 输入: [B*256, T, 64] -> 输出: [B*256, T, hidden_size*2]
        self.lstm = nn.LSTM(
            input_size=64,              # 输入特征维度（来自前置CNN）
            hidden_size=hidden_size,    # 隐藏层大小
            num_layers=num_layers,      # LSTM层数
            batch_first=True,           # 输入格式为(batch, seq, feature)
            bidirectional=True,         # 双向LSTM，同时考虑前后文
            dropout=0.2 if num_layers > 1 else 0,  # 多层时使用dropout
        )

        # =================== 后置卷积层 ===================
        # 作用：将LSTM输出转换回频谱图格式并生成掩膜
        # 输入: [B, hidden_size*2, 256, T] -> 输出: [B, 1, 256, T]
        self.post_conv = nn.Sequential(
            # 第1层：hidden_size*2通道 -> 64通道
            nn.Conv2d(hidden_size * 2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 第2层：64通道 -> 32通道
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 第3层（输出层）：32通道 -> 1通道
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
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

        # 获取输入维度信息
        B, C, F, T = x_log.shape  # [B, 1, 256, T]

        # =================== 步骤2：前置CNN提取特征 ===================
        # 提取频谱的局部空间特征
        conv_out = self.pre_conv(x_log)  # [B, 1, 256, T] -> [B, 64, 256, T]

        # =================== 步骤3：维度重组（适配LSTM）===================
        # LSTM需要的输入格式是 [batch, seq_len, features]
        # 我们将每个频率bin看作一个独立的序列

        # [B, 64, 256, T] -> [B, 256, T, 64]
        conv_out = conv_out.permute(0, 2, 3, 1)

        # [B, 256, T, 64] -> [B*256, T, 64]
        # 这样每个频率bin都有T个时间步，每步64个特征
        conv_out = conv_out.reshape(B * F, T, -1)

        # =================== 步骤4：LSTM时序建模 ===================
        # 对每个频率bin的时序进行处理
        lstm_out, _ = self.lstm(conv_out)
        # 输出: [B*256, T, hidden_size*2] = [B*256, T, 256]

        # =================== 步骤5：维度恢复 ===================
        # 将LSTM输出转换回频谱图格式

        # [B*256, T, 256] -> [B, 256, T, 256]
        lstm_out = lstm_out.reshape(B, F, T, -1)

        # [B, 256, T, 256] -> [B, 256, 256, T]
        lstm_out = lstm_out.permute(0, 3, 1, 2)

        # =================== 步骤6：后置CNN生成掩膜 ===================
        mask = self.post_conv(lstm_out)  # [B, 256, 256, T] -> [B, 1, 256, T]

        # =================== 步骤7：应用掩膜 ===================
        output = x * mask

        return output
