"""时序建模模块

U-Net时序建模组件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """时序注意力机制
    
    沿时间维度计算注意力权重，动态聚焦不同时间片段
    """
    
    def __init__(self, channels: int, reduction: int = 8, use_residual: bool = True):
        """初始化时序注意力"""
        super(TemporalAttention, self).__init__()
        
        self.channels = channels
        self.use_residual = use_residual
        
        # 频率维度池化
        self.avg_pool = nn.AdaptiveAvgPool2d((None, 1))
        
        # 注意力网络
        intermediate_channels = max(channels // reduction, 1)
        self.attention = nn.Sequential(
            nn.Linear(channels, intermediate_channels),
            nn.ReLU(inplace=True),
            nn.Linear(intermediate_channels, channels),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        residual = x
        
        # 聚合频率维度
        x_pool = self.avg_pool(x.permute(0, 1, 3, 2))
        x_pool = x_pool.squeeze(-1)
        x_pool = x_pool.permute(0, 2, 1)
        
        # 计算注意力
        attention_scores = self.attention(x_pool)
        attention_weights = F.softmax(attention_scores, dim=1)
        attention_weights = attention_weights.permute(0, 2, 1)
        
        # 应用注意力权重
        attention_weights = attention_weights.unsqueeze(2)
        attention_weights = attention_weights.expand_as(x)
        output = x * attention_weights
        
        if self.use_residual:
            output = output + residual
        
        return output


class ConvLSTMCell(nn.Module):
    """卷积LSTM单元
    
    结合卷积和LSTM的时空建模
    """
    
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        padding: int = 1
    ):
        """初始化ConvLSTM"""
        super(ConvLSTMCell, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        
        # 输入门
        self.conv_input = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )
        
        # 遗忘门
        self.conv_forget = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )
        
        # 细胞状态
        self.conv_cell = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )
        
        # 输出门
        self.conv_output = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
    
    def forward(self, x: torch.Tensor, state: tuple = None) -> tuple:
        """前向传播"""
        batch_size, _, height, width = x.size()
        
        # 初始化状态
        if state is None or state[0] is None or state[1] is None:
            h_prev = torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device)
            c_prev = torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device)
        else:
            h_prev, c_prev = state
        
        # 拼接输入和隐藏状态
        combined = torch.cat([x, h_prev], dim=1)
        
        # 计算门控
        i_t = self.sigmoid(self.conv_input(combined))
        f_t = self.sigmoid(self.conv_forget(combined))
        g_t = self.tanh(self.conv_cell(combined))
        o_t = self.sigmoid(self.conv_output(combined))
        
        # 更新状态
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * self.tanh(c_t)
        
        return h_t, c_t


class TemporalPooling(nn.Module):
    """多尺度时序池化
    
    捕获不同时间尺度的上下文信息
    """
    
    def __init__(self, channels: int, pool_sizes: list = [3, 5, 7]):
        """初始化时序池化"""
        super(TemporalPooling, self).__init__()
        
        self.channels = channels
        self.pool_sizes = pool_sizes
        
        # 创建池化层
        self.pools = nn.ModuleList()
        for pool_size in pool_sizes:
            padding = pool_size // 2
            pool = nn.AvgPool2d(kernel_size=(1, pool_size), stride=1, padding=(0, padding))
            self.pools.append(pool)
        
        # 降维卷积
        output_channels = channels * len(pool_sizes)
        self.conv_reduce = nn.Sequential(
            nn.Conv2d(output_channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 应用多尺度池化
        pooled_features = []
        for pool in self.pools:
            pooled = pool(x)
            pooled_features.append(pooled)
        
        # 拼接并降维
        combined = torch.cat(pooled_features, dim=1)
        output = self.conv_reduce(combined)
        
        return output


class TemporalConvBlock(nn.Module):
    """空洞卷积时序块
    
    多尺度空洞卷积捕获时序依赖
    """
    
    def __init__(
        self,
        channels: int,
        dilations: list = [1, 2, 4, 8],
        kernel_size: int = 3
    ):
        """初始化时序卷积块"""
        super(TemporalConvBlock, self).__init__()
        
        self.channels = channels
        self.dilations = dilations
        
        # 并行空洞卷积
        self.dilated_convs = nn.ModuleList()
        for dilation in dilations:
            padding = (kernel_size // 2) * dilation
            conv = nn.Sequential(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    padding=(1, padding),
                    dilation=(1, dilation),
                    bias=False
                ),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
            self.dilated_convs.append(conv)
        
        # 融合层
        fusion_channels = channels * len(dilations)
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        features = []
        for conv in self.dilated_convs:
            features.append(conv(x))
        
        combined = torch.cat(features, dim=1)
        output = self.fusion(combined)
        
        return output


if __name__ == "__main__":
    print("Testing temporal modules...\n")
    
    # Test TemporalAttention
    print("Testing TemporalAttention...")
    temp_att = TemporalAttention(channels=256)
    x = torch.randn(4, 256, 32, 100)
    out = temp_att(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  ✓ TemporalAttention test passed\n")
    
    # Test ConvLSTMCell
    print("Testing ConvLSTMCell...")
    conv_lstm = ConvLSTMCell(input_channels=256, hidden_channels=128)
    x_t = torch.randn(4, 256, 8, 100)
    h_prev = torch.randn(4, 128, 8, 100)
    c_prev = torch.randn(4, 128, 8, 100)
    h_next, c_next = conv_lstm(x_t, (h_prev, c_prev))
    print(f"  Input shape: {x_t.shape}")
    print(f"  Hidden state shape: {h_next.shape}")
    print(f"  Cell state shape: {c_next.shape}")
    print(f"  ✓ ConvLSTMCell test passed\n")
    
    # Test TemporalPooling
    print("Testing TemporalPooling...")
    temp_pool = TemporalPooling(channels=256, pool_sizes=[3, 5, 7])
    x = torch.randn(4, 256, 32, 100)
    out = temp_pool(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  ✓ TemporalPooling test passed\n")
    
    # Test TemporalConvBlock
    print("Testing TemporalConvBlock...")
    temp_conv = TemporalConvBlock(channels=256, dilations=[1, 2, 4, 8])
    x = torch.randn(4, 256, 32, 100)
    out = temp_conv(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  ✓ TemporalConvBlock test passed\n")
    
    print("All temporal module tests completed successfully! ✓")