"""注意力与残差模块

U-Net改进的核心构建块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    """注意力门机制
    
    用于U-Net跳跃连接的注意力门，让模型自动聚焦于相关特征
    """
    
    def __init__(self, F_g: int, F_l: int, F_int: int):
        """初始化注意力块"""
        super(AttentionBlock, self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class ResidualBlock(nn.Module):
    """残差块
    
    带有残差连接的双层卷积块，缓解梯度消失问题
    """
    
    def __init__(self, channels: int):
        """初始化残差块"""
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        residual = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += residual
        out = self.relu(out)
        
        return out


class AtrousConvBlock(nn.Module):
    """空洞卷积块
    
    多尺度特征提取，增大感受野而不增加参数量
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation_rates: list = [2, 4, 8],
        kernel_size: int = 3
    ):
        """初始化空洞卷积块"""
        super(AtrousConvBlock, self).__init__()
        
        self.dilated_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels // len(dilation_rates),
                    kernel_size=kernel_size,
                    padding=dilation,
                    dilation=dilation,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels // len(dilation_rates)),
                nn.ReLU(inplace=True)
            )
            for dilation in dilation_rates
        ])
        
        actual_concat_channels = (out_channels // len(dilation_rates)) * len(dilation_rates)
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(actual_concat_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        outputs = []
        for conv in self.dilated_convs:
            outputs.append(conv(x))
        
        out = torch.cat(outputs, dim=1)
        out = self.final_conv(out)
        
        return out


if __name__ == "__main__":
    print("Testing AttentionBlock...")
    att_block = AttentionBlock(F_g=128, F_l=64, F_int=32)
    g = torch.randn(2, 128, 16, 100)
    x = torch.randn(2, 64, 32, 100)
    out = att_block(g, x)
    print(f"  Input shapes: g={g.shape}, x={x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  ✓ AttentionBlock test passed\n")
    
    print("Testing ResidualBlock...")
    res_block = ResidualBlock(channels=64)
    x = torch.randn(2, 64, 128, 100)
    out = res_block(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  ✓ ResidualBlock test passed\n")
    
    print("Testing AtrousConvBlock...")
    atrous_block = AtrousConvBlock(256, 256, dilation_rates=[2, 4, 8])
    x = torch.randn(2, 256, 8, 100)
    out = atrous_block(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  ✓ AtrousConvBlock test passed\n")
    
    print("All tests completed successfully! ✓")