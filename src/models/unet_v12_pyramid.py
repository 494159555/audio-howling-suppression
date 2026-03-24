"""
U-Net v12 Model - Pyramid Pooling U-Net for Audio Howling Suppression

This module implements a U-Net architecture with a Pyramid Pooling Module (PPM)
at the bottleneck layer to capture multi-scale context information.

Author: Research Team
Date: 2026-3-24
Version: 12.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PyramidPoolingModule(nn.Module):
    """Pyramid Pooling Module (PPM) for multi-scale context aggregation.
    
    This module captures multi-scale context information by applying
    adaptive average pooling at different scales, then upsampling and
    concatenating the features.
    
    Based on: "Pyramid Scene Parsing Network" (Zhao et al., 2017)
    
    Args:
        in_channels (int): Number of input channels
        pool_sizes (tuple): List of pooling output sizes (default: (1, 2, 3, 6))
        reduction_ratio (int): Channel reduction ratio for efficiency
    """
    
    def __init__(self, in_channels, pool_sizes=(1, 2, 3, 6), reduction_ratio=4):
        super(PyramidPoolingModule, self).__init__()
        
        # Number of pyramid levels
        self.pool_sizes = pool_sizes
        self.num_levels = len(pool_sizes)
        
        # Calculate reduced channels per level
        reduced_channels = in_channels // reduction_ratio
        
        # Create pyramid branches
        self.pyramid_branches = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=size),
                nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
                nn.BatchNorm2d(reduced_channels),
                nn.ReLU(inplace=True)
            )
            for size in pool_sizes
        ])
        
        # Output projection to restore original channel dimension
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels + reduced_channels * self.num_levels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Apply pyramid pooling module.
        
        Args:
            x (torch.Tensor): Input feature map [B, C, H, W]
            
        Returns:
            torch.Tensor: Output feature map with multi-scale context [B, C, H, W]
        """
        input_size = x.size()[2:]
        
        # Apply each pyramid level
        pyramid_features = []
        for branch in self.pyramid_branches:
            # Pool and project
            pooled = branch(x)
            # Upsample to original size
            upsampled = F.interpolate(
                pooled,
                size=input_size,
                mode='bilinear',
                align_corners=True
            )
            pyramid_features.append(upsampled)
        
        # Concatenate original and pyramid features
        pyramid_concat = torch.cat([x] + pyramid_features, dim=1)
        
        # Project back to original channels
        output = self.out_conv(pyramid_concat)
        
        return output


class AudioUNet5Pyramid(nn.Module):
    """5-layer U-Net with Pyramid Pooling for audio howling suppression.
    
    This model extends the standard 5-layer U-Net by adding a Pyramid Pooling
    Module at the bottleneck layer, enabling the model to capture multi-scale
    context information for better howling suppression.
    
    Network Architecture:
        Encoder (Downsampling):
            enc1: [B,1,256,T] -> [B,16,128,T]
            enc2: [B,16,128,T] -> [B,32,64,T]
            enc3: [B,32,64,T] -> [B,64,32,T]
            enc4: [B,64,32,T] -> [B,128,16,T]
            enc5: [B,128,16,T] -> [B,256,8,T] (bottleneck)
            
        Pyramid Pooling Module:
            Applied at bottleneck [B,256,8,T]
            Multi-scale pooling with levels (1, 2, 3, 6)
            
        Decoder (Upsampling):
            dec5: [B,256,8,T] -> [B,128,16,T] + enc4 skip connection
            dec4: [B,256,16,T] -> [B,64,32,T] + enc3 skip connection
            dec3: [B,128,32,T] -> [B,32,64,T] + enc2 skip connection
            dec2: [B,64,64,T] -> [B,16,128,T] + enc1 skip connection
            dec1: [B,32,128,T] -> [B,1,256,T] + enc1 skip connection
    
    Key Features:
        - Multi-scale context aggregation at bottleneck
        - Captures both local and global information
        - Improves feature representation
        - Minimal parameter increase
    
    Args:
        pyramid_levels (tuple): Pooling sizes for PPM (default: (1, 2, 3, 6))
        reduction_ratio (int): Channel reduction ratio for PPM (default: 4)
    """
    
    def __init__(self, pyramid_levels=(1, 2, 3, 6), reduction_ratio=4):
        """
        Initialize the U-Net with Pyramid Pooling.
        
        Args:
            pyramid_levels (tuple): Pooling sizes for pyramid pooling module
            reduction_ratio (int): Channel reduction ratio for efficiency
        """
        super(AudioUNet5Pyramid, self).__init__()

        # ==========================
        # Encoder (Downsampling) - 5 Layers
        # ==========================
        
        # Layer 1: [B, 1, 256, T] -> [B, 16, 128, T]
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Layer 2: [B, 16, 128, T] -> [B, 32, 64, T]
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Layer 3: [B, 32, 64, T] -> [B, 64, 32, T]
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Layer 4: [B, 64, 32, T] -> [B, 128, 16, T]
        self.enc4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Layer 5: [B, 128, 16, T] -> [B, 256, 8, T] (Bottleneck)
        self.enc5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # ==========================
        # Pyramid Pooling Module at Bottleneck
        # ==========================
        self.ppm = PyramidPoolingModule(
            in_channels=256,
            pool_sizes=pyramid_levels,
            reduction_ratio=reduction_ratio
        )

        # ==========================
        # Decoder (Upsampling) - 5 Layers
        # ==========================
        
        # Layer 5: [B, 256, 8, T] -> [B, 128, 16, T]
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(
                256, 128, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Layer 4: [B, 128+128, 16, T] -> [B, 64, 32, T]
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(
                256, 64, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Layer 3: [B, 64+64, 32, T] -> [B, 32, 64, T]
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(
                128, 32, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Layer 2: [B, 32+32, 64, T] -> [B, 16, 128, T]
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(
                64, 16, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        # Layer 1: [B, 16+16, 128, T] -> [B, 1, 256, T]
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(
                32, 1, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)
            ),
            nn.Sigmoid(),  # Output multiplicative mask in [0, 1] range
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the U-Net with Pyramid Pooling.
        
        Args:
            x (torch.Tensor): Input spectrogram [B, 1, 256, T]
            
        Returns:
            torch.Tensor: Output spectrogram [B, 1, 256, T]
        """
        # ==========================
        # Log-domain Feature Extraction
        # ==========================
        x_log = torch.log10(x + 1e-8)

        # ==========================
        # Encoder Forward Pass
        # ==========================
        e1 = self.enc1(x_log)    # [B, 16, 128, T]
        e2 = self.enc2(e1)        # [B, 32, 64, T]
        e3 = self.enc3(e2)        # [B, 64, 32, T]
        e4 = self.enc4(e3)        # [B, 128, 16, T]
        e5 = self.enc5(e4)        # [B, 256, 8, T] - Bottleneck

        # ==========================
        # Apply Pyramid Pooling Module at Bottleneck
        # ==========================
        # Capture multi-scale context information
        e5_ppm = self.ppm(e5)     # [B, 256, 8, T]

        # ==========================
        # Decoder Forward Pass with Skip Connections
        # ==========================
        # Decoder Layer 5 + Skip Connection 4
        d5 = self.dec5(e5_ppm)    # [B, 128, 16, T]
        d5_cat = torch.cat([d5, e4], dim=1)  # [B, 256, 16, T]

        # Decoder Layer 4 + Skip Connection 3
        d4 = self.dec4(d5_cat)    # [B, 64, 32, T]
        d4_cat = torch.cat([d4, e3], dim=1)  # [B, 128, 32, T]

        # Decoder Layer 3 + Skip Connection 2
        d3 = self.dec3(d4_cat)    # [B, 32, 64, T]
        d3_cat = torch.cat([d3, e2], dim=1)  # [B, 64, 64, T]

        # Decoder Layer 2 + Skip Connection 1
        d2 = self.dec2(d3_cat)    # [B, 16, 128, T]
        d2_cat = torch.cat([d2, e1], dim=1)  # [B, 32, 128, T]

        # Final Decoder Layer - Generate Mask
        mask = self.dec1(d2_cat)  # [B, 1, 256, T]

        # ==========================
        # Multiplicative Masking
        # ==========================
        output = x * mask
        return output