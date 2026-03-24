"""
U-Net v11 Model - Multi-scale U-Net for Audio Howling Suppression

This module implements a multi-scale U-Net architecture that uses multiple
U-Nets of different depths to process different frequency ranges, providing
specialized processing for low, mid, and high frequency bands.

Author: Research Team
Date: 2026-3-24
Version: 11.0.0
"""

import torch
import torch.nn as nn


class AudioUNet3(nn.Module):
    """3-layer U-Net for low-frequency processing."""
    
    def __init__(self):
        super(AudioUNet3, self).__init__()

        # Encoder
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

        # Decoder
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)
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

    def forward(self, x):
        x_log = torch.log10(x + 1e-8)
        
        e1 = self.enc1(x_log)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        d3 = self.dec3(e3)
        d3_cat = torch.cat([d3, e2], dim=1)
        
        d2 = self.dec2(d3_cat)
        d2_cat = torch.cat([d2, e1], dim=1)
        
        mask = self.dec1(d2_cat)
        output = x * mask
        return output


class AudioUNet7(nn.Module):
    """7-layer U-Net for high-frequency processing."""
    
    def __init__(self):
        super(AudioUNet7, self).__init__()

        # Encoder
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

        self.enc6 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.enc7 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Decoder
        self.dec7 = nn.Sequential(
            nn.ConvTranspose2d(
                1024, 512, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.dec6 = nn.Sequential(
            nn.ConvTranspose2d(
                1024, 256, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(
                512, 128, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)
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

    def forward(self, x):
        x_log = torch.log10(x + 1e-8)
        
        e1 = self.enc1(x_log)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        
        d7 = self.dec7(e7)
        d7_cat = torch.cat([d7, e6], dim=1)
        
        d6 = self.dec6(d7_cat)
        d6_cat = torch.cat([d6, e5], dim=1)
        
        d5 = self.dec5(d6_cat)
        d5_cat = torch.cat([d5, e4], dim=1)
        
        d4 = self.dec4(d5_cat)
        d4_cat = torch.cat([d4, e3], dim=1)
        
        d3 = self.dec3(d4_cat)
        d3_cat = torch.cat([d3, e2], dim=1)
        
        d2 = self.dec2(d3_cat)
        d2_cat = torch.cat([d2, e1], dim=1)
        
        mask = self.dec1(d2_cat)
        output = x * mask
        return output


class AudioUNet5(nn.Module):
    """5-layer U-Net for mid-frequency processing."""
    
    def __init__(self):
        super(AudioUNet5, self).__init__()

        # Encoder
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

        # Decoder
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

    def forward(self, x):
        x_log = torch.log10(x + 1e-8)
        
        e1 = self.enc1(x_log)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        
        d5 = self.dec5(e5)
        d5_cat = torch.cat([d5, e4], dim=1)
        
        d4 = self.dec4(d5_cat)
        d4_cat = torch.cat([d4, e3], dim=1)
        
        d3 = self.dec3(d4_cat)
        d3_cat = torch.cat([d3, e2], dim=1)
        
        d2 = self.dec2(d3_cat)
        d2_cat = torch.cat([d2, e1], dim=1)
        
        mask = self.dec1(d2_cat)
        output = x * mask
        return output


class AudioUNet5MultiScale(nn.Module):
    """Multi-scale U-Net for audio howling suppression.
    
    This model uses multiple U-Nets of different depths to process different
    frequency ranges:
        - Low frequencies (0-64 bins): 3-layer U-Net (shallow, fast)
        - Mid frequencies (64-192 bins): 5-layer U-Net (balanced)
        - High frequencies (192-256 bins): 7-layer U-Net (deep, detailed)
    
    The outputs are then fused using a 1x1 convolution to produce the final
    multiplicative mask.
    
    Network Architecture:
        Input: [B, 1, 256, T]
        Split into:
            - Low freq: [B, 1, 64, T] -> AudioUNet3
            - Mid freq: [B, 1, 128, T] -> AudioUNet5
            - High freq: [B, 1, 64, T] -> AudioUNet7
        Fuse: Concatenate [B, 3, 256, T] -> 1x1 Conv -> [B, 1, 256, T]
        
    Key Features:
        - Specialized processing for different frequency bands
        - Efficient use of computational resources
        - Better frequency resolution
        - Flexible architecture for different audio characteristics
    """
    
    def __init__(self):
        """Initialize the multi-scale U-Net model."""
        super(AudioUNet5MultiScale, self).__init__()

        # Initialize three U-Nets of different depths
        self.unet_low = AudioUNet3()      # For low frequencies (0-64 bins)
        self.unet_mid = AudioUNet5()      # For mid frequencies (64-192 bins)
        self.unet_high = AudioUNet7()     # For high frequencies (192-256 bins)

        # Fusion layer: combine outputs from all scales
        self.fusion = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the multi-scale U-Net model.
        
        Args:
            x (torch.Tensor): Input spectrogram [B, 1, 256, T]
            
        Returns:
            torch.Tensor: Output spectrogram [B, 1, 256, T]
        """
        # Split input into three frequency bands
        # Low: [0:64], Mid: [64:192], High: [192:256]
        x_low = x[:, :, 0:64, :]       # [B, 1, 64, T]
        x_mid = x[:, :, 64:192, :]     # [B, 1, 128, T]
        x_high = x[:, :, 192:256, :]   # [B, 1, 64, T]

        # Process each frequency band with its specialized U-Net
        out_low = self.unet_low(x_low)     # [B, 1, 64, T]
        out_mid = self.unet_mid(x_mid)     # [B, 1, 128, T]
        out_high = self.unet_high(x_high)  # [B, 1, 64, T]

        # Concatenate outputs along channel dimension
        out_concat = torch.cat([out_low, out_mid, out_high], dim=1)  # [B, 3, 256, T]

        # Apply fusion layer
        mask = self.fusion(out_concat)  # [B, 1, 256, T]

        # Apply multiplicative mask
        output = x * mask

        return output