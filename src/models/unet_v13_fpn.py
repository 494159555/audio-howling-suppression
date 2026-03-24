"""
U-Net v13 Model - Feature Pyramid Network U-Net for Audio Howling Suppression

This module implements a U-Net architecture with a Feature Pyramid Network (FPN)
that enables multi-scale feature fusion with a top-down pathway and lateral
connections.

Author: Research Team
Date: 2026-3-24
Version: 13.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioUNet5FPN(nn.Module):
    """5-layer U-Net with Feature Pyramid Network for audio howling suppression.
    
    This model enhances the standard U-Net by incorporating a Feature Pyramid
    Network (FPN) that enables powerful multi-scale feature fusion through:
    1. Bottom-up pathway (encoder): Extracts multi-scale features
    2. Top-down pathway (lateral upsampling): Propagates high-level semantic info
    3. Lateral connections: Combines spatial information from encoder
    
    Based on: "Feature Pyramid Networks for Object Detection" (Lin et al., 2017)
    
    Network Architecture:
        Encoder (Bottom-up pathway):
            C1: [B,1,256,T] -> [B,16,128,T]
            C2: [B,16,128,T] -> [B,32,64,T]
            C3: [B,32,64,T] -> [B,64,32,T]
            C4: [B,64,32,T] -> [B,128,16,T]
            C5: [B,128,16,T] -> [B,256,8,T] (bottleneck)
            
        Lateral Connections (1x1 convolutions):
            L5: C5 -> [B,64,8,T]
            L4: C4 -> [B,64,16,T]
            L3: C3 -> [B,64,32,T]
            L2: C2 -> [B,64,64,T]
            L1: C1 -> [B,64,128,T]
            
        Top-down Pathway (upsampling + addition):
            P5: L5
            P4: Upsample(P5) + L4
            P3: Upsample(P4) + L3
            P2: Upsample(P3) + L2
            P1: Upsample(P2) + L1
            
        Decoder (using FPN features):
            Uses P5, P4, P3, P2, P1 for reconstruction
            
    Key Features:
        - Strong multi-scale feature fusion
        - High-level semantic information at all scales
        - Improved feature representation
        - Better detection of howling at different scales
    
    Args:
        fpn_channels (int): Number of channels in FPN feature maps (default: 64)
        use_fpn_fusion (bool): Whether to use FPN fusion (default: True)
    """
    
    def __init__(self, fpn_channels=64, use_fpn_fusion=True):
        """
        Initialize the U-Net with Feature Pyramid Network.
        
        Args:
            fpn_channels (int): Number of channels for FPN feature maps
            use_fpn_fusion (bool): Whether to use FPN fusion or standard U-Net
        """
        super(AudioUNet5FPN, self).__init__()

        self.use_fpn_fusion = use_fpn_fusion

        # ==========================
        # Encoder (Bottom-up pathway)
        # ==========================
        
        # C1: [B, 1, 256, T] -> [B, 16, 128, T]
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # C2: [B, 16, 128, T] -> [B, 32, 64, T]
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # C3: [B, 32, 64, T] -> [B, 64, 32, T]
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # C4: [B, 64, 32, T] -> [B, 128, 16, T]
        self.enc4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # C5: [B, 128, 16, T] -> [B, 256, 8, T] (Bottleneck)
        self.enc5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # ==========================
        # Lateral Connections (1x1 convolutions)
        # ==========================
        # Reduce channels to fpn_channels for efficient fusion
        self.lateral5 = nn.Conv2d(256, fpn_channels, kernel_size=1)
        self.lateral4 = nn.Conv2d(128, fpn_channels, kernel_size=1)
        self.lateral3 = nn.Conv2d(64, fpn_channels, kernel_size=1)
        self.lateral2 = nn.Conv2d(32, fpn_channels, kernel_size=1)
        self.lateral1 = nn.Conv2d(16, fpn_channels, kernel_size=1)

        # ==========================
        # Smooth layers (3x3 convolutions for FPN output)
        # ==========================
        self.smooth5 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        self.smooth4 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        self.smooth3 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        self.smooth2 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        self.smooth1 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)

        # ==========================
        # Decoder (using FPN features)
        # ==========================
        
        # Decoder Layer 5
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(
                fpn_channels, 128, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Decoder Layer 4
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(
                256, 64, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Decoder Layer 3
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(
                128, 32, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Decoder Layer 2
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(
                64, 16, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        # Final Layer
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(
                32, 1, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)
            ),
            nn.Sigmoid(),
        )

    def _build_fpn(self, c5, c4, c3, c2, c1):
        """
        Build Feature Pyramid Network features.
        
        Args:
            c5, c4, c3, c2, c1: Encoder feature maps
            
        Returns:
            tuple: (p5, p4, p3, p2, p1) FPN feature maps
        """
        # Apply lateral connections
        l5 = self.lateral5(c5)  # [B, 64, 8, T]
        l4 = self.lateral4(c4)  # [B, 64, 16, T]
        l3 = self.lateral3(c3)  # [B, 64, 32, T]
        l2 = self.lateral2(c2)  # [B, 64, 64, T]
        l1 = self.lateral1(c1)  # [B, 64, 128, T]

        # Top-down pathway (upsampling and addition)
        # Start from top (P5)
        p5 = self.smooth5(l5)  # [B, 64, 8, T]

        # P4: Upsample P5 and add L4
        p5_up = F.interpolate(p5, size=l4.shape[2:], mode='bilinear', align_corners=True)
        p4 = self.smooth4(p5_up + l4)  # [B, 64, 16, T]

        # P3: Upsample P4 and add L3
        p4_up = F.interpolate(p4, size=l3.shape[2:], mode='bilinear', align_corners=True)
        p3 = self.smooth3(p4_up + l3)  # [B, 64, 32, T]

        # P2: Upsample P3 and add L2
        p3_up = F.interpolate(p3, size=l2.shape[2:], mode='bilinear', align_corners=True)
        p2 = self.smooth2(p3_up + l2)  # [B, 64, 64, T]

        # P1: Upsample P2 and add L1
        p2_up = F.interpolate(p2, size=l1.shape[2:], mode='bilinear', align_corners=True)
        p1 = self.smooth1(p2_up + l1)  # [B, 64, 128, T]

        return p5, p4, p3, p2, p1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the U-Net with Feature Pyramid Network.
        
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
        # Encoder Forward Pass (Bottom-up)
        # ==========================
        c1 = self.enc1(x_log)    # [B, 16, 128, T]
        c2 = self.enc2(c1)        # [B, 32, 64, T]
        c3 = self.enc3(c2)        # [B, 64, 32, T]
        c4 = self.enc4(c3)        # [B, 128, 16, T]
        c5 = self.enc5(c4)        # [B, 256, 8, T] - Bottleneck

        # ==========================
        # Build Feature Pyramid Network
        # ==========================
        if self.use_fpn_fusion:
            p5, p4, p3, p2, p1 = self._build_fpn(c5, c4, c3, c2, c1)
            decoder_inputs = (p5, p4, p3, p2, p1)
        else:
            # Use standard encoder features without FPN
            decoder_inputs = (c5, c4, c3, c2, c1)

        # ==========================
        # Decoder Forward Pass
        # ==========================
        
        # Decoder Layer 5
        d5 = self.dec5(decoder_inputs[0])    # [B, 128, 16, T]
        d5_cat = torch.cat([d5, decoder_inputs[1]], dim=1)  # [B, 256, 16, T]

        # Decoder Layer 4
        d4 = self.dec4(d5_cat)    # [B, 64, 32, T]
        d4_cat = torch.cat([d4, decoder_inputs[2]], dim=1)  # [B, 128, 32, T]

        # Decoder Layer 3
        d3 = self.dec3(d4_cat)    # [B, 32, 64, T]
        d3_cat = torch.cat([d3, decoder_inputs[3]], dim=1)  # [B, 64, 64, T]

        # Decoder Layer 2
        d2 = self.dec2(d3_cat)    # [B, 16, 128, T]
        d2_cat = torch.cat([d2, decoder_inputs[4]], dim=1)  # [B, 32, 128, T]

        # Final Layer - Generate Mask
        mask = self.dec1(d2_cat)  # [B, 1, 256, T]

        # ==========================
        # Multiplicative Masking
        # ==========================
        output = x * mask
        return output