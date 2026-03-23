"""
U-Net v4 Model - 5-Layer U-Net with Residual Connections

This module implements a 5-layer U-Net architecture with residual blocks
for audio howling suppression. The residual connections help alleviate the
vanishing gradient problem in deep networks and enable more stable training.

Key Improvements:
- Residual blocks in encoder (5 residual blocks)
- Alleviates vanishing gradient problem
- Enables deeper network architectures
- More stable training convergence

Author: Research Team
Date: 2026-3-23
Version: 4.0.0
"""

# Standard library imports
# None

# Third-party imports
import torch
import torch.nn as nn

# Local imports
from .attention_modules import ResidualBlock


class AudioUNet5Residual(nn.Module):
    """5-layer U-Net with residual blocks for audio howling suppression.
    
    This model enhances the standard U-Net architecture by replacing each
    encoder layer with a residual block. The residual connections (skip
    connections within each block) help alleviate the vanishing gradient
    problem and enable more stable training of deep networks.
    
    Network Architecture:
        Encoder (Downsampling with Residual Blocks):
            enc1: [B,1,256,T] -> [B,16,128,T] (residual block + downsample)
            enc2: [B,16,128,T] -> [B,32,64,T] (residual block + downsample)
            enc3: [B,32,64,T] -> [B,64,32,T] (residual block + downsample)
            enc4: [B,64,32,T] -> [B,128,16,T] (residual block + downsample)
            enc5: [B,128,16,T] -> [B,256,8,T] (residual block + downsample, bottleneck)
            
        Decoder (Upsampling):
            dec5: [B,256,8,T] -> [B,128,16,T] + enc4 skip connection
            dec4: [B,256,16,T] -> [B,64,32,T] + enc3 skip connection
            dec3: [B,128,32,T] -> [B,32,64,T] + enc2 skip connection
            dec2: [B,64,64,T] -> [B,16,128,T] + enc1 skip connection
            dec1: [B,32,128,T] -> [B,1,256,T]
    
    Key Features:
        - Residual blocks in all encoder layers (5 residual blocks)
        - Alleviates vanishing gradient problem
        - Enables deeper network architectures
        - More stable training convergence
        - Log-domain processing for numerical stability
        - Multiplicative masking mechanism
        
    Attributes:
        enc1-enc5 (nn.Sequential): Encoder layers with ResidualBlock, Conv2d, BatchNorm2d, LeakyReLU
        dec1-dec5 (nn.Sequential): Decoder layers with ConvTranspose2d, BatchNorm2d, ReLU
        res1-res5 (ResidualBlock): Residual blocks for each encoder layer
    """
    
    def __init__(self):
        """Initialize the 5-layer U-Net with residual connections."""
        super(AudioUNet5Residual, self).__init__()

        # ==========================
        # Encoder (Downsampling with Residual Blocks) - 5 Layers
        # ==========================
        
        # Layer 1: [B, 1, 256, T] -> [B, 16, 128, T]
        self.enc1_down = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.res1 = ResidualBlock(channels=16)

        # Layer 2: [B, 16, 128, T] -> [B, 32, 64, T]
        self.enc2_down = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.res2 = ResidualBlock(channels=32)

        # Layer 3: [B, 32, 64, T] -> [B, 64, 32, T]
        self.enc3_down = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.res3 = ResidualBlock(channels=64)

        # Layer 4: [B, 64, 32, T] -> [B, 128, 16, T]
        self.enc4_down = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.res4 = ResidualBlock(channels=128)

        # Layer 5: [B, 128, 16, T] -> [B, 256, 8, T] (Bottleneck)
        self.enc5_down = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.res5 = ResidualBlock(channels=256)

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
        """Forward pass of the U-Net with residual connections.
        
        Processes input spectrogram through encoder-decoder architecture with
        residual blocks in encoder to generate a multiplicative mask for
        howling suppression.
        
        Args:
            x (torch.Tensor): Input spectrogram with shape [B, 1, 256, T]
                             representing linear magnitude spectrum
            
        Returns:
            torch.Tensor: Output spectrogram with shape [B, 1, 256, T]
                         representing howling-suppressed audio spectrum
        """
        # ==========================
        # Log-domain Feature Extraction
        # ==========================
        # [ALGORITHM] Extract log features for numerical stability
        # Reason: Log domain provides better dynamic range and numerical stability
        x_log = torch.log10(x + 1e-8)

        # ==========================
        # Encoder Forward Pass with Residual Blocks
        # ==========================
        # Each encoder layer: downsample -> residual block
        
        # Encoder Layer 1
        e1_down = self.enc1_down(x_log)  # [B, 16, 128, T]
        e1 = self.res1(e1_down)           # [B, 16, 128, T]
        
        # Encoder Layer 2
        e2_down = self.enc2_down(e1)     # [B, 32, 64, T]
        e2 = self.res2(e2_down)           # [B, 32, 64, T]
        
        # Encoder Layer 3
        e3_down = self.enc3_down(e2)     # [B, 64, 32, T]
        e3 = self.res3(e3_down)           # [B, 64, 32, T]
        
        # Encoder Layer 4
        e4_down = self.enc4_down(e3)     # [B, 128, 16, T]
        e4 = self.res4(e4_down)           # [B, 128, 16, T]
        
        # Encoder Layer 5 (Bottleneck)
        e5_down = self.enc5_down(e4)     # [B, 256, 8, T]
        e5 = self.res5(e5_down)           # [B, 256, 8, T] - Bottleneck

        # ==========================
        # Decoder Forward Pass with Skip Connections
        # ==========================
        
        # Decoder Layer 5 + Skip Connection 4
        d5 = self.dec5(e5)                # [B, 128, 16, T]
        d5_cat = torch.cat([d5, e4], dim=1)  # [B, 256, 16, T]

        # Decoder Layer 4 + Skip Connection 3
        d4 = self.dec4(d5_cat)            # [B, 64, 32, T]
        d4_cat = torch.cat([d4, e3], dim=1)  # [B, 128, 32, T]

        # Decoder Layer 3 + Skip Connection 2
        d3 = self.dec3(d4_cat)            # [B, 32, 64, T]
        d3_cat = torch.cat([d3, e2], dim=1)  # [B, 64, 64, T]

        # Decoder Layer 2 + Skip Connection 1
        d2 = self.dec2(d3_cat)            # [B, 16, 128, T]
        d2_cat = torch.cat([d2, e1], dim=1)  # [B, 32, 128, T]

        # Final Decoder Layer - Generate Mask
        mask = self.dec1(d2_cat)          # [B, 1, 256, T]

        # ==========================
        # Multiplicative Masking
        # ==========================
        # [ALGORITHM] Apply multiplicative mask in linear domain
        # Reason: Mask preserves phase information and provides interpretable results
        # 1.0 = complete preservation, 0.0 = complete suppression
        output = x * mask
        return output


if __name__ == "__main__":
    """Test the model with sample input."""
    print("Testing AudioUNet5Residual...")
    model = AudioUNet5Residual()
    
    # Create sample input: [Batch=2, Channels=1, Freq=256, Time=100]
    x = torch.randn(2, 1, 256, 100)
    
    # Forward pass
    output = model(x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  ✓ AudioUNet5Residual test passed\n")