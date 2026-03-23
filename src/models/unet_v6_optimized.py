"""
U-Net v6 Model - Optimized 5-Layer U-Net with All Improvements

This module implements the most advanced version of the U-Net architecture
for audio howling suppression, combining all three key improvements:
attention mechanisms, residual connections, and atrous convolutions.

Key Improvements:
- Attention gates on all skip connections (4 attention blocks)
- Residual blocks in encoder (5 residual blocks)
- Atrous convolutions in bottleneck with dilation rates [2, 4, 8]
- Combines benefits of all three improvements
- Best expected performance for howling suppression

Author: Research Team
Date: 2026-3-23
Version: 6.0.0
"""

# Standard library imports
# None

# Third-party imports
import torch
import torch.nn as nn

# Local imports
from .attention_modules import AttentionBlock, ResidualBlock, AtrousConvBlock


class AudioUNet5Optimized(nn.Module):
    """Optimized 5-layer U-Net with attention, residual, and atrous convolutions.
    
    This is the most advanced version of the U-Net architecture for audio
    howling suppression, combining all three key improvements:
    1. Attention gates on skip connections for dynamic feature fusion
    2. Residual blocks in encoder to alleviate vanishing gradient problem
    3. Atrous convolutions in bottleneck to increase receptive field
    
    Network Architecture:
        Encoder (Downsampling with Residual Blocks):
            enc1: [B,1,256,T] -> [B,16,128,T] (residual block + downsample)
            enc2: [B,16,128,T] -> [B,32,64,T] (residual block + downsample)
            enc3: [B,32,64,T] -> [B,64,32,T] (residual block + downsample)
            enc4: [B,64,32,T] -> [B,128,16,T] (residual block + downsample)
            enc5: [B,128,16,T] -> [B,256,8,T] (residual + atrous conv, bottleneck)
            
        Decoder (Upsampling with Attention Gates):
            dec5: [B,256,8,T] -> [B,128,16,T] + att4(enc4, dec5)
            dec4: [B,256,16,T] -> [B,64,32,T] + att3(enc3, dec4)
            dec3: [B,128,32,T] -> [B,32,64,T] + att2(enc2, dec3)
            dec2: [B,64,64,T] -> [B,16,128,T] + att1(enc1, dec2)
            dec1: [B,32,128,T] -> [B,1,256,T]
    
    Key Features:
        - Attention gates on all skip connections (4 attention blocks)
        - Residual blocks in all encoder layers (5 residual blocks)
        - Atrous convolution block in bottleneck with dilation rates [2, 4, 8]
        - Combines benefits of all three improvements
        - Automatic focus on howling-related frequency bands
        - Alleviates vanishing gradient problem
        - Captures long-range temporal dependencies
        - Log-domain processing for numerical stability
        - Multiplicative masking mechanism
        
    Attributes:
        enc1-enc4 (nn.Sequential): Encoder downsampling layers
        res1-res5 (ResidualBlock): Residual blocks for each encoder layer
        atrous_block (AtrousConvBlock): Multi-scale dilated convolution block
        dec1-dec5 (nn.Sequential): Decoder layers
        att1-att4 (AttentionBlock): Attention gates for skip connections
    """
    
    def __init__(self, dilation_rates: list = [2, 4, 8]):
        """Initialize the optimized 5-layer U-Net.
        
        Args:
            dilation_rates: List of dilation rates for the atrous convolution block.
                           Default is [2, 4, 8] for multi-scale feature extraction.
        """
        super(AudioUNet5Optimized, self).__init__()

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
        
        # Atrous convolution block in bottleneck for multi-scale feature extraction
        self.atrous_block = AtrousConvBlock(
            in_channels=256,
            out_channels=256,
            dilation_rates=dilation_rates,
            kernel_size=3
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

        # ==========================
        # Attention Gates for Skip Connections
        # ==========================
        # Attention gates compute weights based on both decoder and encoder features
        
        # Att4: Gate between dec5 and enc4
        self.att4 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        
        # Att3: Gate between dec4 and enc3
        self.att3 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        
        # Att2: Gate between dec3 and enc2
        self.att2 = AttentionBlock(F_g=32, F_l=32, F_int=16)
        
        # Att1: Gate between dec2 and enc1
        self.att1 = AttentionBlock(F_g=16, F_l=16, F_int=8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the optimized U-Net.
        
        Processes input spectrogram through encoder-decoder architecture with
        residual blocks, attention gates, and atrous convolutions to generate
        a multiplicative mask for howling suppression.
        
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
        e5_res = self.res5(e5_down)      # [B, 256, 8, T]
        e5 = self.atrous_block(e5_res)   # [B, 256, 8, T] - Atrous conv for multi-scale features

        # ==========================
        # Decoder Forward Pass with Attention Gates
        # ==========================
        
        # Decoder Layer 5 + Attention Gate 4
        d5 = self.dec5(e5)                # [B, 128, 16, T]
        # Apply attention to encoder features before concatenation
        e4_att = self.att4(d5, e4)        # [B, 128, 16, T] - weighted encoder features
        d5_cat = torch.cat([d5, e4_att], dim=1)  # [B, 256, 16, T]

        # Decoder Layer 4 + Attention Gate 3
        d4 = self.dec4(d5_cat)            # [B, 64, 32, T]
        # Apply attention to encoder features before concatenation
        e3_att = self.att3(d4, e3)        # [B, 64, 32, T] - weighted encoder features
        d4_cat = torch.cat([d4, e3_att], dim=1)  # [B, 128, 32, T]

        # Decoder Layer 3 + Attention Gate 2
        d3 = self.dec3(d4_cat)            # [B, 32, 64, T]
        # Apply attention to encoder features before concatenation
        e2_att = self.att2(d3, e2)        # [B, 32, 64, T] - weighted encoder features
        d3_cat = torch.cat([d3, e2_att], dim=1)  # [B, 64, 64, T]

        # Decoder Layer 2 + Attention Gate 1
        d2 = self.dec2(d3_cat)            # [B, 16, 128, T]
        # Apply attention to encoder features before concatenation
        e1_att = self.att1(d2, e1)        # [B, 16, 128, T] - weighted encoder features
        d2_cat = torch.cat([d2, e1_att], dim=1)  # [B, 32, 128, T]

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
    print("Testing AudioUNet5Optimized...")
    model = AudioUNet5Optimized(dilation_rates=[2, 4, 8])
    
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
    print(f"  ✓ AudioUNet5Optimized test passed\n")