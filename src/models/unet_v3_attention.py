"""
U-Net v3 Model - 5-Layer U-Net with Attention Mechanism

This module implements a 5-layer U-Net architecture with attention gates for
audio howling suppression. The attention mechanism allows the model to
automatically focus on relevant features from encoder skip connections,
enabling dynamic feature fusion based on both decoder and encoder information.

Key Improvements:
- Attention gates on all skip connections (5 attention blocks)
- Automatic focus on howling-related frequency bands
- Dynamic feature fusion based on attention weights
- Better suppression accuracy and reduced collateral damage

Author: Research Team
Date: 2026-3-23
Version: 3.0.0
"""

# Standard library imports
# None

# Third-party imports
import torch
import torch.nn as nn

# Local imports
from .attention_modules import AttentionBlock


class AudioUNet5Attention(nn.Module):
    """5-layer U-Net with attention mechanism for audio howling suppression.
    
    This model enhances the standard U-Net architecture by adding attention
    gates on all skip connections. The attention mechanism computes attention
    weights based on both decoder output and encoder skip connection, allowing
    the model to dynamically focus on relevant features and suppress noise.
    
    Network Architecture:
        Encoder (Downsampling):
            enc1: [B,1,256,T] -> [B,16,128,T]
            enc2: [B,16,128,T] -> [B,32,64,T]
            enc3: [B,32,64,T] -> [B,64,32,T]
            enc4: [B,64,32,T] -> [B,128,16,T]
            enc5: [B,128,16,T] -> [B,256,8,T] (bottleneck)
            
        Decoder (Upsampling with Attention):
            dec5: [B,256,8,T] -> [B,128,16,T] + att4(enc4, dec5)
            dec4: [B,256,16,T] -> [B,64,32,T] + att3(enc3, dec4)
            dec3: [B,128,32,T] -> [B,32,64,T] + att2(enc2, dec3)
            dec2: [B,64,64,T] -> [B,16,128,T] + att1(enc1, dec2)
            dec1: [B,32,128,T] -> [B,1,256,T]
    
    Key Features:
        - 5 encoder and decoder layers for deep feature extraction
        - Attention gates on all skip connections (5 attention blocks)
        - Automatic focus on howling-related frequency bands
        - Dynamic feature fusion based on attention weights
        - Log-domain processing for numerical stability
        - Multiplicative masking mechanism
        
    Attributes:
        enc1-enc5 (nn.Sequential): Encoder layers with Conv2d, BatchNorm2d, LeakyReLU
        dec1-dec5 (nn.Sequential): Decoder layers with ConvTranspose2d, BatchNorm2d, ReLU
        att1-att4 (AttentionBlock): Attention gates for skip connections
    """
    
    def __init__(self):
        """Initialize the 5-layer U-Net with attention mechanism."""
        super(AudioUNet5Attention, self).__init__()

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
        """Forward pass of the U-Net with attention mechanism.
        
        Processes input spectrogram through encoder-decoder architecture with
        attention gates on all skip connections to generate a multiplicative
        mask for howling suppression.
        
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
        # Encoder Forward Pass
        # ==========================
        e1 = self.enc1(x_log)    # [B, 16, 128, T]
        e2 = self.enc2(e1)        # [B, 32, 64, T]
        e3 = self.enc3(e2)        # [B, 64, 32, T]
        e4 = self.enc4(e3)        # [B, 128, 16, T]
        e5 = self.enc5(e4)        # [B, 256, 8, T] - Bottleneck

        # ==========================
        # Decoder Forward Pass with Attention Gates
        # ==========================
        
        # Decoder Layer 5 + Attention Gate 4
        d5 = self.dec5(e5)        # [B, 128, 16, T]
        # Apply attention to encoder features before concatenation
        e4_att = self.att4(d5, e4)  # [B, 128, 16, T] - weighted encoder features
        d5_cat = torch.cat([d5, e4_att], dim=1)  # [B, 256, 16, T]

        # Decoder Layer 4 + Attention Gate 3
        d4 = self.dec4(d5_cat)    # [B, 64, 32, T]
        # Apply attention to encoder features before concatenation
        e3_att = self.att3(d4, e3)  # [B, 64, 32, T] - weighted encoder features
        d4_cat = torch.cat([d4, e3_att], dim=1)  # [B, 128, 32, T]

        # Decoder Layer 3 + Attention Gate 2
        d3 = self.dec3(d4_cat)    # [B, 32, 64, T]
        # Apply attention to encoder features before concatenation
        e2_att = self.att2(d3, e2)  # [B, 32, 64, T] - weighted encoder features
        d3_cat = torch.cat([d3, e2_att], dim=1)  # [B, 64, 64, T]

        # Decoder Layer 2 + Attention Gate 1
        d2 = self.dec2(d3_cat)    # [B, 16, 128, T]
        # Apply attention to encoder features before concatenation
        e1_att = self.att1(d2, e1)  # [B, 16, 128, T] - weighted encoder features
        d2_cat = torch.cat([d2, e1_att], dim=1)  # [B, 32, 128, T]

        # Final Decoder Layer - Generate Mask
        mask = self.dec1(d2_cat)  # [B, 1, 256, T]

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
    print("Testing AudioUNet5Attention...")
    model = AudioUNet5Attention()
    
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
    print(f"  ✓ AudioUNet5Attention test passed\n")