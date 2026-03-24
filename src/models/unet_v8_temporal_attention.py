"""
U-Net v8 Model - 5-Layer U-Net with Temporal Attention

This module implements a 5-layer U-Net architecture with temporal attention
mechanism for audio howling suppression. The temporal attention is applied
at the bottleneck to dynamically focus on different time segments.

Key Improvements:
- Temporal attention at bottleneck for dynamic time focusing
- Improved temporal consistency
- Better handling of dynamic howling patterns
- Interpretable attention weights

Author: Research Team
Date: 2026-3-23
Version: 8.0.0
"""

# Standard library imports
# None

# Third-party imports
import torch
import torch.nn as nn

# Local imports
from .temporal_modules import TemporalAttention


class AudioUNet5TemporalAttention(nn.Module):
    """5-layer U-Net with temporal attention for audio howling suppression.
    
    This model enhances the standard U-Net by adding a temporal attention
    mechanism at the bottleneck. The attention computes weights along the
    time dimension, allowing the model to dynamically focus on relevant
    time segments (e.g., periods with howling activity).
    
    Network Architecture:
        Encoder (Downsampling):
            enc1: [B,1,256,T] -> [B,16,128,T]
            enc2: [B,16,128,T] -> [B,32,64,T]
            enc3: [B,32,64,T] -> [B,64,32,T]
            enc4: [B,64,32,T] -> [B,128,16,T]
            enc5: [B,128,16,T] -> [B,256,8,T] (bottleneck)
            
        Temporal Attention (Bottleneck):
            temporal_att: [B,256,8,T] -> [B,256,8,T]
            
        Decoder (Upsampling):
            dec5: [B,256,8,T] -> [B,128,16,T] + enc4 skip connection
            dec4: [B,256,16,T] -> [B,64,32,T] + enc3 skip connection
            dec3: [B,128,32,T] -> [B,32,64,T] + enc2 skip connection
            dec2: [B,64,64,T] -> [B,16,128,T] + enc1 skip connection
            dec1: [B,32,128,T] -> [B,1,256,T]
    
    Key Features:
        - 5 encoder and decoder layers for deep feature extraction
        - Temporal attention at bottleneck
        - Dynamic focus on time segments
        - Improved temporal consistency
        - Log-domain processing for numerical stability
        - Multiplicative masking mechanism
        
    Attributes:
        enc1-enc5 (nn.Sequential): Encoder layers
        dec1-dec5 (nn.Sequential): Decoder layers
        temporal_attention (TemporalAttention): Temporal attention module
    """
    
    def __init__(self, reduction: int = 8):
        """Initialize the 5-layer U-Net with temporal attention.
        
        Args:
            reduction: Reduction ratio for temporal attention (default: 8)
        """
        super(AudioUNet5TemporalAttention, self).__init__()

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
        # Temporal Attention (Bottleneck)
        # ==========================
        self.temporal_attention = TemporalAttention(
            channels=256,
            reduction=reduction,
            use_residual=True
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
        """Forward pass of the U-Net with temporal attention.
        
        Processes input spectrogram through encoder-attention-decoder architecture
        to generate a multiplicative mask for howling suppression.
        
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
        # Temporal Attention at Bottleneck
        # ==========================
        # Apply temporal attention to bottleneck features
        e5_attended = self.temporal_attention(e5)  # [B, 256, 8, T]

        # ==========================
        # Decoder Forward Pass with Skip Connections
        # ==========================
        
        # Decoder Layer 5 + Skip Connection 4
        d5 = self.dec5(e5_attended)  # [B, 128, 16, T]
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


if __name__ == "__main__":
    """Test the model with sample input."""
    print("Testing AudioUNet5TemporalAttention...")
    model = AudioUNet5TemporalAttention(reduction=8)
    
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
    
    # Test temporal attention component
    print(f"  Temporal attention reduction ratio: 8")
    print(f"  ✓ AudioUNet5TemporalAttention test passed\n")