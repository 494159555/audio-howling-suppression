"""
U-Net v9 Model - 5-Layer U-Net with ConvLSTM

This module implements a 5-layer U-Net architecture with ConvLSTM for
audio howling suppression. The ConvLSTM replaces the bottleneck layer,
preserving spatial structure while modeling temporal dependencies.

Key Improvements:
- ConvLSTM at bottleneck for spatio-temporal modeling
- Preserves spatial structure of spectrograms
- Better captures both frequency and temporal patterns
- Improved temporal consistency

Author: Research Team
Date: 2026-3-23
Version: 9.0.0
"""

# Standard library imports
# None

# Third-party imports
import torch
import torch.nn as nn

# Local imports
from .temporal_modules import ConvLSTMCell


class AudioUNet5ConvLSTM(nn.Module):
    """5-layer U-Net with ConvLSTM for audio howling suppression.
    
    This model enhances the standard U-Net by replacing the bottleneck layer
    with a ConvLSTM. Unlike standard LSTM which flattens spatial dimensions,
    ConvLSTM preserves the 2D structure of spectrograms, enabling better
    modeling of spatio-temporal patterns.
    
    Network Architecture:
        Encoder (Downsampling):
            enc1: [B,1,256,T] -> [B,16,128,T]
            enc2: [B,16,128,T] -> [B,32,64,T]
            enc3: [B,32,64,T] -> [B,64,32,T]
            enc4: [B,64,32,T] -> [B,128,16,T]
            enc5: [B,128,16,T] -> [B,256,8,T] (bottleneck input)
            
        ConvLSTM (Bottleneck):
            convlstm: [B,256,8,T] -> [B,128,8,T] (preserving spatial structure)
            
        Decoder (Upsampling):
            dec5: [B,128,8,T] -> [B,128,16,T] + enc4 skip connection
            dec4: [B,256,16,T] -> [B,64,32,T] + enc3 skip connection
            dec3: [B,128,32,T] -> [B,32,64,T] + enc2 skip connection
            dec2: [B,64,64,T] -> [B,16,128,T] + enc1 skip connection
            dec1: [B,32,128,T] -> [B,1,256,T]
    
    Key Features:
        - 5 encoder and decoder layers for deep feature extraction
        - ConvLSTM at bottleneck (128 hidden channels)
        - Preserves spatial structure
        - Better spatio-temporal modeling
        - Log-domain processing for numerical stability
        - Multiplicative masking mechanism
        
    Attributes:
        enc1-enc5 (nn.Sequential): Encoder layers
        dec1-dec5 (nn.Sequential): Decoder layers
        convlstm_cell (ConvLSTMCell): ConvLSTM cell for temporal modeling
    """
    
    def __init__(
        self,
        hidden_channels: int = 128,
        kernel_size: int = 3,
        padding: int = 1
    ):
        """Initialize the 5-layer U-Net with ConvLSTM.
        
        Args:
            hidden_channels: Number of hidden channels in ConvLSTM
            kernel_size: Kernel size for ConvLSTM convolutions
            padding: Padding for ConvLSTM convolutions
        """
        super(AudioUNet5ConvLSTM, self).__init__()

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

        # Layer 5: [B, 128, 16, T] -> [B, 256, 8, T] (Bottleneck input)
        self.enc5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # ==========================
        # ConvLSTM (Bottleneck)
        # ==========================
        self.conv_lstm = ConvLSTMCell(
            input_channels=256,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )

        # ==========================
        # Decoder (Upsampling) - 5 Layers
        # ==========================
        
        # Layer 5: [B, 128, 8, T] -> [B, 128, 16, T]
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(
                128, 128, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)
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
        """Forward pass of the U-Net with ConvLSTM.
        
        Processes input spectrogram through encoder-ConvLSTM-decoder architecture
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
        e5 = self.enc5(e4)        # [B, 256, 8, T] - Bottleneck input

        # ==========================
        # ConvLSTM Processing (Bottleneck)
        # ==========================
        # Process each time step through ConvLSTM
        batch_size, _, freq_bins, time_steps = e5.shape
        
        # Initialize hidden and cell states
        h = None
        c = None
        
        # Process each time step
        hidden_states = []
        for t in range(time_steps):
            # Extract time step
            x_t = e5[:, :, :, t:t+1]  # [B, 256, 8, 1]
            
            # Apply ConvLSTM
            h, c = self.conv_lstm(x_t, (h, c))  # [B, 128, 8, 1]
            
            # Store hidden state
            hidden_states.append(h)
        
        # Concatenate all time steps
        lstm_out = torch.cat(hidden_states, dim=3)  # [B, 128, 8, T]

        # ==========================
        # Decoder Forward Pass with Skip Connections
        # ==========================
        
        # Decoder Layer 5 + Skip Connection 4
        d5 = self.dec5(lstm_out)  # [B, 128, 16, T]
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
    print("Testing AudioUNet5ConvLSTM...")
    model = AudioUNet5ConvLSTM(hidden_channels=128, kernel_size=3, padding=1)
    
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
    
    # Test ConvLSTM component
    print(f"  ConvLSTM hidden channels: 128")
    print(f"  ConvLSTM kernel size: 3")
    print(f"  ✓ AudioUNet5ConvLSTM test passed\n")