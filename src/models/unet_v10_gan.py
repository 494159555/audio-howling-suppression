"""
U-Net v10 Model - 5-Layer U-Net with GAN Framework

This module implements a 5-layer U-Net architecture with GAN (Generative
Adversarial Network) framework for audio howling suppression. The generator
is a U-Net, and a discriminator distinguishes between real and fake
spectrograms.

Key Improvements:
- GAN framework for improved generation quality
- Adversarial loss for more natural audio
- Combined reconstruction and adversarial loss
- Better perceptual quality

Author: Research Team
Date: 2026-3-23
Version: 10.0.0
"""

# Standard library imports
# None

# Third-party imports
import torch
import torch.nn as nn

# Local imports
from .loss_functions import AdversarialLoss


class AudioUNet5GAN(nn.Module):
    """5-layer U-Net with GAN framework for audio howling suppression.
    
    This model implements a GAN framework where:
    - Generator: 5-layer U-Net that generates howling-suppressed spectrograms
    - Discriminator: CNN network that classifies real vs fake spectrograms
    
    The generator is trained with combined reconstruction loss and adversarial loss,
    while the discriminator is trained to distinguish between real (clean) and
    fake (generated) spectrograms.
    
    Network Architecture:
        Generator (U-Net):
            Encoder: [B,1,256,T] -> ... -> [B,256,8,T]
            Decoder: [B,256,8,T] -> ... -> [B,1,256,T]
            
        Discriminator (CNN):
            Input: [B,1,256,T]
            Conv1: [B,1,256,T] -> [B,64,128,T]
            Conv2: [B,64,128,T] -> [B,128,64,T]
            Conv3: [B,128,64,T] -> [B,256,32,T]
            Output: [B,1,1,1] (real/fake probability)
    
    Key Features:
        - GAN framework for improved generation quality
        - Combined reconstruction + adversarial loss
        - More natural audio output
        - Better perceptual quality
        - Log-domain processing for numerical stability
        - Multiplicative masking mechanism
        
    Attributes:
        generator (nn.Module): U-Net generator
        discriminator (nn.Module): CNN discriminator
    """
    
    def __init__(self):
        """Initialize the U-Net GAN model."""
        super(AudioUNet5GAN, self).__init__()
        
        # ==========================
        # Generator (U-Net)
        # ==========================
        self.generator = self._build_generator()
        
        # ==========================
        # Discriminator (CNN)
        # ==========================
        self.discriminator = self._build_discriminator()
    
    def _build_generator(self):
        """Build the U-Net generator."""
        class Generator(nn.Module):
            def __init__(self):
                super(Generator, self).__init__()

                # Encoder (Downsampling) - 5 Layers
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

                # Decoder (Upsampling) - 5 Layers
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

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """Forward pass of generator."""
                x_log = torch.log10(x + 1e-8)

                # Encoder
                e1 = self.enc1(x_log)    # [B, 16, 128, T]
                e2 = self.enc2(e1)        # [B, 32, 64, T]
                e3 = self.enc3(e2)        # [B, 64, 32, T]
                e4 = self.enc4(e3)        # [B, 128, 16, T]
                e5 = self.enc5(e4)        # [B, 256, 8, T]

                # Decoder with skip connections
                d5 = self.dec5(e5)        # [B, 128, 16, T]
                d5_cat = torch.cat([d5, e4], dim=1)

                d4 = self.dec4(d5_cat)    # [B, 64, 32, T]
                d4_cat = torch.cat([d4, e3], dim=1)

                d3 = self.dec3(d4_cat)    # [B, 32, 64, T]
                d3_cat = torch.cat([d3, e2], dim=1)

                d2 = self.dec2(d3_cat)    # [B, 16, 128, T]
                d2_cat = torch.cat([d2, e1], dim=1)

                mask = self.dec1(d2_cat)    # [B, 1, 256, T]
                output = x * mask
                return output

        return Generator()
    
    def _build_discriminator(self):
        """Build the CNN discriminator."""
        class Discriminator(nn.Module):
            def __init__(self):
                super(Discriminator, self).__init__()

                self.conv_layers = nn.Sequential(
                    # Layer 1
                    nn.Conv2d(1, 64, kernel_size=4, stride=(2, 1), padding=1),
                    nn.BatchNorm2d(64),
                    nn.LeakyReLU(0.2, inplace=True),

                    # Layer 2
                    nn.Conv2d(64, 128, kernel_size=4, stride=(2, 1), padding=1),
                    nn.BatchNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True),

                    # Layer 3
                    nn.Conv2d(128, 256, kernel_size=4, stride=(2, 1), padding=1),
                    nn.BatchNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True),
                )

                self.final_layer = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(256, 1),
                    nn.Sigmoid()
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """Forward pass of discriminator."""
                features = self.conv_layers(x)
                output = self.final_layer(features)
                return output

        return Discriminator()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of generator only.
        
        Args:
            x: Input noisy spectrogram [B, 1, 256, T]
            
        Returns:
            Generated clean spectrogram [B, 1, 256, T]
        """
        return self.generator(x)


if __name__ == "__main__":
    """Test the model with sample input."""
    print("Testing AudioUNet5GAN...")
    model = AudioUNet5GAN()
    
    # Create sample input: [Batch=2, Channels=1, Freq=256, Time=100]
    noisy_spec = torch.randn(2, 1, 256, 100).abs()
    clean_spec = torch.randn(2, 1, 256, 100).abs()
    
    # Forward pass through generator
    pred_spec = model.generator(noisy_spec)
    
    # Forward pass through discriminator
    real_score = model.discriminator(clean_spec)
    fake_score = model.discriminator(pred_spec)
    
    print(f"  Input noisy shape: {noisy_spec.shape}")
    print(f"  Generated clean shape: {pred_spec.shape}")
    print(f"  Real score shape: {real_score.shape}")
    print(f"  Fake score shape: {fake_score.shape}")
    print(f"  Real score mean: {real_score.mean().item():.4f}")
    print(f"  Fake score mean: {fake_score.mean().item():.4f}")
    
    # Count parameters
    gen_params = sum(p.numel() for p in model.generator.parameters())
    disc_params = sum(p.numel() for p in model.discriminator.parameters())
    total_params = gen_params + disc_params
    print(f"  Generator parameters: {gen_params:,}")
    print(f"  Discriminator parameters: {disc_params:,}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  ✓ AudioUNet5GAN test passed\n")