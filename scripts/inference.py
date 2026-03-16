#!/usr/bin/env python3
"""
Audio Howling Suppression Inference Script

This script performs audio howling suppression using a trained AudioUNet5 model.
It supports both Griffin-Lim and ISTFT phase reconstruction methods with
comprehensive audio preprocessing and postprocessing.

Author: Research Team
Date: 2024-12-14
Version: 2.0.0
"""

# Standard library imports
import os
import sys
from pathlib import Path

# Third-party imports
import torch
import torchaudio
import argparse

# Local imports
from src.models import AudioUNet5


def inference(model_path: str, input_wav: str, output_wav: str, 
              device: str = "cpu", use_griffin_lim: bool = True) -> bool:
    """Perform audio howling suppression inference using trained model.
    
    This function loads a trained AudioUNet5 model and processes input audio
    to suppress howling artifacts. It supports two phase reconstruction methods:
    Griffin-Lim for better quality (slower) and ISTFT for faster processing.
    
    Args:
        model_path (str): Path to the trained model checkpoint file
        input_wav (str): Path to input audio file for processing
        output_wav (str): Path to save the processed audio output
        device (str, optional): Computation device ('cpu' or 'cuda'). Defaults to "cpu".
        use_griffin_lim (bool, optional): Whether to use Griffin-Lim algorithm.
                                        True for better quality but slower,
                                        False for faster processing with original phase.
                                        Defaults to True.
    
    Returns:
        bool: True if inference succeeded, False otherwise
        
    Raises:
        FileNotFoundError: If model or input audio files don't exist
        RuntimeError: If model loading or audio processing fails
    """
    print(f"🔄 Processing: {input_wav} ...")

    # ==========================
    # 1. Model Loading and Setup
    # ==========================
    model = AudioUNet5().to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Model loaded successfully: {model_path}")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False
    
    model.eval()

    # ==========================
    # 2. Audio Loading and Preprocessing
    # ==========================
    try:
        waveform, sr = torchaudio.load(input_wav)
    except Exception as e:
        print(f"❌ Audio loading failed: {e}")
        return False

    # Resample to 16kHz if needed (model's expected sample rate)
    if sr != 16000:
        print(f"⚠️ Resampling from {sr}Hz to 16000Hz")
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)
        sr = 16000

    # ==========================
    # 3. Spectral Analysis Parameters
    # ==========================
    # [CRITICAL] These parameters must match training configuration exactly
    n_fft = 512
    hop_length = 128  # High overlap rate for better time resolution
    win_length = n_fft
    window = torch.hann_window(n_fft).to(device)

    # ==========================
    # 4. Magnitude Spectrum Extraction
    # ==========================
    spec_transform = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        power=2.0,  # Power spectrum
    ).to(device)

    # Move to device for faster computation
    waveform = waveform.to(device)
    mag = spec_transform(waveform).sqrt()  # Convert to magnitude: [channel, 257, Time]

    # ==========================
    # 5. Input Preprocessing (Log + Normalization + Cropping)
    # ==========================
    # [ALGORITHM] Apply same preprocessing as training
    eps = 1e-8  # Numerical stability
    norm_min = -11.5  # Based on empirical audio data
    norm_max = 2.5

    # Log transformation
    mag_log = torch.log10(mag + eps)
    
    # Normalization to [0, 1]
    mag_norm = (mag_log - norm_min) / (norm_max - norm_min)
    
    # Crop frequency dimension to match U-Net input (257 -> 256)
    input_mag = mag_norm[:, :-1, :]  # [channel, 256, Time]

    # Add batch dimension
    input_tensor = input_mag.unsqueeze(0)  # [1, channel, 256, Time]

    # ==========================
    # 6. Time Dimension Padding for U-Net
    # ==========================
    # Pad time dimension to be divisible by 16 (2^4 for 4 downsampling layers)
    original_len = input_tensor.shape[-1]
    pad_len = 0
    if original_len % 16 != 0:
        pad_len = 16 - (original_len % 16)
        input_tensor = torch.nn.functional.pad(input_tensor, (0, pad_len))

    # ==========================
    # 7. Model Inference
    # ==========================
    print("🧠 Model inference in progress...")
    with torch.no_grad():
        pred_norm = model(input_tensor)

    # ==========================
    # 8. Output Postprocessing
    # ==========================
    # Remove batch dimension and padding
    pred_norm = pred_norm.squeeze(0)  # [channel, 256, Time_padded]
    if pad_len > 0:
        pred_norm = pred_norm[..., :original_len]

    # Denormalization and inverse log transformation
    pred_log = pred_norm * (norm_max - norm_min) + norm_min
    pred_linear = torch.pow(10, pred_log)

    # Restore frequency dimension (256 -> 257)
    padding_freq = torch.zeros(pred_linear.shape[0], 1, pred_linear.shape[2]).to(device)
    pred_linear = torch.cat([pred_linear, padding_freq], dim=1)

    # ==========================
    # 9. Phase Reconstruction and Waveform Synthesis
    # ==========================
    if use_griffin_lim:
        print("✨ Using Griffin-Lim for phase reconstruction (enhanced quality)...")
        griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=n_fft,
            n_iter=32,  # 32 iterations usually sufficient
            win_length=win_length,
            hop_length=hop_length,  # Must match analysis parameters
            power=1.0,  # Input is magnitude spectrum
        ).to(device)

        new_waveform = griffin_lim(pred_linear)
    else:
        print("⚠️ Using original phase reconstruction (ISTFT)...")
        # Compute original phase only when needed (saves computation)
        stft_complex = torch.stft(
            waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True,
        )
        phase = torch.angle(stft_complex)

        # Reconstruct complex spectrum
        new_stft_complex = pred_linear * torch.exp(1j * phase)

        # Inverse STFT
        new_waveform = torch.istft(
            new_stft_complex,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            length=waveform.shape[1],
        )

    # ==========================
    # 10. Smart Normalization (Clipping Prevention)
    # ==========================
    # [ALGORITHM] Intelligent normalization to prevent clipping and noise amplification
    # Strategy: If max > 1.0 (clipping), normalize down; if max is small (silence), don't amplify
    max_val = torch.max(torch.abs(new_waveform))
    if max_val > 1.0:
        new_waveform = new_waveform / max_val
        print(f"🔊 Amplitude overflow detected ({max_val:.2f}), auto-limited.")

    # ==========================
    # 11. Save Output
    # ==========================
    try:
        os.makedirs(os.path.dirname(output_wav), exist_ok=True)
        torchaudio.save(output_wav, new_waveform.cpu(), sr)
        print(f"✅ Processing completed! Saved to: {output_wav}")
        return True
    except Exception as e:
        print(f"❌ Save failed: {e}")
        return False


def main() -> None:
    """Main function for command-line interface.
    
    Parses command-line arguments and performs audio howling suppression inference.
    Supports automatic device detection and various processing options.
    """
    parser = argparse.ArgumentParser(description="Audio Howling Suppression Inference Script")
    parser.add_argument("--model", type=str, required=True, 
                       help="Path to trained model checkpoint file")
    parser.add_argument("--input", type=str, required=True, 
                       help="Path to input audio file")
    parser.add_argument("--output", type=str, required=True, 
                       help="Path to save processed audio output")
    parser.add_argument("--device", type=str, default="auto", 
                       choices=["auto", "cpu", "cuda"], 
                       help="Computation device (auto, cpu, or cuda)")
    parser.add_argument("--use_griffin_lim", action="store_true", default=True,
                       help="Use Griffin-Lim algorithm for phase reconstruction")
    
    args = parser.parse_args()
    
    # ==========================
    # Device Selection
    # ==========================
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"🎯 Using device: {device}")
    
    # ==========================
    # Input Validation
    # ==========================
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        return
    
    if not os.path.exists(args.input):
        print(f"❌ Input audio file not found: {args.input}")
        return
    
    # ==========================
    # Execute Inference
    # ==========================
    success = inference(
        model_path=args.model,
        input_wav=args.input,
        output_wav=args.output,
        device=device,
        use_griffin_lim=args.use_griffin_lim
    )
    
    if success:
        print("🎉 Inference completed successfully!")
    else:
        print("💥 Inference failed!")


if __name__ == "__main__":
    main()
