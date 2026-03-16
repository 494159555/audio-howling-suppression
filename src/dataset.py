"""
Audio Howling Suppression Dataset Module

This module implements the HowlingDataset class for loading and preprocessing
audio data, supporting paired clean and howling audio files with comprehensive
preprocessing capabilities including spectral transformation, log scaling,
and normalization.

Author: Research Team
Date: 2024-12-14
Version: 2.0.0
"""

# Standard library imports
import os

# Third-party imports
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

# Local imports
from src.config import cfg


class HowlingDataset(Dataset):
    """Audio howling suppression dataset.
    
    Inherits from torch.utils.data.Dataset for loading paired clean and howling
    audio files with comprehensive preprocessing capabilities including spectral
    transformation, log scaling, and normalization.
    
    Args:
        clean_dir (str or Path): Directory path for clean audio files
        howling_dir (str or Path): Directory path for howling audio files  
        sample_rate (int, optional): Audio sample rate in Hz. Defaults to None (uses cfg.SAMPLE_RATE).
        chunk_len (float, optional): Audio chunk length in seconds. Defaults to None (uses cfg.CHUNK_LEN).
        n_fft (int, optional): FFT window size. Defaults to None (uses cfg.N_FFT).
        hop_length (int, optional): Hop length for STFT. Defaults to None (uses cfg.HOP_LENGTH).
        
    Attributes:
        clean_dir (Path): Clean audio directory path
        howling_dir (Path): Howling audio directory path
        sample_rate (int): Audio sample rate
        chunk_len (float): Audio chunk length in seconds
        chunk_size (int): Number of samples per chunk
        n_fft (int): FFT window size
        hop_length (int): Hop length for STFT
        spec_transform (Spectrogram): STFT transformation object
        filenames (list): List of audio filenames
    """
    
    def __init__(
        self,
        clean_dir,
        howling_dir,
        sample_rate=None,
        chunk_len=None,
        n_fft=None,
        hop_length=None,
    ):
        self.clean_dir = clean_dir
        self.howling_dir = howling_dir

        # Use provided parameters or fall back to global config
        self.sample_rate = sample_rate if sample_rate is not None else cfg.SAMPLE_RATE
        self.chunk_len = chunk_len if chunk_len is not None else cfg.CHUNK_LEN
        self.n_fft = n_fft if n_fft is not None else cfg.N_FFT
        self.hop_length = hop_length if hop_length is not None else cfg.HOP_LENGTH

        # Calculate chunk size (number of samples)
        self.chunk_size = int(self.sample_rate * self.chunk_len)

        # Validate directory existence
        if not os.path.exists(str(self.howling_dir)):
            raise FileNotFoundError(f"Directory does not exist: {self.howling_dir}")

        # Get sorted list of filenames
        self.filenames = sorted(os.listdir(str(self.howling_dir)))

        # Initialize spectral transformation
        # Note: power=2.0 returns power spectrum, we'll take sqrt for magnitude
        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft, hop_length=self.hop_length, power=2.0
        )

    def __len__(self) -> int:
        """Get the total number of samples in the dataset.
        
        Returns:
            int: Number of audio files in the dataset
        """
        return len(self.filenames)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get data sample by index.
        
        Loads and preprocesses audio files at the specified index, returning
        preprocessed spectrogram pairs with log transformation and normalization.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple[torch.Tensor, torch.Tensor]: (noisy_mag, clean_mag) 
                Preprocessed spectrogram pair with shape [1, 256, T]
                
        Raises:
            FileNotFoundError: If audio files are not found
            RuntimeError: If audio loading fails
        """
        # ==========================
        # 1. Audio Loading
        # ==========================
        file_name = self.filenames[idx]
        howling_path = os.path.join(self.howling_dir, file_name)
        clean_path = os.path.join(self.clean_dir, file_name)

        try:
            howling_wave, sr_h = torchaudio.load(howling_path)
            clean_wave, sr_c = torchaudio.load(clean_path)
        except Exception as e:
            print(f"Error loading {file_name}: {e}")
            # Return zero tensors to prevent training interruption
            return self._get_zero_tensors()

        # TODO: Add sample rate conversion if needed
        # Check sample rate consistency
        if sr_h != self.sample_rate:
            # In production, add torchaudio.transforms.Resample here
            pass

        # ==========================
        # 2. Audio Length Normalization
        # ==========================
        # Pad or truncate to fixed length
        if howling_wave.shape[1] < self.chunk_size:
            pad_len = self.chunk_size - howling_wave.shape[1]
            howling_wave = F.pad(howling_wave, (0, pad_len))
            clean_wave = F.pad(clean_wave, (0, pad_len))
        else:
            howling_wave = howling_wave[:, : self.chunk_size]
            clean_wave = clean_wave[:, : self.chunk_size]

        # ==========================
        # 3. Spectral Transformation
        # ==========================
        # Convert to magnitude spectrum
        howling_mag = self.spec_transform(howling_wave).sqrt()
        clean_mag = self.spec_transform(clean_wave).sqrt()

        # ==========================
        # 4. Log Transformation and Normalization
        # ==========================
        # [ALGORITHM] Log transformation with numerical stability
        # Reason: Direct linear domain training can cause numerical overflow
        eps = 1e-8  # Small value to prevent log(0)
        
        # Log10 transformation
        howling_log = torch.log10(howling_mag + eps)
        clean_log = torch.log10(clean_mag + eps)

        # Normalization parameters based on empirical audio data
        # log10(1e-8) = -8, considering smaller noise, set lower bound to -11.5 (~-230dB)
        # log10(100) = 2, set upper bound to 2.5
        norm_min = -11.5
        norm_max = 2.5

        # Normalize to [0, 1] range: (x - min) / (max - min)
        howling_norm = (howling_log - norm_min) / (norm_max - norm_min)
        clean_norm = (clean_log - norm_min) / (norm_max - norm_min)

        # ==========================
        # 5. Dimension Adjustment for U-Net
        # ==========================
        # [ALGORITHM] Crop last frame to fit U-Net architecture
        # Reason: 256 = 2^8, suitable for multiple downsampling in U-Net
        howling_out = howling_norm[:, :-1, :]  # Crop from 257 to 256 frequency bins
        clean_out = clean_norm[:, :-1, :]

        return howling_out, clean_out

    def _get_zero_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate zero tensors as fallback for failed audio loading.
        
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Zero tensors with appropriate shape
        """
        freq_bins = self.n_fft // 2 + 1
        # Rough time frame estimation
        time_frames = int(self.chunk_size / (self.n_fft / 2)) + 1
        
        # Crop to match U-Net input requirements
        freq_bins_cropped = freq_bins - 1  # Crop from 257 to 256
        
        return (
            torch.zeros(1, freq_bins_cropped, time_frames),
            torch.zeros(1, freq_bins_cropped, time_frames)
        )
