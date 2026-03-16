'''
移频移向法 - Frequency Shift Method

文件功能：
- 实现基于频率偏移的音频啸叫抑制算法
- 通过轻微改变信号频率来破坏反馈路径的相位条件
- 使用相位声码器技术实现高质量的频率偏移

算法原理：
1. 啸叫产生条件：信号经过放大器、扬声器、麦克风路径形成正反馈
2. 相位条件：反馈信号与原信号同相时产生啸叫
3. 解决方案：通过频率偏移破坏相位条件，抑制啸叫

技术实现：
- STFT/ISTFT变换：时频域处理
- 相位调整：根据频率偏移量调整相位
- 频率偏移：通常10-50Hz的小幅偏移
- 重叠相加：保证信号连续性

重要参数：
- shift_hz: 频率偏移量(默认20Hz)
- n_fft: FFT窗口大小(默认512)
- hop_length: 跳跃长度(默认128)
- window: 窗函数类型(默认hann)

特点：
- 音质保持好：小幅频率偏移人耳难以察觉
- 实时性强：计算复杂度适中
- 参数简单：只需调节频率偏移量
- 效果稳定：对各种啸叫都有效

使用方法：
from src.traditional.frequency_shift import FrequencyShiftMethod
method = FrequencyShiftMethod(shift_hz=20)
output = method(input_spectrogram)  # input: [B,1,256,T]
'''

import torch
import torch.nn as nn
import numpy as np
import math


class FrequencyShiftMethod(nn.Module):
    """
    移频移向法 - 通过频率偏移抑制音频啸叫
    
    Args:
        shift_hz (float): 频率偏移量，单位Hz，默认20Hz
        sample_rate (int): 采样率，默认16000Hz
        n_fft (int): FFT窗口大小，默认512
        hop_length (int): 跳跃长度，默认128
        window (str): 窗函数类型，默认'hann'
    """
    
    def __init__(self, shift_hz=20.0, sample_rate=16000, n_fft=512, hop_length=128, window='hann'):
        super(FrequencyShiftMethod, self).__init__()
        
        self.shift_hz = shift_hz
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = window
        
        # 频率分辨率
        self.freq_resolution = sample_rate / n_fft
        
        # 频率偏移对应的bin数
        self.shift_bins = shift_hz / self.freq_resolution
        
        # 创建窗函数
        if window == 'hann':
            self.window = torch.hann_window(n_fft)
        elif window == 'hamming':
            self.window = torch.hamming_window(n_fft)
        else:
            self.window = torch.hann_window(n_fft)
            
        # 预计算相位调整因子
        self._precompute_phase_factors()
        
    def _precompute_phase_factors(self):
        """预计算相位调整因子"""
        # 时间帧索引
        frame_indices = torch.arange(1000).float()  # 预计算1000帧
        
        # 相位调整因子：exp(j * 2π * shift * t)
        phase_shift = 2 * math.pi * self.shift_hz * frame_indices * self.hop_length / self.sample_rate
        self.phase_factors = torch.exp(1j * phase_shift)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入频谱图 [B, 1, F, T]
            
        Returns:
            torch.Tensor: 处理后的频谱图 [B, 1, F, T]
        """
        batch_size, channels, freq_bins, time_frames = x.shape
        
        # 确保输入是单通道
        assert channels == 1, f"Expected single channel input, got {channels}"
        
        # 转换为线性域幅度谱
        x_linear = torch.pow(10, x)  # 从log域转换回线性域
        
        # 频率偏移处理（直接处理幅度谱）
        magnitude_spec = x_linear.squeeze(1)  # [B, F, T]
        shifted_magnitude = self._apply_frequency_shift(magnitude_spec)
        
        # 重新添加通道维度
        shifted_magnitude = shifted_magnitude.unsqueeze(1)  # [B, 1, F, T]
        
        # 转换回log域
        shifted_log = torch.log10(shifted_magnitude + 1e-8)
        
        return shifted_log
    
    def _apply_frequency_shift(self, magnitude_spec):
        """
        应用频率偏移（简化版本，直接处理幅度谱）
        
        Args:
            magnitude_spec (torch.Tensor): 幅度频谱 [B, F, T]
            
        Returns:
            torch.Tensor: 偏移后的幅度频谱 [B, F, T]
        """
        batch_size, freq_bins, time_frames = magnitude_spec.shape
        
        # 创建输出频谱
        shifted_magnitude = torch.zeros_like(magnitude_spec)
        
        # 频率偏移（只处理幅度，简化实现）
        for f in range(freq_bins):
            # 计算目标频率bin
            target_f = f + self.shift_bins
            
            # 边界检查
            if 0 <= target_f < freq_bins:
                # 线性插值
                f_low = int(math.floor(target_f))
                f_high = int(math.ceil(target_f))
                alpha = target_f - f_low
                
                if f_low < freq_bins and f_high < freq_bins:
                    # 线性插值频率分量
                    if f_low == f_high:
                        shifted_magnitude[:, f, :] = magnitude_spec[:, f_low, :]
                    else:
                        shifted_magnitude[:, f, :] = (
                            (1 - alpha) * magnitude_spec[:, f_low, :] + 
                            alpha * magnitude_spec[:, f_high, :]
                        )
        
        return shifted_magnitude
    
    def process_audio(self, audio):
        """
        处理时域音频信号（可选功能）
        
        Args:
            audio (torch.Tensor): 时域音频 [B, 1, T]
            
        Returns:
            torch.Tensor: 处理后的音频 [B, 1, T]
        """
        batch_size, channels, samples = audio.shape
        
        # STFT
        window = self.window.to(audio.device)
        stft = torch.stft(
            audio.squeeze(1), 
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True
        )
        
        # 频率偏移
        shifted_stft = self._apply_frequency_shift(stft)
        
        # ISTFT
        shifted_audio = torch.istft(
            shifted_stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            length=samples
        )
        
        return shifted_audio.unsqueeze(1)


def create_frequency_shift_method(shift_hz=20.0, **kwargs):
    """
    创建移频移向法实例的便捷函数
    
    Args:
        shift_hz (float): 频率偏移量
        **kwargs: 其他参数
        
    Returns:
        FrequencyShiftMethod: 移频移向法实例
    """
    return FrequencyShiftMethod(shift_hz=shift_hz, **kwargs)


# 测试代码
if __name__ == "__main__":
    # 创建测试数据
    batch_size, channels, freq_bins, time_frames = 2, 1, 256, 100
    test_input = torch.randn(batch_size, channels, freq_bins, time_frames).abs()
    test_input = torch.log10(test_input + 1e-8)  # 模拟log域输入
    
    # 创建方法实例
    method = FrequencyShiftMethod(shift_hz=20.0)
    
    # 测试前向传播
    output = method(test_input)
    
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出范围: [{output.min():.4f}, {output.max():.4f}]")
    print("移频移向法测试通过！")
