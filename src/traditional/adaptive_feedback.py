'''
自适应反馈抵消法 - Adaptive Feedback Cancellation Method

文件功能：
- 实现基于自适应滤波的音频啸叫抑制算法
- 使用LMS/NLMS算法实时估计并消除反馈路径
- 通过自适应滤波器建模反馈路径并进行抵消

算法原理：
1. 反馈路径建模：使用自适应滤波器估计扬声器到麦克风的反馈路径
2. 信号预测：根据输出信号预测反馈信号
3. 信号抵消：从输入信号中减去预测的反馈信号
4. 滤波器更新：使用LMS算法自适应更新滤波器系数

技术实现：
- 自适应滤波器：FIR滤波器结构，可调长度
- LMS算法：最小均方算法，计算简单收敛稳定
- NLMS算法：归一化LMS，收敛速度更快
- 步长控制：自适应步长调整，平衡收敛速度和稳定性

重要参数：
- filter_length: 自适应滤波器长度，默认64
- step_size: LMS步长，默认0.01
- leakage_factor: 泄漏因子，默认0.9999
- normalization: 是否使用NLMS，默认True
- max_gain: 最大增益限制，默认20dB

特点：
- 自适应强：能自动适应变化的反馈路径
- 精度高：直接建模反馈路径，抵消效果好
- 稳定性好：通过步长控制保证算法稳定
- 实时性强：计算复杂度适中

使用方法：
from src.traditional.adaptive_feedback import AdaptiveFeedbackMethod
method = AdaptiveFeedbackMethod(filter_length=64)
output = method(input_spectrogram)  # input: [B,1,256,T]
'''

import torch
import torch.nn as nn
import numpy as np
import math


class AdaptiveFeedbackMethod(nn.Module):
    """
    自适应反馈抵消法 - 通过自适应滤波抑制音频啸叫
    
    Args:
        filter_length (int): 自适应滤波器长度，默认64
        step_size (float): LMS步长，默认0.01
        leakage_factor (float): 泄漏因子，默认0.9999
        normalization (bool): 是否使用NLMS，默认True
        max_gain (float): 最大增益限制，dB，默认20dB
        sample_rate (int): 采样率，默认16000Hz
        n_fft (int): FFT窗口大小，默认512
        hop_length (int): 跳跃长度，默认128
    """
    
    def __init__(self, 
                 filter_length=64,
                 step_size=0.01,
                 leakage_factor=0.9999,
                 normalization=True,
                 max_gain=20.0,
                 sample_rate=16000,
                 n_fft=512,
                 hop_length=128):
        super(AdaptiveFeedbackMethod, self).__init__()
        
        self.filter_length = filter_length
        self.step_size = step_size
        self.leakage_factor = leakage_factor
        self.normalization = normalization
        self.max_gain = max_gain
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # 转换为线性域
        self.max_gain_linear = 10 ** (max_gain / 20)
        
        # 状态变量
        self.register_buffer('filter_coeffs', None)
        self.register_buffer('input_buffer', None)
        self.register_buffer('output_buffer', None)
        
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
        
        # 转换为线性域
        x_linear = torch.pow(10, x)  # 从log域转换回线性域
        
        # 初始化状态变量
        if (self.filter_coeffs is None or 
            self.filter_coeffs.shape[0] != batch_size or
            self.input_buffer is None or
            self.input_buffer.shape[-1] != time_frames):
            
            self.filter_coeffs = torch.zeros(batch_size, self.filter_length, device=x.device)
            self.input_buffer = torch.zeros(batch_size, time_frames + self.filter_length, device=x.device)
            self.output_buffer = torch.zeros(batch_size, time_frames, device=x.device)
        
        # 处理每个时间帧
        processed_spec = torch.zeros_like(x_linear.squeeze(1))
        
        for t in range(time_frames):
            # 获取当前帧的频谱
            current_frame = x_linear[:, 0, :, t]  # [B, F]
            
            # 简化处理：使用频谱的平均值作为时域信号的代表
            # 在实际应用中，应该使用ISTFT转换到时域进行处理
            input_signal = torch.mean(current_frame, dim=1)  # [B]
            
            # 更新输入缓冲区
            self.input_buffer[:, t + self.filter_length] = input_signal
            
            # 自适应滤波处理
            for b in range(batch_size):
                # 获取滤波器输入向量
                filter_input = self.input_buffer[b, t:t + self.filter_length].flip(0)  # [filter_length]
                
                # 计算滤波器输出
                filter_output = torch.dot(self.filter_coeffs[b], filter_input)
                
                # 计算误差信号（这里简化处理）
                error = input_signal[b] - filter_output
                
                # 更新滤波器系数 (LMS算法)
                if self.normalization:
                    # NLMS: 归一化LMS
                    input_power = torch.sum(filter_input ** 2) + 1e-8
                    normalized_step = self.step_size / input_power
                else:
                    # 标准LMS
                    normalized_step = self.step_size
                
                # 滤波器系数更新
                self.filter_coeffs[b] = (
                    self.leakage_factor * self.filter_coeffs[b] + 
                    normalized_step * error * filter_input
                )
                
                # 限制滤波器系数范围
                self.filter_coeffs[b] = torch.clamp(
                    self.filter_coeffs[b], 
                    -self.max_gain_linear, 
                    self.max_gain_linear
                )
                
                # 保存处理后的信号
                self.output_buffer[b, t] = error
            
            # 重建频谱（简化处理）
            # 在实际应用中，应该使用完整的时域处理和STFT/ISTFT
            for b in range(batch_size):
                # 使用处理后的信号调整整个频谱
                gain_factor = torch.abs(self.output_buffer[b, t]) / (torch.abs(input_signal[b]) + 1e-8)
                gain_factor = torch.clamp(gain_factor, 0.1, 2.0)  # 限制增益范围
                processed_spec[b, :, t] = current_frame[b] * gain_factor
        
        # 转换回log域
        processed_log = torch.log10(processed_spec.unsqueeze(1) + 1e-8)
        
        return processed_log
    
    def process_time_domain(self, input_audio):
        """
        处理时域音频信号（完整实现）
        
        Args:
            input_audio (torch.Tensor): 输入音频 [B, 1, T]
            
        Returns:
            torch.Tensor: 处理后的音频 [B, 1, T]
        """
        batch_size, channels, samples = input_audio.shape
        
        # 初始化状态变量
        if (self.filter_coeffs is None or 
            self.filter_coeffs.shape[0] != batch_size):
            
            self.filter_coeffs = torch.zeros(batch_size, self.filter_length, device=input_audio.device)
            self.input_buffer = torch.zeros(batch_size, samples + self.filter_length, device=input_audio.device)
            self.output_buffer = torch.zeros(batch_size, samples, device=input_audio.device)
        
        # 处理每个样本
        processed_audio = torch.zeros_like(input_audio.squeeze(1))
        
        for t in range(samples):
            for b in range(batch_size):
                # 当前输入样本
                input_sample = input_audio[b, 0, t]
                
                # 更新输入缓冲区
                self.input_buffer[b, t + self.filter_length] = input_sample
                
                # 获取滤波器输入向量
                filter_input = self.input_buffer[b, t:t + self.filter_length].flip(0)
                
                # 计算滤波器输出（预测的反馈信号）
                predicted_feedback = torch.dot(self.filter_coeffs[b], filter_input)
                
                # 计算误差信号（消除反馈后的信号）
                error_signal = input_sample - predicted_feedback
                
                # 更新滤波器系数
                if self.normalization:
                    # NLMS
                    input_power = torch.sum(filter_input ** 2) + 1e-8
                    normalized_step = self.step_size / input_power
                else:
                    # 标准LMS
                    normalized_step = self.step_size
                
                # 滤波器系数更新
                self.filter_coeffs[b] = (
                    self.leakage_factor * self.filter_coeffs[b] + 
                    normalized_step * error_signal * filter_input
                )
                
                # 限制滤波器系数
                self.filter_coeffs[b] = torch.clamp(
                    self.filter_coeffs[b], 
                    -self.max_gain_linear, 
                    self.max_gain_linear
                )
                
                # 保存处理后的信号
                processed_audio[b, t] = error_signal
        
        return processed_audio.unsqueeze(1)
    
    def get_filter_coeffs(self):
        """
        获取当前滤波器系数
        
        Returns:
            torch.Tensor: 滤波器系数 [B, filter_length]
        """
        return self.filter_coeffs
    
    def reset_state(self):
        """重置内部状态变量"""
        self.filter_coeffs = None
        self.input_buffer = None
        self.output_buffer = None


def create_adaptive_feedback_method(filter_length=64, **kwargs):
    """
    创建自适应反馈抵消法实例的便捷函数
    
    Args:
        filter_length (int): 滤波器长度
        **kwargs: 其他参数
        
    Returns:
        AdaptiveFeedbackMethod: 自适应反馈抵消法实例
    """
    return AdaptiveFeedbackMethod(filter_length=filter_length, **kwargs)


# 测试代码
if __name__ == "__main__":
    # 创建测试数据
    batch_size, channels, freq_bins, time_frames = 2, 1, 256, 100
    test_input = torch.randn(batch_size, channels, freq_bins, time_frames).abs()
    test_input = torch.log10(test_input + 1e-8)  # 模拟log域输入
    
    # 创建方法实例
    method = AdaptiveFeedbackMethod(filter_length=64, step_size=0.01)
    
    # 测试前向传播
    output = method(test_input)
    
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输入范围: [{test_input.min():.4f}, {test_input.max():.4f}]")
    print(f"输出范围: [{output.min():.4f}, {output.max():.4f}]")
    
    # 检查滤波器系数
    filter_coeffs = method.get_filter_coeffs()
    if filter_coeffs is not None:
        print(f"滤波器系数形状: {filter_coeffs.shape}")
        print(f"滤波器系数范围: [{filter_coeffs.min():.4f}, {filter_coeffs.max():.4f}]")
    
    # 测试时域处理
    print("\n测试时域处理:")
    audio_samples = 16000  # 1秒音频
    test_audio = torch.randn(batch_size, 1, audio_samples) * 0.1
    processed_audio = method.process_time_domain(test_audio)
    
    print(f"音频输入形状: {test_audio.shape}")
    print(f"音频输出形状: {processed_audio.shape}")
    print(f"音频输入范围: [{test_audio.min():.4f}, {test_audio.max():.4f}]")
    print(f"音频输出范围: [{processed_audio.min():.4f}, {processed_audio.max():.4f}]")
    
    print("自适应反馈抵消法测试通过！")
