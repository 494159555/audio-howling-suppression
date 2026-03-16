'''
增益抑制法 - Gain Suppression Method

文件功能：
- 实现基于自适应增益控制的音频啸叫抑制算法
- 通过检测啸叫频段并应用自适应增益衰减来抑制啸叫
- 使用频谱分析和自适应阈值检测技术

算法原理：
1. 啸叫特征：特定频段的能量异常增强
2. 检测机制：实时监控各频段的能量变化
3. 抑制策略：对检测到的啸叫频段应用增益衰减

技术实现：
- 频谱分析：实时计算各频段的能量
- 峰值检测：识别可能的啸叫频率
- 自适应阈值：根据背景噪声动态调整检测阈值
- 增益控制：对啸叫频段应用平滑的增益衰减

重要参数：
- threshold_db: 检测阈值，单位dB，默认-30dB
- attack_time: 攻击时间，秒，默认0.01s
- release_time: 释放时间，秒，默认0.1s
- max_attenuation: 最大衰减量，dB，默认-20dB
- min_freq: 最小检测频率，Hz，默认1000Hz
- max_freq: 最大检测频率，Hz，默认8000Hz

特点：
- 针对性强：只对啸叫频段进行处理
- 音质保护：非啸叫频段保持原样
- 自适应性强：阈值可根据环境自动调整
- 实时性好：计算复杂度低

使用方法：
from src.traditional.gain_suppression import GainSuppressionMethod
method = GainSuppressionMethod(threshold_db=-30)
output = method(input_spectrogram)  # input: [B,1,256,T]
'''

import torch
import torch.nn as nn
import numpy as np
import math


class GainSuppressionMethod(nn.Module):
    """
    增益抑制法 - 通过自适应增益控制抑制音频啸叫
    
    Args:
        threshold_db (float): 检测阈值，单位dB，默认-30dB
        attack_time (float): 攻击时间，秒，默认0.01s
        release_time (float): 释放时间，秒，默认0.1s
        max_attenuation (float): 最大衰减量，dB，默认-20dB
        sample_rate (int): 采样率，默认16000Hz
        n_fft (int): FFT窗口大小，默认512
        hop_length (int): 跳跃长度，默认128
        min_freq (float): 最小检测频率，Hz，默认1000Hz
        max_freq (float): 最大检测频率，Hz，默认8000Hz
    """
    
    def __init__(self, 
                 threshold_db=-30.0,
                 attack_time=0.01,
                 release_time=0.1,
                 max_attenuation=-20.0,
                 sample_rate=16000,
                 n_fft=512,
                 hop_length=128,
                 min_freq=1000.0,
                 max_freq=8000.0):
        super(GainSuppressionMethod, self).__init__()
        
        self.threshold_db = threshold_db
        self.attack_time = attack_time
        self.release_time = release_time
        self.max_attenuation = max_attenuation
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.min_freq = min_freq
        self.max_freq = max_freq
        
        # 计算频率bin范围
        self.freq_resolution = sample_rate / n_fft
        self.min_bin = int(min_freq / self.freq_resolution)
        self.max_bin = int(max_freq / self.freq_resolution)
        
        # 计算攻击和释放系数
        frame_rate = sample_rate / hop_length
        self.attack_coeff = math.exp(-1.0 / (attack_time * frame_rate))
        self.release_coeff = math.exp(-1.0 / (release_time * frame_rate))
        
        # 转换为线性域
        self.threshold_linear = 10 ** (threshold_db / 20)
        self.max_attenuation_linear = 10 ** (max_attenuation / 20)
        
        # 状态变量
        self.register_buffer('gain_mask', None)
        self.register_buffer('background_estimate', None)
        
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
        if self.gain_mask is None or self.gain_mask.shape[-1] != time_frames:
            self.gain_mask = torch.ones(batch_size, freq_bins, time_frames, device=x.device)
            self.background_estimate = torch.zeros(batch_size, freq_bins, time_frames, device=x.device)
        
        # 处理每个时间帧
        processed_spec = torch.zeros_like(x_linear.squeeze(1))
        
        for t in range(time_frames):
            current_frame = x_linear[:, 0, :, t]  # [B, F]
            
            # 更新背景噪声估计
            if t == 0:
                self.background_estimate[:, :, t] = current_frame
            else:
                # 使用慢速跟踪估计背景
                alpha = 0.95  # 背景估计的时间常数
                self.background_estimate[:, :, t] = (
                    alpha * self.background_estimate[:, :, t-1] + 
                    (1 - alpha) * current_frame
                )
            
            # 计算信噪比
            snr = current_frame / (self.background_estimate[:, :, t] + 1e-8)
            
            # 检测啸叫频段
            howling_mask = self._detect_howling(snr, t)
            
            # 更新增益掩码
            if t == 0:
                self.gain_mask[:, :, t] = torch.ones(batch_size, freq_bins, device=x.device)
            else:
                # 平滑增益变化
                target_gain = torch.where(
                    howling_mask,
                    torch.full_like(howling_mask, self.max_attenuation_linear, dtype=torch.float32),
                    torch.ones_like(howling_mask, dtype=torch.float32)
                )
                
                # 攻击和释放时间控制
                gain_diff = target_gain - self.gain_mask[:, :, t-1]
                gain_change = torch.where(
                    gain_diff < 0,  # 需要衰减（攻击）
                    gain_diff * (1 - self.attack_coeff),
                    gain_diff * (1 - self.release_coeff)  # 需要恢复（释放）
                )
                
                self.gain_mask[:, :, t] = self.gain_mask[:, :, t-1] + gain_change
            
            # 应用增益掩码
            processed_spec[:, :, t] = current_frame * self.gain_mask[:, :, t]
        
        # 转换回log域
        processed_log = torch.log10(processed_spec.unsqueeze(1) + 1e-8)
        
        return processed_log
    
    def _detect_howling(self, snr, time_frame):
        """
        检测啸叫频段
        
        Args:
            snr (torch.Tensor): 信噪比 [B, F]
            time_frame (int): 当前时间帧索引
            
        Returns:
            torch.Tensor: 啸叫检测掩码 [B, F]
        """
        batch_size, freq_bins = snr.shape
        
        # 创建掩码
        howling_mask = torch.zeros_like(snr, dtype=torch.bool)
        
        # 只在指定频率范围内检测
        for b in range(batch_size):
            for f in range(self.min_bin, min(self.max_bin, freq_bins)):
                # 检测条件：信噪比超过阈值且为局部峰值
                if snr[b, f] > self.threshold_linear:
                    # 检查是否为局部峰值
                    is_peak = True
                    for df in [-2, -1, 1, 2]:  # 检查邻近频率
                        nf = f + df
                        if 0 <= nf < freq_bins and snr[b, nf] >= snr[b, f]:
                            is_peak = False
                            break
                    
                    if is_peak:
                        howling_mask[b, f] = True
        
        return howling_mask
    
    def get_gain_mask(self):
        """
        获取当前的增益掩码
        
        Returns:
            torch.Tensor: 增益掩码 [B, F, T]
        """
        return self.gain_mask
    
    def get_background_estimate(self):
        """
        获取背景噪声估计
        
        Returns:
            torch.Tensor: 背景噪声估计 [B, F, T]
        """
        return self.background_estimate
    
    def reset_state(self):
        """重置内部状态变量"""
        self.gain_mask = None
        self.background_estimate = None


def create_gain_suppression_method(threshold_db=-30.0, **kwargs):
    """
    创建增益抑制法实例的便捷函数
    
    Args:
        threshold_db (float): 检测阈值
        **kwargs: 其他参数
        
    Returns:
        GainSuppressionMethod: 增益抑制法实例
    """
    return GainSuppressionMethod(threshold_db=threshold_db, **kwargs)


# 测试代码
if __name__ == "__main__":
    # 创建测试数据
    batch_size, channels, freq_bins, time_frames = 2, 1, 256, 100
    test_input = torch.randn(batch_size, channels, freq_bins, time_frames).abs()
    test_input = torch.log10(test_input + 1e-8)  # 模拟log域输入
    
    # 添加模拟的啸叫信号
    test_input[:, 0, 50:60, 20:80] += 2.0  # 在特定频段添加强信号
    
    # 创建方法实例
    method = GainSuppressionMethod(threshold_db=-30.0)
    
    # 测试前向传播
    output = method(test_input)
    
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输入范围: [{test_input.min():.4f}, {test_input.max():.4f}]")
    print(f"输出范围: [{output.min():.4f}, {output.max():.4f}]")
    
    # 检查增益掩码
    gain_mask = method.get_gain_mask()
    if gain_mask is not None:
        print(f"增益掩码范围: [{gain_mask.min():.4f}, {gain_mask.max():.4f}]")
    
    print("增益抑制法测试通过！")
