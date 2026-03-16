'''
音频评估指标模块

实现各种音频质量评估指标，包括：
- 客观指标：SNR、PSNR、STOI、PESQ等
- 啸叫抑制专用指标
- 计算效率指标
'''

import torch
import numpy as np
import torchaudio
from typing import Dict, List, Tuple, Optional
import time
import psutil
import os


class AudioMetrics:
    """音频质量评估指标计算类"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.eps = 1e-8
        
    def calculate_snr(self, clean: torch.Tensor, enhanced: torch.Tensor, 
                     noisy: torch.Tensor) -> float:
        """
        计算信噪比改善 (SNR Improvement)
        
        Args:
            clean: 纯净音频 [B, 1, T]
            enhanced: 处理后音频 [B, 1, T] 
            noisy: 带噪音频 [B, 1, T]
            
        Returns:
            SNR改善值 (dB)
        """
        # 计算原始SNR
        snr_input = 10 * torch.log10(
            torch.mean(clean ** 2, dim=-1) / (torch.mean((noisy - clean) ** 2, dim=-1) + self.eps)
        )
        
        # 计算处理后SNR
        snr_output = 10 * torch.log10(
            torch.mean(clean ** 2, dim=-1) / (torch.mean((enhanced - clean) ** 2, dim=-1) + self.eps)
        )
        
        # SNR改善
        snr_improvement = snr_output - snr_input
        
        # 确保返回标量值
        result = snr_improvement.mean()
        if hasattr(result, 'item'):
            return result.item()
        else:
            return float(result)
    
    def calculate_psnr(self, clean: torch.Tensor, enhanced: torch.Tensor) -> float:
        """
        计算峰值信噪比 (PSNR)
        
        Args:
            clean: 纯净音频
            enhanced: 处理后音频
            
        Returns:
            PSNR值 (dB)
        """
        mse = torch.mean((clean - enhanced) ** 2)
        if mse == 0:
            return float('inf')
            
        max_val = torch.max(clean)
        psnr = 20 * torch.log10(max_val / torch.sqrt(mse + self.eps))
        
        # 确保返回标量值
        if hasattr(psnr, 'item'):
            return psnr.item()
        else:
            return float(psnr)
    
    def calculate_stoi(self, clean: torch.Tensor, enhanced: torch.Tensor) -> float:
        """
        计算短时客观可懂度 (STOI)
        简化版本，实际应用中建议使用pystoi库
        
        Args:
            clean: 纯净音频
            enhanced: 处理后音频
            
        Returns:
            STOI值 (0-1)
        """
        # 简化的STOI计算
        # 实际建议使用: from pystoi import stoi
        
        clean_np = clean.detach().cpu().numpy()
        enhanced_np = enhanced.detach().cpu().numpy()
        
        # 计算相关性作为简化指标
        if clean_np.ndim > 1:
            clean_np = clean_np.flatten()
            enhanced_np = enhanced_np.flatten()
            
        correlation = np.corrcoef(clean_np, enhanced_np)[0, 1]
        
        # 确保在0-1范围内
        stoi_score = max(0, min(1, (correlation + 1) / 2))
        
        return float(stoi_score)
    
    def calculate_howling_reduction(self, noisy: torch.Tensor, enhanced: torch.Tensor,
                                  freq_bins: List[int] = None) -> Dict[str, float]:
        """
        计算啸叫抑制效果
        
        Args:
            noisy: 带噪音频频谱
            enhanced: 处理后音频频谱
            freq_bins: 啸叫频段列表
            
        Returns:
            啸叫抑制指标字典
        """
        if freq_bins is None:
            # 默认高频段作为啸叫频段
            freq_bins = list(range(200, 256))  # 假设512点FFT，256个频bin
            
        # 计算频域能量
        noisy_power = torch.mean(noisy[..., freq_bins, :] ** 2, dim=(-2, -1))
        enhanced_power = torch.mean(enhanced[..., freq_bins, :] ** 2, dim=(-2, -1))
        
        # 啸叫衰减率
        reduction_ratio = 1 - (enhanced_power / (noisy_power + self.eps))
        
        # 确保reduction_ratio在合理范围内
        reduction_ratio = torch.clamp(reduction_ratio, min=0, max=1)
        
        # 频谱平滑度 (简化计算)
        noisy_smoothness = self._calculate_spectral_smoothness(noisy)
        enhanced_smoothness = self._calculate_spectral_smoothness(enhanced)
        
        # 确保返回标量值
        howling_reduction_db = (10 * torch.log10(reduction_ratio + self.eps)).mean()
        spectral_smoothness_improvement = enhanced_smoothness - noisy_smoothness
        high_frequency_reduction = reduction_ratio.mean()
        
        return {
            'howling_reduction_db': howling_reduction_db.item() if hasattr(howling_reduction_db, 'item') else float(howling_reduction_db),
            'spectral_smoothness_improvement': spectral_smoothness_improvement.item() if hasattr(spectral_smoothness_improvement, 'item') else float(spectral_smoothness_improvement),
            'high_frequency_reduction': high_frequency_reduction.item() if hasattr(high_frequency_reduction, 'item') else float(high_frequency_reduction)
        }
    
    def _calculate_spectral_smoothness(self, spectrum: torch.Tensor) -> float:
        """计算频谱平滑度"""
        # 计算相邻频bin的差异
        diff = torch.diff(spectrum, dim=-2)
        smoothness = 1 / (torch.mean(diff ** 2) + self.eps)
        result = smoothness
        if hasattr(result, 'item'):
            return result.item()
        else:
            return float(result)
    
    def calculate_computational_metrics(self, method_name: str, input_data: torch.Tensor,
                                      processing_func, **kwargs) -> Dict[str, float]:
        """
        计算计算效率指标
        
        Args:
            method_name: 方法名称
            input_data: 输入数据
            processing_func: 处理函数
            **kwargs: 处理函数参数
            
        Returns:
            计算效率指标字典
        """
        # 内存使用
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # 处理时间
        start_time = time.time()
        
        with torch.no_grad():
            output = processing_func(input_data, **kwargs)
            
        processing_time = time.time() - start_time
        
        # 内存使用后
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = memory_after - memory_before
        
        # 参数量 (如果是模型)
        param_count = 0
        if hasattr(processing_func, 'parameters'):
            param_count = sum(p.numel() for p in processing_func.parameters())
        
        return {
            'processing_time_ms': processing_time * 1000,
            'memory_usage_mb': memory_usage,
            'parameter_count': param_count,
            'throughput_samples_per_sec': input_data.shape[0] / processing_time if processing_time > 0 else 0
        }
    
    def calculate_all_metrics(self, clean: torch.Tensor, noisy: torch.Tensor, 
                            enhanced: torch.Tensor, method_name: str = "unknown",
                            processing_func=None, **kwargs) -> Dict[str, float]:
        """
        计算所有评估指标
        
        Args:
            clean: 纯净音频
            noisy: 带噪音频  
            enhanced: 处理后音频
            method_name: 方法名称
            processing_func: 处理函数(用于计算效率指标)
            **kwargs: 处理函数参数
            
        Returns:
            所有指标的字典
        """
        metrics = {}
        
        # 音频质量指标
        metrics['snr_improvement_db'] = self.calculate_snr(clean, enhanced, noisy)
        metrics['psnr_db'] = self.calculate_psnr(clean, enhanced)
        metrics['stoi_score'] = self.calculate_stoi(clean, enhanced)
        
        # 啸叫抑制指标
        howling_metrics = self.calculate_howling_reduction(noisy, enhanced)
        metrics.update(howling_metrics)
        
        # 计算效率指标
        if processing_func is not None:
            comp_metrics = self.calculate_computational_metrics(
                method_name, noisy, processing_func, **kwargs
            )
            metrics.update(comp_metrics)
        
        return metrics


def calculate_mos_score(metrics: Dict[str, float]) -> float:
    """
    基于客观指标估算MOS分数
    
    Args:
        metrics: 客观指标字典
        
    Returns:
        估算的MOS分数 (1-5)
    """
    # 简化的MOS估算公式
    # 实际应用中建议使用主观评估数据训练回归模型
    
    snr_weight = 0.3
    stoi_weight = 0.4
    psnr_weight = 0.2
    howling_weight = 0.1
    
    # 归一化指标到0-1范围
    snr_norm = min(1.0, max(0.0, metrics.get('snr_improvement_db', 0) / 20))
    stoi_norm = metrics.get('stoi_score', 0)
    psnr_norm = min(1.0, max(0.0, metrics.get('psnr_db', 0) / 40))
    howling_norm = min(1.0, max(0.0, metrics.get('howling_reduction_db', 0) / 10))
    
    # 加权平均
    quality_score = (snr_weight * snr_norm + 
                    stoi_weight * stoi_norm + 
                    psnr_weight * psnr_norm + 
                    howling_weight * howling_norm)
    
    # 映射到MOS 1-5分
    mos_score = 1 + quality_score * 4
    
    return mos_score
