#!/usr/bin/env python3
"""
简化的评估测试脚本
"""

import sys
import torch
import numpy as np
from pathlib import Path

# 添加src到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "src"))

from src.evaluation.metrics import AudioMetrics
from src.traditional.frequency_shift import FrequencyShiftMethod
from src.traditional.gain_suppression import GainSuppressionMethod
from src.traditional.adaptive_feedback import AdaptiveFeedbackMethod

def test_traditional_methods():
    """测试传统方法的评估"""
    print("="*60)
    print("测试传统方法评估")
    print("="*60)
    
    # 创建评估器
    metrics = AudioMetrics()
    
    # 创建传统方法
    methods = {
        'frequency_shift': FrequencyShiftMethod(),
        'gain_suppression': GainSuppressionMethod(),
        'adaptive_feedback': AdaptiveFeedbackMethod()
    }
    
    # 创建测试数据
    batch_size = 2
    freq_bins = 256
    time_frames = 100
    
    # 模拟log域频谱图数据
    clean_spectrogram = torch.randn(batch_size, 1, freq_bins, time_frames).abs()
    clean_spectrogram = torch.log10(clean_spectrogram + 1e-8)
    
    # 添加啸叫（高频增强）
    noisy_spectrogram = clean_spectrogram.clone()
    # 在高频段添加能量模拟啸叫
    noisy_spectrogram[:, :, 200:, :] += 0.5
    
    print(f"测试数据形状: {clean_spectrogram.shape}")
    print(f"纯净信号范围: [{clean_spectrogram.min():.3f}, {clean_spectrogram.max():.3f}]")
    print(f"带噪信号范围: [{noisy_spectrogram.min():.3f}, {noisy_spectrogram.max():.3f}]")
    
    # 测试每个方法
    results = {}
    
    for method_name, method in methods.items():
        print(f"\n测试 {method_name}...")
        
        try:
            # 处理信号
            enhanced = method(noisy_spectrogram)
            
            # 计算指标
            method_metrics = {}
            
            # SNR改善
            snr_improvement = metrics.calculate_snr(clean_spectrogram, enhanced, noisy_spectrogram)
            method_metrics['snr_improvement_db'] = snr_improvement
            
            # PSNR
            psnr = metrics.calculate_psnr(clean_spectrogram, enhanced)
            method_metrics['psnr_db'] = psnr
            
            # STOI（简化版本）
            stoi = metrics.calculate_stoi(clean_spectrogram, enhanced)
            method_metrics['stoi_score'] = stoi
            
            # 啸叫抑制指标
            howling_metrics = metrics.calculate_howling_reduction(noisy_spectrogram, enhanced)
            method_metrics.update(howling_metrics)
            
            # 计算效率指标
            comp_metrics = metrics.calculate_computational_metrics(
                method_name, noisy_spectrogram, method
            )
            method_metrics.update(comp_metrics)
            
            results[method_name] = method_metrics
            
            print(f"  SNR改善: {snr_improvement:.2f} dB")
            print(f"  PSNR: {psnr:.2f} dB")
            print(f"  STOI: {stoi:.3f}")
            print(f"  啸叫衰减: {howling_metrics['howling_reduction_db']:.2f} dB")
            print(f"  处理时间: {comp_metrics['processing_time_ms']:.2f} ms")
            print(f"  内存使用: {comp_metrics['memory_usage_mb']:.2f} MB")
            
        except Exception as e:
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()
    
    # 输出对比结果
    print("\n" + "="*60)
    print("方法对比结果")
    print("="*60)
    
    metrics_to_compare = ['snr_improvement_db', 'psnr_db', 'stoi_score', 
                         'howling_reduction_db', 'processing_time_ms']
    
    for metric in metrics_to_compare:
        print(f"\n{metric}:")
        for method_name, method_results in results.items():
            if metric in method_results:
                value = method_results[metric]
                print(f"  {method_name}: {value:.3f}")
    
    # 找出最佳方法
    print("\n" + "="*60)
    print("最佳方法推荐")
    print("="*60)
    
    best_methods = {}
    
    # SNR改善最佳
    best_snr = max(results.items(), key=lambda x: x[1].get('snr_improvement_db', -float('inf')))
    best_methods['SNR改善'] = best_snr[0]
    
    # 啸叫抑制最佳
    best_howling = max(results.items(), key=lambda x: x[1].get('howling_reduction_db', -float('inf')))
    best_methods['啸叫抑制'] = best_howling[0]
    
    # 速度最佳
    fastest = min(results.items(), key=lambda x: x[1].get('processing_time_ms', float('inf')))
    best_methods['处理速度'] = fastest[0]
    
    for category, method in best_methods.items():
        print(f"  {category}: {method}")
    
    return results

if __name__ == "__main__":
    try:
        results = test_traditional_methods()
        print("\n评估测试完成！")
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
