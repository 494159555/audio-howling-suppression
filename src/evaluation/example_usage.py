'''
评估系统使用示例

演示如何使用音频啸叫抑制方法科学评估系统
'''

import torch
import numpy as np
from pathlib import Path

# 导入评估系统
from .test_runner import (
    run_comprehensive_evaluation,
    run_quick_evaluation,
    evaluate_all_methods,
    evaluate_traditional_methods
)
from .metrics import AudioMetrics, calculate_mos_score
from .visualizer import AudioVisualizer
from .comparator import MethodComparator


def example_basic_usage():
    """基础使用示例"""
    print("="*60)
    print("基础使用示例")
    print("="*60)
    
    # 1. 快速评估（仅测试传统方法，少量样本）
    print("\n1. 运行快速评估...")
    quick_results = run_quick_evaluation(
        methods=['frequency_shift', 'gain_suppression', 'adaptive_feedback'],
        num_samples=5
    )
    
    if quick_results:
        print("✅ 快速评估完成")
        print(f"结果: {quick_results}")
    else:
        print("❌ 快速评估失败")
    
    return quick_results


def example_comprehensive_evaluation():
    """全面评估示例"""
    print("\n" + "="*60)
    print("全面评估示例")
    print("="*60)
    
    # 2. 全面评估所有方法
    print("\n2. 运行全面评估...")
    try:
        comprehensive_results = run_comprehensive_evaluation(
            methods=['frequency_shift', 'gain_suppression', 'adaptive_feedback'],
            batch_size=2,  # 减少批大小以节省内存
            save_results=True,
            generate_visualizations=True
        )
        
        if comprehensive_results:
            print("✅ 全面评估完成")
            return comprehensive_results
        else:
            print("❌ 全面评估失败")
            return None
            
    except Exception as e:
        print(f"❌ 全面评估出错: {e}")
        return None


def example_manual_metrics():
    """手动计算指标示例"""
    print("\n" + "="*60)
    print("手动计算指标示例")
    print("="*60)
    
    # 创建模拟数据
    batch_size, freq_bins, time_frames = 2, 256, 128
    
    # 模拟音频数据
    clean_audio = torch.randn(batch_size, 1, freq_bins, time_frames)
    noisy_audio = clean_audio + 0.1 * torch.randn_like(clean_audio)
    enhanced_audio = clean_audio + 0.05 * torch.randn_like(clean_audio)
    
    # 计算指标
    metrics_calculator = AudioMetrics()
    
    # 计算各项指标
    snr_improvement = metrics_calculator.calculate_snr(clean_audio, enhanced_audio, noisy_audio)
    psnr = metrics_calculator.calculate_psnr(clean_audio, enhanced_audio)
    stoi = metrics_calculator.calculate_stoi(clean_audio, enhanced_audio)
    
    print(f"SNR改善: {snr_improvement:.2f} dB")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"STOI: {stoi:.3f}")
    
    # 计算MOS分数
    metrics = {
        'snr_improvement_db': snr_improvement,
        'psnr_db': psnr,
        'stoi_score': stoi,
        'howling_reduction_db': 5.0
    }
    
    mos_score = calculate_mos_score(metrics)
    print(f"MOS分数估算: {mos_score:.2f}")
    
    return metrics


def example_visualization():
    """可视化示例"""
    print("\n" + "="*60)
    print("可视化示例")
    print("="*60)
    
    # 创建模拟结果数据
    results_dict = {
        'frequency_shift': {
            'snr_improvement_db': 8.5,
            'psnr_db': 25.3,
            'stoi_score': 0.82,
            'howling_reduction_db': 6.2,
            'processing_time_ms': 15.5,
            'memory_usage_mb': 45.2
        },
        'gain_suppression': {
            'snr_improvement_db': 7.2,
            'psnr_db': 23.1,
            'stoi_score': 0.78,
            'howling_reduction_db': 8.5,
            'processing_time_ms': 8.3,
            'memory_usage_mb': 32.1
        },
        'adaptive_feedback': {
            'snr_improvement_db': 9.1,
            'psnr_db': 26.8,
            'stoi_score': 0.85,
            'howling_reduction_db': 7.8,
            'processing_time_ms': 25.7,
            'memory_usage_mb': 68.9
        }
    }
    
    # 创建可视化器
    visualizer = AudioVisualizer(save_dir="example_visualizations")
    
    try:
        # 生成各种图表
        visualizer.plot_metrics_comparison(results_dict)
        print("✅ 指标对比图生成完成")
        
        visualizer.plot_radar_chart(results_dict)
        print("✅ 雷达图生成完成")
        
        visualizer.plot_computational_comparison(results_dict)
        print("✅ 计算效率对比图生成完成")
        
        visualizer.generate_comprehensive_report(results_dict)
        print("✅ 综合报告生成完成")
        
    except Exception as e:
        print(f"❌ 可视化生成失败: {e}")
    
    return results_dict


def example_method_comparison():
    """方法对比示例"""
    print("\n" + "="*60)
    print("方法对比示例")
    print("="*60)
    
    # 模拟多个方法的结果
    results_dict = {
        'unet': {
            'snr_improvement_db': 12.5,
            'psnr_db': 32.1,
            'stoi_score': 0.91,
            'howling_reduction_db': 10.2,
            'processing_time_ms': 125.3,
            'memory_usage_mb': 245.6,
            'parameter_count': 1500000
        },
        'frequency_shift': {
            'snr_improvement_db': 8.5,
            'psnr_db': 25.3,
            'stoi_score': 0.82,
            'howling_reduction_db': 6.2,
            'processing_time_ms': 15.5,
            'memory_usage_mb': 45.2,
            'parameter_count': 1000
        },
        'gain_suppression': {
            'snr_improvement_db': 7.2,
            'psnr_db': 23.1,
            'stoi_score': 0.78,
            'howling_reduction_db': 8.5,
            'processing_time_ms': 8.3,
            'memory_usage_mb': 32.1,
            'parameter_count': 500
        },
        'adaptive_feedback': {
            'snr_improvement_db': 9.1,
            'psnr_db': 26.8,
            'stoi_score': 0.85,
            'howling_reduction_db': 7.8,
            'processing_time_ms': 25.7,
            'memory_usage_mb': 68.9,
            'parameter_count': 2000
        }
    }
    
    # 创建对比器
    comparator = MethodComparator()
    
    try:
        # 进行对比分析
        comparison_results = comparator.compare_methods(results_dict)
        
        print("✅ 方法对比分析完成")
        
        # 生成对比表格
        comparison_table = comparator.generate_comparison_table()
        print("\n对比表格:")
        print(comparison_table.to_string())
        
        # 保存对比报告
        comparator.save_comparison_report("example_comparison_report.json")
        print("✅ 对比报告已保存")
        
        return comparison_results
        
    except Exception as e:
        print(f"❌ 方法对比失败: {e}")
        return None


def example_custom_evaluation():
    """自定义评估示例"""
    print("\n" + "="*60)
    print("自定义评估示例")
    print("="*60)
    
    # 模拟自定义方法的结果
    custom_results = {
        'my_method': {
            'snr_improvement_db': 10.5,
            'psnr_db': 28.5,
            'stoi_score': 0.87,
            'howling_reduction_db': 9.2,
            'processing_time_ms': 45.6,
            'memory_usage_mb': 78.3,
            'parameter_count': 50000
        }
    }
    
    # 与基线方法对比
    baseline_results = {
        'baseline': {
            'snr_improvement_db': 5.2,
            'psnr_db': 20.1,
            'stoi_score': 0.72,
            'howling_reduction_db': 4.1,
            'processing_time_ms': 12.3,
            'memory_usage_mb': 25.6,
            'parameter_count': 100
        }
    }
    
    # 合并结果
    all_results = {**custom_results, **baseline_results}
    
    # 进行对比
    comparator = MethodComparator()
    comparison_results = comparator.compare_methods(all_results)
    
    print("✅ 自定义方法对比完成")
    
    # 输出推荐
    if comparison_results.get('recommendations'):
        rec = comparison_results['recommendations']
        print(f"\n推荐结果:")
        print(f"综合最佳: {rec.get('best_overall', '未确定')}")
        print(f"质量最佳: {rec.get('best_quality', '未确定')}")
        print(f"效率最高: {rec.get('most_efficient', '未确定')}")
    
    return comparison_results


def run_all_examples():
    """运行所有示例"""
    print("🚀 音频啸叫抑制方法科学评估系统 - 示例演示")
    print("="*80)
    
    examples = [
        ("基础使用", example_basic_usage),
        ("手动指标计算", example_manual_metrics),
        ("可视化演示", example_visualization),
        ("方法对比", example_method_comparison),
        ("自定义评估", example_custom_evaluation)
    ]
    
    results = {}
    
    for name, example_func in examples:
        try:
            print(f"\n📋 运行示例: {name}")
            result = example_func()
            results[name] = result
            print(f"✅ {name} 示例完成")
        except Exception as e:
            print(f"❌ {name} 示例失败: {e}")
            results[name] = None
    
    print("\n" + "="*80)
    print("🎉 所有示例运行完成！")
    print("="*80)
    
    return results


if __name__ == "__main__":
    # 运行所有示例
    run_all_examples()
    
    print("\n📖 更多使用方法:")
    print("1. 运行全面评估: from src.evaluation.test_runner import evaluate_all_methods; evaluate_all_methods()")
    print("2. 运行快速评估: from src.evaluation.test_runner import run_quick_evaluation; run_quick_evaluation()")
    print("3. 仅评估传统方法: from src.evaluation.test_runner import evaluate_traditional_methods; evaluate_traditional_methods()")
