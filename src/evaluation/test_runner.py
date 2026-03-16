'''
统一测试运行器模块

提供一键运行全面评估的功能，包括：
- 自动加载和测试所有方法
- 生成可视化报告
- 输出对比结果
- 保存评估报告
'''

import torch
import numpy as np
from pathlib import Path
import time
from typing import Dict, List, Optional

from .benchmark import BenchmarkRunner
from .comparator import MethodComparator
from .visualizer import AudioVisualizer
from .metrics import AudioMetrics


def run_comprehensive_evaluation(
    methods: List[str] = None,
    test_data_dir: str = None,
    save_results: bool = True,
    generate_visualizations: bool = True,
    batch_size: int = 4
) -> Dict:
    """
    运行全面的评估流程
    
    Args:
        methods: 要评估的方法列表，None表示评估所有方法
        test_data_dir: 测试数据目录
        save_results: 是否保存结果
        generate_visualizations: 是否生成可视化
        batch_size: 批大小
        
    Returns:
        完整的评估结果
    """
    print("="*80)
    print("音频啸叫抑制方法科学评估系统")
    print("="*80)
    
    # 1. 初始化组件
    print("\n1. 初始化评估组件...")
    benchmark_runner = BenchmarkRunner(test_data_dir=test_data_dir, batch_size=batch_size)
    comparator = MethodComparator()
    visualizer = AudioVisualizer()
    
    # 2. 运行基准测试
    print("\n2. 运行基准测试...")
    benchmark_results = benchmark_runner.run_comprehensive_benchmark(methods)
    
    if not benchmark_results.get('test_summary', {}).get('methods_tested'):
        print("❌ 没有成功测试任何方法")
        return {}
    
    print(f"✅ 成功测试 {len(benchmark_results['test_summary']['methods_tested'])} 个方法")
    
    # 3. 方法对比分析
    print("\n3. 进行方法对比分析...")
    
    # 提取基准测试的原始结果
    raw_results = {}
    for method_name in benchmark_results['test_summary']['methods_tested']:
        # 这里需要从benchmark_runner获取原始结果
        if hasattr(benchmark_runner, 'results') and method_name in benchmark_runner.results:
            raw_results[method_name] = benchmark_runner.results[method_name]
    
    if raw_results:
        comparison_results = comparator.compare_methods(raw_results)
        print("✅ 对比分析完成")
    else:
        print("⚠️ 无法进行对比分析（缺少原始结果）")
        comparison_results = {}
    
    # 4. 生成可视化报告
    if generate_visualizations and raw_results:
        print("\n4. 生成可视化报告...")
        
        try:
            # 生成各种对比图
            visualizer.plot_metrics_comparison(raw_results)
            visualizer.plot_radar_chart(raw_results)
            visualizer.plot_computational_comparison(raw_results)
            visualizer.generate_comprehensive_report(raw_results)
            
            print("✅ 可视化报告生成完成")
        except Exception as e:
            print(f"⚠️ 可视化生成失败: {e}")
    
    # 5. 保存结果
    if save_results:
        print("\n5. 保存评估结果...")
        
        try:
            # 保存对比结果
            if comparison_results:
                comparator.save_comparison_report("evaluation_results/comparison_report.json")
            
            print("✅ 结果保存完成")
        except Exception as e:
            print(f"⚠️ 结果保存失败: {e}")
    
    # 6. 输出总结报告
    print("\n6. 生成总结报告...")
    summary = generate_summary_report(benchmark_results, comparison_results, raw_results)
    
    print("\n" + "="*80)
    print("评估完成！")
    print("="*80)
    
    return {
        'benchmark_results': benchmark_results,
        'comparison_results': comparison_results,
        'raw_results': raw_results,
        'summary': summary
    }


def run_quick_evaluation(
    methods: List[str] = None,
    num_samples: int = 10
) -> Dict:
    """
    运行快速评估（仅测试少量样本）
    
    Args:
        methods: 要评估的方法列表
        num_samples: 测试样本数量
        
    Returns:
        快速评估结果
    """
    print("="*60)
    print("快速评估模式")
    print("="*60)
    
    benchmark_runner = BenchmarkRunner()
    
    # 运行快速测试
    quick_results = benchmark_runner.run_quick_test(num_samples)
    
    if not quick_results:
        print("❌ 快速评估失败")
        return {}
    
    # 输出结果表格
    print(benchmark_runner.get_method_comparison_table())
    
    return {
        'quick_results': quick_results,
        'comparison_table': benchmark_runner.get_method_comparison_table()
    }


def run_method_comparison(
    results_dict: Dict[str, Dict[str, float]],
    save_report: bool = True
) -> Dict:
    """
    仅运行方法对比分析（基于已有结果）
    
    Args:
        results_dict: 方法结果字典
        save_report: 是否保存报告
        
    Returns:
        对比分析结果
    """
    print("="*60)
    print("方法对比分析")
    print("="*60)
    
    comparator = MethodComparator()
    comparison_results = comparator.compare_methods(results_dict)
    
    # 生成可视化
    visualizer = AudioVisualizer()
    visualizer.plot_metrics_comparison(results_dict)
    visualizer.plot_radar_chart(results_dict)
    visualizer.generate_comprehensive_report(results_dict)
    
    # 保存报告
    if save_report:
        comparator.save_comparison_report("evaluation_results/method_comparison.json")
    
    # 输出对比表格
    comparison_table = comparator.generate_comparison_table()
    print("\n方法对比表格:")
    print(comparison_table.to_string())
    
    return comparison_results


def generate_summary_report(
    benchmark_results: Dict,
    comparison_results: Dict,
    raw_results: Dict
) -> Dict:
    """生成总结报告"""
    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'key_findings': [],
        'recommendations': {},
        'performance_ranking': {},
        'best_practices': []
    }
    
    # 关键发现
    if benchmark_results.get('test_summary', {}).get('methods_tested'):
        methods_tested = benchmark_results['test_summary']['methods_tested']
        summary['key_findings'].append(f"成功评估了 {len(methods_tested)} 种方法: {', '.join(methods_tested)}")
    
    if comparison_results.get('summary'):
        comp_summary = comparison_results['summary']
        if 'overall_winner' in comp_summary:
            summary['key_findings'].append(f"综合最佳方法: {comp_summary['overall_winner']}")
        
        if 'key_findings' in comp_summary:
            summary['key_findings'].extend(comp_summary['key_findings'])
    
    # 推荐
    if comparison_results.get('recommendations'):
        rec = comparison_results['recommendations']
        summary['recommendations'] = {
            '综合最佳': rec.get('best_overall', '未确定'),
            '质量最佳': rec.get('best_quality', '未确定'),
            '效率最高': rec.get('most_efficient', '未确定'),
            '实时应用': rec.get('best_for_realtime', '未确定'),
            '高质量应用': rec.get('best_for_high_quality', '未确定')
        }
    
    # 性能排名
    if raw_results:
        from .metrics import calculate_mos_score
        
        method_scores = {}
        for method, metrics in raw_results.items():
            mos_score = calculate_mos_score(metrics)
            processing_time = metrics.get('processing_time_ms', 1000)
            
            method_scores[method] = {
                'mos_score': mos_score,
                'processing_time_ms': processing_time,
                'snr_improvement': metrics.get('snr_improvement_db', 0),
                'howling_reduction': metrics.get('howling_reduction_db', 0)
            }
        
        # 按MOS分数排序
        sorted_methods = sorted(method_scores.items(), key=lambda x: x[1]['mos_score'], reverse=True)
        summary['performance_ranking'] = {
            method: rank + 1 for rank, (method, _) in enumerate(sorted_methods)
        }
    
    # 最佳实践建议
    summary['best_practices'] = [
        "根据具体应用场景选择合适的方法",
        "实时应用优先考虑处理时间",
        "高质量应用优先考虑音质指标",
        "资源受限环境优先考虑内存和计算效率",
        "建议在实际部署前进行充分测试"
    ]
    
    # 输出总结
    print("\n" + "="*60)
    print("评估总结报告")
    print("="*60)
    
    print("\n🔍 关键发现:")
    for finding in summary['key_findings']:
        print(f"  • {finding}")
    
    print("\n🏆 推荐方法:")
    for category, method in summary['recommendations'].items():
        print(f"  • {category}: {method}")
    
    if summary['performance_ranking']:
        print("\n📊 性能排名 (按MOS分数):")
        sorted_ranking = sorted(summary['performance_ranking'].items(), key=lambda x: x[1])
        for method, rank in sorted_ranking:
            print(f"  {rank}. {method}")
    
    print("\n💡 最佳实践建议:")
    for practice in summary['best_practices']:
        print(f"  • {practice}")
    
    print("\n" + "="*60)
    
    return summary


def run_evaluation_pipeline(
    evaluation_type: str = "comprehensive",
    methods: List[str] = None,
    **kwargs
) -> Dict:
    """
    运行评估流水线
    
    Args:
        evaluation_type: 评估类型 ("comprehensive", "quick", "comparison")
        methods: 要评估的方法列表
        **kwargs: 其他参数
        
    Returns:
        评估结果
    """
    if evaluation_type == "comprehensive":
        return run_comprehensive_evaluation(methods, **kwargs)
    elif evaluation_type == "quick":
        return run_quick_evaluation(methods, **kwargs)
    elif evaluation_type == "comparison":
        if 'results_dict' not in kwargs:
            raise ValueError("comparison模式需要提供results_dict参数")
        return run_method_comparison(kwargs['results_dict'], **kwargs)
    else:
        raise ValueError(f"未知的评估类型: {evaluation_type}")


# 便捷函数
def evaluate_all_methods():
    """评估所有可用方法"""
    return run_comprehensive_evaluation()


def evaluate_traditional_methods():
    """仅评估传统方法"""
    return run_comprehensive_evaluation(
        methods=['frequency_shift', 'gain_suppression', 'adaptive_feedback']
    )


def compare_with_baseline(baseline_results: Dict):
    """与基线方法对比"""
    return run_method_comparison(baseline_results)


if __name__ == "__main__":
    # 示例用法
    print("音频啸叫抑制方法科学评估系统")
    print("使用方法:")
    print("1. run_comprehensive_evaluation() - 全面评估")
    print("2. run_quick_evaluation() - 快速评估")
    print("3. run_method_comparison(results_dict) - 方法对比")
    print("\n示例:")
    print("python -c \"from src.evaluation.test_runner import evaluate_all_methods; evaluate_all_methods()\"")
