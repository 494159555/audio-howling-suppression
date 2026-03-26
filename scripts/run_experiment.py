#!/usr/bin/env python3
"""音频啸叫抑制方法对比实验脚本

本脚本用于运行科学评估实验，对比不同音频啸叫抑制方法的性能。

主要功能：
    - 支持多种评估模式（快速、全面、传统方法、自定义）
    - 对比深度学习模型和传统信号处理方法
    - 生成详细的评估报告和可视化图表
    - 提供方法推荐和性能分析

可评估的方法：
    深度学习模型：
    - unet_v1 ~ unet_v13: 各代U-Net变体
    - cnn, rnn: 基线模型

    传统方法：
    - frequency_shift: 频率移位法
    - gain_suppression: 增益抑制法
    - adaptive_feedback: 自适应反馈抑制法

使用方法：
    # 快速评估（默认10个样本）
    python scripts/run_experiment.py --mode quick

    # 全面评估（整个测试集）
    python scripts/run_experiment.py --mode comprehensive

    # 仅评估传统方法
    python scripts/run_experiment.py --mode traditional

    # 自定义方法组合
    python scripts/run_experiment.py --mode custom --methods unet_v2 frequency_shift

输出结果：
    - 评估结果保存在 evaluation_results/ 目录
    - 可视化图表包括：对比柱状图、趋势图、热力图等
    - 生成Markdown格式的评估报告

作者：音频处理实验室
版本：1.0
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# 添加src目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation.test_runner import (
    run_quick_evaluation,
    run_comprehensive_evaluation,
    evaluate_traditional_methods,
    run_evaluation_pipeline
)


def print_experiment_header(args) -> None:
    """打印实验配置信息

    Args:
        args: 命令行参数对象
    """
    print("="*80)
    print("🔬 音频啸叫抑制方法科学评估实验")
    print("="*80)
    print(f"\n📋 实验配置:")
    print(f"   实验模式:        {args.mode}")
    print(f"   批大小:          {args.batch_size}")
    print(f"   保存结果:        {'是' if args.save_results else '否'}")
    print(f"   生成可视化:      {'是' if args.generate_visualizations else '否'}")

    if args.mode == "quick":
        print(f"   评估样本数:      {args.num_samples}")

    if args.methods:
        print(f"   评估方法:")
        for method in args.methods:
            print(f"      • {method}")

    print("\n" + "-"*80)


def run_quick_mode(args) -> Optional[dict]:
    """运行快速评估模式

    快速模式仅评估少量样本，用于快速验证和初步对比。

    Args:
        args: 命令行参数对象

    Returns:
        评估结果字典，如果失败返回None
    """
    print(f"🚀 启动快速评估模式")
    print(f"   说明: 使用少量样本快速验证方法性能")
    print(f"   样本数: {args.num_samples}")

    if not args.methods:
        print("\n⚠️  未指定方法，将使用默认方法进行评估")
        print("   提示: 使用 --methods 参数指定要评估的方法")
        print("   示例: --methods unet_v2 frequency_shift gain_suppression\n")

    try:
        results = run_quick_evaluation(
            methods=args.methods,
            num_samples=args.num_samples
        )
        return results
    except Exception as e:
        print(f"\n❌ 快速评估失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_comprehensive_mode(args) -> Optional[dict]:
    """运行全面评估模式

    全面模式评估整个测试集，生成详细的报告和可视化。

    Args:
        args: 命令行参数对象

    Returns:
        评估结果字典，如果失败返回None
    """
    print(f"🚀 启动全面评估模式")
    print(f"   说明: 对整个测试集进行全面评估")
    print(f"   批大小: {args.batch_size}")
    print(f"   将生成详细报告和可视化图表\n")

    if not args.methods:
        print("⚠️  未指定方法，将评估所有可用方法\n")

    try:
        results = run_comprehensive_evaluation(
            methods=args.methods,
            batch_size=args.batch_size,
            save_results=args.save_results,
            generate_visualizations=args.generate_visualizations
        )
        return results
    except Exception as e:
        print(f"\n❌ 全面评估失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_traditional_mode(args) -> Optional[dict]:
    """运行传统方法评估模式

    仅评估传统信号处理方法，不包括深度学习模型。

    Args:
        args: 命令行参数对象

    Returns:
        评估结果字典，如果失败返回None
    """
    print(f"🚀 启动传统方法评估模式")
    print(f"   说明: 仅评估传统信号处理方法")
    print(f"   评估方法:")
    print(f"      • frequency_shift    - 频率移位法")
    print(f"      • gain_suppression   - 增益抑制法")
    print(f"      • adaptive_feedback  - 自适应反馈抑制法\n")

    try:
        results = run_comprehensive_evaluation(
            methods=['frequency_shift', 'gain_suppression', 'adaptive_feedback'],
            batch_size=args.batch_size,
            save_results=args.save_results,
            generate_visualizations=args.generate_visualizations
        )
        return results
    except Exception as e:
        print(f"\n❌ 传统方法评估失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_custom_mode(args) -> Optional[dict]:
    """运行自定义方法评估模式

    使用用户指定的方法列表进行评估。

    Args:
        args: 命令行参数对象

    Returns:
        评估结果字典，如果失败返回None
    """
    if not args.methods:
        print("❌ 错误: 自定义模式必须使用 --methods 参数指定评估方法")
        print("\n💡 使用示例:")
        print('   python scripts/run_experiment.py --mode custom --methods unet_v2 frequency_shift')
        print('\n可用方法:')
        print('   深度学习: unet_v1 ~ unet_v13, cnn, rnn')
        print('   传统方法: frequency_shift, gain_suppression, adaptive_feedback')
        return None

    print(f"🚀 启动自定义方法评估模式")
    print(f"   评估方法: {args.methods}\n")

    try:
        results = run_comprehensive_evaluation(
            methods=args.methods,
            batch_size=args.batch_size,
            save_results=args.save_results,
            generate_visualizations=args.generate_visualizations
        )
        return results
    except Exception as e:
        print(f"\n❌ 自定义评估失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_experiment_summary(results: Optional[dict]) -> None:
    """打印实验结果摘要

    Args:
        results: 评估结果字典
    """
    print("\n" + "="*80)
    print("📊 实验结果摘要")
    print("="*80 + "\n")

    if results is None:
        print("❌ 实验失败，无法生成结果摘要")
        return

    if 'summary' in results:
        summary = results['summary']

        # 打印推荐结果
        if 'recommendations' in summary:
            print("💡 方法推荐:")
            for category, method in summary['recommendations'].items():
                print(f"   • {category}: {method}")
            print()

        # 打印最佳方法
        if 'best_method' in summary:
            print(f"🏆 最佳方法: {summary['best_method']}")
            if 'best_score' in summary:
                print(f"   得分: {summary['best_score']:.4f}")
            print()

    # 打印结果保存位置
    print("📁 结果文件位置:")
    if 'results_dir' in results:
        print(f"   结果目录: {results['results_dir']}")
    if 'report_path' in results:
        print(f"   评估报告: {results['report_path']}")
    if 'visualization_dir' in results:
        print(f"   可视化图表: {results['visualization_dir']}")

    print("\n" + "="*80 + "\n")


def main() -> None:
    """主函数：解析参数并运行实验"""
    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description="音频啸叫抑制方法科学评估实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 快速评估（默认10个样本）
  python scripts/run_experiment.py --mode quick

  # 快速评估，指定样本数
  python scripts/run_experiment.py --mode quick --num_samples 20

  # 全面评估
  python scripts/run_experiment.py --mode comprehensive --batch_size 8

  # 仅评估传统方法
  python scripts/run_experiment.py --mode traditional

  # 自定义方法评估
  python scripts/run_experiment.py --mode custom --methods unet_v2 frequency_shift

  # 不保存结果和可视化
  python scripts/run_experiment.py --mode quick --no-save_results --no-generate_visualizations

可用方法:
  深度学习: unet_v1, unet_v2, unet_v3_attention, unet_v4_residual, unet_v5_dilated,
            unet_v6_optimized, unet_v7_lstm, unet_v8_temporal_attention,
            unet_v9_convlstm, unet_v10_gan, unet_v11_multiscale, unet_v12_pyramid,
            unet_v13_fpn, cnn, rnn
  传统方法: frequency_shift, gain_suppression, adaptive_feedback
        """
    )

    # 添加命令行参数
    parser.add_argument("--mode", type=str, default="quick",
                       choices=["quick", "comprehensive", "traditional", "custom"],
                       help="实验模式: quick(快速), comprehensive(全面), "
                            "traditional(仅传统方法), custom(自定义)")

    parser.add_argument("--methods", type=str, nargs="+",
                       help="指定要评估的方法列表 (仅在custom模式或覆盖默认方法时使用)")

    parser.add_argument("--num_samples", type=int, default=10,
                       help="快速评估模式下的样本数 (默认: 10)")

    parser.add_argument("--batch_size", type=int, default=4,
                       help="批大小，影响内存使用和速度 (默认: 4)")

    parser.add_argument("--save_results", action="store_true", default=True,
                       help="是否保存评估结果到文件 (默认: True)")

    parser.add_argument("--no_save_results", action="store_false", dest="save_results",
                       help="不保存评估结果")

    parser.add_argument("--generate_visualizations", action="store_true", default=True,
                       help="是否生成可视化图表 (默认: True)")

    parser.add_argument("--no_generate_visualizations", action="store_false", dest="generate_visualizations",
                       help="不生成可视化图表")

    # 解析参数
    args = parser.parse_args()

    # 打印实验配置
    print_experiment_header(args)

    # 根据模式执行不同的评估流程
    results = None

    if args.mode == "quick":
        results = run_quick_mode(args)

    elif args.mode == "comprehensive":
        results = run_comprehensive_mode(args)

    elif args.mode == "traditional":
        results = run_traditional_mode(args)

    elif args.mode == "custom":
        results = run_custom_mode(args)

    # 打印结果摘要
    print_experiment_summary(results)


if __name__ == "__main__":
    main()