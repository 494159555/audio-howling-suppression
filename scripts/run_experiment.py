#!/usr/bin/env python3
"""音频啸叫抑制方法对比实验脚本

提供多种评估模式对比不同抑制方法
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from src.evaluation.test_runner import (
    run_quick_evaluation,
    run_comprehensive_evaluation,
    evaluate_traditional_methods,
    run_evaluation_pipeline
)


def main() -> None:
    """运行音频啸叫抑制对比实验"""
    parser = argparse.ArgumentParser(description="音频啸叫抑制方法科学评估实验")
    parser.add_argument("--mode", type=str, default="quick", 
                       choices=["quick", "comprehensive", "traditional", "custom"],
                       help="实验模式")
    parser.add_argument("--methods", type=str, nargs="+", 
                       help="指定评估方法列表")
    parser.add_argument("--num_samples", type=int, default=10,
                       help="快速评估样本数")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="批大小")
    parser.add_argument("--save_results", action="store_true", default=True,
                       help="保存结果")
    parser.add_argument("--generate_visualizations", action="store_true", default=True,
                       help="生成可视化")
    
    args = parser.parse_args()
    
    print("="*80)
    print("音频啸叫抑制方法科学评估实验")
    print("="*80)
    print(f"实验模式: {args.mode}")
    print(f"批大小: {args.batch_size}")
    print(f"保存结果: {args.save_results}")
    print(f"生成可视化: {args.generate_visualizations}")
    
    try:
        if args.mode == "quick":
            # 快速评估
            print(f"\n运行快速评估 (样本数: {args.num_samples})")
            results = run_quick_evaluation(
                methods=args.methods,
                num_samples=args.num_samples
            )
            
        elif args.mode == "comprehensive":
            # 全面评估
            print("\n运行全面评估")
            results = run_comprehensive_evaluation(
                methods=args.methods,
                batch_size=args.batch_size,
                save_results=args.save_results,
                generate_visualizations=args.generate_visualizations
            )
            
        elif args.mode == "traditional":
            # 传统方法
            print("\n仅评估传统方法")
            results = run_comprehensive_evaluation(
                methods=['frequency_shift', 'gain_suppression', 'adaptive_feedback'],
                batch_size=args.batch_size,
                save_results=args.save_results,
                generate_visualizations=args.generate_visualizations
            )
            
        elif args.mode == "custom":
            # 自定义方法
            if not args.methods:
                print("自定义模式需要指定 --methods 参数")
                return
            
            print(f"\n运行自定义方法评估: {args.methods}")
            results = run_comprehensive_evaluation(
                methods=args.methods,
                batch_size=args.batch_size,
                save_results=args.save_results,
                generate_visualizations=args.generate_visualizations
            )
        
        # 输出结果摘要
        if results:
            print("\n" + "="*80)
            print("实验完成！")
            print("="*80)
            
            if 'summary' in results:
                summary = results['summary']
                if 'recommendations' in summary:
                    print("\n推荐结果:")
                    for category, method in summary['recommendations'].items():
                        print(f"  • {category}: {method}")
            
            print(f"\n结果保存在: evaluation_results/ 目录")
            print(f"可视化图表保存在: evaluation_results/ 目录")
            
        else:
            print("实验失败，请检查错误信息")
            
    except Exception as e:
        print(f"实验出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()