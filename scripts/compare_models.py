"""U-Net模型对比脚本

初始化并对比测试所有U-Net模型，统计参数量并输出对比结果
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

from src.models import (
    AudioUNet3,
    AudioUNet5,
    AudioUNet5Attention,
    AudioUNet5Residual,
    AudioUNet5Dilated,
    AudioUNet5Optimized,
)
from src.config import Config, cfg


def count_parameters(model: nn.Module) -> dict:
    """统计模型参数数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': non_trainable,
    }


def format_number(num: int) -> str:
    """格式化大数字"""
    if num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    else:
        return str(num)


def test_model(model_class, model_name: str, input_tensor: torch.Tensor) -> dict:
    """测试模型"""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")
    
    try:
        # 初始化模型
        model = model_class()
        
        # 移动到设备
        model = model.to(cfg.DEVICE)
        input_tensor = input_tensor.to(cfg.DEVICE)
        
        # 统计参数
        params = count_parameters(model)
        
        # 测试前向传播
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
        
        # 收集结果
        results = {
            'name': model_name,
            'input_shape': tuple(input_tensor.shape),
            'output_shape': tuple(output.shape),
            'params': params,
            'success': True,
            'error': None,
        }
        
        # 打印结果
        print(f"✓ 模型初始化成功")
        print(f"  输入形状:  {results['input_shape']}")
        print(f"  输出形状: {results['output_shape']}")
        print(f"  总参数量:       {format_number(params['total']):>10} ({params['total']:,})")
        print(f"  可训练参数:    {format_number(params['trainable']):>10} ({params['trainable']:,})")
        print(f"  不可训练参数:   {format_number(params['non_trainable']):>10} ({params['non_trainable']:,})")
        print(f"✓ 前向传播成功")
        
    except Exception as e:
        results = {
            'name': model_name,
            'input_shape': tuple(input_tensor.shape),
            'output_shape': None,
            'params': None,
            'success': False,
            'error': str(e),
        }
        print(f"✗ 模型测试失败:")
        print(f"  {e}")
    
    return results


def print_summary_table(results: list, config: Config):
    """打印模型对比汇总表"""
    print(f"\n\n{'='*80}")
    print("模型对比汇总表")
    print(f"{'='*80}\n")
    
    # 打印表头
    header = f"{'模型':<25} {'总参数':>15} {'可训练':>15} {'状态':>15}"
    print(header)
    print("-" * 80)
    
    # 打印每个模型
    for result in results:
        if result['success']:
            params_str = format_number(result['params']['total'])
            trainable_str = format_number(result['params']['trainable'])
            status = "✓ 通过"
        else:
            params_str = "N/A"
            trainable_str = "N/A"
            status = "✗ 失败"
        
        row = f"{result['name']:<25} {params_str:>15} {trainable_str:>15} {status:>15}"
        print(row)
    
    # 打印分割线
    print("-" * 80)
    
    # 计算统计信息
    successful = sum(1 for r in results if r['success'])
    total = len(results)
    
    if successful > 0:
        total_params = [r['params']['total'] for r in results if r['success']]
        min_params = min(total_params)
        max_params = max(total_params)
        avg_params = sum(total_params) / len(total_params)
        
        print(f"\n统计信息:")
        print(f"  成功模型数: {successful}/{total}")
        print(f"  最少参数:  {format_number(min_params):>10} ({min_params:,})")
        print(f"  最多参数:  {format_number(max_params):>10} ({max_params:,})")
        print(f"  平均参数:  {format_number(int(avg_params)):>10} ({int(avg_params):,})")
    
    # 打印模型描述
    print(f"\n\n{'='*80}")
    print("模型描述")
    print(f"{'='*80}\n")
    
    for model_key, description in config.MODEL_DESCRIPTIONS.items():
        print(f"  {model_key:<20} : {description}")


def main():
    """主函数"""
    print("\n" + "="*80)
    print("U-Net模型对比脚本")
    print("="*80)
    print(f"\n设备: {cfg.DEVICE}")
    print(f"PyTorch版本: {torch.__version__}")
    
    # 定义待测试模型
    models_to_test = [
        (AudioUNet3, 'AudioUNet3 (3层基线)'),
        (AudioUNet5, 'AudioUNet5 (5层基线)'),
        (AudioUNet5Attention, 'AudioUNet5Attention (注意力机制)'),
        (AudioUNet5Residual, 'AudioUNet5Residual (残差连接)'),
        (AudioUNet5Dilated, 'AudioUNet5Dilated (空洞卷积)'),
        (AudioUNet5Optimized, 'AudioUNet5Optimized (综合改进)'),
    ]
    
    # 创建样本输入
    # 形状: [Batch=2, Channels=1, Freq=256, Time=100]
    sample_input = torch.randn(2, 1, 256, 100)
    print(f"\n样本输入形状: {tuple(sample_input.shape)}")
    
    # 测试所有模型
    results = []
    for model_class, model_name in models_to_test:
        result = test_model(model_class, model_name, sample_input)
        results.append(result)
    
    # 打印汇总表
    print_summary_table(results, cfg)
    
    # 打印结论
    print(f"\n\n{'='*80}")
    print("结论")
    print(f"{'='*80}\n")
    
    successful = sum(1 for r in results if r['success'])
    if successful == len(results):
        print("✓ 所有模型初始化和测试成功!")
        print("\n可以使用以下任意模型进行训练:")
        for model_key, description in cfg.AVAILABLE_MODELS.items():
            print(f"  - {model_key}: {description}")
    else:
        print(f"✗ {len(results) - successful} 个模型初始化失败")
        print("\n请检查上述错误信息")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()