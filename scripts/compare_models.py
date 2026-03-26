#!/usr/bin/env python3
"""U-Net模型对比脚本

本脚本用于全面对比所有U-Net模型变体，提供详细的性能和参数分析。

主要功能：
    - 对比所有U-Net模型的参数量
    - 测试模型的前向传播性能
    - 生成详细的对比报告
    - 提供模型选择建议

使用方法：
    python scripts/compare_models.py

输出信息：
    - 每个模型的参数统计（总数、可训练、不可训练）
    - 模型对比汇总表
    - 统计信息（最少/最多/平均参数量）
    - 模型描述和特点说明

适用场景：
    - 选择最适合当前任务的模型
    - 评估模型的计算复杂度
    - 对比不同变体的改进效果

作者：音频处理实验室
版本：1.1
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径，确保可以导入src模块
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

# 导入所有要对比的U-Net模型
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
    """统计模型的参数数量

    分别统计总参数、可训练参数和不可训练参数（如BatchNorm的running stats）。

    Args:
        model: PyTorch模型实例

    Returns:
        包含参数统计的字典：
        {
            'total': 总参数数,
            'trainable': 可训练参数数,
            'non_trainable': 不可训练参数数
        }
    """
    # 统计所有参数
    total = sum(p.numel() for p in model.parameters())

    # 只统计需要梯度的参数（可训练参数）
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 不可训练参数 = 总参数 - 可训练参数
    non_trainable = total - trainable

    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': non_trainable,
    }


def format_number(num: int) -> str:
    """格式化大数字，转换为易读格式

    Args:
        num: 要格式化的数字

    Returns:
        格式化后的字符串：
        - >= 1,000,000: 显示为X.XXM (如1.23M)
        - >= 1,000: 显示为X.XXK (如456.78K)
        - < 1,000: 直接显示数字

    Examples:
        >>> format_number(1234567)
        '1.23M'
        >>> format_number(45678)
        '45.68K'
        >>> format_number(123)
        '123'
    """
    if num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    else:
        return str(num)


def test_model(model_class, model_name: str, input_tensor: torch.Tensor) -> dict:
    """测试单个模型

    执行完整的模型测试流程：
    1. 初始化模型
    2. 移动到指定设备（GPU/CPU）
    3. 统计参数量
    4. 测试前向传播
    5. 收集所有结果

    Args:
        model_class: 模型类（如AudioUNet5）
        model_name: 模型的显示名称
        input_tensor: 测试用的输入张量

    Returns:
        包含测试结果的字典：
        {
            'name': 模型名称,
            'input_shape': 输入形状,
            'output_shape': 输出形状,
            'params': 参数统计字典,
            'success': 是否成功,
            'error': 错误信息（如果失败）
        }
    """
    print(f"\n{'='*70}")
    print(f"🧪 测试模型: {model_name}")
    print(f"{'='*70}")

    try:
        # ============ 步骤1: 初始化模型 ============
        model = model_class()
        print(f"✓ 模型初始化成功")

        # ============ 步骤2: 移动到设备 ============
        model = model.to(cfg.DEVICE)
        input_tensor = input_tensor.to(cfg.DEVICE)
        print(f"✓ 模型已移动到设备: {cfg.DEVICE}")

        # ============ 步骤3: 统计参数 ============
        params = count_parameters(model)
        print(f"✓ 参数统计完成:")
        print(f"  └─ 总参数:      {format_number(params['total']):>10} ({params['total']:,})")
        print(f"  └─ 可训练:    {format_number(params['trainable']):>10} ({params['trainable']:,})")
        print(f"  └─ 冻结参数:  {format_number(params['non_trainable']):>10} ({params['non_trainable']:,})")

        # ============ 步骤4: 前向传播测试 ============
        model.eval()  # 设置为评估模式
        print(f"✓ 正在执行前向传播...")

        with torch.no_grad():  # 不计算梯度，节省内存
            output = model(input_tensor)

        # ============ 步骤5: 收集结果 ============
        results = {
            'name': model_name,
            'input_shape': tuple(input_tensor.shape),
            'output_shape': tuple(output.shape),
            'params': params,
            'success': True,
            'error': None,
        }

        print(f"✓ 前向传播成功")
        print(f"  └─ 输入形状:  {results['input_shape']}")
        print(f"  └─ 输出形状: {results['output_shape']}")

    except Exception as e:
        # 如果测试失败，记录错误信息
        results = {
            'name': model_name,
            'input_shape': tuple(input_tensor.shape),
            'output_shape': None,
            'params': None,
            'success': False,
            'error': str(e),
        }
        print(f"✗ 模型测试失败:")
        print(f"  错误: {e}")

    return results


def print_summary_table(results: list, config: Config) -> None:
    """打印模型对比汇总表

    以表格形式展示所有模型的对比结果，包括参数量和测试状态。

    Args:
        results: 测试结果列表
        config: 配置对象，用于获取模型描述
    """
    print(f"\n\n{'='*90}")
    print("📊 模型对比汇总表")
    print(f"{'='*90}\n")

    # 打印表头
    header = f"{'模型名称':<28} {'总参数':>15} {'可训练参数':>15} {'状态':>12}"
    print(header)
    print("-" * 90)

    # 打印每个模型的结果
    for result in results:
        if result['success']:
            params_str = format_number(result['params']['total'])
            trainable_str = format_number(result['params']['trainable'])
            status = "✅ 通过"
        else:
            params_str = "N/A"
            trainable_str = "N/A"
            status = "❌ 失败"

        row = f"{result['name']:<28} {params_str:>15} {trainable_str:>15} {status:>12}"
        print(row)

    # 打印分割线
    print("-" * 90)

    # ============ 计算并显示统计信息 ============
    successful = [r for r in results if r['success']]
    successful_count = len(successful)
    total_count = len(results)

    if successful_count > 0:
        # 提取所有成功模型的参数量
        total_params = [r['params']['total'] for r in successful]
        min_params = min(total_params)
        max_params = max(total_params)
        avg_params = sum(total_params) / len(total_params)

        print(f"\n📈 统计信息:")
        print(f"  成功模型数:     {successful_count}/{total_count}")
        print(f"  最少参数量:     {format_number(min_params):>12} ({min_params:,})")
        print(f"  最多参数量:     {format_number(max_params):>12} ({max_params:,})")
        print(f"  平均参数量:     {format_number(int(avg_params)):>12} ({int(avg_params):,})")

        # 计算参数增长倍数（相对于最小模型）
        if min_params > 0:
            ratio = max_params / min_params
            print(f"  参数量差距:     {ratio:.2f}x (最大 vs 最小)")

    # ============ 打印模型详细描述 ============
    print(f"\n\n{'='*90}")
    print("📚 模型详细描述")
    print(f"{'='*90}\n")

    for model_key, description in config.MODEL_DESCRIPTIONS.items():
        print(f"  🔹 {model_key:<20} : {description}")


def print_recommendations(results: list) -> None:
    """根据测试结果给出模型选择建议

    Args:
        results: 测试结果列表
    """
    print(f"\n\n{'='*90}")
    print("💡 模型选择建议")
    print(f"{'='*90}\n")

    successful = [r for r in results if r['success']]

    if len(successful) == 0:
        print("❌ 没有可用的模型，请检查错误信息")
        return

    # 按参数量排序
    successful_sorted = sorted(successful, key=lambda x: x['params']['total'])

    print("根据不同需求，推荐以下模型:\n")

    # 参数量最少的模型（适合快速实验）
    lightest = successful_sorted[0]
    print(f"🚀 快速原型开发 (参数最少):")
    print(f"   {lightest['name']}")
    print(f"   参数量: {format_number(lightest['params']['total'])}\n")

    # 参数量中等的模型（平衡性能和速度）
    if len(successful_sorted) >= 3:
        mid_index = len(successful_sorted) // 2
        balanced = successful_sorted[mid_index]
        print(f"⚖️  平衡性能和速度:")
        print(f"   {balanced['name']}")
        print(f"   参数量: {format_number(balanced['params']['total'])}\n")

    # 参数量最多的模型（最佳性能）
    heaviest = successful_sorted[-1]
    print(f"🏆 追求最佳性能 (参数最多):")
    print(f"   {heaviest['name']}")
    print(f"   参数量: {format_number(heaviest['params']['total'])}\n")

    print("使用示例:")
    print(f"  python src/train.py --model unet_v1          # 轻量级模型")
    print(f"  python src/train.py --model unet_v2          # 标准模型")
    print(f"  python src/train.py --model unet_v6_optimized # 高性能模型")


def main():
    """主函数：执行模型对比流程"""
    print("\n" + "="*90)
    print("🔬 U-Net模型对比脚本")
    print("="*90)

    # 打印环境信息
    print(f"\n🖥️  环境信息:")
    print(f"   设备:        {cfg.DEVICE}")
    print(f"   PyTorch版本: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"   GPU:         {torch.cuda.get_device_name(0)}")
        print(f"   显存:       {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 定义要测试的模型列表
    # 格式: (模型类, 显示名称)
    models_to_test = [
        (AudioUNet3, 'AudioUNet3 (3层基线U-Net)'),
        (AudioUNet5, 'AudioUNet5 (5层基线U-Net)'),
        (AudioUNet5Attention, 'AudioUNet5Attention (+注意力门)'),
        (AudioUNet5Residual, 'AudioUNet5Residual (+残差连接)'),
        (AudioUNet5Dilated, 'AudioUNet5Dilated (+空洞卷积)'),
        (AudioUNet5Optimized, 'AudioUNet5Optimized (综合改进版)'),
    ]

    print(f"\n📋 待测试模型: {len(models_to_test)} 个")

    # 创建样本输入
    # 形状: [Batch=2, Channels=1, Frequency=256, Time=100]
    # 这是音频频谱图的标准输入格式
    sample_input = torch.randn(2, 1, 256, 100)
    print(f"📊 样本输入形状: {tuple(sample_input.shape)}")
    print(f"   解释: [批次=2, 通道=1, 频率=256, 时间=100]")

    # 测试所有模型
    results = []
    for i, (model_class, model_name) in enumerate(models_to_test, 1):
        print(f"\n[{i}/{len(models_to_test)}] {model_name}")
        result = test_model(model_class, model_name, sample_input)
        results.append(result)

    # 打印汇总表
    print_summary_table(results, cfg)

    # 打印推荐
    print_recommendations(results)

    # 打印总结
    print(f"\n{'='*90}")
    print("📌 测试总结")
    print(f"{'='*90}\n")

    successful = sum(1 for r in results if r['success'])
    total = len(results)

    if successful == total:
        print("✅ 恭喜！所有模型测试通过！\n")
        print("💡 您可以使用以下任意模型进行训练:")
        for model_key, class_name in cfg.AVAILABLE_MODELS.items():
            print(f"   python src/train.py --model {model_key}")
    else:
        failed = total - successful
        print(f"⚠️  警告: {failed} 个模型测试失败\n")
        print("💡 请检查:")
        print("   1. 模型代码是否有语法错误")
        print("   2. 模型是否正确导入到 src/models/__init__.py")
        print("   3. 模型是否在 src/config.py 中正确注册")

    print("\n" + "="*90 + "\n")


if __name__ == "__main__":
    main()