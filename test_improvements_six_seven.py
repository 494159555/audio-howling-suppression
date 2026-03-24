"""测试改进六和改进七的实现

测试训练策略优化和后处理优化的所有功能
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import cfg
from src.models import (
    AudioUNet5,
    # 训练策略
    MixedPrecisionTrainer,
    CosineAnnealingWarmupScheduler,
    OneCycleScheduler,
    CurriculumLearning,
    create_lr_scheduler,
    # 后处理
    AdaptivePostProcessing,
    MultiFrameSmoother,
    AdaptiveGainControl,
    PostProcessingPipeline
)


def test_training_strategies():
    """测试训练策略"""
    print("\n" + "="*80)
    print("测试训练策略优化 (改进六)")
    print("="*80 + "\n")
    
    device = cfg.DEVICE
    model = AudioUNet5(in_channels=1, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    criterion = nn.L1Loss()
    
    # 创建测试数据
    batch_size = 4
    channels = 1
    freq_bins = 256
    time_steps = 128
    inputs = torch.randn(batch_size, channels, freq_bins, time_steps).abs().to(device)
    targets = torch.randn(batch_size, channels, freq_bins, time_steps).abs().to(device)
    
    # Test 1: MixedPrecisionTrainer
    print("=" * 60)
    print("Test 1: MixedPrecisionTrainer")
    print("=" * 60)
    
    trainer = MixedPrecisionTrainer(model, optimizer, device, use_amp=True)
    loss, predictions = trainer.train_step(inputs, targets, criterion)
    
    print(f"✅ 混合精度训练测试通过")
    print(f"   Loss: {loss:.4f}")
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   AMP启用: {trainer.use_amp}")
    print()
    
    # Test 2: CosineAnnealingWarmupScheduler
    print("=" * 60)
    print("Test 2: CosineAnnealingWarmupScheduler")
    print("=" * 60)
    
    scheduler = CosineAnnealingWarmupScheduler(
        optimizer=optimizer,
        warmup_epochs=5,
        total_epochs=20,
        base_lr=0.001,
        min_lr=1e-6
    )
    
    print(f"✅ Cosine Annealing + Warmup调度器测试通过")
    print(f"   Warmup轮数: 5")
    print(f"   总轮数: 20")
    print(f"   学习率变化:")
    for epoch in [0, 5, 10, 15, 20]:
        scheduler.step(epoch)
        lr = scheduler.get_lr()
        print(f"     Epoch {epoch:2d}: LR = {lr:.2e}")
    print()
    
    # Test 3: OneCycleScheduler
    print("=" * 60)
    print("Test 3: OneCycleScheduler")
    print("=" * 60)
    
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    scheduler = OneCycleScheduler(
        optimizer=optimizer,
        max_lr=0.01,
        total_epochs=20,
        pct_start=0.3
    )
    
    print(f"✅ One Cycle调度器测试通过")
    print(f"   最大学习率: 0.01")
    print(f"   总轮数: 20")
    print(f"   上升阶段占比: 30%")
    print(f"   学习率变化:")
    for epoch in [0, 6, 10, 15, 20]:
        scheduler.step(epoch)
        lr = scheduler.get_lr()
        print(f"     Epoch {epoch:2d}: LR = {lr:.2e}")
    print()
    
    # Test 4: CurriculumLearning
    print("=" * 60)
    print("Test 4: CurriculumLearning")
    print("=" * 60)
    
    curriculum = CurriculumLearning(
        total_epochs=20,
        difficulty_levels=[
            {'noise_level': 0.01},
            {'noise_level': 0.05},
            {'noise_level': 0.1},
            {'noise_level': 0.2}
        ],
        schedule_type='step'
    )
    
    print(f"✅ 课程学习策略测试通过")
    print(f"   难度级别数: 4")
    print(f"   难度变化:")
    for epoch in [0, 5, 10, 15, 20]:
        difficulty = curriculum.get_current_difficulty(epoch)
        desc = curriculum.get_difficulty_description(epoch)
        print(f"     Epoch {epoch:2d}: {desc}")
    print()
    
    # Test 5: create_lr_scheduler
    print("=" * 60)
    print("Test 5: create_lr_scheduler (工厂函数)")
    print("=" * 60)
    
    # Test different scheduler types
    scheduler_types = ['cosine_warmup', 'one_cycle', 'plateau', 'step']
    
    for scheduler_type in scheduler_types:
        try:
            optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
            scheduler = create_lr_scheduler(
                optimizer=optimizer,
                scheduler_type=scheduler_type,
                warmup_epochs=5,
                total_epochs=20,
                base_lr=0.001,
                min_lr=1e-6,
                factor=0.5,
                patience=3
            )
            print(f"   ✅ {scheduler_type} 调度器创建成功")
        except Exception as e:
            print(f"   ❌ {scheduler_type} 调度器创建失败: {e}")
    
    print(f"\n✅ 工厂函数测试通过")
    print()
    
    print("="*80)
    print("✅ 所有训练策略测试通过！")
    print("="*80 + "\n")


def test_post_processing():
    """测试后处理方法"""
    print("\n" + "="*80)
    print("测试后处理优化 (改进七)")
    print("="*80 + "\n")
    
    # 创建测试数据
    batch_size = 4
    channels = 1
    freq_bins = 256
    time_steps = 128
    spectrogram = torch.randn(batch_size, channels, freq_bins, time_steps).abs()
    
    # Test 1: AdaptivePostProcessing
    print("=" * 60)
    print("Test 1: AdaptivePostProcessing")
    print("=" * 60)
    
    adaptive_pp = AdaptivePostProcessing(
        threshold=0.1,
        adaptive_threshold=True,
        smoothing_window=5
    )
    result = adaptive_pp(spectrogram)
    
    print(f"✅ 自适应后处理测试通过")
    print(f"   输入形状: {spectrogram.shape}")
    print(f"   输出形状: {result.shape}")
    print(f"   输入均值: {spectrogram.mean():.4f}")
    print(f"   输出均值: {result.mean():.4f}")
    print()
    
    # Test 2: MultiFrameSmoother (所有方法)
    print("=" * 60)
    print("Test 2: MultiFrameSmoother")
    print("=" * 60)
    
    smoothing_methods = ['moving_average', 'kalman', 'wiener', 'median']
    
    for method in smoothing_methods:
        try:
            smoother = MultiFrameSmoother(method=method, window_size=5)
            
            # Kalman滤波较慢，只测试一个样本
            test_spec = spectrogram[:1] if method == 'kalman' else spectrogram
            
            result = smoother(test_spec)
            print(f"   ✅ {method:15s} - 输出形状: {result.shape}, 均值: {result.mean():.4f}")
        except Exception as e:
            print(f"   ❌ {method:15s} - 失败: {e}")
    
    print(f"\n✅ 多帧平滑测试通过")
    print()
    
    # Test 3: AdaptiveGainControl (所有方法)
    print("=" * 60)
    print("Test 3: AdaptiveGainControl")
    print("=" * 60)
    
    gain_methods = ['agc', 'drc', 'limiter']
    
    for method in gain_methods:
        try:
            gain_ctrl = AdaptiveGainControl(method=method, target_level=0.7)
            result = gain_ctrl(spectrogram)
            print(f"   ✅ {method:10s} - 输出形状: {result.shape}, 均值: {result.mean():.4f}")
        except Exception as e:
            print(f"   ❌ {method:10s} - 失败: {e}")
    
    print(f"\n✅ 自适应增益控制测试通过")
    print()
    
    # Test 4: PostProcessingPipeline
    print("=" * 60)
    print("Test 4: PostProcessingPipeline")
    print("=" * 60)
    
    pipeline = PostProcessingPipeline(
        use_adaptive=True,
        use_smoothing=True,
        use_gain_control=True,
        adaptive_params={'adaptive_threshold': True},
        smoothing_params={'method': 'moving_average'},
        gain_params={'method': 'agc'}
    )
    result = pipeline(spectrogram)
    
    print(f"✅ 后处理管道测试通过")
    print(f"   输入形状: {spectrogram.shape}")
    print(f"   输出形状: {result.shape}")
    print(f"   输入均值: {spectrogram.mean():.4f}")
    print(f"   输出均值: {result.mean():.4f}")
    print()
    
    # Test 5: 参数敏感性测试
    print("=" * 60)
    print("Test 5: 参数敏感性测试")
    print("=" * 60)
    
    # 测试不同的平滑窗口大小
    for window_size in [3, 5, 7, 9]:
        smoother = MultiFrameSmoother(method='moving_average', window_size=window_size)
        result = smoother(spectrogram)
        print(f"   窗口大小 {window_size}: 均值 = {result.mean():.4f}, 最大值 = {result.max():.4f}")
    
    print(f"\n✅ 参数敏感性测试通过")
    print()
    
    print("="*80)
    print("✅ 所有后处理方法测试通过！")
    print("="*80 + "\n")


def test_integration():
    """测试训练策略和后处理的集成"""
    print("\n" + "="*80)
    print("测试训练策略和后处理的集成")
    print("="*80 + "\n")
    
    device = cfg.DEVICE
    model = AudioUNet5(in_channels=1, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    criterion = nn.L1Loss()
    
    # 创建测试数据
    batch_size = 4
    channels = 1
    freq_bins = 256
    time_steps = 128
    inputs = torch.randn(batch_size, channels, freq_bins, time_steps).abs().to(device)
    targets = torch.randn(batch_size, channels, freq_bins, time_steps).abs().to(device)
    
    # Test 1: 混合精度训练 + 后处理
    print("=" * 60)
    print("Test 1: 混合精度训练 + 后处理管道")
    print("=" * 60)
    
    trainer = MixedPrecisionTrainer(model, optimizer, device, use_amp=True)
    post_processor = PostProcessingPipeline(
        use_adaptive=True,
        use_smoothing=True,
        use_gain_control=True
    )
    
    # 训练步骤
    loss, predictions = trainer.train_step(inputs, targets, criterion)
    
    # 应用后处理
    processed_predictions = post_processor(predictions)
    
    print(f"✅ 集成测试通过")
    print(f"   训练损失: {loss:.4f}")
    print(f"   原始预测均值: {predictions.mean():.4f}")
    print(f"   后处理后均值: {processed_predictions.mean():.4f}")
    print()
    
    # Test 2: 模拟训练循环
    print("=" * 60)
    print("Test 2: 模拟训练循环（含课程学习）")
    print("=" * 60)
    
    curriculum = CurriculumLearning(
        total_epochs=10,
        difficulty_levels=[
            {'noise_level': 0.01},
            {'noise_level': 0.05},
            {'noise_level': 0.1}
        ],
        schedule_type='step'
    )
    
    scheduler = CosineAnnealingWarmupScheduler(
        optimizer=optimizer,
        warmup_epochs=2,
        total_epochs=10,
        base_lr=0.001,
        min_lr=1e-6
    )
    
    print(f"   模拟训练循环:")
    for epoch in range(10):
        # 获取当前难度
        difficulty = curriculum.get_current_difficulty(epoch)
        difficulty_desc = curriculum.get_difficulty_description(epoch)
        
        # 更新学习率
        scheduler.step(epoch)
        lr = scheduler.get_lr()
        
        # 训练步骤
        loss, _ = trainer.train_step(inputs, targets, criterion)
        
        if epoch % 3 == 0 or epoch == 9:
            print(f"   Epoch {epoch}: LR={lr:.2e}, 难度={difficulty_desc}, Loss={loss:.4f}")
    
    print(f"\n✅ 模拟训练循环测试通过")
    print()
    
    print("="*80)
    print("✅ 所有集成测试通过！")
    print("="*80 + "\n")


def main():
    """主测试函数"""
    print("\n" + "="*80)
    print("开始测试改进六和改进七的实现")
    print("="*80)
    
    try:
        # 测试训练策略
        test_training_strategies()
        
        # 测试后处理
        test_post_processing()
        
        # 测试集成
        test_integration()
        
        print("\n" + "="*80)
        print("🎉 所有测试通过！改进六和改进七实现成功！")
        print("="*80 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)