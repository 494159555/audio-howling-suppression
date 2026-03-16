'''
基准测试运行器模块

提供标准化的基准测试流程，包括：
- 数据加载和预处理
- 方法性能测试
- 结果收集和统计
- 批量测试支持
'''

import torch
import torchaudio
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Callable
import time
import json
from pathlib import Path
from tqdm import tqdm

from src.config import cfg
from src.dataset import HowlingDataset
from .metrics import AudioMetrics


class BenchmarkRunner:
    """基准测试运行器"""
    
    def __init__(self, test_data_dir: str = None, batch_size: int = 4):
        self.test_data_dir = test_data_dir or str(cfg.VAL_CLEAN_DIR).replace('/clean', '')
        self.batch_size = batch_size
        self.metrics_calculator = AudioMetrics()
        
        # 测试结果存储
        self.results = {}
        self.detailed_results = {}
        
    def load_test_data(self, clean_dir: str = None, noisy_dir: str = None) -> Tuple[DataLoader, DataLoader]:
        """
        加载测试数据
        
        Args:
            clean_dir: 纯净音频目录
            noisy_dir: 带噪音频目录
            
        Returns:
            (clean_loader, noisy_loader)
        """
        if clean_dir is None:
            clean_dir = str(cfg.VAL_CLEAN_DIR)
        if noisy_dir is None:
            noisy_dir = str(cfg.VAL_NOISY_DIR)
            
        # 创建数据集
        clean_dataset = HowlingDataset(
            clean_dir=clean_dir,
            howling_dir=noisy_dir,
            sample_rate=cfg.SAMPLE_RATE,
            chunk_len=cfg.CHUNK_LEN,
            n_fft=cfg.N_FFT,
            hop_length=cfg.HOP_LENGTH
        )
        
        # 创建数据加载器
        clean_loader = DataLoader(
            clean_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=cfg.NUM_WORKERS
        )
        
        return clean_loader, clean_loader  # 返回相同的loader，因为dataset已经包含clean和noisy
    
    def benchmark_method(self, method_name: str, processing_func: Callable,
                       test_loader: DataLoader, **method_params) -> Dict:
        """
        对单个方法进行基准测试
        
        Args:
            method_name: 方法名称
            processing_func: 处理函数
            test_loader: 测试数据加载器
            **method_params: 方法参数
            
        Returns:
            测试结果字典
        """
        print(f"\n正在测试方法: {method_name}")
        
        all_metrics = []
        processing_times = []
        
        device = cfg.DEVICE
        
        with torch.no_grad():
            for batch_idx, (noisy_mag, clean_mag) in enumerate(tqdm(test_loader, desc=f"测试 {method_name}")):
                noisy_mag = noisy_mag.to(device)
                clean_mag = clean_mag.to(device)
                
                # 计算处理时间
                start_time = time.time()
                
                try:
                    # 处理音频
                    enhanced_mag = processing_func(noisy_mag, **method_params)
                    
                    processing_time = time.time() - start_time
                    processing_times.append(processing_time)
                    
                    # 计算各项指标
                    batch_metrics = self.metrics_calculator.calculate_all_metrics(
                        clean_mag, noisy_mag, enhanced_mag, method_name, processing_func, **method_params
                    )
                    
                    all_metrics.append(batch_metrics)
                    
                except Exception as e:
                    print(f"处理批次 {batch_idx} 时出错: {e}")
                    continue
        
        # 汇总结果
        if all_metrics:
            avg_metrics = self._average_metrics(all_metrics)
            avg_metrics['processing_time_ms'] = np.mean(processing_times) * 1000
            avg_metrics['processing_time_std'] = np.std(processing_times) * 1000
            avg_metrics['total_samples'] = len(all_metrics) * self.batch_size
            avg_metrics['method_name'] = method_name
            
            return avg_metrics
        else:
            print(f"方法 {method_name} 测试失败")
            return {}
    
    def benchmark_multiple_methods(self, methods_config: Dict[str, Dict]) -> Dict:
        """
        对多个方法进行基准测试
        
        Args:
            methods_config: 方法配置字典 {method_name: {'func': func, 'params': {}}}
            
        Returns:
            所有方法的测试结果
        """
        # 加载测试数据
        test_loader, _ = self.load_test_data()
        
        results = {}
        
        for method_name, config in methods_config.items():
            processing_func = config['func']
            method_params = config.get('params', {})
            
            try:
                method_results = self.benchmark_method(
                    method_name, processing_func, test_loader, **method_params
                )
                
                if method_results:
                    results[method_name] = method_results
                    print(f"✓ {method_name} 测试完成")
                else:
                    print(f"✗ {method_name} 测试失败")
                    
            except Exception as e:
                print(f"✗ {method_name} 测试出错: {e}")
                continue
        
        self.results = results
        return results
    
    def _average_metrics(self, metrics_list: List[Dict]) -> Dict:
        """计算平均指标"""
        if not metrics_list:
            return {}
        
        avg_metrics = {}
        
        # 获取所有指标名称
        all_keys = set()
        for metrics in metrics_list:
            all_keys.update(metrics.keys())
        
        # 计算每个指标的平均值
        for key in all_keys:
            values = [metrics.get(key, 0) for metrics in metrics_list]
            if isinstance(values[0], (int, float)):
                avg_metrics[key] = np.mean(values)
                avg_metrics[f"{key}_std"] = np.std(values)
            else:
                # 非数值型指标，取第一个
                avg_metrics[key] = values[0]
        
        return avg_metrics
    
    def run_comprehensive_benchmark(self, methods_to_test: List[str] = None) -> Dict:
        """
        运行综合基准测试
        
        Args:
            methods_to_test: 要测试的方法列表，None表示测试所有方法
            
        Returns:
            完整的测试结果
        """
        if methods_to_test is None:
            methods_to_test = ['unet', 'frequency_shift', 'gain_suppression', 'adaptive_feedback']
        
        # 准备方法配置
        methods_config = self._prepare_methods_config(methods_to_test)
        
        print("开始综合基准测试...")
        print(f"测试方法: {list(methods_config.keys())}")
        
        # 运行测试
        results = self.benchmark_multiple_methods(methods_config)
        
        # 生成详细报告
        detailed_results = self._generate_detailed_report(results)
        self.detailed_results = detailed_results
        
        # 保存结果
        self._save_benchmark_results(results, detailed_results)
        
        return detailed_results
    
    def _prepare_methods_config(self, methods_to_test: List[str]) -> Dict:
        """准备方法配置"""
        methods_config = {}
        
        for method_name in methods_to_test:
            if method_name == 'unet':
                # 深度学习方法
                methods_config[method_name] = {
                    'func': self._load_unet_model(),
                    'params': {}
                }
            elif method_name == 'frequency_shift':
                # 移频移向法
                from src.traditional import FrequencyShiftMethod
                methods_config[method_name] = {
                    'func': FrequencyShiftMethod(shift_hz=20.0),
                    'params': {}
                }
            elif method_name == 'gain_suppression':
                # 增益抑制法
                from src.traditional import GainSuppressionMethod
                methods_config[method_name] = {
                    'func': GainSuppressionMethod(threshold_db=-30.0),
                    'params': {}
                }
            elif method_name == 'adaptive_feedback':
                # 自适应反馈抵消法
                from src.traditional import AdaptiveFeedbackMethod
                methods_config[method_name] = {
                    'func': AdaptiveFeedbackMethod(filter_length=64),
                    'params': {}
                }
            else:
                print(f"未知方法: {method_name}")
        
        return methods_config
    
    def _load_unet_model(self):
        """加载UNet模型"""
        from src.models.unet_v2 import AudioUNet5
        
        device = cfg.DEVICE
        model = AudioUNet5().to(device)
        
        # 尝试加载最佳模型
        best_model_path = None
        exp_dir = cfg.EXP_DIR
        
        if exp_dir.exists():
            # 查找最新的实验
            exp_dirs = [d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith('exp_')]
            if exp_dirs:
                latest_exp = max(exp_dirs, key=lambda x: x.stat().st_mtime)
                checkpoint_path = latest_exp / 'checkpoints' / 'best_model.pth'
                
                if checkpoint_path.exists():
                    best_model_path = checkpoint_path
                    print(f"加载模型: {best_model_path}")
        
        if best_model_path:
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model
    
    def _generate_detailed_report(self, results: Dict) -> Dict:
        """生成详细报告"""
        detailed_report = {
            'test_summary': {
                'total_methods': len(results),
                'methods_tested': list(results.keys()),
                'test_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'device': str(cfg.DEVICE),
                'batch_size': self.batch_size
            },
            'performance_summary': {},
            'ranking': {},
            'recommendations': {}
        }
        
        if not results:
            return detailed_report
        
        # 性能摘要
        methods = list(results.keys())
        for metric in ['snr_improvement_db', 'psnr_db', 'stoi_score', 'howling_reduction_db', 'processing_time_ms']:
            values = [results[method].get(metric, 0) for method in methods]
            detailed_report['performance_summary'][metric] = {
                'best_method': methods[np.argmax(values)] if metric != 'processing_time_ms' else methods[np.argmin(values)],
                'best_value': max(values) if metric != 'processing_time_ms' else min(values),
                'worst_method': methods[np.argmin(values)] if metric != 'processing_time_ms' else methods[np.argmax(values)],
                'worst_value': min(values) if metric != 'processing_time_ms' else max(values),
                'average': np.mean(values),
                'std': np.std(values)
            }
        
        # 综合排名
        from .metrics import calculate_mos_score
        
        method_scores = {}
        for method in methods:
            mos_score = calculate_mos_score(results[method])
            processing_time = results[method].get('processing_time_ms', 1000)
            
            # 综合评分 (质量70% + 效率30%)
            quality_score = mos_score / 5.0  # 归一化到0-1
            efficiency_score = max(0, 1 - processing_time / 1000)  # 归一化到0-1
            
            comprehensive_score = 0.7 * quality_score + 0.3 * efficiency_score
            method_scores[method] = {
                'comprehensive_score': comprehensive_score,
                'mos_score': mos_score,
                'quality_score': quality_score,
                'efficiency_score': efficiency_score
            }
        
        # 按综合评分排序
        sorted_methods = sorted(method_scores.items(), key=lambda x: x[1]['comprehensive_score'], reverse=True)
        detailed_report['ranking'] = {
            method: rank + 1 for rank, (method, _) in enumerate(sorted_methods)
        }
        
        # 推荐
        best_overall = sorted_methods[0][0]
        best_quality = max(method_scores.items(), key=lambda x: x[1]['mos_score'])[0]
        most_efficient = max(method_scores.items(), key=lambda x: x[1]['efficiency_score'])[0]
        
        detailed_report['recommendations'] = {
            'best_overall': best_overall,
            'best_quality': best_quality,
            'most_efficient': most_efficient,
            'detailed_scores': method_scores
        }
        
        return detailed_report
    
    def _save_benchmark_results(self, results: Dict, detailed_results: Dict):
        """保存基准测试结果"""
        save_dir = Path("benchmark_results")
        save_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        # 保存原始结果
        results_file = save_dir / f"benchmark_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 保存详细报告
        report_file = save_dir / f"benchmark_report_{timestamp}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n基准测试结果已保存:")
        print(f"原始结果: {results_file}")
        print(f"详细报告: {report_file}")
    
    def get_method_comparison_table(self) -> str:
        """生成方法对比表格"""
        if not self.results:
            return "请先运行基准测试"
        
        # 创建表格
        table_lines = []
        table_lines.append("\n" + "="*80)
        table_lines.append("方法性能对比表")
        table_lines.append("="*80)
        
        # 表头
        header = f"{'方法':<20} {'SNR改善(dB)':<12} {'PSNR(dB)':<10} {'STOI':<8} {'啸叫抑制(dB)':<12} {'处理时间(ms)':<12}"
        table_lines.append(header)
        table_lines.append("-" * len(header))
        
        # 数据行
        for method, metrics in self.results.items():
            row = f"{method:<20} {metrics.get('snr_improvement_db', 0):<12.2f} {metrics.get('psnr_db', 0):<10.2f} {metrics.get('stoi_score', 0):<8.3f} {metrics.get('howling_reduction_db', 0):<12.2f} {metrics.get('processing_time_ms', 0):<12.2f}"
            table_lines.append(row)
        
        table_lines.append("="*80)
        
        return "\n".join(table_lines)
    
    def run_quick_test(self, num_samples: int = 10) -> Dict:
        """
        运行快速测试（仅测试少量样本）
        
        Args:
            num_samples: 测试样本数量
            
        Returns:
            快速测试结果
        """
        print(f"运行快速测试 (样本数: {num_samples})")
        
        # 加载少量数据
        test_loader, _ = self.load_test_data()
        
        # 限制样本数量
        limited_loader = []
        for i, (noisy, clean) in enumerate(test_loader):
            if i >= num_samples:
                break
            limited_loader.append((noisy, clean))
        
        # 测试所有方法
        methods_config = self._prepare_methods_config(['frequency_shift', 'gain_suppression', 'adaptive_feedback'])
        
        quick_results = {}
        for method_name, config in methods_config.items():
            print(f"快速测试: {method_name}")
            
            try:
                processing_func = config['func']
                method_params = config.get('params', {})
                
                all_metrics = []
                for noisy_mag, clean_mag in limited_loader:
                    noisy_mag = noisy_mag.to(cfg.DEVICE)
                    clean_mag = clean_mag.to(cfg.DEVICE)
                    
                    # 处理
                    enhanced_mag = processing_func(noisy_mag, **method_params)
                    
                    # 计算指标
                    metrics = self.metrics_calculator.calculate_all_metrics(
                        clean_mag, noisy_mag, enhanced_mag, method_name
                    )
                    all_metrics.append(metrics)
                
                # 平均结果
                if all_metrics:
                    avg_metrics = self._average_metrics(all_metrics)
                    quick_results[method_name] = avg_metrics
                    
            except Exception as e:
                print(f"快速测试 {method_name} 失败: {e}")
        
        return quick_results
