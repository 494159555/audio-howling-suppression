# Tests 目录说明

本目录包含项目的自动化测试套件，用于验证代码正确性和功能完整性。

---

## 📁 目录结构

```
tests/
├── README.md         # 本文档
├── __init__.py       # 包初始化文件
└── run_tests.py      # 主测试脚本
```

---

## 🎯 测试套件功能

### 快速测试 (quick)
验证基础功能，适合快速检查项目状态。

```bash
python tests/run_tests.py
# 或
python tests/run_tests.py --mode quick
```

**测试内容**:
- ✅ 模块导入测试
- ✅ 数据可用性检查

**运行时间**: ~5秒

---

### 评估测试 (evaluation)
测试传统方法和评估系统。

```bash
python tests/run_tests.py --mode evaluation
```

**测试内容**:
- ✅ 传统方法基本功能
- ✅ 评估系统指标计算

**运行时间**: ~10秒

---

### 全面测试 (full)
完整的功能测试，涵盖所有主要模块。

```bash
python tests/run_tests.py --mode full
```

**测试内容**:
- ✅ 模块导入测试
- ✅ 数据可用性检查
- ✅ 传统方法基本功能
- ✅ 深度学习模型测试
- ✅ 评估系统测试

**运行时间**: ~30秒

---

### 模型测试 (models)
测试所有U-Net模型变体和损失函数。

```bash
python tests/run_tests.py --mode models
```

**测试内容**:
- ✅ 所有U-Net模型变体（v1-v10）
- ✅ 所有损失函数

**运行时间**: ~60秒

---

## 🔬 与 Scripts 目录的区别

### Tests 目录（tests/）
**目的**: 自动化测试，验证代码正确性

| 特点 | 说明 |
|-----|------|
| 快速 | 测试脚本快速运行 |
| 自动化 | 适合CI/CD集成 |
| 验证性 | 检查通过/失败 |
| 开发导向 | 帮助开发者发现问题 |

**使用场景**:
- 代码修改后验证功能
- CI/CD自动测试
- 发布前质量检查
- 快速定位问题

### Scripts 目录（scripts/）
**目的**: 辅助工具，支持开发和实验

| 特点 | 说明 |
|-----|------|
| 详细 | 提供详细的信息和报告 |
| 交互性 | 支持参数配置和自定义 |
| 分析性 | 性能分析和对比 |
| 用户导向 | 帮助理解和决策 |

**使用场景**:
- 模型性能对比
- 详细参数统计
- 生成实验报告
- 音频推理处理

---

## 📊 功能对比表

| 功能 | tests/ | scripts/ | 推荐使用 |
|-----|--------|----------|---------|
| **模型能否运行** | run_tests.py | - | ✅ tests（快速） |
| **模型参数统计** | - | compare_models.py | ✅ scripts（详细） |
| **基本功能测试** | run_tests.py | - | ✅ tests |
| **科学评估实验** | - | run_experiment.py | ✅ scripts |
| **模型对比分析** | - | compare_models.py | ✅ scripts |
| **音频推理** | - | inference.py | ✅ scripts |

---

## 💡 使用建议

### 开发流程

```bash
# 1. 修改代码后，运行测试验证
python tests/run_tests.py --mode full

# 2. 测试通过后，使用scripts分析性能
python scripts/compare_models.py

# 3. 训练模型
python src/train.py --config configs/unet_v2.yaml

# 4. 评估模型效果
python scripts/run_experiment.py --mode comprehensive
```

### CI/CD集成

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.8'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run tests
        run: |
          python tests/run_tests.py --mode full
```

---

## 🧪 添加新测试

### 步骤1: 定义测试函数

在 `tests/run_tests.py` 中添加新的测试函数：

```python
def test_new_feature():
    """测试新功能"""
    print("\n" + "="*60)
    print("🆕 测试: 新功能")
    print("="*60)

    try:
        # 测试代码
        from src.new_module import NewClass

        obj = NewClass()
        result = obj.method()

        print(f"✅ 新功能测试通过: {result}")
        return True

    except Exception as e:
        print(f"❌ 新功能测试失败: {e}")
        return False
```

### 步骤2: 添加到测试套件

```python
TEST_SUITES = {
    'quick': [
        ("模块导入", test_imports),
        ("数据可用性", test_data_availability),
        ("新功能", test_new_feature),  # 添加到快速测试
    ],
    # ...
}
```

### 步骤3: 运行测试

```bash
python tests/run_tests.py
```

---

## 📋 测试检查清单

### 代码修改后的测试流程

- [ ] 运行快速测试 (`--mode quick`)
- [ ] 运行全面测试 (`--mode full`)
- [ ] 如果修改了模型，运行模型测试 (`--mode models`)
- [ ] 检查所有测试是否通过
- [ ] 提交代码前确保测试通过

### 发布前检查

- [ ] 所有测试模式通过
- [ ] 在不同环境下测试（CPU/GPU）
- [ ] 验证数据目录正确
- [ ] 检查依赖版本兼容性

---

## ❓ 常见问题

### Q: 测试失败怎么办？
**A**:
1. 查看详细的错误信息
2. 检查数据目录是否存在
3. 验证依赖是否正确安装
4. 使用单模块测试定位问题

### Q: 为什么测试和scripts功能重复？
**A**:
- **tests**: 快速验证（能否运行？）
- **scripts**: 详细分析（运行得如何？）
- 目的不同，各司其职

### Q: 如何跳过某些测试？
**A**:
修改 `TEST_SUITES` 字典，临时注释掉不需要的测试。

### Q: 测试能在CPU上运行吗？
**A**:
可以，所有测试都支持CPU和GPU。会自动检测设备。

---

## 🔗 相关文档

- [Scripts目录说明](../scripts/README.md)
- [配置文件说明](../configs/README.md)
- [项目文档](../项目文档.md)

---

## 📝 维护说明

### 测试原则
1. **快速**: 测试应该快速完成
2. **独立**: 每个测试应该独立运行
3. **明确**: 测试结果应该清晰明确
4. **有用**: 测试应该能发现实际问题

### 测试更新
- 添加新功能时同步添加测试
- 修复bug后添加回归测试
- 定期检查测试覆盖率
- 保持测试代码的可维护性

---

**最后更新**: 2025-03-26
