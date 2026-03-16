# 代码注释规范化规则 (Code Annotation Standards)

## 1. 总体原则

### 1.1 注释语言策略
- **文档字符串**：使用英文（便于国际化和工具兼容）
- **技术细节注释**：可使用中文（便于理解和维护）
- **变量/函数名**：保持英文（符合Python命名规范）

### 1.2 注释层次结构
```
文件级 → 模块级 → 类级 → 函数级 → 代码块级 → 行级
```

## 2. 具体规范

### 2.1 文件级注释（File-level）

每个Python文件开头必须包含：

```python
"""
[Module Name] Module

[Brief description of module functionality]

[More detailed description if needed]

Author: [Author Name]
Date: YYYY-MM-DD
Version: X.X.X
"""

# Standard library imports
import os
from pathlib import Path

# Third-party imports
import torch
import torchaudio

# Local imports
from src.config import cfg
```

**要求：**
- 使用三重双引号 `"""`
- 模块名使用英文，简洁明了
- 包含作者、日期、版本信息
- 导入语句按标准库、第三方库、本地库分组

### 2.2 类级注释（Class-level）

```python
class ClassName(ParentClass):
    """Brief class description.
    
    Detailed description of class functionality, purpose, and usage.
    
    Args:
        param1 (type): Description of parameter
        param2 (type, optional): Description of optional parameter with default
        
    Attributes:
        attr1 (type): Description of attribute
        attr2 (type): Description of attribute
    """
```

**要求：**
- 第一行简短描述
- 详细描述（可选）
- Args部分说明所有构造函数参数
- Attributes部分说明重要属性
- 包含类型提示

### 2.3 函数级注释（Function-level）

```python
def function_name(param1: type, param2: type = default) -> return_type:
    """Brief function description.
    
    Detailed description of function functionality and algorithm.
    
    Args:
        param1 (type): Description of parameter
        param2 (type, optional): Description of optional parameter
        
    Returns:
        return_type: Description of return value
        
    Raises:
        ExceptionType: Description of when exception is raised
    """
```

**要求：**
- 包含完整的类型提示
- Args、Returns、Raises部分完整
- 参数和返回值都要有类型和描述

### 2.4 代码块级注释（Block-level）

```python
# ==========================
# 1. Block Description
# ==========================
# Code block here

# --------------------------
# 2. Sub-block Description  
# --------------------------
# Sub-block code here
```

**要求：**
- 使用等号分隔主要代码块
- 使用连字符分隔次要代码块
- 每个代码块有清晰的描述

### 2.5 行级注释（Line-level）

```python
# 计算chunk大小（采样点数）
self.chunk_size = int(self.sample_rate * self.chunk_len)

# 关键：添加极小值防止log(0)数值错误
eps = 1e-8
x_log = torch.log10(x + eps)

# 适配U-Net架构：裁剪最后一帧使频率维度为256（2^8）
output = input[:, :-1, :]
```

**要求：**
- 注释在代码上方或右侧
- 解释"为什么"而不是"是什么"
- 关键算法步骤必须注释

## 3. 特殊注释规范

### 3.1 TODO/FIXME注释

```python
# TODO: 添加采样率转换功能
# FIXME: 这里需要优化内存使用  
# NOTE: 关键算法步骤，不要轻易修改
# WARNING: 确保输入数据已经预处理
# HACK: 临时解决方案，需要重构
```

### 3.2 算法关键步骤注释

```python
# [ALGORITHM] Log域变换：提高数值稳定性
# 原因：直接在线性域训练容易产生数值溢出
x_log = torch.log10(x + eps)

# [ALGORITHM] 掩膜机制：输出0-1范围的乘法掩膜
# 1.0 = 完全保留，0.0 = 完全消除
mask = self.dec1(d2_cat)
output = x * mask
```

### 3.3 配置参数注释

```python
class Config:
    # ==========================
    # Audio Processing Parameters
    # ==========================
    SAMPLE_RATE = 16000      # 采样率：CD质量的一半，适合语音处理
    CHUNK_LEN = 3.0          # 音频片段长度：3秒包含足够上下文信息
    N_FFT = 512              # FFT窗口：2的9次方，适合频谱分析
    HOP_LENGTH = 128         # 跳跃长度：N_FFT的1/4，保证时频分辨率平衡
```

## 4. 质量检查清单

### 4.1 文件级检查
- [ ] 文件头包含模块描述、作者、日期、版本
- [ ] 导入语句按标准库、第三方库、本地库分组
- [ ] 模块级变量有类型提示和说明

### 4.2 类级检查
- [ ] 类有完整的docstring描述
- [ ] 所有参数和属性都有说明
- [ ] 复杂方法有详细注释

### 4.3 函数级检查
- [ ] 函数有类型提示
- [ ] 参数、返回值、异常都有说明
- [ ] 算法关键步骤有注释

### 4.4 代码级检查
- [ ] 魔法数字有常量定义或注释
- [ ] 复杂逻辑有行级注释
- [ ] 代码块有分隔注释

## 5. 工具和自动化

### 5.1 代码检查工具

```bash
# 安装工具
pip install flake8 black isort mypy

# 代码格式化
black src/
isort src/

# 类型检查
mypy src/

# 注释检查
flake8 src/ --select=D
```

### 5.2 文档生成

```bash
# 安装Sphinx
pip install sphinx sphinx-rtd-theme

# 生成文档
sphinx-build -b html docs/ docs/_build/
```

## 6. 实施优先级

### 高优先级（核心模块）
1. `src/config.py` - 配置文件
2. `src/dataset.py` - 数据集模块
3. `src/models/unet_v2.py` - 主要模型

### 中优先级（训练评估）
1. `src/train.py` / `src/train_v2.py` - 训练脚本
2. `src/evaluate.py` - 评估模块
3. `inference.py` - 推理脚本

### 低优先级（辅助模块）
1. `src/models/` 下的其他模型文件
2. `src/evaluation/` 下的评估工具
3. `tests/` 下的测试文件

## 7. 示例代码

### 完整的类示例

```python
"""
Audio Howling Suppression Dataset Module

This module implements the HowlingDataset class for loading and preprocessing
audio data, supporting paired clean and howling audio files.

Author: Research Team
Date: 2024-12-14
Version: 2.0.0
"""

import os
import torch
import torchaudio
from torch.utils.data import Dataset
from src.config import cfg


class HowlingDataset(Dataset):
    """Audio howling suppression dataset.
    
    Inherits from torch.utils.data.Dataset for loading paired clean and howling
    audio files with preprocessing capabilities.
    
    Args:
        clean_dir (str or Path): Directory path for clean audio files
        howling_dir (str or Path): Directory path for howling audio files  
        sample_rate (int, optional): Audio sample rate in Hz. Defaults to 16000.
        chunk_len (float, optional): Audio chunk length in seconds. Defaults to 3.0.
        
    Attributes:
        clean_dir (Path): Clean audio directory path
        howling_dir (Path): Howling audio directory path
        sample_rate (int): Audio sample rate
        chunk_size (int): Number of samples per chunk
    """
    
    def __init__(self, clean_dir, howling_dir, sample_rate=None, chunk_len=None):
        self.clean_dir = clean_dir
        self.howling_dir = howling_dir
        
        # 使用配置文件默认值
        self.sample_rate = sample_rate or cfg.SAMPLE_RATE
        self.chunk_len = chunk_len or cfg.CHUNK_LEN
        
        # 计算chunk大小（采样点数）
        self.chunk_size = int(self.sample_rate * self.chunk_len)
        
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get data sample by index.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple[torch.Tensor, torch.Tensor]: (noisy_mag, clean_mag) 
                Preprocessed spectrogram pair
        """
        # ==========================
        # 1. Audio Loading
        # ==========================
        file_name = self.filenames[idx]
        # ... implementation
        
        # ==========================
        # 2. Audio Preprocessing  
        # ==========================
        # 关键：Log域变换提高数值稳定性
        x_log = torch.log10(x + 1e-8)
        # ... implementation
```

---

**注意：本规则文档应与代码同步更新，确保所有新代码都遵循此标准。**
