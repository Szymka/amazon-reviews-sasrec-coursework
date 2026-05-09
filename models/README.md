# Models

本目录包含推荐系统模型的实现。

---

## 📁 当前模型

### SeqRec

**路径**: `models/seqrec/`

**说明**: SeqRec (Sequential Recommendation) 模型，遵循 SASRec 论文的架构思想（Transformer + 位置编码 + 因果注意力），但在实现细节和技术栈上完全独立。

**文件结构**:

```text
models/seqrec/
├── __init__.py      # 模块初始化
├── model.py         # SeqRec 模型实现
└── dataset.py       # 数据集加载和预处理
```

**核心组件**:

- `SASRec` - 主模型类
- `PositionalEncoding` - 位置编码
- `MultiHeadAttention` - 多头注意力机制
- `TransformerBlock` - Transformer 块
- `build_category_datasets` - 数据集构建函数

**实现说明**:

- 框架: PyTorch (独立实现，非原始 TensorFlow SASRec 的 fork)
- 损失函数: CrossEntropyLoss
- 注意力机制: PyTorch-native 因果掩码

---

## 📊 模型对比

| 模型       | 类型     | 特点                           | 状态        |
| ---------- | -------- | ------------------------------ | ----------- |
| **SeqRec** | 序列推荐 | 基于 Transformer，自注意力机制 | ✅ 已实现   |
| PopRec     | 基线     | 基于流行度                     | 📋 可选实现 |
| BPR        | 矩阵分解 | 贝叶斯个性化排序               | 📋 可选实现 |
| GRU4Rec    | 序列推荐 | 基于 RNN                       | 📋 可选实现 |

---

## 🔧 使用示例

### SeqRec

```python
from models.seqrec.model import SASRec
from models.seqrec.dataset import build_category_datasets

# 加载数据
train_dataset, dev_dataset, test_dataset = build_category_datasets(
    category='Industrial_and_Scientific',
    processed_root='data/processed',
    maxlen=50
)

# 创建模型
model = SASRec(
    num_items=25848,
    maxlen=50,
    hidden_units=64,
    num_blocks=2,
    num_heads=2,
    dropout_rate=0.2,
    padding_id=0
)
```

---

## 📂 目录结构

```text
models/
├── README.md              # 本文件
├── seqrec/                # SeqRec 模型
│   ├── README.md
│   ├── __init__.py
│   ├── model.py
│   └── dataset.py
└── baselines/             # 基线模型（可选）
    └── README.md
```

---

## 📝 开发规范

1. **路径设置**: 使用相对路径，不要写死绝对路径
2. **配置读取**: 从 `configs/` 读取数据路径和超参数
3. **代码风格**: 遵循 PEP 8 规范
4. **文档注释**: 为类和函数添加 docstring

---

## 📚 相关文档

- [SeqRec 详细文档](seqrec/README.md)
- [模型训练](../docs/MODEL_TRAINING.md)
- [配置文件](../configs/README.md)

---

_最后更新：2026年5月9日_
