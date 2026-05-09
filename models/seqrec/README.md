# SeqRec

本目录包含 SeqRec (Sequential Recommendation) 模型的完整实现。该模型遵循 SASRec 论文的架构思想（Transformer + 位置编码 + 因果注意力），但在实现细节和技术栈上完全独立。

---

## 📁 文件结构

| 文件 | 说明 |
|------|------|
| `model.py` | SeqRec 模型实现（包含 Transformer 架构） |
| `dataset.py` | 数据集加载和预处理 |
| `__init__.py` | 模块初始化 |

---

## 🧠 模型架构

SeqRec 基于 Transformer 架构，包含以下核心组件：

### 1. 位置编码 (PositionalEncoding)
- 使用正弦/余弦函数编码序列位置信息
- 帮助模型理解序列顺序

### 2. 多头注意力 (MultiHeadAttention)
- 并行计算多个注意力头
- 捕捉不同类型的依赖关系
- 使用因果掩码防止未来信息泄露

### 3. 前馈网络 (PointWiseFFN)
- 两层全连接网络
- GELU 激活函数
- 增强模型表达能力

### 4. Transformer 块 (TransformerBlock)
- 多头注意力 + 残差连接 + LayerNorm
- 前馈网络 + 残差连接 + LayerNorm
- 使用 Pre-LayerNorm 结构提升训练稳定性

### 5. SeqRec 主模型
- Item Embedding 层
- 位置编码层
- 堆叠 Transformer 块
- 输出层归一化

---

## 📊 模型参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `num_items` | 商品总数 | - |
| `maxlen` | 序列最大长度 | 50 |
| `hidden_units` | 嵌入维度 | 64 |
| `num_blocks` | Transformer 块数 | 2 |
| `num_heads` | 注意力头数 | 2 |
| `dropout_rate` | Dropout 率 | 0.2 |
| `padding_id` | Padding 标识 | 0 |

---

## 🔧 使用方法

### 导入模型

```python
from models.seqrec.model import SASRec
from models.seqrec.dataset import build_category_datasets

# 构建数据集
train_dataset, dev_dataset, test_dataset = build_category_datasets(
    category='Industrial_and_Scientific',
    processed_root='data/processed',
    maxlen=50
)

# 创建模型
model = SASRec(
    num_items=25848,  # 从 stats.json 获取
    maxlen=50,
    hidden_units=64,
    num_blocks=2,
    num_heads=2,
    dropout_rate=0.2,
    padding_id=0
)

# 前向传播
output = model(input_ids)  # shape: (batch_size, seq_len, hidden_units)

# 预测
logits = model.predict(input_ids)  # shape: (batch_size, num_items)
```

### 数据集使用

```python
from models.seqrec.dataset import build_category_datasets

# 加载数据
train_dataset, dev_dataset, test_dataset = build_category_datasets(
    category='Industrial_and_Scientific',
    processed_root='data/processed',
    maxlen=50
)

# 使用 DataLoader
from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# 迭代数据
for batch in train_loader:
    input_ids = batch['input_ids']  # 历史序列
    target_id = batch['target_id']  # 目标商品
    # 训练逻辑...
```

---

## 📥 输入数据格式

### 训练数据 (train.tsv)

```
user_id_int	target_id	rating	timestamp	seq_ids	raw_user_id	raw_parent_asin
```

| 字段 | 说明 |
|------|------|
| `user_id_int` | 用户整数 ID |
| `target_id` | 目标商品 ID |
| `rating` | 评分 |
| `timestamp` | 时间戳 |
| `seq_ids` | 空格分隔的历史商品序列 |
| `raw_user_id` | 原始用户 ID |
| `raw_parent_asin` | 原始商品 ASIN |

### 序列处理

- 按 `maxlen` 截断历史序列
- 使用 `0` 作为 padding
- 序列格式：`[item1_id, item2_id, ..., itemN_id]`

---

## 📝 注意事项

1. **路径设置**：使用相对路径，不要写死绝对路径
2. **数据划分**：训练/验证/测试集已按作业要求划分
3. **Padding**：使用 0 作为 padding_id，在模型中会被忽略
4. **因果掩码**：模型内部自动处理因果注意力掩码

---

## 📚 参考

- SASRec 论文: [Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781)
- 配置文件: `configs/seqrec_*.yaml`
- 训练脚本: `train/train_seqrec.py`

---

*最后更新：2026年5月9日*
