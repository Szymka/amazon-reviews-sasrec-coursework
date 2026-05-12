# Model Training

本文档面向模型训练同学，说明如何使用现有的训练代码进行实验。

## 当前状态

✅ 已完成的工作：

- LLMRank 顺序模型（`models/llmrank/model/sasrec.py`）
- 训练脚本（`train/train_llmrank.py`）
- 配置文件（`configs/llmrank_*.yaml`）
- 评估函数（`evaluation/metrics.py`）
- 数据加载模块（`models/llmrank/dataset.py`）

## 代码结构

| 目录/文件                  | 说明                                         |
| -------------------------- | -------------------------------------------- |
| `train/train_llmrank.py`    | 主训练脚本，支持多数据集切换、早停、模型保存 |
| `models/llmrank/model/sasrec.py` | LLMRank 顺序骨干（PyTorch）            |
| `models/llmrank/dataset.py` | 数据集加载和预处理                           |
| `evaluation/metrics.py`    | 评估指标计算                                 |
| `configs/llmrank_*.yaml`   | 超参数配置文件                               |

## 训练命令

### 基础训练命令

```powershell
# 训练单个类别（使用默认配置）
python -m train.train_llmrank --config configs/llmrank_industrial.yaml

# 明确指定使用 GPU
python -m train.train_llmrank --config configs/llmrank_industrial.yaml --device cuda

# 训练所有类别
python -m train.train_llmrank --config configs/llmrank_industrial.yaml --device cuda
python -m train.train_llmrank --config configs/llmrank_musical.yaml --device cuda
python -m train.train_llmrank --config configs/llmrank_cds.yaml --device cuda
```

### 使用部分数据快速测试

```powershell
# 使用 1000 个用户快速验证
python -m train.train_llmrank --config configs/llmrank_industrial.yaml --device cuda --max-users 1000

# 使用一半数据量
python -m train.train_llmrank --config configs/llmrank_industrial.yaml --device cuda --max-users 25000
```

## 超参数配置

### 配置文件位置

基础配置已放在：

```text
configs/llmrank_industrial.yaml
configs/llmrank_musical.yaml
configs/llmrank_cds.yaml
```

### 超参数说明

| 参数                  | 说明            | 默认值 | 建议搜索范围        |
| --------------------- | --------------- | ------ | ------------------- |
| `maxlen`              | 序列最大长度    | 50     | 30, 50, 100         |
| `hidden_units`        | 嵌入维度        | 64     | 64, 128, 256        |
| `num_blocks`          | Transformer块数 | 2      | 1, 2, 4             |
| `num_heads`           | 注意力头数      | 2      | 2, 4, 8             |
| `dropout_rate`        | Dropout率       | 0.2    | 0.1, 0.2, 0.3       |
| `learning_rate`       | 学习率          | 0.001  | 0.0001, 0.001, 0.01 |
| `batch_size`          | 批次大小        | 128    | 64, 128, 256        |
| `num_epochs`          | 最大训练轮数    | 100    | 50, 100, 200        |
| `topk`                | Top-K评估       | 10     | 10（固定）          |
| `early_stop_patience` | 早停耐心值      | 5      | 3, 5, 10            |
| `seed`                | 随机种子        | 42     | 42, 123, 2024       |

### 修改配置文件

编辑对应的 YAML 文件调整超参数：

```yaml
category: Industrial_and_Scientific
maxlen: 50
hidden_units: 64
num_blocks: 2
num_heads: 2
dropout_rate: 0.2
learning_rate: 0.001
batch_size: 128
num_epochs: 100
topk: 10
early_stop_patience: 5
early_stop_min_delta: 0.0
```

## 训练流程

### 1. 数据加载

训练脚本自动从以下位置加载数据：

```text
data/processed/<category>/train.tsv
data/processed/<category>/dev.tsv
data/processed/<category>/test.tsv
```

数据集统计信息从 `stats.json` 读取。

### 2. 模型初始化

根据配置文件中的超参数创建模型：

```python
from models.llmrank.model import LLMRankSequentialModel

model = LLMRankSequentialModel(
    num_items=stats['num_items'],
    maxlen=args.maxlen,
    hidden_units=args.hidden_units,
    num_blocks=args.num_blocks,
    num_heads=args.num_heads,
    dropout_rate=args.dropout_rate,
    padding_id=0
)
```

### 3. 训练循环

- 使用交叉熵损失函数
- 使用 Adam 优化器
- 每轮结束后在验证集上评估 NDCG@10
- 启用早停机制，当验证集 NDCG@10 连续 `early_stop_patience` 轮无提升时停止训练

### 4. 模型保存

- 自动保存最佳模型到 `train/llmrank_{category}_best.pth`
- 同时保存配置信息到 `train/llmrank_{category}_best_config.json`

## 评估与输出

### 测试集评估

训练完成后自动在测试集上评估：

```powershell
Test Results for Industrial_and_Scientific:
  Loss: 38.5822
  NDCG@10: 0.3448
  HitRate@10: 0.7523
  Recall@10: 0.7523
  MRR@10: 0.1548
  Precision@10: 0.0752
```

### 结果保存位置

测试结果会自动保存到两个位置：

1. **训练目录**（用于模型检查）：

   ```text
   train/llmrank_{category}_test_results.json
   ```

2. **结果目录**（用于报告）：

   ```text
   results/tables/llmrank_{category}_test_results.json
   ```

### JSON 输出格式

```json
{
  "category": "Industrial_and_Scientific",
  "best_epoch": 5,
  "test_loss": 38.5822,
  "test_metrics": {
    "hit_rate": 0.7523,
    "ndcg": 0.3448,
    "recall": 0.7523,
    "mrr": 0.1548,
    "precision": 0.0752
  },
  "hyperparameters": {
    "maxlen": 50,
    "hidden_units": 64,
    "num_blocks": 2,
    "num_heads": 2,
    "dropout_rate": 0.2,
    "learning_rate": 0.001,
    "batch_size": 128,
    "num_epochs": 100,
    "topk": 10,
    "seed": 42
  }
}
```

## 实验记录

### 建议的实验记录格式

| 实验ID | 类别       | maxlen | hidden_units | num_heads | dropout_rate | lr    | batch_size | Dev NDCG@10 | Test NDCG@10 |
| ------ | ---------- | ------ | ------------ | --------- | ------------ | ----- | ---------- | ----------- | ------------ |
| EXP001 | Industrial | 50     | 64           | 2         | 0.2          | 0.001 | 128        | 0.352       | 0.348        |
| EXP002 | Industrial | 50     | 128          | 4         | 0.2          | 0.001 | 128        | 0.365       | 0.361        |

### 多次实验

如果使用多次随机种子，应报告：

- **均值**：各指标的平均值
- **标准差**：各指标的标准差

## 注意事项

### 路径设置

- ✅ 使用相对路径（如 `data/processed/`）
- ❌ 不要写死绝对路径（如 `D:\data\processed\`）

### GPU 使用

- 建议使用 GPU 训练以加快速度
- 如果遇到 OOM 错误，减小 `batch_size` 或 `hidden_units`
- 使用 `--device cuda` 明确指定使用 GPU

### 随机种子

- 固定随机种子以确保实验可复现
- 建议使用不同的种子进行多次实验
- 在配置文件中设置 `seed` 参数

### 早停机制

- 训练脚本已内置早停功能
- 监控验证集 NDCG@10
- 连续 `early_stop_patience` 轮无提升时自动停止

### GitHub 提交

- ❌ 不要上传模型检查点（`.pth`、`.pt` 文件）
- ❌ 不要上传训练日志
- ✅ 只上传代码、配置、文档和轻量结果
- ✅ 使用 `.gitignore` 排除大文件

## 常见问题

### Q: 如何验证 GPU 是否可用？

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

### Q: 遇到 OOM 错误怎么办？

减小以下参数：

- `batch_size`：从 128 减到 64 或 32
- `hidden_units`：从 64 减到 32
- `maxlen`：从 50 减到 30

### Q: 如何加载已训练的模型？

```powershell
python -m evaluation.evaluate_topk \
    --category Industrial_and_Scientific \
    --model-path train/llmrank_Industrial_and_Scientific_best.pth \
    --processed-root data/processed
```

### Q: 如何查看训练进度？

训练过程中会实时显示：

```
Epoch 1/100 | Time: 2.5s | Train Loss: 22.7158 | Dev Loss: 43.5127 | Dev NDCG@10: 0.5655 | Dev HR@10: 1.0000
  Best model saved to train\llmrank_Industrial_and_Scientific_best.pth
```
