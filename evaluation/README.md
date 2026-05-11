# Evaluation

本文档说明如何使用评估模块进行模型评估和结果分析。

---

## 📁 文件结构

| 文件               | 说明                                                                 |
| ------------------ | -------------------------------------------------------------------- |
| `metrics.py`       | 评估指标实现（NDCG@10, HitRate@10, Recall@10, MRR@10, Precision@10） |
| `evaluate_topk.py` | Top-K 评估脚本，支持加载已训练模型进行评估                           |

---

## 📊 评估指标

### 核心指标

| 指标             | 说明               | 计算公式              |
| ---------------- | ------------------ | --------------------- |
| **NDCG@10**      | 归一化折损累积增益 | 1/log2(rank + 1)      |
| **HitRate@10**   | 命中率             | 命中样本数 / 总样本数 |
| **Recall@10**    | 召回率             | 命中样本数 / 总样本数 |
| **MRR@10**       | 平均倒数排名       | 平均(1 / rank)        |
| **Precision@10** | 精确率             | 命中数 / 10           |

### NDCG@10 详细说明

NDCG@10 是核心评估指标，计算方式为：

```
NDCG@10 = 1/log2(rank + 1)  （当真实目标在第rank位时）
```

| 排名       | NDCG贡献 | 说明     |
| ---------- | -------- | -------- |
| 第1位      | 1.000    | 最佳情况 |
| 第2位      | 0.500    |          |
| 第3位      | 0.333    |          |
| 第5位      | 0.231    |          |
| 第10位     | 0.105    | 勉强命中 |
| 不在Top-10 | 0        | 未命中   |

---

## 🔧 使用方法

### 方式一：训练时自动评估

训练脚本 `train/train_seqrec.py` 会在训练完成后自动在测试集上评估，并将结果保存到：

```text
train/seqrec_{category}_test_results.json
results/tables/seqrec_{category}_test_results.json
```

### 方式二：单独评估已训练模型

```powershell
# 评估单个类别
python -m evaluation.evaluate_topk \
    --category Industrial_and_Scientific \
    --model-path train/seqrec_Industrial_and_Scientific_best.pth \
    --processed-root data/processed

# 评估所有类别
python -m evaluation.evaluate_topk --category Industrial_and_Scientific --model-path train/seqrec_Industrial_and_Scientific_best.pth --processed-root data/processed
python -m evaluation.evaluate_topk --category Musical_Instruments --model-path train/seqrec_Musical_Instruments_best.pth --processed-root data/processed
python -m evaluation.evaluate_topk --category CDs_and_Vinyl --model-path train/seqrec_CDs_and_Vinyl_best.pth --processed-root data/processed
```

### 方式三：在代码中使用评估函数

```python
from evaluation.metrics import evaluate, ndcg_at_k, hit_rate_at_k
import torch

# 模型预测结果 (batch_size, num_items)
logits = model.predict(input_ids)

# 真实目标 (batch_size,)
targets = batch['target_id']

# 计算所有指标
metrics = evaluate(logits, targets, k=10)
print(f"NDCG@10: {metrics['ndcg']:.4f}")
print(f"HitRate@10: {metrics['hit_rate']:.4f}")

# 单独计算某个指标
ndcg = ndcg_at_k(logits, targets, k=10)
hr = hit_rate_at_k(logits, targets, k=10)
```

---

## 📥 评估输入

### 必需文件

| 文件       | 路径                                     | 说明                       |
| ---------- | ---------------------------------------- | -------------------------- |
| 测试集     | `data/processed/<category>/test.tsv`     | 包含用户历史序列和目标商品 |
| 模型检查点 | `train/seqrec_{category}_best.pth`       | 已训练模型权重             |
| 统计信息   | `data/processed/<category>/stats.json`   | 商品数量等统计信息         |
| ID映射     | `data/processed/<category>/item2id.json` | 商品ID映射                 |

### test.tsv 格式

```
user_id_int	target_id	rating	timestamp	seq_ids	raw_user_id	raw_parent_asin
1	123	5	1609459200	"1 2 3 4 5"	user_001	B00001
```

---

## 📤 评估输出

### 控制台输出

```
Test Results for Industrial_and_Scientific:
  Loss: 38.5822
  NDCG@10: 0.3448
  HitRate@10: 0.7523
  Recall@10: 0.7523
  MRR@10: 0.1548
  Precision@10: 0.0752
```

### JSON 输出文件

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

---

## 📈 结果保存

### 表格结果

保存到 `results/tables/`：

```text
results/tables/seqrec_{category}_test_results.json
results/tables/comparison_table.csv
results/tables/hyperparameter_study.csv
```

### 图表结果

保存到 `results/figures/`：

```text
results/figures/ndcg_convergence.png
results/figures/loss_curve.png
results/figures/attention_visualization.png
```

---

## 🎯 三个类别评估

需要对以下三个类别分别进行评估：

1. **Industrial_and_Scientific**
2. **Musical_Instruments**
3. **CDs_and_Vinyl**

### 数据集统计

| 类别                      | 用户数  | 商品数 | 测试样本数 | 平均序列长度 |
| ------------------------- | ------- | ------ | ---------- | ------------ |
| Industrial_and_Scientific | 50,985  | 25,848 | 50,985     | 8.10         |
| Musical_Instruments       | 57,439  | 24,587 | 57,439     | 8.91         |
| CDs_and_Vinyl             | 123,876 | 89,370 | 123,876    | 12.53        |

---

## ⚠️ 注意事项

1. **Top-K 设置**：评估时使用 `k=10`，与作业要求一致
2. **数据划分**：测试集为每个用户的最后一个交互（第N个）
3. **结果一致性**：确保训练和评估使用相同的随机种子
4. **GPU加速**：建议使用GPU进行评估，加快推理速度
5. **路径设置**：使用相对路径，不要写死绝对路径

---

## 🔍 常见问题

### Q: 如何验证评估结果正确性？

使用 `examples/tiny_sample/` 数据进行测试，预期结果：

- HitRate@10 应该接近 1.0（因为数据简单）
- NDCG@10 应该在 0.3-0.5 之间

### Q: 评估时遇到 OOM 错误怎么办？

减小 `batch_size` 参数：

```powershell
python -m evaluation.evaluate_topk \
    --category Industrial_and_Scientific \
    --model-path train/seqrec_Industrial_and_Scientific_best.pth \
    --processed-root data/processed \
    --batch-size 64
```

### Q: 如何生成对比表格？

收集三个类别的结果后，使用 Python 或 Excel 制作对比表格：

```python
import json
import pandas as pd

categories = ['Industrial_and_Scientific', 'Musical_Instruments', 'CDs_and_Vinyl']
results = []

for cat in categories:
    with open(f'results/tables/seqrec_{cat}_test_results.json') as f:
        data = json.load(f)
        results.append({
            'Category': cat,
            'NDCG@10': data['test_metrics']['ndcg'],
            'HitRate@10': data['test_metrics']['hit_rate'],
            'MRR@10': data['test_metrics']['mrr']
        })

df = pd.DataFrame(results)
print(df.to_string(index=False))
```

---

_最后更新：2026年5月9日_
