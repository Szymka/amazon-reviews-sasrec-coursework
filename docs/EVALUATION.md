# Evaluation

本文档面向评估同学，说明推荐系统实验结果的计算和保存方式。

## 推荐指标

### 主要指标

训练脚本自动计算以下指标：

| 指标             | 计算公式              | 说明                           |
| ---------------- | --------------------- | ------------------------------ |
| **HitRate@10**   | 命中样本数 / 总样本数 | 真实目标是否出现在Top-10推荐中 |
| **NDCG@10**      | 归一化折损累积增益    | 衡量推荐列表的排序质量         |
| **Recall@10**    | 命中样本数 / 总样本数 | 与HitRate@10相同               |
| **MRR@10**       | 平均倒数排名          | 真实目标排名的倒数平均值       |
| **Precision@10** | 命中数 / 10           | Top-10推荐中的正确比例         |

### NDCG@10 计算逻辑

NDCG@10 是核心评估指标，计算方式为：

```
NDCG@10 = 1/log2(rank + 1)  （当真实目标在第rank位时）
```

| 排名       | NDCG贡献 |
| ---------- | -------- |
| 第1位      | 1.000    |
| 第2位      | 0.500    |
| 第3位      | 0.333    |
| 第10位     | 0.105    |
| 不在Top-10 | 0        |

## 评估函数位置

评估函数实现位于：

```text
evaluation/metrics.py
```

包含以下函数：

- `hit_rate_at_k(predicted, target, k=10)`
- `ndcg_at_k(predicted, target, k=10)`
- `recall_at_k(predicted, target, k=10)`
- `mrr_at_k(predicted, target, k=10)`
- `precision_at_k(predicted, target, k=10)`
- `evaluate(logits, targets, k=10)` - 返回所有指标的字典

## 测试集位置

测试集文件在：

```text
data/processed/<category>/test.tsv
```

每行包含字段：

- `user_id_int` - 用户整数ID
- `target_id` - 目标商品ID
- `rating` - 评分
- `timestamp` - 时间戳
- `seq_ids` - 空格分隔的历史商品序列
- `raw_user_id` - 原始用户ID
- `raw_parent_asin` - 原始商品ASIN

## 训练脚本输出

### 测试结果文件

训练完成后，测试结果会自动保存到两个位置：

1. **训练目录**（用于模型检查）：

   ```text
   train/seqrec_{category}_test_results.json
   ```

2. **结果目录**（用于报告）：
   ```text
   results/tables/seqrec_{category}_test_results.json
   ```

### 输出格式

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

## 结果保存建议

### 表格

表格保存到：

```text
results/tables/
```

建议包含的表格：

- 不同类别对比表
- 不同超参数实验结果表
- 指标均值和标准差表

### 图表

图表保存到：

```text
results/figures/
```

建议生成的图表：

- 训练/验证损失曲线
- 验证集NDCG@10变化曲线
- 不同类别指标对比图
- 超参数敏感性分析图

## 三个类别分别汇报

报告中需要对三个类别分别汇报结果：

1. **Industrial_and_Scientific**
2. **Musical_Instruments**
3. **CDs_and_Vinyl**

### 数据集统计

| 类别                      | 用户数  | 商品数 | 交互数    | 平均序列长度 |
| ------------------------- | ------- | ------ | --------- | ------------ |
| Industrial_and_Scientific | 50,985  | 25,848 | 412,947   | 8.10         |
| Musical_Instruments       | 57,439  | 24,587 | 511,836   | 8.91         |
| CDs_and_Vinyl             | 123,876 | 89,370 | 1,552,764 | 12.53        |

## 多次实验结果处理

如果使用多次随机种子，应报告：

- **均值**：各指标的平均值
- **标准差**：各指标的标准差

结果文件应保留在 `results/tables/` 中，格式可以是 CSV 或 Markdown 表格。

## 评估命令示例

### 运行训练并评估

```powershell
# 训练并评估单个类别
python -m train.train_seqrec --config configs/seqrec_industrial.yaml --device cuda

# 训练并评估所有类别
python -m train.train_seqrec --config configs/seqrec_industrial.yaml --device cuda
python -m train.train_seqrec --config configs/seqrec_musical.yaml --device cuda
python -m train.train_seqrec --config configs/seqrec_cds.yaml --device cuda
```

### 单独评估已训练模型

```powershell
python -m evaluation.evaluate_topk \
    --category Industrial_and_Scientific \
    --model-path train/seqrec_Industrial_and_Scientific_best.pth \
    --processed-root data/processed
```

## 注意事项

1. **Top-K 设置**：评估时使用 `topk=10`，与作业要求一致
2. **数据划分**：测试集为每个用户的最后一个交互（第N个）
3. **结果一致性**：确保训练和评估使用相同的随机种子
4. **GPU加速**：建议使用GPU进行评估，加快推理速度
