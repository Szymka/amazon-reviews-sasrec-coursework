# Evaluation

本文档面向评估同学，说明推荐系统实验结果的计算和保存方式。

## 推荐指标

建议至少汇报：

- `HitRate@10`
- `NDCG@10`
- `Recall@10`

如有时间，可以补充 `MRR@10`、`Precision@10`、训练时间和推理时间。

## 测试集位置

测试集文件在：

```text
data/processed/<category>/test.tsv
```

每行包含当前用户的历史序列 `seq_ids` 和目标商品 `target_id`。评估时需要结合模型输出的 Top-K 推荐列表计算指标。

## 结果保存建议

表格保存到：

```text
results/tables/
```

图表保存到：

```text
results/figures/
```

报告中需要对三个类别分别汇报结果：

- `Industrial_and_Scientific`
- `Musical_Instruments`
- `CDs_and_Vinyl`

如果使用多次随机种子，应报告均值和标准差，并在 `results/tables/` 中保留可复查的 CSV 或 Markdown 表格。
