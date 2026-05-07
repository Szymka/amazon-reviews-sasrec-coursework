# Evaluation

评估脚本放在本目录。

建议后续新增：

```text
evaluation/evaluate_topk.py
evaluation/metrics.py
```

推荐指标：

- `HitRate@10`
- `NDCG@10`
- `Recall@10`

评估输入通常包括：

- `data/processed/<category>/test.tsv`
- 模型生成的 Top-K 推荐结果

最终表格放入 `results/tables/`，图表放入 `results/figures/`。
