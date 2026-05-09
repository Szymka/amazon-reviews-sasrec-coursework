# Evaluation

评估脚本放在本目录。

已添加：

- `evaluation/metrics.py` - 评估指标实现（NDCG@10, HitRate@10, Recall@10, MRR@10, Precision@10）
- `evaluation/evaluate_topk.py` - Top-K 评估脚本

推荐指标：

- `HitRate@10`
- `NDCG@10`
- `Recall@10`

评估输入通常包括：

- `data/processed/<category>/test.tsv`
- 模型生成的 Top-K 推荐结果

基本调用形式：

```powershell
python evaluation/evaluate_topk.py --config configs/sasrec_industrial.yaml --model-path train/sasrec_Industrial_and_Scientific_best.pth
```

最终表格放入 `results/tables/`，图表放入 `results/figures/`。
