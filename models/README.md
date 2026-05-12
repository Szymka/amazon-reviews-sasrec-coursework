# Models

## `llmrank/`

**LLMRank** 论文中的顺序候选生成骨干（与上游 `LLMRank/llmrank/model/sasrec.py` 中 RecBole 包装器同构的打分逻辑；本仓库为 **无 RecBole** 的 PyTorch 实现）。从 `data/processed/<category>/` 读取 `train.tsv` / `dev.tsv` / `test.tsv` 与 `stats.json`，由 `CourseworkSequenceDataset` 提供 `input_ids` / `target_id`。

```text
models/llmrank/
├── __init__.py
├── dataset.py          # CourseworkSequenceDataset / build_*_datasets / load_simple_yaml
├── model/
│   ├── __init__.py
│   └── sasrec.py       # LLMRankSequentialModel, build_llmrank_model, predict_on_subsets
└── README.md
```

训练入口：**`python -m train.train_llmrank --config configs/llmrank_<suffix>.yaml`**（例如 `llmrank_industrial.yaml`）。
