# SASRec

本目录用于放置 SASRec 模型实现。

建议后续结构：

```text
models/sasrec/
├── model.py
├── dataset.py
├── losses.py
└── utils.py
```

训练输入可使用：

- `data/processed/<category>/sasrec_sequence.txt`
- `data/processed/<category>/train.tsv`
- `data/processed/<category>/dev.tsv`
- `data/processed/<category>/test.tsv`

默认超参数见 `configs/sasrec_*.yaml`，训练同学可按验证集结果调整。
