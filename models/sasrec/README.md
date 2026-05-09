# SASRec

本目录用于放置 SASRec 模型实现。

当前已添加：

```text
models/sasrec/
└── dataset.py
```

其中 `dataset.py` 已支持：

- 读取 `data/processed/<category>/train.tsv`
- 读取 `data/processed/<category>/dev.tsv`
- 读取 `data/processed/<category>/test.tsv`
- 按 `maxlen` 截断历史序列
- 使用 `0` 作为 padding
- 根据 config 或类别目录构建 train/dev/test 数据集

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
