# Models

模型实现放在本目录下。

当前已添加：

- `models/sasrec/dataset.py` - SASRec 数据集处理
- `models/sasrec/model.py` - SASRec 模型实现
- `models/sasrec/__init__.py` - SASRec 模块初始化

评估指标已移至：

- `evaluation/metrics.py` - NDCG@10, HitRate@10 等评估指标

可选 baseline：

- `models/baselines/` - 可选 baseline，例如 PopRec、BPR、GRU4Rec

不要在模型代码中写死本机绝对路径。训练入口应从 `configs/` 读取数据路径和超参数。
