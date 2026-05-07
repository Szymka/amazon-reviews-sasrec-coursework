# Models

模型实现放在本目录下。当前预处理数据已经支持 SASRec 风格序列推荐，后续可新增：

- `models/sasrec/`：SASRec 模型实现
- `models/baselines/`：可选 baseline，例如 PopRec、BPR、GRU4Rec

不要在模型代码中写死本机绝对路径。训练入口应从 `configs/` 读取数据路径和超参数。
