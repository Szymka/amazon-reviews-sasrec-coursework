# Model Training

本文档面向模型训练同学，说明后续训练代码应该如何接入当前 processed 数据。

## 推荐代码位置

- 训练入口脚本建议放在 `train/`。
- 模型实现建议放在 `models/`。
- SASRec 相关实现建议放在 `models/sasrec/`。
- 配置文件放在 `configs/`。

不要在训练脚本里写死某个同学电脑的绝对路径。统一使用命令行参数或相对路径，例如 `data/processed/<category>/train.tsv`。

## 推荐训练输入

SASRec 训练可以优先读取：

```text
data/processed/<category>/sasrec_sequence.txt
```

如果训练逻辑需要显式 train/dev/test 样本，也可以读取：

```text
data/processed/<category>/train.tsv
data/processed/<category>/dev.tsv
data/processed/<category>/test.tsv
```

常用映射和统计文件：

```text
data/processed/<category>/user2id.json
data/processed/<category>/item2id.json
data/processed/<category>/stats.json
```

## 配置文件

基础配置已放在：

```text
configs/sasrec_industrial.yaml
configs/sasrec_musical.yaml
configs/sasrec_cds.yaml
```

其中包含 `maxlen`、`hidden_units`、`num_blocks`、`num_heads`、`dropout_rate`、`learning_rate`、`batch_size`、`num_epochs`、`topk` 和 `seed` 等默认建议参数。训练同学可以根据实验资源和验证集结果调整。

## 评估衔接

训练脚本应保存模型在 dev/test 上的预测结果或指标摘要，供 `evaluation/` 中的评估脚本计算和汇总。

推荐指标：

- `HitRate@10`
- `NDCG@10`
- `Recall@10`

模型 checkpoint 不应提交到 GitHub。请把大文件保存在本地、实验服务器或小组共享盘，并在 `results/` 中只保留表格、图和轻量摘要。
