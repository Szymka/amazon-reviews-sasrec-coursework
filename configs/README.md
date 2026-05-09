# Configs

本目录保存实验配置。当前 YAML 文件是 SASRec 的默认建议参数，不是最终调参结果。

配置文件：

- `seqrec_industrial.yaml`
- `seqrec_musical.yaml`
- `seqrec_cds.yaml`

训练同学可以调整 `maxlen`、`hidden_units`、`num_blocks`、`dropout_rate`、`learning_rate`、`batch_size`、`num_epochs` 等参数，但应保留相对路径，避免写死本机路径。
