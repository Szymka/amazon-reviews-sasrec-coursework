# Train

训练入口脚本放在本目录。

建议后续新增：

```text
train/check_data_loading.py
train/train_sasrec.py
```

当前已添加：

```powershell
python train/check_data_loading.py --config configs/sasrec_industrial.yaml
```

该脚本用于在训练前检查 processed 数据是否能被正确读取，并验证 `maxlen`、padding 和 train/dev/test 划分。

基本调用形式建议为：

```powershell
python train/train_sasrec.py --config configs/sasrec_industrial.yaml
```

训练脚本应读取配置文件中的相对路径，不要写死本机绝对路径。模型 checkpoint、日志、`runs/`、`wandb/` 不提交到 GitHub。
