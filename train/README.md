# Train

训练入口脚本放在本目录。

建议后续新增：

```text
train/train_sasrec.py
```

基本调用形式建议为：

```powershell
python train/train_sasrec.py --config configs/sasrec_industrial.yaml
```

训练脚本应读取配置文件中的相对路径，不要写死本机绝对路径。模型 checkpoint、日志、`runs/`、`wandb/` 不提交到 GitHub。
