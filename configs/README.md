# 配置文件（LLMRank 顺序骨干）

`llmrank_*.yaml` 为 **Amazon coursework** 扁平键值配置：指向 `data/processed/<category>/` 下的 `train.tsv`、`dev.tsv`、`test.tsv`、`stats.json`，并设置 `train/train_llmrank.py` 的超参数。

| 文件 | `category` |
| --- | --- |
| `llmrank_industrial.yaml` | `Industrial_and_Scientific` |
| `llmrank_musical.yaml` | `Musical_Instruments` |
| `llmrank_cds.yaml` | `CDs_and_Vinyl` |
| `llmrank_tiny.yaml` | `examples/tiny_sample`（冒烟） |

训练：

```powershell
conda activate llmrec
python -m train.train_llmrank --config configs/llmrank_industrial.yaml --device cuda
```
