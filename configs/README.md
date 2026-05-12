# 配置文件

## PyTorch 骨干（`llmrank_*.yaml`）

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

## 完整 LLMRank Rank（RecBole + OpenAI）

- `llmrank_recbole/`：导出 RecBole 原子文件、在 **test 划分用户** 上采样候选等（Python 包，位于仓库根目录）。
- `configs/llmrank_recbole/amazon_sequential.yaml`：RecBole `load_col`（与上游 ML-1M 风格一致）。
- `configs/llmrank_recbole/openai_api.yaml.example`：复制到 `secrets/openai_api.yaml` 后填写 `api_key`。

运行评估（会调用 OpenAI，产生费用）：

```powershell
conda activate llmrec
pip install -r requirements-llmrank-recbole.txt
python scripts/run_llmrank_recbole_rank.py --category Industrial_and_Scientific --prepare --num-eval-users 200 --device cuda
```
