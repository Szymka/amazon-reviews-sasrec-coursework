# LLMRank（coursework 适配）

本目录布局对齐上游仓库 [`LLMRank`](https://github.com/RUCAIBox/LLMRank) 的 `llmrank/` 包：`model/sasrec.py` 对应上游顺序骨干；**数据**则直接消费 Amazon Reviews coursework 预处理得到的 **TSV**（见 `dataset.py`），无需 RecBole `.inter` 格式。

训练入口：

```powershell
conda activate llmrec
python -m train.train_llmrank --config configs/llmrank_industrial.yaml --device cuda
```

论文中的 **Rank**（大模型零样本重排）依赖 OpenAI / RecBole 流水线；本作业仓库仅实现 **可监督训练的顺序骨干** 并在 `data/processed/` 上完成 Top-K 评估。
