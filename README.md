# Amazon Reviews 2023 Recommendation System Coursework

Team coursework scaffold: Amazon Reviews 2023 5-Core → processed sequential splits under `data/processed/`, trained with the **LLMRank** sequential backbone (PyTorch, no RecBole).

---

## Current status

### Done

- Raw 5-Core files for three categories (local `data/raw/`)
- Preprocessing to `data/processed/<category>/{train,dev,test}.tsv` + `stats.json` (+ auxiliary line files)
- **LLMRank** sequential model aligned with upstream `LLMRank/llmrank/model/sasrec.py` logic: `models/llmrank/model/sasrec.py`
- Training / early stopping / checkpoints: `train/train_llmrank.py`
- Top-K metrics: `evaluation/metrics.py`, `evaluation/evaluate_topk.py`
- Configs: `configs/llmrank_*.yaml`

### In progress

- Hyperparameter search on three categories
- Final runs and report

---

## Categories

| Category                    | Users   | Items  | Interactions | Avg Seq Len |
| --------------------------- | ------- | ------ | ------------ | ----------- |
| `Industrial_and_Scientific` | 50,985  | 25,848 | 412,947      | 8.10        |
| `Musical_Instruments`       | 57,439  | 24,587 | 511,836      | 8.91        |
| `CDs_and_Vinyl`             | 123,876 | 89,370 | 1,552,764    | 12.53       |

---

## Top-K sequential setup

Per user, sequence by time: **[i₁, …, i_N]** (item ids 1…`num_items`, **0** = padding).

- **Train** (`train.tsv`): history **[i₁, …, i_{N−2}]**, target next item in the split (see `scripts/preprocess_to_seqrec.py`).
- **Dev** (`dev.tsv`): history **[i₁, …, i_{N−2}]**, target **i_{N−1}**.
- **Test** (`test.tsv`): history **[i₁, …, i_{N−1}]**, target **i_N**.

Metrics: **NDCG@K** and **HitRate@K** with **K = `topk`** in the YAML (default 10), computed in `evaluation/metrics.py` over full-vocabulary scores.

### Model (LLMRank backbone)

Upstream [LLMRank](https://github.com/RUCAIBox/LLMRank) couples an LLM ranker with a sequential candidate generator. This repo ships the **trainable causal Transformer backbone** only: `CourseworkSequenceDataset` maps TSV columns `seq_ids` / `target_id` to `input_ids` / `target_id`. The optional LLM **Rank** stage (API + RecBole) is **not** wired here.

---

## Repository layout

```text
.
├── README.md
├── README_data.md
├── requirements.txt
├── requirements-llmrec.txt   # conda env llmrec (torch + pandas + tqdm)
├── docs/
├── scripts/
│   ├── download_amazon5core.py
│   ├── check_raw_data.py
│   ├── preprocess_to_seqrec.py
│   ├── check_processed_data.py
│   └── test_project.py
├── models/llmrank/
│   ├── dataset.py
│   └── model/
│       └── sasrec.py
├── train/
│   ├── train_llmrank.py
│   └── check_data_loading.py
├── evaluation/
├── configs/
│   ├── llmrank_industrial.yaml
│   ├── llmrank_musical.yaml
│   ├── llmrank_cds.yaml
│   └── llmrank_tiny.yaml
├── results/
├── examples/tiny_sample/
└── data/
    ├── raw/
    └── processed/
```

---

## Quick start

### 1. Conda env `llmrec`

```powershell
conda create -n llmrec python=3.10 -y
conda activate llmrec
pip install -r requirements-llmrec.txt
```

### 2. Data

Place raw gz under `data/raw/<category>/` (see `README_data.md`), then preprocess:

```powershell
conda activate llmrec
python scripts/check_raw_data.py
python scripts/preprocess_to_seqrec.py --categories Industrial_and_Scientific --overwrite
python scripts/preprocess_to_seqrec.py --categories Musical_Instruments --overwrite
python scripts/preprocess_to_seqrec.py --categories CDs_and_Vinyl --overwrite
python scripts/check_processed_data.py --categories Industrial_and_Scientific Musical_Instruments CDs_and_Vinyl
```

### 3. Train / test

```powershell
conda activate llmrec
python -m train.train_llmrank --config configs/llmrank_industrial.yaml --device cuda
python -m train.train_llmrank --config configs/llmrank_musical.yaml --device cuda
python -m train.train_llmrank --config configs/llmrank_cds.yaml --device cuda
```

Quick CPU smoke test on the bundled tiny sample:

```powershell
python -m train.train_llmrank --config configs/llmrank_tiny.yaml --device cpu
```

Artifacts:

```text
train/llmrank_{category}_best.pth
train/llmrank_{category}_best_config.json
train/llmrank_{category}_test_results.json
results/tables/llmrank_{category}_test_results.json
```

### 4. Evaluate a checkpoint

```powershell
python evaluation/evaluate_topk.py --help
```

---

## Hyperparameters (YAML)

| Key                   | Role                          |
| --------------------- | ----------------------------- |
| `maxlen`              | Max sequence length           |
| `hidden_units`        | Embedding size                |
| `num_blocks`          | Transformer blocks            |
| `num_heads`           | Attention heads               |
| `dropout_rate`        | Dropout                       |
| `learning_rate`       | AdamW lr                      |
| `batch_size`          | Batch size                    |
| `num_epochs`          | Max epochs                    |
| `early_stop_patience` | Early stopping                |
| `topk`                | *k* for NDCG@*k* / HR@*k*     |
| `train_loss_mode`     | `sampled` or `full` softmax   |

---

## Documentation

- `README_data.md` — data layout
- `docs/DATA_DOWNLOAD.md`, `docs/DATA_PREPROCESS.md`
- `docs/MODEL_TRAINING.md`, `docs/EVALUATION.md`
- `docs/TEAM_HANDOFF.md`, `docs/GITHUB_COLLABORATION.md`

---

## Notes

- Do not commit full `data/raw/` or `data/processed/`, or large `.pth` checkpoints.
- Use **relative paths** in configs and scripts.

---

## Citation (LLMRank)

```bibtex
@inproceedings{hou2024llmrank,
  title={Large Language Models are Zero-Shot Rankers for Recommender Systems},
  author={Hou, Yupeng and Zhang, Junjie and Lin, Zihan and Lu, Hongyu and Xie, Ruobing and McAuley, Julian and Zhao, Wayne Xin},
  booktitle={ECIR},
  year={2024}
}
```

_Last updated: May 12, 2026_
