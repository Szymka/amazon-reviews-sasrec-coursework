# Amazon Reviews 2023 Recommendation System Coursework

This repository is a team-oriented engineering scaffold for a recommendation-system coursework project based on Amazon Reviews 2023 5-Core data. The current experiment target is **SASRec-style sequential recommendation** implemented in PyTorch on three product categories.

**Implementation Note**: This project follows the architectural ideas of the SASRec paper (Transformer + positional encoding + causal attention), but is completely independent in implementation details and technical stack:

| Aspect            | Original SASRec (TensorFlow)        | This Implementation (PyTorch) |
| ----------------- | ----------------------------------- | ----------------------------- |
| Framework         | TensorFlow                          | PyTorch                       |
| Loss Function     | BPR (Bayesian Personalized Ranking) | CrossEntropyLoss              |
| Attention Mask    | TensorFlow-specific                 | PyTorch-native causal mask    |
| Training Pipeline | Custom TF Estimator                 | PyTorch native training loop  |

This is an **independent reimplementation**, not a fork or adaptation of the original SASRec codebase.

---

## ✅ Current Status

### Completed

- ✅ Raw Amazon Reviews 2023 5-Core files downloaded for three categories
- ✅ All three categories converted to sequential recommendation-ready processed files
- ✅ SeqRec model implementation (`models/seqrec/`)
- ✅ Training script with early stopping and model saving (`train/train_seqrec.py`)
- ✅ Evaluation metrics (NDCG@10, HitRate@10, etc.)
- ✅ Configuration files for all categories (`configs/seqrec_*.yaml`)
- ✅ Data loading and preprocessing pipeline

### In Progress

- 📋 Hyperparameter tuning on three categories
- 📋 Final model training and evaluation
- 📋 Report writing

---

## 📊 Categories

| Category                    | Users   | Items  | Interactions | Avg Seq Len |
| --------------------------- | ------- | ------ | ------------ | ----------- |
| `Industrial_and_Scientific` | 50,985  | 25,848 | 412,947      | 8.10        |
| `Musical_Instruments`       | 57,439  | 24,587 | 511,836      | 8.91        |
| `CDs_and_Vinyl`             | 123,876 | 89,370 | 1,552,764    | 12.53       |

---

## 📁 Repository Layout

```text
.
├── README.md                 # This file
├── README_data.md            # Data documentation
├── requirements.txt          # Python dependencies
├── docs/                     # Documentation
│   ├── DATA_DOWNLOAD.md
│   ├── DATA_PREPROCESS.md
│   ├── MODEL_TRAINING.md
│   ├── EVALUATION.md
│   ├── TEAM_HANDOFF.md
│   └── GITHUB_COLLABORATION.md
├── scripts/                  # Data processing scripts
│   ├── download_amazon5core.py
│   ├── check_raw_data.py
│   ├── preprocess_to_seqrec.py
│   └── check_processed_data.py
├── models/                   # Model implementations
│   └── seqrec/               # SeqRec model (SASRec-style)
│       ├── model.py
│       ├── dataset.py
│       └── __init__.py
├── train/                    # Training scripts
│   └── train_seqrec.py
├── evaluation/               # Evaluation scripts
│   ├── metrics.py
│   └── evaluate_topk.py
├── configs/                  # Configuration files
│   ├── seqrec_industrial.yaml
│   ├── seqrec_musical.yaml
│   ├── seqrec_cds.yaml
│   └── seqrec_tiny.yaml
├── results/                  # Results (tables and figures)
│   ├── tables/
│   └── figures/
├── examples/                 # Example data
│   └── tiny_sample/          # Tiny sample for testing
└── data/                     # Data directories (local only)
    ├── raw/                  # Raw data (ignored by Git)
    └── processed/            # Processed data (ignored by Git)
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```powershell
# Create conda environment (recommended)
conda create -n seqrec python=3.10
conda activate seqrec

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

Place raw files under:

```text
data/raw/Industrial_and_Scientific/
data/raw/Musical_Instruments/
data/raw/CDs_and_Vinyl/
```

Each category should contain:

```text
<category>.train.csv.gz
<category>.valid.csv.gz
<category>.test.csv.gz
<category>.jsonl.gz
meta_<category>.jsonl.gz
```

### 3. Preprocess Data

```powershell
# Check raw data
python scripts/check_raw_data.py

# Preprocess all categories
python scripts/preprocess_to_seqrec.py --categories Industrial_and_Scientific --overwrite
python scripts/preprocess_to_seqrec.py --categories Musical_Instruments --overwrite
python scripts/preprocess_to_seqrec.py --categories CDs_and_Vinyl --overwrite

# Check processed data
python scripts/check_processed_data.py --categories Industrial_and_Scientific Musical_Instruments CDs_and_Vinyl
```

### 4. Train Model

```powershell
# Train on GPU
python -m train.train_seqrec --config configs/seqrec_industrial.yaml --device cuda
python -m train.train_seqrec --config configs/seqrec_musical.yaml --device cuda
python -m train.train_seqrec --config configs/seqrec_cds.yaml --device cuda
```

### 5. Evaluate Model

Training script automatically evaluates on test set. Results are saved to:

```text
train/seqrec_{category}_test_results.json
results/tables/seqrec_{category}_test_results.json
```

---

## 🎯 Model Training

### Training Commands

```powershell
# Train with default config
python -m train.train_seqrec --config configs/seqrec_industrial.yaml

# Train with GPU
python -m train.train_seqrec --config configs/seqrec_industrial.yaml --device cuda

# Train with limited users (for quick test)
python -m train.train_seqrec --config configs/seqrec_industrial.yaml --device cuda --max-users 1000
```

### Hyperparameters

Key hyperparameters in `configs/*.yaml`:

| Parameter             | Description                  | Default | Search Range        |
| --------------------- | ---------------------------- | ------- | ------------------- |
| `maxlen`              | Maximum sequence length      | 50      | 30, 50, 100         |
| `hidden_units`        | Embedding dimension          | 64      | 64, 128, 256        |
| `num_blocks`          | Number of Transformer blocks | 2       | 1, 2, 4             |
| `num_heads`           | Number of attention heads    | 2       | 2, 4, 8             |
| `dropout_rate`        | Dropout rate                 | 0.2     | 0.1, 0.2, 0.3       |
| `learning_rate`       | Learning rate                | 0.001   | 0.0001, 0.001, 0.01 |
| `batch_size`          | Batch size                   | 128     | 64, 128, 256        |
| `num_epochs`          | Number of epochs             | 100     | 50, 100, 200        |
| `early_stop_patience` | Early stopping patience      | 5       | 3, 5, 10            |
| `seed`                | Random seed                  | 42      | 42, 123, 2024       |

### Evaluation Metrics

Training monitors the following metrics:

- **NDCG@10** - Normalized Discounted Cumulative Gain (primary metric)
- **HitRate@10** - Hit rate
- **Recall@10** - Recall (same as HitRate@10 in single-target setting)
- **MRR@10** - Mean Reciprocal Rank
- **Precision@10** - Precision

### Output Files

After training:

- `train/seqrec_{category}_best.pth` - Best model checkpoint
- `train/seqrec_{category}_best_config.json` - Model configuration
- `train/seqrec_{category}_test_results.json` - Test results
- `results/tables/seqrec_{category}_test_results.json` - Test results (for report)

---

## 📚 Documentation

- [Data Download](docs/DATA_DOWNLOAD.md) - How to download raw data
- [Data Preprocess](docs/DATA_PREPROCESS.md) - Data preprocessing pipeline
- [Model Training](docs/MODEL_TRAINING.md) - Detailed training guide
- [Evaluation](docs/EVALUATION.md) - Evaluation metrics and usage
- [Team Handoff](docs/TEAM_HANDOFF.md) - Project status and task assignment
- [GitHub Collaboration](docs/GITHUB_COLLABORATION.md) - Git workflow

---

## ⚠️ Important Notes

### Data Files

- **Do not upload** full `data/raw/` and `data/processed/` directories to GitHub
- These are large local data artifacts
- Each team member should obtain data locally or through shared storage

### Model Checkpoints

- **Do not upload** model checkpoints (`.pth`, `.pt`, `.ckpt` files) to GitHub
- Checkpoints should be saved locally or on shared storage
- Only upload code, configs, docs, and lightweight results

### Path Settings

- Use **relative paths** in all scripts (e.g., `data/processed/`)
- Do not write **absolute paths** (e.g., `D:\data\processed\`)

---

## 🛠️ Development

### Running Tests

```powershell
# Test data loading
python -m train.check_data_loading

# Test full pipeline
python test_project.py
```

### Code Style

- Follow PEP 8 style guide
- Add docstrings to classes and functions
- Use type hints where appropriate

---

## 📅 Timeline

| Phase                 | Status         | Deadline     |
| --------------------- | -------------- | ------------ |
| Data Preprocessing    | ✅ Completed   | -            |
| Model Implementation  | ✅ Completed   | -            |
| Training & Tuning     | 📋 In Progress | -            |
| Evaluation & Analysis | 📋 Pending     | -            |
| Report Writing        | 📋 Pending     | May 18, 2026 |

---

## 👥 Team

See [Team Handoff](docs/TEAM_HANDOFF.md) for task assignment and project status.

---

## 📖 Citation

SASRec Paper:

```
@inproceedings{kang2018self,
  title={Self-attentive sequential recommendation},
  author={Kang, Wang-Cheng and McAuley, Julian},
  booktitle={2018 IEEE International Conference on Data Mining (ICDM)},
  pages={197--206},
  year={2018},
  organization={IEEE}
}
```

---

_Last Updated: May 9, 2026_
