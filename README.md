# Amazon Reviews 2023 Recommendation System Coursework

This repository is a team-oriented engineering scaffold for a recommendation-system coursework project based on Amazon Reviews 2023 5-Core data. The current experiment target is SASRec-style sequential recommendation on three product categories.

## Current Status

- Raw Amazon Reviews 2023 5-Core files have been downloaded locally for three categories.
- All three categories have been converted to SASRec-ready processed files.
- `scripts/check_raw_data.py` and `scripts/check_processed_data.py` have passed on the local full data.
- Full `data/raw/` and `data/processed/` files are not included in GitHub because they are large local data artifacts.

## Categories

- `Industrial_and_Scientific`
- `Musical_Instruments`
- `CDs_and_Vinyl`

## Repository Layout

```text
.
├── README.md
├── README_data.md
├── requirements.txt
├── docs/
├── scripts/
├── models/
├── train/
├── evaluation/
├── configs/
├── results/
├── examples/
└── data/
```

Important directories:

- `scripts/`: data download, raw checks, preprocessing, processed checks.
- `docs/`: data, preprocessing, training, evaluation, handoff, and GitHub collaboration docs.
- `configs/`: starter YAML configs for SASRec experiments.
- `models/`: future model implementations.
- `train/`: future training entrypoints.
- `evaluation/`: future metric scripts.
- `results/`: final tables and figures, not checkpoints.
- `examples/tiny_sample/`: tiny synthetic data for pipeline smoke tests.
- `data/raw/` and `data/processed/`: local-only full data directories ignored by Git.

## Quick Reproduction

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

If `python` is not on PATH in PowerShell, use the known local interpreter path:

```powershell
$env:PYTHONIOENCODING='utf-8'; & "C:\Users\Chess\AppData\Local\Programs\Python\Python312\python.exe" scripts/check_raw_data.py --help
```

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

Check raw files:

```powershell
python scripts/check_raw_data.py
```

Run preprocessing:

```powershell
python scripts/preprocess_to_sasrec.py --categories Industrial_and_Scientific --dry-run
python scripts/preprocess_to_sasrec.py --categories Industrial_and_Scientific --overwrite
python scripts/preprocess_to_sasrec.py --categories Musical_Instruments --overwrite
python scripts/preprocess_to_sasrec.py --categories CDs_and_Vinyl --overwrite
```

Check processed files:

```powershell
python scripts/check_processed_data.py --categories Industrial_and_Scientific Musical_Instruments CDs_and_Vinyl
```

## Files for Model Training

Training code should read from `data/processed/<category>/`:

- `train.tsv`
- `dev.tsv`
- `test.tsv`
- `sasrec_sequence.txt`
- `sasrec_interactions.txt`
- `user2id.json`
- `item2id.json`
- `stats.json`

The complete raw and processed data should be obtained locally by each member or shared through a private file-sharing channel. Do not upload full data to GitHub.

## Documentation

- [Data Download](docs/DATA_DOWNLOAD.md)
- [Data Preprocess](docs/DATA_PREPROCESS.md)
- [Model Training](docs/MODEL_TRAINING.md)
- [Evaluation](docs/EVALUATION.md)
- [Team Handoff](docs/TEAM_HANDOFF.md)
- [GitHub Collaboration](docs/GITHUB_COLLABORATION.md)
