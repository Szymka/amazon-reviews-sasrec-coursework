"""
Evaluate an A-LLMRec pre-trained SASRec checkpoint on coursework train/dev/test splits.

Uses the same NDCG@k definition as evaluation/metrics.py (full-vocabulary ranking).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import evaluate
from models.seqrec.dataset import build_datasets_from_config, load_simple_yaml


class SeqRecTorchDataset(Dataset):
    def __init__(self, base) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.base[index]
        return {
            "input_ids": torch.tensor(sample["input_ids"], dtype=torch.long),
            "target_id": torch.tensor(sample["target_id"], dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True, help="configs/*.yaml for Industrial_and_Scientific")
    p.add_argument("--checkpoint", type=Path, required=True, help="SASRec .pth from A-LLMRec pre_train/sasrec/...")
    p.add_argument(
        "--allmrec-root",
        type=Path,
        default=PROJECT_ROOT / "A-LLMRec",
        help="A-LLMRec repo root (for SASRec class import).",
    )
    p.add_argument("--split", type=str, default="test", choices=("train", "dev", "test"))
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=256)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    raw_cfg = Path(args.config).expanduser()
    config_path: Path | None = None
    if raw_cfg.is_file():
        config_path = raw_cfg.resolve()
    else:
        rebased = raw_cfg
        if rebased.parts and rebased.parts[0] == "amazon-reviews-sasrec-coursework":
            rebased = Path(*rebased.parts[1:])
        for candidate in (
            Path.cwd() / raw_cfg,
            PROJECT_ROOT / rebased,
            PROJECT_ROOT / "configs" / raw_cfg.name,
            Path(__file__).resolve().parent.parent / "configs" / raw_cfg.name,
        ):
            if candidate.is_file():
                config_path = candidate.resolve()
                break
    if config_path is None:
        raise FileNotFoundError(f"config not found: {args.config}")
    args.config = config_path

    # Paths inside yaml are relative to coursework root
    os.chdir(PROJECT_ROOT)

    allmrec_root = args.allmrec_root.resolve()
    if str(allmrec_root) not in sys.path:
        sys.path.insert(0, str(allmrec_root))

    from pre_train.sasrec.model import SASRec

    config = load_simple_yaml(args.config)
    k = int(config.get("topk", 10))

    if args.split == "train":
        raise SystemExit("Use dev or test for evaluation (train is for fitting).")

    ds_map = build_datasets_from_config(args.config)
    dataset = ds_map[args.split]
    loader = DataLoader(
        SeqRecTorchDataset(dataset),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    raw_ckpt = Path(args.checkpoint).expanduser()
    checkpoint_path: Path | None = None
    if raw_ckpt.is_file():
        checkpoint_path = raw_ckpt.resolve()
    else:
        rebased = raw_ckpt
        if rebased.parts and rebased.parts[0] == "amazon-reviews-sasrec-coursework":
            rebased = Path(*rebased.parts[1:])
        for candidate in (Path.cwd() / raw_ckpt, PROJECT_ROOT / rebased):
            if candidate.is_file():
                checkpoint_path = candidate.resolve()
                break
    if checkpoint_path is None:
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    try:
        kwargs, state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        kwargs, state = torch.load(checkpoint_path, map_location="cpu")
    kwargs["args"].device = args.device
    model = SASRec(**kwargs).to(args.device)
    model.load_state_dict(state)
    model.eval()

    all_scores: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(args.device)
            targets = batch["target_id"].to(args.device)
            bsz = input_ids.size(0)
            user_ids = np.zeros(bsz, dtype=np.int64)
            seq = input_ids.cpu().numpy().astype(np.int64)
            log_feat = model.forward(user_ids, seq, None, None, mode="log_only")
            full_logits = torch.matmul(log_feat, model.item_emb.weight.T)
            full_logits[:, 0] = -1e9
            all_scores.append(full_logits.cpu())
            all_targets.append(targets.cpu())

    scores = torch.cat(all_scores, dim=0)
    targets = torch.cat(all_targets, dim=0)
    metrics = evaluate(scores, targets, k=k)
    print(f"split={args.split}  NDCG@{k}={metrics['ndcg']:.4f}  HR@{k}={metrics['hit_rate']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
