from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from evaluation.metrics import evaluate
from models.seqrec.dataset import DEFAULT_CATEGORIES, build_category_datasets
from models.seqrec.model import GRUSeqRec, SASRec


class SASRecDataset(torch.utils.data.Dataset):
    def __init__(self, dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        sample = self.dataset[index]
        return {
            "input_ids": torch.tensor(sample["input_ids"], dtype=torch.long),
            "target_id": torch.tensor(sample["target_id"], dtype=torch.long),
        }


def load_model_from_checkpoint(
    model_path: Path,
    config_path: Path | None,
    device: torch.device,
) -> tuple[torch.nn.Module, dict]:
    if config_path is None:
        config_path = model_path.with_name(model_path.stem + "_config.json")
    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing model config JSON: {config_path}. "
            "Pass --model-config or keep {stem}_best_config.json next to the checkpoint."
        )
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    num_items = int(cfg["num_items"])
    maxlen = int(cfg["maxlen"])
    hidden = int(cfg["hidden_units"])
    dropout = float(cfg["dropout_rate"])
    padding_id = int(cfg.get("padding_id", 0))
    layers = int(cfg.get("gru_num_layers", 1))
    name = str(cfg.get("model_class", "GRUSeqRec")).upper()
    if name == "SASREC":
        model = SASRec(
            num_items=num_items,
            maxlen=maxlen,
            hidden_units=hidden,
            num_blocks=layers,
            num_heads=2,
            dropout_rate=dropout,
            padding_id=padding_id,
        )
    else:
        model = GRUSeqRec(
            num_items=num_items,
            maxlen=maxlen,
            hidden_units=hidden,
            gru_num_layers=layers,
            dropout_rate=dropout,
            padding_id=padding_id,
        )
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    return model, cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate sequential recommender Top-K recommendations")
    parser.add_argument(
        "--category",
        type=str,
        default="Industrial_and_Scientific",
        choices=list(DEFAULT_CATEGORIES),
        help=f"Dataset category. Choices: {', '.join(DEFAULT_CATEGORIES)}",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("data/processed"),
        help="Root directory for processed data",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to saved model checkpoint",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=None,
        help="Optional path to *_best_config.json (default: sibling of checkpoint)",
    )
    parser.add_argument(
        "--maxlen",
        type=int,
        default=50,
        help="Maximum sequence length (must match training)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Top-K for evaluation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cpu or cuda)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    device = torch.device(args.device)
    model, cfg = load_model_from_checkpoint(
        args.model_path,
        args.model_config,
        device,
    )
    maxlen = int(cfg.get("maxlen", args.maxlen))
    topk = int(cfg.get("topk", args.topk))

    print(f"Loading dataset: {args.category}")
    datasets = build_category_datasets(args.processed_root, args.category, maxlen)

    test_dataset = SASRecDataset(datasets["test"])
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model.eval()

    all_logits = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            target_id = batch["target_id"].to(device)

            logits = model.predict(input_ids)

            all_logits.append(logits.cpu())
            all_targets.append(target_id.cpu())

    logits_tensor = torch.cat(all_logits)
    targets_tensor = torch.cat(all_targets)
    metrics = evaluate(logits_tensor, targets_tensor, k=topk)

    print(f"\nTest Results for {args.category}:")
    print(f"  NDCG@{topk}: {metrics['ndcg']:.4f}")
    print(f"  HitRate@{topk}: {metrics['hit_rate']:.4f}")
    print(f"  Recall@{topk}: {metrics['recall']:.4f}")
    print(f"  MRR@{topk}: {metrics['mrr']:.4f}")
    print(f"  Precision@{topk}: {metrics['precision']:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
