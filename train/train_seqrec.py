from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import evaluate
from models.seqrec.dataset import build_datasets_from_config, load_simple_yaml
from models.seqrec.model import GRUSeqRec, SASRec


class SeqRecTorchDataset(Dataset):
    def __init__(self, base: Any) -> None:
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
    parser = argparse.ArgumentParser(description="Train GRU-based sequential recommender (coursework).")
    parser.add_argument("--config", type=Path, required=True, help="Path to configs/*.yaml")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cpu or cuda",
    )
    parser.add_argument("--max-users", type=int, default=None, help="Optional cap on train users for debugging.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("train"),
        help="Directory for checkpoints and local result JSON.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/tables"),
        help="Directory for report-friendly result JSON copies.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_subset_train(dataset: SeqRecTorchDataset, max_users: int | None) -> SeqRecTorchDataset:
    if max_users is None:
        return dataset
    base = dataset.base
    keep = [i for i in range(len(base)) if int(base[i]["user_id"]) < max_users]
    if not keep:
        raise ValueError("--max-users produced an empty training set.")
    return SeqRecTorchDataset(Subset(base, keep))


@torch.no_grad()
def run_eval(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    topk: int,
) -> dict[str, float]:
    model.eval()
    all_scores: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        targets = batch["target_id"].to(device)
        logits = model.predict(input_ids)
        all_scores.append(logits.cpu())
        all_targets.append(targets.cpu())
    scores = torch.cat(all_scores, dim=0)
    targets = torch.cat(all_targets, dim=0)
    return evaluate(scores, targets, k=topk)


def sample_negatives(
    targets: torch.Tensor,
    num_items: int,
    num_neg: int,
    device: torch.device,
) -> torch.Tensor:
    """Uniform negatives in [1, num_items], reshaped (B, num_neg), never equal to target."""
    bsz = targets.size(0)
    max_neg = min(num_neg, max(1, num_items - 1))
    if max_neg < num_neg:
        num_neg = max_neg
    negs = torch.randint(1, num_items + 1, (bsz, num_neg), device=device)
    mask = negs == targets.unsqueeze(1)
    for _ in range(32):
        if not mask.any():
            break
        repl = torch.randint(1, num_items + 1, (bsz, num_neg), device=device)
        negs = torch.where(mask, repl, negs)
        mask = negs == targets.unsqueeze(1)
    return negs


def sampled_softmax_loss(
    model: nn.Module,
    hidden: torch.Tensor,
    targets: torch.Tensor,
    num_items: int,
    num_neg: int,
    padding_id: int,
) -> torch.Tensor:
    """CE over {target, num_neg random items} — scalable training for large item vocab."""
    device = hidden.device
    bsz = targets.size(0)
    negs = sample_negatives(targets, num_items, num_neg, device)
    cand = torch.cat([targets.unsqueeze(1), negs], dim=1)
    emb = model.item_embedding(cand)
    logits = (hidden.unsqueeze(1) * emb).sum(dim=-1)
    logits[:, 1:][cand[:, 1:] == padding_id] = -1e9
    return F.cross_entropy(logits, torch.zeros(bsz, dtype=torch.long, device=device))


def build_model(config: dict[str, Any], num_items: int, padding_id: int) -> nn.Module:
    name = str(config.get("model_name", "GRUSeqRec")).upper()
    maxlen = int(config["maxlen"])
    hidden = int(config["hidden_units"])
    dropout = float(config["dropout_rate"])
    layers = int(config.get("gru_num_layers", config.get("num_blocks", 1)))
    if name in {"SASREC", "SEQREC"}:
        return SASRec(
            num_items=num_items,
            maxlen=maxlen,
            hidden_units=hidden,
            num_blocks=layers,
            num_heads=int(config.get("num_heads", 2)),
            dropout_rate=dropout,
            padding_id=padding_id,
        )
    return GRUSeqRec(
        num_items=num_items,
        maxlen=maxlen,
        hidden_units=hidden,
        gru_num_layers=layers,
        dropout_rate=dropout,
        padding_id=padding_id,
    )


def model_config_blob(
    config: dict[str, Any],
    num_items: int,
    padding_id: int,
    model: nn.Module,
) -> dict[str, Any]:
    layers = int(config.get("gru_num_layers", config.get("num_blocks", 1)))
    blob: dict[str, Any] = {
        "model_class": model.__class__.__name__,
        "category": str(config["category"]),
        "num_items": int(num_items),
        "maxlen": int(config["maxlen"]),
        "hidden_units": int(config["hidden_units"]),
        "gru_num_layers": layers,
        "dropout_rate": float(config["dropout_rate"]),
        "padding_id": int(padding_id),
        "topk": int(config.get("topk", 10)),
        "train_loss_mode": str(config.get("train_loss_mode", "sampled")),
        "num_sampled_negatives": int(config.get("num_sampled_negatives", 512)),
    }
    return blob


def main() -> int:
    args = parse_args()
    config = load_simple_yaml(args.config)
    required = (
        "category",
        "train_file",
        "dev_file",
        "test_file",
        "stats_file",
        "maxlen",
        "hidden_units",
        "dropout_rate",
        "learning_rate",
        "batch_size",
        "num_epochs",
    )
    missing = [key for key in required if key not in config]
    if missing:
        print(f"ERROR: config missing keys: {missing}")
        return 2

    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    datasets = build_datasets_from_config(args.config)
    category = str(config["category"])
    stats_path = Path(str(config["stats_file"]))
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    num_items = int(stats["num_items"])
    padding_id = int(stats.get("padding_id", 0))
    topk = int(config.get("topk", 10))

    train_ds = maybe_subset_train(SeqRecTorchDataset(datasets["train"]), args.max_users)
    dev_ds = SeqRecTorchDataset(datasets["dev"])
    test_ds = SeqRecTorchDataset(datasets["test"])

    train_loader = DataLoader(
        train_ds,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=0,
    )
    dev_loader = DataLoader(dev_ds, batch_size=int(config["batch_size"]), shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=int(config["batch_size"]), shuffle=False, num_workers=0)

    model = build_model(config, num_items=num_items, padding_id=padding_id).to(device)
    lr = float(config["learning_rate"])
    weight_decay = float(config.get("weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loss_mode = str(config.get("train_loss_mode", "sampled")).lower()
    num_sampled_neg = int(config.get("num_sampled_negatives", 512))
    full_ce = train_loss_mode in {"full", "full_softmax", "softmax"}
    criterion = nn.CrossEntropyLoss() if full_ce else None

    patience = int(config.get("early_stop_patience", 5))
    min_delta = float(config.get("early_stop_min_delta", 0.0))
    best_ndcg = -1.0
    stale = 0

    stem = f"seqrec_{category}"
    best_weights_path = args.output_dir / f"{stem}_best.pth"
    best_config_path = args.output_dir / f"{stem}_best_config.json"

    for epoch in range(1, int(config["num_epochs"]) + 1):
        model.train()
        losses: list[float] = []
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{int(config['num_epochs'])}", leave=False)
        for batch in progress:
            input_ids = batch["input_ids"].to(device)
            targets = batch["target_id"].to(device)
            lengths = GRUSeqRec.sequence_lengths(input_ids, padding_id)
            if full_ce:
                logits = model(input_ids, lengths)
                logits[:, padding_id] = -1e9
                loss = criterion(logits, targets)
            else:
                hidden = model.encode(input_ids, lengths)
                loss = sampled_softmax_loss(
                    model,
                    hidden,
                    targets,
                    num_items,
                    num_sampled_neg,
                    padding_id,
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.item()))
            progress.set_postfix(loss=sum(losses) / len(losses))

        dev_metrics = run_eval(model, dev_loader, device, topk)
        print(
            f"epoch {epoch}: train_loss={sum(losses)/max(len(losses),1):.4f} "
            f"dev_ndcg@{topk}={dev_metrics['ndcg']:.4f} dev_hr@{topk}={dev_metrics['hit_rate']:.4f}"
        )

        improved = dev_metrics["ndcg"] > best_ndcg + min_delta
        if improved:
            best_ndcg = dev_metrics["ndcg"]
            stale = 0
            torch.save(model.state_dict(), best_weights_path)
            blob = model_config_blob(config, num_items=num_items, padding_id=padding_id, model=model)
            blob["best_epoch"] = epoch
            blob["best_dev_ndcg"] = best_ndcg
            best_config_path.write_text(json.dumps(blob, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            stale += 1
            if stale >= patience:
                print(f"early stop at epoch {epoch} (no NDCG improvement for {patience} epochs).")
                break

    if not best_weights_path.exists():
        torch.save(model.state_dict(), best_weights_path)
        best_config_path.write_text(
            json.dumps(
                model_config_blob(config, num_items=num_items, padding_id=padding_id, model=model),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    try:
        state = torch.load(best_weights_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(best_weights_path, map_location=device)
    model.load_state_dict(state)
    test_metrics = run_eval(model, test_loader, device, topk)
    print(f"test_ndcg@{topk}={test_metrics['ndcg']:.4f} test_hr@{topk}={test_metrics['hit_rate']:.4f}")

    result_payload = {
        "category": category,
        "config_path": str(args.config.resolve()),
        "checkpoint": str(best_weights_path.resolve()),
        "test_metrics": test_metrics,
        "topk": topk,
    }
    local_results = args.output_dir / f"{stem}_test_results.json"
    local_results.write_text(json.dumps(result_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    report_results = args.results_dir / f"{stem}_test_results.json"
    report_results.write_text(json.dumps(result_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
