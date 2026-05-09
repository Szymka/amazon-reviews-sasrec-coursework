from __future__ import annotations

import argparse
import json
from pathlib import Path

from models.sasrec.dataset import DEFAULT_CATEGORIES
from models.sasrec.dataset import build_category_datasets
from models.sasrec.dataset import build_datasets_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="检查 SASRec processed 数据是否能被正确加载。"
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("data/processed"),
        help="processed 数据根目录，默认: data/processed",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=list(DEFAULT_CATEGORIES),
        help="需要检查的类别列表。",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="可选：直接从 configs/sasrec_*.yaml 读取路径和 maxlen。",
    )
    parser.add_argument(
        "--maxlen",
        type=int,
        default=50,
        help="序列截断长度，默认: 50。",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="展示的样本下标，默认: 0。",
    )
    return parser.parse_args()


def print_sample(split: str, sample: dict[str, object]) -> None:
    non_padding = [item for item in sample["input_ids"] if item != 0]
    tail_preview = sample["input_ids"][-10:]
    print(f"  [{split}] user_id={sample['user_id']} target_id={sample['target_id']}")
    print(
        f"    seq_len={sample['seq_len']} padded_len={len(sample['input_ids'])} "
        f"non_padding={len(non_padding)}"
    )
    print(f"    input_tail(last_10)={tail_preview}")
    print(
        f"    raw_user_id={sample['raw_user_id']} "
        f"raw_parent_asin={sample['raw_parent_asin']}"
    )


def verify_dataset_lengths(category: str, processed_root: Path, datasets: dict[str, object]) -> None:
    stats_file = processed_root / category / "stats.json"
    stats = json.loads(stats_file.read_text(encoding="utf-8"))
    expected = {
        "train": int(stats["num_train_rows"]),
        "dev": int(stats["num_dev_rows"]),
        "test": int(stats["num_test_rows"]),
    }
    for split, dataset in datasets.items():
        actual = len(dataset)
        if actual != expected[split]:
            raise ValueError(
                f"{category} {split} rows mismatch: actual={actual}, expected={expected[split]}"
            )


def inspect_category(
    category: str,
    processed_root: Path,
    maxlen: int,
    sample_index: int,
) -> None:
    datasets = build_category_datasets(processed_root, category, maxlen=maxlen)
    verify_dataset_lengths(category, processed_root, datasets)
    print(f"\n[{category}]")
    for split, dataset in datasets.items():
        summary = dataset.summary()
        print(
            f"  {split}: rows={summary['num_rows']} maxlen={summary['maxlen']} "
            f"padding_id={summary['padding_id']}"
        )
        sample = dataset[min(sample_index, len(dataset) - 1)]
        print_sample(split, sample)


def inspect_from_config(config_path: Path, sample_index: int) -> None:
    datasets = build_datasets_from_config(config_path)
    category = next(iter(datasets.values())).category
    print(f"\n[{category}] from config={config_path}")
    for split, dataset in datasets.items():
        summary = dataset.summary()
        print(
            f"  {split}: rows={summary['num_rows']} maxlen={summary['maxlen']} "
            f"padding_id={summary['padding_id']}"
        )
        sample = dataset[min(sample_index, len(dataset) - 1)]
        print_sample(split, sample)


def main() -> int:
    args = parse_args()
    if args.maxlen <= 0:
        print("ERROR: --maxlen must be positive.")
        return 2
    if args.sample_index < 0:
        print("ERROR: --sample-index must be non-negative.")
        return 2

    try:
        if args.config is not None:
            inspect_from_config(args.config, args.sample_index)
        else:
            for category in args.categories:
                inspect_category(
                    category=category,
                    processed_root=args.processed_root,
                    maxlen=args.maxlen,
                    sample_index=args.sample_index,
                )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
