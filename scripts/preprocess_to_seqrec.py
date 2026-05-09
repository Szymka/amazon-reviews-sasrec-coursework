"""将 Amazon Reviews 2023 5-Core split 转换为 SASRec 可用格式。

本脚本只读取官方 Leave-Last-Out split:
<category>.train.csv.gz、<category>.valid.csv.gz、<category>.test.csv.gz。
raw 阶段使用 valid，processed 阶段输出 dev.tsv。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_CATEGORIES = (
    "Industrial_and_Scientific",
    "Musical_Instruments",
    "CDs_and_Vinyl",
)

RAW_SPLITS = ("train", "valid", "test")
OUTPUT_SPLITS = {
    "train": "train.tsv",
    "valid": "dev.tsv",
    "test": "test.tsv",
}

TSV_COLUMNS = (
    "user_id_int",
    "target_id",
    "rating",
    "timestamp",
    "seq_ids",
    "raw_user_id",
    "raw_parent_asin",
)

REQUIRED_RAW_COLUMNS = ("user_id", "parent_asin", "rating", "timestamp", "history")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="把 Amazon Reviews 2023 5-Core split 预处理为 SASRec 格式。"
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="原始数据根目录，默认: data/raw",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="处理后数据输出根目录，默认: data/processed",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=list(DEFAULT_CATEGORIES),
        help="需要预处理的商品类别列表。",
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=None,
        help="按 test split 文件顺序抽取的最大用户数；默认处理 test 中全部用户。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖目标 processed 类别目录下的输出文件。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只读取和检查小样本逻辑，不写入 processed 文件。",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
        help="筛选 raw split 时的 pandas 分块行数，默认: 200000。",
    )
    return parser.parse_args()


def validate_categories(categories: Iterable[str]) -> list[str]:
    """确认类别名在当前项目约定范围内。"""
    requested = list(categories)
    unknown = sorted(set(requested) - set(DEFAULT_CATEGORIES))
    if unknown:
        allowed = ", ".join(DEFAULT_CATEGORIES)
        raise ValueError(f"未知类别: {', '.join(unknown)}。允许值: {allowed}")
    return requested


def raw_paths(raw_dir: Path, category: str) -> dict[str, Path]:
    """返回官方 raw split 文件路径。"""
    category_dir = raw_dir / category
    return {
        split: category_dir / f"{category}.{split}.csv.gz"
        for split in RAW_SPLITS
    }


def output_paths(processed_dir: Path, category: str) -> dict[str, Path]:
    """返回 processed 输出文件路径。"""
    category_dir = processed_dir / category
    paths = {split: category_dir / file_name for split, file_name in OUTPUT_SPLITS.items()}
    paths.update(
        {
            "seqrec_sequence": category_dir / "seqrec_sequence.txt",
            "seqrec_interactions": category_dir / "seqrec_interactions.txt",
            "user_mapping": category_dir / "user2id.json",
            "id_to_user_mapping": category_dir / "id2user.json",
            "item_mapping": category_dir / "item2id.json",
            "id_to_item_mapping": category_dir / "id2item.json",
            "stats": category_dir / "stats.json",
        }
    )
    return paths


def ensure_raw_files_exist(paths: dict[str, Path]) -> None:
    """确认需要的 raw split 文件存在且非空。"""
    for split, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"缺少 raw {split} 文件: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"raw {split} 文件为 0 字节: {path}")


def select_users_from_test(test_path: Path, max_users: int | None, chunksize: int) -> list[str]:
    """从 test split 按文件顺序抽取用户，保持可复现。"""
    selected: list[str] = []
    seen: set[str] = set()

    for chunk in pd.read_csv(test_path, usecols=["user_id"], chunksize=chunksize):
        for user_id in chunk["user_id"]:
            user_text = str(user_id)
            if user_text in seen:
                continue
            selected.append(user_text)
            seen.add(user_text)
            if max_users is not None and len(selected) >= max_users:
                return selected

    return selected


def read_selected_split(
    path: Path,
    selected_user_set: set[str],
    chunksize: int,
) -> pd.DataFrame:
    """按 selected users 过滤 split 文件，不把其它用户写入 processed。"""
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, chunksize=chunksize):
        missing_columns = [column for column in REQUIRED_RAW_COLUMNS if column not in chunk.columns]
        if missing_columns:
            raise ValueError(f"{path} 缺少字段: {missing_columns}")
        chunk["user_id"] = chunk["user_id"].astype(str)
        filtered = chunk[chunk["user_id"].isin(selected_user_set)].copy()
        if not filtered.empty:
            parts.append(filtered.loc[:, REQUIRED_RAW_COLUMNS])

    if not parts:
        return pd.DataFrame(columns=REQUIRED_RAW_COLUMNS)
    return pd.concat(parts, ignore_index=True)


def parse_history(value: object) -> list[str]:
    """把 history 字段解析为 parent_asin 列表，兼容 NaN、None 和空字符串。"""
    if value is None or pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [item for item in text.split() if item]


def add_mapping(mapping: dict[str, int], raw_id: str) -> None:
    """按首次出现顺序建立从 1 开始的连续整数 ID。"""
    if raw_id not in mapping:
        mapping[raw_id] = len(mapping) + 1


def build_mappings(
    selected_users: list[str],
    split_frames: dict[str, pd.DataFrame],
) -> tuple[dict[str, int], dict[str, int]]:
    """为 selected users 和其所有 target/history item 建立共享映射。"""
    user2id = {user_id: index for index, user_id in enumerate(selected_users, start=1)}
    item2id: dict[str, int] = {}

    for split in RAW_SPLITS:
        frame = split_frames[split]
        for row in frame.itertuples(index=False):
            for item_id in parse_history(row.history):
                add_mapping(item2id, item_id)
            add_mapping(item2id, str(row.parent_asin))

    return user2id, item2id


def build_tsv_frame(
    frame: pd.DataFrame,
    user2id: dict[str, int],
    item2id: dict[str, int],
) -> pd.DataFrame:
    """将一个 split 转换为样本级 TSV 数据。"""
    rows: list[dict[str, object]] = []
    for row in frame.itertuples(index=False):
        raw_user_id = str(row.user_id)
        raw_parent_asin = str(row.parent_asin)
        history_ids = [str(item2id[item]) for item in parse_history(row.history)]
        rows.append(
            {
                "user_id_int": user2id[raw_user_id],
                "target_id": item2id[raw_parent_asin],
                "rating": row.rating,
                "timestamp": row.timestamp,
                "seq_ids": " ".join(history_ids),
                "raw_user_id": raw_user_id,
                "raw_parent_asin": raw_parent_asin,
            }
        )

    return pd.DataFrame(rows, columns=TSV_COLUMNS)


def build_sequences(
    test_frame: pd.DataFrame,
    selected_users: list[str],
    user2id: dict[str, int],
    item2id: dict[str, int],
) -> dict[int, list[int]]:
    """优先从 test 行重构完整序列: test.history + test.parent_asin。"""
    test_by_user: dict[str, pd.Series] = {}
    for _, row in test_frame.iterrows():
        raw_user_id = str(row["user_id"])
        if raw_user_id not in test_by_user:
            test_by_user[raw_user_id] = row

    sequences: dict[int, list[int]] = {}
    for raw_user_id in selected_users:
        if raw_user_id not in test_by_user:
            raise ValueError(f"selected user 在 test split 中缺失: {raw_user_id}")
        row = test_by_user[raw_user_id]
        sequence = [item2id[item] for item in parse_history(row["history"])]
        sequence.append(item2id[str(row["parent_asin"])])
        sequences[user2id[raw_user_id]] = sequence

    return sequences


def build_stats(
    category: str,
    max_users: int | None,
    raw_file_paths: dict[str, Path],
    tsv_frames: dict[str, pd.DataFrame],
    user2id: dict[str, int],
    item2id: dict[str, int],
    sequences: dict[int, list[int]],
) -> dict[str, object]:
    """生成 stats.json 内容。"""
    sequence_lengths = [len(sequence) for sequence in sequences.values()]
    interaction_count = sum(sequence_lengths)
    return {
        "category": category,
        "max_users": max_users,
        "num_users": len(user2id),
        "num_items": len(item2id),
        "num_train_rows": int(len(tsv_frames["train"])),
        "num_dev_rows": int(len(tsv_frames["valid"])),
        "num_test_rows": int(len(tsv_frames["test"])),
        "num_seqrec_sequences": len(sequences),
        "num_seqrec_interactions": interaction_count,
        "min_sequence_length": min(sequence_lengths) if sequence_lengths else 0,
        "max_sequence_length": max(sequence_lengths) if sequence_lengths else 0,
        "avg_sequence_length": (
            round(sum(sequence_lengths) / len(sequence_lengths), 6)
            if sequence_lengths
            else 0
        ),
        "source_train_file": str(raw_file_paths["train"]),
        "source_valid_file": str(raw_file_paths["valid"]),
        "source_test_file": str(raw_file_paths["test"]),
        "id_start_from": 1,
        "padding_id": 0,
    }


def write_outputs(
    paths: dict[str, Path],
    tsv_frames: dict[str, pd.DataFrame],
    sequences: dict[int, list[int]],
    user2id: dict[str, int],
    item2id: dict[str, int],
    stats: dict[str, object],
    overwrite: bool,
) -> None:
    """写入 processed 文件，默认不覆盖已存在文件。"""
    target_dir = paths["train"].parent
    target_dir.mkdir(parents=True, exist_ok=True)
    planned_paths = list(paths.values())
    existing = [path for path in planned_paths if path.exists()]
    if existing and not overwrite:
        existing_text = "\n".join(f"  - {path}" for path in existing)
        raise FileExistsError(
            "目标输出已存在，请确认后使用 --overwrite 覆盖:\n" + existing_text
        )

    for split, frame in tsv_frames.items():
        frame.to_csv(paths[split], sep="\t", index=False)

    with paths["seqrec_sequence"].open("w", encoding="utf-8", newline="\n") as file_obj:
        for user_id_int, sequence in sequences.items():
            values = [str(user_id_int), *(str(item_id) for item_id in sequence)]
            file_obj.write(" ".join(values) + "\n")

    with paths["seqrec_interactions"].open("w", encoding="utf-8", newline="\n") as file_obj:
        for user_id_int, sequence in sequences.items():
            for item_id in sequence:
                file_obj.write(f"{user_id_int} {item_id}\n")

    id2user = {str(user_id_int): raw_user_id for raw_user_id, user_id_int in user2id.items()}
    id2item = {str(item_id_int): raw_item_id for raw_item_id, item_id_int in item2id.items()}
    json_payloads = {
        "user_mapping": user2id,
        "id_to_user_mapping": id2user,
        "item_mapping": item2id,
        "id_to_item_mapping": id2item,
        "stats": stats,
    }
    for key, payload in json_payloads.items():
        paths[key].write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def process_category(args: argparse.Namespace, category: str) -> dict[str, object]:
    """处理单个类别；dry-run 只做读取和统计，不写文件。"""
    raw_file_paths = raw_paths(args.raw_dir, category)
    processed_paths = output_paths(args.processed_dir, category)
    ensure_raw_files_exist(raw_file_paths)

    selected_users = select_users_from_test(
        raw_file_paths["test"],
        args.max_users,
        args.chunksize,
    )
    if not selected_users:
        raise ValueError(f"{category} 未选中任何用户。")
    selected_user_set = set(selected_users)

    split_frames = {
        split: read_selected_split(path, selected_user_set, args.chunksize)
        for split, path in raw_file_paths.items()
    }
    user2id, item2id = build_mappings(selected_users, split_frames)
    tsv_frames = {
        split: build_tsv_frame(frame, user2id, item2id)
        for split, frame in split_frames.items()
    }
    sequences = build_sequences(
        split_frames["test"],
        selected_users,
        user2id,
        item2id,
    )
    stats = build_stats(
        category,
        args.max_users,
        raw_file_paths,
        tsv_frames,
        user2id,
        item2id,
        sequences,
    )

    print(f"\n[{category}]")
    print(f"  selected_users={len(selected_users)}")
    print(f"  num_items={len(item2id)}")
    print(f"  train_rows={len(tsv_frames['train'])}")
    print(f"  dev_rows={len(tsv_frames['valid'])}  source={raw_file_paths['valid'].name}")
    print(f"  test_rows={len(tsv_frames['test'])}")
    print(f"  seqrec_sequences={len(sequences)}")
    print(f"  seqrec_interactions={stats['num_seqrec_interactions']}")
    print(
        "  sequence_length="
        f"min:{stats['min_sequence_length']} "
        f"max:{stats['max_sequence_length']} "
        f"avg:{stats['avg_sequence_length']}"
    )

    if args.dry_run:
        print("  DRY-RUN: 未写入 processed 文件。")
    else:
        write_outputs(
            processed_paths,
            tsv_frames,
            sequences,
            user2id,
            item2id,
            stats,
            args.overwrite,
        )
        print(f"  wrote={processed_paths['train'].parent}")

    return stats


def main() -> int:
    args = parse_args()
    if args.max_users is not None and args.max_users <= 0:
        print("ERROR: --max-users 必须大于 0。")
        return 2
    if args.chunksize <= 0:
        print("ERROR: --chunksize 必须大于 0。")
        return 2
    if not args.dry_run and not args.overwrite:
        print("ERROR: 实际写入前请显式传入 --overwrite，避免误覆盖 processed 文件。")
        return 2

    try:
        categories = validate_categories(args.categories)
        print("SASRec 数据预处理")
        print(f"raw_dir={args.raw_dir}")
        print(f"processed_dir={args.processed_dir}")
        print(f"categories={categories}")
        print(f"max_users={args.max_users}")
        print(f"dry_run={args.dry_run}")
        print(f"overwrite={args.overwrite}")

        for category in categories:
            process_category(args, category)
    except Exception as exc:  # noqa: BLE001 - 命令行脚本需要给出清楚错误。
        print(f"ERROR: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
