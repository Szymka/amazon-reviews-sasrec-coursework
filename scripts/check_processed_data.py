"""检查预处理后的 SASRec processed 数据文件。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_CATEGORIES = (
    "Industrial_and_Scientific",
    "Musical_Instruments",
    "CDs_and_Vinyl",
)

REQUIRED_FILES = (
    "train.tsv",
    "dev.tsv",
    "test.tsv",
    "seqrec_sequence.txt",
    "seqrec_interactions.txt",
    "user2id.json",
    "id2user.json",
    "item2id.json",
    "id2item.json",
    "stats.json",
)

TSV_COLUMNS = [
    "user_id_int",
    "target_id",
    "rating",
    "timestamp",
    "seq_ids",
    "raw_user_id",
    "raw_parent_asin",
]

REQUIRED_STATS_KEYS = (
    "category",
    "max_users",
    "num_users",
    "num_items",
    "num_train_rows",
    "num_dev_rows",
    "num_test_rows",
    "num_seqrec_sequences",
    "num_seqrec_interactions",
    "min_sequence_length",
    "max_sequence_length",
    "avg_sequence_length",
    "source_train_file",
    "source_valid_file",
    "source_test_file",
    "id_start_from",
    "padding_id",
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="检查预处理后的 SASRec 数据和映射文件。"
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="处理后数据根目录，默认: data/processed",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=list(DEFAULT_CATEGORIES),
        help="需要检查的商品类别列表。",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="展示样本行数，默认: 5。",
    )
    return parser.parse_args()


def expected_files(processed_dir: Path, category: str) -> dict[str, Path]:
    """返回某个类别需要检查的文件。"""
    category_dir = processed_dir / category
    return {file_name: category_dir / file_name for file_name in REQUIRED_FILES}


def load_json(path: Path) -> dict[str, Any]:
    """读取 UTF-8 JSON 对象。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} 不是 JSON object。")
    return data


def check_contiguous_ids(name: str, mapping: dict[str, Any]) -> list[str]:
    """检查映射值是否为从 1 开始的连续整数，且没有 0。"""
    issues: list[str] = []
    values = [int(value) for value in mapping.values()]
    if not values:
        issues.append(f"{name} 为空。")
        return issues
    if 0 in values:
        issues.append(f"{name} 把 0 分配给了真实 ID。")
    expected = list(range(1, len(values) + 1))
    if sorted(values) != expected:
        issues.append(f"{name} 不是从 1 开始的连续整数。")
    return issues


def parse_seq_ids(value: Any) -> list[int]:
    """解析 TSV 中的 seq_ids 字段。"""
    if value is None or pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [int(item) for item in text.split()]


def read_space_lines(path: Path) -> list[list[int]]:
    """读取空格分隔整数文件。"""
    rows: list[list[int]] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            text = line.strip()
            if not text:
                raise ValueError(f"{path} 第 {line_number} 行为空。")
            try:
                rows.append([int(part) for part in text.split()])
            except ValueError as exc:
                raise ValueError(f"{path} 第 {line_number} 行不是整数列。") from exc
    return rows


def check_tsv_frame(
    name: str,
    frame: pd.DataFrame,
    user2id: dict[str, Any],
    item2id: dict[str, Any],
) -> list[str]:
    """检查 TSV 表头、映射一致性和 ID 范围。"""
    issues: list[str] = []
    if list(frame.columns) != TSV_COLUMNS:
        issues.append(f"{name} 表头不匹配: {list(frame.columns)}")
        return issues

    user_values = {int(value) for value in user2id.values()}
    item_values = {int(value) for value in item2id.values()}
    for row_number, row in enumerate(frame.itertuples(index=False), start=2):
        user_id_int = int(row.user_id_int)
        target_id = int(row.target_id)
        raw_user_id = str(row.raw_user_id)
        raw_parent_asin = str(row.raw_parent_asin)
        if user_id_int == 0 or target_id == 0:
            issues.append(f"{name} 第 {row_number} 行出现真实 ID=0。")
            break
        if user_id_int not in user_values:
            issues.append(f"{name} 第 {row_number} 行 user_id_int 不在 user2id 中。")
            break
        if target_id not in item_values:
            issues.append(f"{name} 第 {row_number} 行 target_id 不在 item2id 中。")
            break
        if raw_user_id not in user2id or int(user2id[raw_user_id]) != user_id_int:
            issues.append(f"{name} 第 {row_number} 行 raw_user_id 与 user2id 不一致。")
            break
        if raw_parent_asin not in item2id or int(item2id[raw_parent_asin]) != target_id:
            issues.append(f"{name} 第 {row_number} 行 raw_parent_asin 与 item2id 不一致。")
            break
        for item_id in parse_seq_ids(row.seq_ids):
            if item_id == 0 or item_id not in item_values:
                issues.append(f"{name} 第 {row_number} 行 seq_ids 中存在非法 item_id。")
                return issues
    return issues


def check_sequence_rows(
    sequence_rows: list[list[int]],
    user_values: set[int],
    item_values: set[int],
) -> list[str]:
    """检查 seqrec_sequence.txt。"""
    issues: list[str] = []
    for line_number, row in enumerate(sequence_rows, start=1):
        if len(row) < 2:
            issues.append(f"seqrec_sequence.txt 第 {line_number} 行少于 2 列。")
            break
        user_id_int = row[0]
        item_ids = row[1:]
        if user_id_int == 0 or user_id_int not in user_values:
            issues.append(f"seqrec_sequence.txt 第 {line_number} 行 user_id 非法。")
            break
        if any(item_id == 0 or item_id not in item_values for item_id in item_ids):
            issues.append(f"seqrec_sequence.txt 第 {line_number} 行 item_id 非法。")
            break
    return issues


def check_interaction_rows(
    interaction_rows: list[list[int]],
    user_values: set[int],
    item_values: set[int],
) -> list[str]:
    """检查 seqrec_interactions.txt。"""
    issues: list[str] = []
    for line_number, row in enumerate(interaction_rows, start=1):
        if len(row) != 2:
            issues.append(f"seqrec_interactions.txt 第 {line_number} 行不是两列。")
            break
        user_id_int, item_id_int = row
        if user_id_int == 0 or user_id_int not in user_values:
            issues.append(f"seqrec_interactions.txt 第 {line_number} 行 user_id 非法。")
            break
        if item_id_int == 0 or item_id_int not in item_values:
            issues.append(f"seqrec_interactions.txt 第 {line_number} 行 item_id 非法。")
            break
    return issues


def check_stats(
    category: str,
    stats: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    user2id: dict[str, Any],
    item2id: dict[str, Any],
    sequence_rows: list[list[int]],
    interaction_rows: list[list[int]],
) -> list[str]:
    """检查 stats.json 的关键字段和数量是否匹配。"""
    issues: list[str] = []
    for key in REQUIRED_STATS_KEYS:
        if key not in stats:
            issues.append(f"stats.json 缺少字段: {key}")

    if issues:
        return issues

    expected_counts = {
        "category": category,
        "num_users": len(user2id),
        "num_items": len(item2id),
        "num_train_rows": len(frames["train"]),
        "num_dev_rows": len(frames["dev"]),
        "num_test_rows": len(frames["test"]),
        "num_seqrec_sequences": len(sequence_rows),
        "num_seqrec_interactions": len(interaction_rows),
        "id_start_from": 1,
        "padding_id": 0,
    }
    for key, expected in expected_counts.items():
        if stats.get(key) != expected:
            issues.append(f"stats.json {key}={stats.get(key)}，期望 {expected}。")

    if not str(stats.get("source_valid_file", "")).endswith(".valid.csv.gz"):
        issues.append("stats.json source_valid_file 未指向 raw valid.csv.gz。")

    sequence_lengths = [len(row) - 1 for row in sequence_rows]
    if sequence_lengths:
        if stats.get("min_sequence_length") != min(sequence_lengths):
            issues.append("stats.json min_sequence_length 不匹配。")
        if stats.get("max_sequence_length") != max(sequence_lengths):
            issues.append("stats.json max_sequence_length 不匹配。")
    return issues


def print_samples(paths: dict[str, Path], sample_size: int) -> None:
    """打印少量输出样本，便于人工复核。"""
    for file_name in ("train.tsv", "dev.tsv", "test.tsv"):
        frame = pd.read_csv(paths[file_name], sep="\t", nrows=sample_size)
        print(f"\n  {file_name} first_{min(sample_size, len(frame))}_rows:")
        print(frame.to_string(index=False))

    for file_name in ("seqrec_sequence.txt", "seqrec_interactions.txt"):
        print(f"\n  {file_name} first_{sample_size}_lines:")
        with paths[file_name].open("r", encoding="utf-8") as file_obj:
            for index, line in enumerate(file_obj, start=1):
                if index > sample_size:
                    break
                print("  " + line.rstrip())


def check_category(processed_dir: Path, category: str, sample_size: int) -> bool:
    """检查单个类别 processed 输出。"""
    print(f"\n[{category}]")
    paths = expected_files(processed_dir, category)
    issues: list[str] = []

    for file_name, path in paths.items():
        exists = path.exists()
        size = path.stat().st_size if exists else None
        print(f"  file {file_name}: exists={exists} size={size}")
        if not exists:
            issues.append(f"缺少文件: {path}")
        elif size == 0:
            issues.append(f"文件为 0 字节: {path}")

    if issues:
        for issue in issues:
            print(f"  ERROR: {issue}")
        return False

    frames = {
        "train": pd.read_csv(paths["train.tsv"], sep="\t"),
        "dev": pd.read_csv(paths["dev.tsv"], sep="\t"),
        "test": pd.read_csv(paths["test.tsv"], sep="\t"),
    }
    user2id = load_json(paths["user2id.json"])
    id2user = load_json(paths["id2user.json"])
    item2id = load_json(paths["item2id.json"])
    id2item = load_json(paths["id2item.json"])
    stats = load_json(paths["stats.json"])
    sequence_rows = read_space_lines(paths["seqrec_sequence.txt"])
    interaction_rows = read_space_lines(paths["seqrec_interactions.txt"])

    issues.extend(check_contiguous_ids("user2id", user2id))
    issues.extend(check_contiguous_ids("item2id", item2id))
    if len(id2user) != len(user2id):
        issues.append("id2user 数量与 user2id 不一致。")
    if len(id2item) != len(item2id):
        issues.append("id2item 数量与 item2id 不一致。")

    for split, frame in frames.items():
        issues.extend(check_tsv_frame(f"{split}.tsv", frame, user2id, item2id))

    user_values = {int(value) for value in user2id.values()}
    item_values = {int(value) for value in item2id.values()}
    issues.extend(check_sequence_rows(sequence_rows, user_values, item_values))
    issues.extend(check_interaction_rows(interaction_rows, user_values, item_values))
    issues.extend(
        check_stats(
            category,
            stats,
            frames,
            user2id,
            item2id,
            sequence_rows,
            interaction_rows,
        )
    )

    print("  TSV columns:", TSV_COLUMNS)
    print(f"  train/dev/test rows: {len(frames['train'])}/{len(frames['dev'])}/{len(frames['test'])}")
    print(f"  users/items: {len(user2id)}/{len(item2id)}")
    print(f"  sequences/interactions: {len(sequence_rows)}/{len(interaction_rows)}")
    print(f"  dev source: {stats.get('source_valid_file')}")

    if issues:
        for issue in issues:
            print(f"  ERROR: {issue}")
        return False

    print("  PASS: processed 数据检查通过。")
    print_samples(paths, sample_size)
    return True


def main() -> int:
    args = parse_args()
    if args.sample_size <= 0:
        print("ERROR: --sample-size 必须大于 0。")
        return 2

    ok = True
    print("processed 数据检查")
    print(f"processed_dir={args.processed_dir}")
    print(f"categories={args.categories}")
    for category in args.categories:
        ok = check_category(args.processed_dir, category, args.sample_size) and ok

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
