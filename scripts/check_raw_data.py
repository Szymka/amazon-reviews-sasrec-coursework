"""Lightweight raw-data checks for Amazon Reviews 2023 5-Core files.

只检查文件存在性、文件大小和少量样本行；不会下载数据、删除文件、训练模型，
也不会执行全量预处理。
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_CATEGORIES = (
    "Industrial_and_Scientific",
    "Musical_Instruments",
    "CDs_and_Vinyl",
)

SPLITS = ("train", "valid", "test")
REQUIRED_CSV_COLUMNS = ("user_id", "parent_asin", "rating", "timestamp", "history")

REVIEW_COMMON_FIELDS = (
    "rating",
    "title",
    "text",
    "asin",
    "parent_asin",
    "user_id",
    "timestamp",
)

META_COMMON_FIELDS = (
    "main_category",
    "title",
    "average_rating",
    "rating_number",
    "features",
    "description",
    "price",
    "images",
    "videos",
    "store",
    "categories",
    "details",
    "parent_asin",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "检查 Amazon Reviews 2023 三个 5-Core 类别的 raw 文件命名、大小、"
            "CSV 字段和 JSONL 样本字段。"
        )
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="raw 数据根目录，默认: data/raw",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=list(DEFAULT_CATEGORIES),
        help="需要检查的类别，默认检查三个大作业类别。",
    )
    parser.add_argument(
        "--csv-rows",
        type=int,
        default=5,
        help="每个 csv.gz 抽样读取的行数，默认: 5",
    )
    parser.add_argument(
        "--jsonl-rows",
        type=int,
        default=3,
        help="每个 jsonl.gz 抽样读取的行数，默认: 3",
    )
    return parser.parse_args()


def expected_files(raw_dir: Path, category: str) -> dict[str, Path]:
    category_dir = raw_dir / category
    files = {
        split: category_dir / f"{category}.{split}.csv.gz" for split in SPLITS
    }
    files["review"] = category_dir / f"{category}.jsonl.gz"
    files["meta"] = category_dir / f"meta_{category}.jsonl.gz"
    return files


def format_size(byte_count: int) -> str:
    size = float(byte_count)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{byte_count} B"


def truncate_value(value: Any, limit: int = 120) -> Any:
    if value is None:
        return None
    text = str(value)
    if len(text) <= limit:
        return value
    return text[: limit - 3] + "..."


def compact_records(df: pd.DataFrame, row_count: int = 2) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in df.head(row_count).to_dict(orient="records"):
        records.append({key: truncate_value(value) for key, value in record.items()})
    return records


def read_jsonl_records(path: Path, row_count: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if len(records) >= row_count:
                break
    return records


def print_inventory(raw_dir: Path, categories: list[str]) -> tuple[bool, dict[str, dict[str, Path]]]:
    print("== 1. raw directory and expected file inventory ==")
    all_ok = True
    files_by_category: dict[str, dict[str, Path]] = {}

    for category in categories:
        category_dir = raw_dir / category
        print(f"\n[{category}] directory_exists={category_dir.is_dir()} path={category_dir}")
        if not category_dir.is_dir():
            all_ok = False

        files = expected_files(raw_dir, category)
        files_by_category[category] = files
        for label, path in files.items():
            exists = path.exists()
            size = path.stat().st_size if exists else None
            zero = size == 0 if size is not None else None
            size_text = format_size(size) if size is not None else "missing"
            print(
                f"  {label:>6}: exists={exists} size={size_text} bytes={size} "
                f"zero_bytes={zero} file={path.name}"
            )
            if not exists or zero:
                all_ok = False

    part_files = sorted(raw_dir.rglob("*.part")) if raw_dir.exists() else []
    print("\n== 2. .part files ==")
    if part_files:
        for path in part_files:
            print(f"  PART: {path} ({format_size(path.stat().st_size)})")
        all_ok = False
    else:
        print("  no .part files found under data/raw")

    return all_ok, files_by_category


def check_csv_files(files_by_category: dict[str, dict[str, Path]], csv_rows: int) -> bool:
    print("\n== 3. csv.gz sample checks ==")
    all_ok = True
    required = set(REQUIRED_CSV_COLUMNS)

    for category, files in files_by_category.items():
        for split in SPLITS:
            path = files[split]
            print(f"\n[{category}] {path.name}")
            try:
                df = pd.read_csv(path, nrows=csv_rows)
            except Exception as exc:  # noqa: BLE001 - report exact local data error.
                print(f"  READ_ERROR: {exc}")
                all_ok = False
                continue

            columns = list(df.columns)
            missing = [column for column in REQUIRED_CSV_COLUMNS if column not in columns]
            print(f"  columns: {columns}")
            print(f"  required_columns_present={not missing} missing={missing}")
            print(
                "  first_2_rows: "
                + json.dumps(compact_records(df, 2), ensure_ascii=False, default=str)
            )
            if missing:
                all_ok = False

    return all_ok


def check_jsonl_files(files_by_category: dict[str, dict[str, Path]], jsonl_rows: int) -> bool:
    print("\n== 4. jsonl.gz sample field checks ==")
    all_ok = True

    for category, files in files_by_category.items():
        for label, common_fields in (
            ("review", REVIEW_COMMON_FIELDS),
            ("meta", META_COMMON_FIELDS),
        ):
            path = files[label]
            print(f"\n[{category}] {label} {path.name}")
            try:
                records = read_jsonl_records(path, jsonl_rows)
            except Exception as exc:  # noqa: BLE001 - report exact local data error.
                print(f"  READ_ERROR: {exc}")
                all_ok = False
                continue

            field_union = sorted({field for record in records for field in record.keys()})
            common_present = [field for field in common_fields if field in field_union]
            common_missing = [field for field in common_fields if field not in field_union]
            print(f"  rows_read={len(records)}")
            print(f"  actual_fields_union: {field_union}")
            print(f"  common_present: {common_present}")
            print(f"  common_missing_in_first_{jsonl_rows}_rows: {common_missing}")
            for index, record in enumerate(records, start=1):
                print(f"  row_{index}_fields: {sorted(record.keys())}")
            if not records:
                all_ok = False

    return all_ok


def main() -> int:
    args = parse_args()
    if args.csv_rows <= 0 or args.jsonl_rows <= 0:
        print("ERROR: --csv-rows and --jsonl-rows must be positive.")
        return 2

    inventory_ok, files_by_category = print_inventory(args.raw_dir, args.categories)
    if not inventory_ok:
        print("\nSTOP: raw 文件不完整、存在 0 字节文件，或发现 .part 残留。")
        print("未继续执行 CSV/JSONL 抽样读取，更不会执行预处理。")
        return 1

    csv_ok = check_csv_files(files_by_category, args.csv_rows)
    jsonl_ok = check_jsonl_files(files_by_category, args.jsonl_rows)

    print("\n== 5. summary ==")
    print(f"directories_and_files_ok={inventory_ok}")
    print(f"csv_sample_checks_ok={csv_ok}")
    print(f"jsonl_sample_checks_ok={jsonl_ok}")
    print("preprocessing_ran=False")
    return 0 if inventory_ok and csv_ok and jsonl_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
