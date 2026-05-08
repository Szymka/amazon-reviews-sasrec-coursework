from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_CATEGORIES = (
    "Industrial_and_Scientific",
    "Musical_Instruments",
    "CDs_and_Vinyl",
)

SPLIT_TO_FILE = {
    "train": "train.tsv",
    "dev": "dev.tsv",
    "test": "test.tsv",
}

REQUIRED_COLUMNS = (
    "user_id_int",
    "target_id",
    "rating",
    "timestamp",
    "seq_ids",
    "raw_user_id",
    "raw_parent_asin",
)


def _coerce_scalar(value: str) -> Any:
    text = value.strip()
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def load_simple_yaml(path: str | Path) -> dict[str, Any]:
    """Load flat key-value config files used in this coursework repo."""
    config: dict[str, Any] = {}
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        config[key.strip()] = _coerce_scalar(value)
    return config


def parse_seq_ids(value: object) -> list[int]:
    if value is None or pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [int(part) for part in text.split()]


def pad_history(sequence: list[int], maxlen: int, padding_id: int = 0) -> list[int]:
    if maxlen <= 0:
        raise ValueError("maxlen must be positive.")
    trimmed = sequence[-maxlen:]
    return [padding_id] * (maxlen - len(trimmed)) + trimmed


@dataclass(frozen=True)
class ProcessedCategoryPaths:
    category: str
    train_file: Path
    dev_file: Path
    test_file: Path
    stats_file: Path

    @classmethod
    def from_category(
        cls,
        processed_root: str | Path,
        category: str,
    ) -> "ProcessedCategoryPaths":
        category_dir = Path(processed_root) / category
        return cls(
            category=category,
            train_file=category_dir / "train.tsv",
            dev_file=category_dir / "dev.tsv",
            test_file=category_dir / "test.tsv",
            stats_file=category_dir / "stats.json",
        )

    @classmethod
    def from_config(cls, config_path: str | Path) -> tuple["ProcessedCategoryPaths", int]:
        config = load_simple_yaml(config_path)
        required = ("category", "train_file", "dev_file", "test_file", "stats_file", "maxlen")
        missing = [key for key in required if key not in config]
        if missing:
            raise ValueError(f"config missing required keys: {missing}")
        return (
            cls(
                category=str(config["category"]),
                train_file=Path(str(config["train_file"])),
                dev_file=Path(str(config["dev_file"])),
                test_file=Path(str(config["test_file"])),
                stats_file=Path(str(config["stats_file"])),
            ),
            int(config["maxlen"]),
        )

    def split_file(self, split: str) -> Path:
        if split not in SPLIT_TO_FILE:
            raise ValueError(f"unknown split: {split}")
        return {
            "train": self.train_file,
            "dev": self.dev_file,
            "test": self.test_file,
        }[split]

    def validate(self) -> None:
        missing = [
            path for path in (self.train_file, self.dev_file, self.test_file, self.stats_file)
            if not path.exists()
        ]
        if missing:
            missing_text = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(f"missing processed files: {missing_text}")


class SASRecProcessedDataset:
    """Read processed TSV data and expose SASRec-ready padded histories."""

    def __init__(
        self,
        split_file: str | Path,
        split: str,
        maxlen: int,
        padding_id: int = 0,
        category: str | None = None,
        stats_file: str | Path | None = None,
    ) -> None:
        if split not in SPLIT_TO_FILE:
            raise ValueError(f"unsupported split: {split}")
        self.split = split
        self.maxlen = maxlen
        self.category = category
        self.split_file = Path(split_file)
        self.stats_file = Path(stats_file) if stats_file is not None else None
        self.stats = self._load_stats(self.stats_file)
        self.padding_id = int(
            self.stats.get("padding_id", padding_id) if self.stats is not None else padding_id
        )

        self.frame = pd.read_csv(self.split_file, sep="\t")
        self._validate_frame()

    @staticmethod
    def _load_stats(stats_file: Path | None) -> dict[str, Any] | None:
        if stats_file is None or not stats_file.exists():
            return None
        loaded = json.loads(stats_file.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"{stats_file} must contain a JSON object.")
        return loaded

    def _validate_frame(self) -> None:
        missing = [column for column in REQUIRED_COLUMNS if column not in self.frame.columns]
        if missing:
            raise ValueError(f"{self.split_file} missing required columns: {missing}")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        history = parse_seq_ids(row["seq_ids"])
        padded_history = pad_history(history, self.maxlen, self.padding_id)
        return {
            "category": self.category,
            "split": self.split,
            "user_id": int(row["user_id_int"]),
            "input_ids": padded_history,
            "target_id": int(row["target_id"]),
            "seq_len": min(len(history), self.maxlen),
            "rating": float(row["rating"]),
            "timestamp": int(row["timestamp"]),
            "raw_user_id": str(row["raw_user_id"]),
            "raw_parent_asin": str(row["raw_parent_asin"]),
        }

    def summary(self) -> dict[str, Any]:
        summary = {
            "category": self.category,
            "split": self.split,
            "num_rows": len(self),
            "maxlen": self.maxlen,
            "padding_id": self.padding_id,
        }
        if self.stats is not None:
            summary["num_users"] = self.stats.get("num_users")
            summary["num_items"] = self.stats.get("num_items")
        return summary


def build_category_datasets(
    processed_root: str | Path,
    category: str,
    maxlen: int,
) -> dict[str, SASRecProcessedDataset]:
    paths = ProcessedCategoryPaths.from_category(processed_root, category)
    paths.validate()
    stats = json.loads(paths.stats_file.read_text(encoding="utf-8"))
    padding_id = int(stats.get("padding_id", 0))
    return {
        split: SASRecProcessedDataset(
            split_file=paths.split_file(split),
            split=split,
            maxlen=maxlen,
            padding_id=padding_id,
            category=category,
            stats_file=paths.stats_file,
        )
        for split in SPLIT_TO_FILE
    }


def build_datasets_from_config(
    config_path: str | Path,
) -> dict[str, SASRecProcessedDataset]:
    paths, maxlen = ProcessedCategoryPaths.from_config(config_path)
    paths.validate()
    stats = json.loads(paths.stats_file.read_text(encoding="utf-8"))
    padding_id = int(stats.get("padding_id", 0))
    return {
        split: SASRecProcessedDataset(
            split_file=paths.split_file(split),
            split=split,
            maxlen=maxlen,
            padding_id=padding_id,
            category=paths.category,
            stats_file=paths.stats_file,
        )
        for split in SPLIT_TO_FILE
    }
