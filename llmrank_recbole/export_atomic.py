"""Export RecBole sequential atomic files from coursework ``seqrec_sequence.txt`` + ``id2item.json``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def export_category(processed_root: Path, category: str, recbole_root: Path) -> Path:
    seq_path = processed_root / category / "seqrec_sequence.txt"
    id2item_path = processed_root / category / "id2item.json"
    if not seq_path.exists():
        raise FileNotFoundError(f"missing {seq_path}")
    if not id2item_path.exists():
        raise FileNotFoundError(f"missing {id2item_path}")

    id2item = json.loads(id2item_path.read_text(encoding="utf-8"))
    out_dir = recbole_root / category
    out_dir.mkdir(parents=True, exist_ok=True)

    item_titles: dict[str, str] = {}
    rows: list[tuple[str, str, float, float]] = []
    ts = 0.0
    for line in seq_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        uid = parts[0]
        for iid in parts[1:]:
            ts += 1.0
            rows.append((uid, iid, 1.0, ts))
            title = id2item.get(str(iid), f"item_{iid}")
            item_titles[str(iid)] = str(title).replace("\t", " ").replace("\n", " ")[:512]

    inter_path = out_dir / f"{category}.inter"
    with inter_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("user_id:token\titem_id:token\trating:float\ttimestamp:float\n")
        for uid, iid, rating, t in rows:
            f.write(f"{uid}\t{iid}\t{rating}\t{t}\n")

    item_path = out_dir / f"{category}.item"
    with item_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("item_id:token\ttitle:token_seq\n")
        f.write("0\t<pad>\n")
        for iid in sorted(item_titles.keys(), key=lambda x: int(x)):
            f.write(f"{iid}\t{item_titles[iid]}\n")

    return out_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    p.add_argument("--recbole-root", type=Path, default=Path("data/recbole_llmrank"))
    p.add_argument(
        "--categories",
        nargs="+",
        default=["Industrial_and_Scientific", "Musical_Instruments", "CDs_and_Vinyl"],
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    for cat in args.categories:
        out = export_category(args.processed_root, cat, args.recbole_root)
        print(f"[OK] {cat} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
