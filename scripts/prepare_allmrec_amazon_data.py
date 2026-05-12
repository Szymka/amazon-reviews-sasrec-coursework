"""Build A-LLMRec-style data/amazon files from coursework-processed splits."""

from __future__ import annotations

import argparse
import pickle
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export Industrial (or other) category into A-LLMRec data/amazon layout.",
    )
    p.add_argument(
        "--category",
        type=str,
        default="Industrial_and_Scientific",
        help="Subfolder under data/processed",
    )
    p.add_argument(
        "--coursework-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="amazon-reviews-sasrec-coursework root",
    )
    p.add_argument(
        "--allmrec-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "A-LLMRec",
        help="Embedded A-LLMRec root (contains data/amazon).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    processed = args.coursework_root / "data" / "processed" / args.category
    interactions = processed / "sasrec_interactions.txt"
    id2item_path = processed / "id2item.json"
    if not interactions.is_file():
        raise FileNotFoundError(f"missing {interactions}")
    if not id2item_path.is_file():
        raise FileNotFoundError(f"missing {id2item_path}")

    out_dir = args.allmrec_root / "data" / "amazon"
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_out = out_dir / f"{args.category}.txt"
    shutil.copyfile(interactions, txt_out)

    import json

    id2item = json.loads(id2item_path.read_text(encoding="utf-8"))
    name_dict: dict[str, dict[int, str]] = {"title": {}, "description": {}}
    for sid, asin in id2item.items():
        idx = int(sid)
        asin_s = str(asin)
        name_dict["title"][idx] = asin_s
        name_dict["description"][idx] = f"Amazon product {asin_s}"

    dict_out = out_dir / f"{args.category}_text_name_dict.json.gz"
    with open(dict_out, "wb") as f:
        pickle.dump(name_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Wrote {txt_out}")
    print(f"Wrote {dict_out} (pickle; title=ASIN, synthetic description)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
