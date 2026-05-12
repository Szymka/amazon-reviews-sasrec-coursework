"""从 `data/processed/<类别>/` 生成 A-LLMRec 所需的 `data/amazon/` 文件。

为每个类别写入：
  - `data/amazon/<类别名>.txt`：与 A-LLMRec `data_partition` 一致的用户-物品交互行（`user_id item_id`），
    直接复制 `sasrec_interactions.txt`。
  - `data/amazon/<类别名>_text_name_dict.json.gz`：与原仓库相同，为 **pickle** 序列化的
    `{'title': {item_id: str}, 'description': {item_id: str}}`。若无商品文案，则用 parent ASIN 构造占位标题。

用法（在项目根目录）::

    python scripts/prepare_allmrec_amazon.py
    python scripts/prepare_allmrec_amazon.py --categories Industrial_and_Scientific
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path

DEFAULT_CATEGORIES = (
    "Industrial_and_Scientific",
    "Musical_Instruments",
    "CDs_and_Vinyl",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="为 A-LLMRec 准备 data/amazon 下的 .txt 与文本字典。")
    p.add_argument(
        "--processed-root",
        type=Path,
        default=Path("data/processed"),
        help="预处理数据根目录（含各类别子目录）。",
    )
    p.add_argument(
        "--amazon-root",
        type=Path,
        default=Path("data/amazon"),
        help="输出目录，对应 A-LLMRec 默认的 ./data/amazon/。",
    )
    p.add_argument(
        "--categories",
        nargs="+",
        default=list(DEFAULT_CATEGORIES),
        help="类别目录名，与 data/processed 下文件夹一致。",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已存在的目标文件。",
    )
    return p.parse_args()


def build_text_dict(id2item_path: Path) -> dict:
    raw = json.loads(id2item_path.read_text(encoding="utf-8"))
    titles: dict[int, str] = {}
    descriptions: dict[int, str] = {}
    for k, asin in raw.items():
        iid = int(k)
        asin_s = str(asin)
        titles[iid] = f"Amazon product {asin_s}"
        descriptions[iid] = "Description not available in this coursework export."
    return {"title": titles, "description": descriptions}


def main() -> None:
    args = parse_args()
    args.amazon_root.mkdir(parents=True, exist_ok=True)

    for cat in args.categories:
        proc = args.processed_root / cat
        inter_src = proc / "sasrec_interactions.txt"
        id2item = proc / "id2item.json"
        if not inter_src.is_file():
            raise FileNotFoundError(f"缺少交互文件: {inter_src}")
        if not id2item.is_file():
            raise FileNotFoundError(f"缺少 id2item: {id2item}")

        txt_dst = args.amazon_root / f"{cat}.txt"
        dict_dst = args.amazon_root / f"{cat}_text_name_dict.json.gz"

        if txt_dst.exists() and not args.overwrite:
            raise FileExistsError(f"已存在（加 --overwrite 覆盖）: {txt_dst}")
        if dict_dst.exists() and not args.overwrite:
            raise FileExistsError(f"已存在（加 --overwrite 覆盖）: {dict_dst}")

        shutil.copyfile(inter_src, txt_dst)
        blob = pickle.dumps(build_text_dict(id2item), protocol=pickle.HIGHEST_PROTOCOL)
        dict_dst.write_bytes(blob)
        print(f"OK: {txt_dst} ({txt_dst.stat().st_size} bytes)")
        print(f"OK: {dict_dst} ({dict_dst.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
