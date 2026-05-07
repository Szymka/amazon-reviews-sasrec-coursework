"""生成数据预处理说明文档和 summary.md 的脚本骨架。

本轮只提供命令行参数、路径规划和简要模板。后续可以接入 processed
统计信息，自动生成每个类别的数据摘要。
"""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_CATEGORIES = (
    "Industrial_and_Scientific",
    "Musical_Instruments",
    "CDs_and_Vinyl",
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="生成数据预处理说明文档和 summary.md。"
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="处理后数据根目录，默认: data/processed",
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path("docs"),
        help="文档输出目录，默认: docs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/summary.md"),
        help="汇总文档输出路径，默认: docs/summary.md",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=list(DEFAULT_CATEGORIES),
        help="需要写入文档的商品类别列表。",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="实际写入 summary.md；默认只打印将要生成的内容。",
    )
    return parser.parse_args()


def build_summary(categories: list[str], processed_dir: Path) -> str:
    """构造 summary.md 的占位内容。"""
    lines = [
        "# 数据预处理 Summary",
        "",
        "本文件后续由 scripts/make_data_readme.py 自动生成。",
        "",
        "每个类别的输出文件统一为 train.tsv、dev.tsv、test.tsv、sasrec_sequence.txt、user2id.json、id2user.json、item2id.json、id2item.json、stats.json。",
        "",
        "样本级 TSV 字段: user_id_int, target_id, rating, timestamp, seq_ids, raw_user_id, raw_parent_asin。",
        "",
        "## 类别",
    ]
    for category in categories:
        lines.extend(
            [
                f"- {category}",
                f"  - processed 目录: {processed_dir / category}",
                "  - 用户数: TODO",
                "  - 商品数: TODO",
                "  - train/dev/test 样本数: TODO",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    summary = build_summary(args.categories, args.processed_dir)

    if args.write:
        args.docs_dir.mkdir(parents=True, exist_ok=True)
        args.output.write_text(summary, encoding="utf-8")
        print(f"已写入: {args.output}")
        return

    print("summary.md 生成脚本骨架")
    print(f"文档目录: {args.docs_dir}")
    print(f"输出文件: {args.output}")
    print("默认不写入文件；如需写入占位 summary.md，可追加 --write。")
    print("\n--- 预览 ---")
    print(summary)


if __name__ == "__main__":
    main()
