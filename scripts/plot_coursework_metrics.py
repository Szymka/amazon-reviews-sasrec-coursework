"""从 `run_three_categories_sasrec.py` 或单次 SASRec 训练生成的 `*_metrics.jsonl` 画评估曲线 / 柱状图。

示例::

    conda activate llmrec
    python scripts/plot_coursework_metrics.py --metrics_dir results/coursework
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="根据 metrics jsonl 生成 NDCG@10 / HR@10 图表。")
    p.add_argument(
        "--metrics_dir",
        type=Path,
        default=Path("results/coursework"),
        help="包含 <类别>_metrics.jsonl 的目录。",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="图表输出目录，默认与 metrics_dir 相同。",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or args.metrics_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.metrics_dir.glob("*_metrics.jsonl"))
    if not files:
        raise FileNotFoundError(f"未找到 *_metrics.jsonl: {args.metrics_dir}")

    series: dict[str, list[dict]] = {}
    finals: list[tuple[str, dict]] = []
    for fp in files:
        name = fp.name.replace("_metrics.jsonl", "")
        rows = load_jsonl(fp)
        if not rows:
            continue
        series[name] = rows
        finals.append((name, rows[-1]))

    # 1) 每个类别：epoch — NDCG@10 / HR@10（验证 + 测试）
    fig1, axes = plt.subplots(len(series), 1, figsize=(8, 3.2 * max(1, len(series))), squeeze=False)
    for ax_row, (name, rows) in zip(axes, series.items()):
        ax = ax_row[0]
        xs = [r["epoch"] for r in rows]
        ax.plot(xs, [r["valid_ndcg10"] for r in rows], marker="o", label="valid NDCG@10")
        ax.plot(xs, [r["test_ndcg10"] for r in rows], marker="s", label="test NDCG@10")
        ax.plot(xs, [r["valid_hr10"] for r in rows], linestyle="--", label="valid HR@10")
        ax.plot(xs, [r["test_hr10"] for r in rows], linestyle=":", label="test HR@10")
        ax.set_title(name)
        ax.set_xlabel("epoch")
        ax.set_ylabel("metric")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig1.tight_layout()
    p1 = out_dir / "metrics_curves_ndcg_hr.png"
    fig1.savefig(p1, dpi=150)
    plt.close(fig1)
    print(f"Wrote {p1.resolve()}")

    # 2) 各类别最后一轮：测试集 NDCG@10 / HR@10 柱状图
    if finals:
        names = [n for n, _ in finals]
        ndcg = [float(r["test_ndcg10"]) for _, r in finals]
        hr = [float(r["test_hr10"]) for _, r in finals]
        x = range(len(names))
        w = 0.35
        fig2, ax = plt.subplots(figsize=(9, 4))
        ax.bar([i - w / 2 for i in x], ndcg, width=w, label="test NDCG@10")
        ax.bar([i + w / 2 for i in x], hr, width=w, label="test HR@10")
        ax.set_xticks(list(x))
        ax.set_xticklabels(names, rotation=15, ha="right")
        ax.set_ylabel("score")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig2.tight_layout()
        p2 = out_dir / "final_test_ndcg10_hr10_bar.png"
        fig2.savefig(p2, dpi=150)
        plt.close(fig2)
        print(f"Wrote {p2.resolve()}")


if __name__ == "__main__":
    main()
