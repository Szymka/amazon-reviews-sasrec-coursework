"""在三个课程类别上依次训练 SASRec，写出 metrics jsonl 并可选调用画图脚本。

在 conda 环境 `llmrec`、仓库根目录下执行::

    conda activate llmrec
    python scripts/prepare_allmrec_amazon.py --overwrite
    python scripts/run_three_categories_sasrec.py --num_epochs 5 --device cuda:0 --batch_size 256 --n_workers 1
    python scripts/plot_coursework_metrics.py --metrics_dir results/coursework
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_CATEGORIES = (
    "Industrial_and_Scientific",
    "Musical_Instruments",
    "CDs_and_Vinyl",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="三类别 SASRec 顺序训练并记录 NDCG@10 指标。")
    p.add_argument(
        "--categories",
        nargs="+",
        default=list(DEFAULT_CATEGORIES),
        help="与 data/amazon/<name>.txt 一致的类别名。",
    )
    p.add_argument("--device", default="cuda:0", type=str)
    p.add_argument("--num_epochs", default=5, type=int)
    p.add_argument("--batch_size", default=256, type=int)
    p.add_argument("--n_workers", default=1, type=int)
    p.add_argument("--eval_every", default=1, type=int, help="每多少个 epoch 评估一次；快速实验可设为 1。")
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path("results/coursework"),
        help="metrics jsonl 输出目录。",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="三类别全部跑完后执行 scripts/plot_coursework_metrics.py。",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    sasrec_main = repo / "pre_train" / "sasrec" / "main.py"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for cat in args.categories:
        metrics_path = args.out_dir / f"{cat}_metrics.jsonl"
        if metrics_path.exists():
            metrics_path.unlink()

        cmd = [
            sys.executable,
            str(sasrec_main),
            "--dataset",
            cat,
            "--skip_preprocess",
            "--device",
            args.device,
            "--num_epochs",
            str(args.num_epochs),
            "--batch_size",
            str(args.batch_size),
            "--n_workers",
            str(args.n_workers),
            "--eval_every",
            str(args.eval_every),
            "--metrics_jsonl",
            str(metrics_path.resolve()),
        ]
        print("Running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=str(repo), check=True)

    if args.plot:
        plot_script = repo / "scripts" / "plot_coursework_metrics.py"
        subprocess.run(
            [sys.executable, str(plot_script), "--metrics_dir", str(args.out_dir.resolve())],
            cwd=str(repo),
            check=True,
        )


if __name__ == "__main__":
    main()
