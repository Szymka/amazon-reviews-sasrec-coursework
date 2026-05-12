"""Prepare data, train A-LLMRec SASRec backbone, evaluate NDCG@10 on coursework splits."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALLMREC_ROOT = PROJECT_ROOT / "A-LLMRec"
CONFIG = PROJECT_ROOT / "configs" / "seqrec_industrial.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--category", default="Industrial_and_Scientific")
    p.add_argument("--num-epochs", type=int, default=50)
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip-prepare", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--eval-split", default="test", choices=("dev", "test"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    py = sys.executable

    if not args.skip_prepare:
        r = subprocess.run(
            [py, str(PROJECT_ROOT / "scripts" / "prepare_allmrec_amazon_data.py"), "--category", args.category],
            cwd=str(PROJECT_ROOT),
        )
        if r.returncode != 0:
            return r.returncode

    if not args.skip_train:
        r = subprocess.run(
            [
                py,
                str(ALLMREC_ROOT / "train_sasrec_coursework.py"),
                "--dataset",
                args.category,
                "--num_epochs",
                str(args.num_epochs),
                "--device",
                args.device,
            ],
            cwd=str(ALLMREC_ROOT),
        )
        if r.returncode != 0:
            return r.returncode

    save_dir = ALLMREC_ROOT / "pre_train" / "sasrec" / args.category
    def _epoch_key(p: Path) -> int:
        m = re.search(r"epoch=(\d+)", p.name)
        return int(m.group(1)) if m else -1

    checkpoints = sorted(save_dir.glob("SASRec.epoch=*.pth"), key=_epoch_key)
    if not checkpoints:
        print(f"No checkpoint in {save_dir}; cannot evaluate.", file=sys.stderr)
        return 1
    ckpt = checkpoints[-1]

    r = subprocess.run(
        [
            py,
            str(PROJECT_ROOT / "evaluation" / "eval_allmrec_sasrec_ndcg.py"),
            "--config",
            str(CONFIG),
            "--checkpoint",
            str(ckpt),
            "--split",
            args.eval_split,
            "--device",
            args.device,
        ],
        cwd=str(PROJECT_ROOT),
    )
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
