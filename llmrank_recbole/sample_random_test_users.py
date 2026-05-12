"""Build ``<dataset>.random`` using users that actually appear in RecBole's **test** split."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path


def write_random_candidates_test_split(
    *,
    llmrank_root: Path,
    coursework_root: Path,
    dataset_name: str,
    data_path: Path,
    out_path: Path,
    num_users: int,
    num_candidates: int,
    seed: int,
) -> None:
    import torch
    from recbole.config import Config
    from recbole.data import data_preparation
    from recbole.data.dataset.sequential_dataset import SequentialDataset
    from recbole.model.sequential_recommender import SASRec
    from recbole.utils import init_logger, init_seed

    props = [
        str(llmrank_root / "props" / "overall.yaml"),
        str(llmrank_root / "props" / "SASRec.yaml"),
        str(coursework_root / "configs" / "llmrank_recbole" / "amazon_sequential.yaml"),
    ]
    config = Config(
        model=SASRec,
        dataset=dataset_name,
        config_file_list=props,
        config_dict={"data_path": str(data_path.resolve()), "device": "cpu"},
    )
    init_seed(seed, config["reproducibility"])
    init_logger(config)

    dataset = SequentialDataset(config)
    _train_data, _valid_data, test_data = data_preparation(config, dataset)

    u_col = test_data.dataset.inter_feat["user_id"]
    uniq = torch.unique(u_col).tolist()
    uid_tokens = [dataset.field2id_token["user_id"][int(i)] for i in uniq if int(i) != 0]
    if not uid_tokens:
        raise RuntimeError("no users in RecBole test split")

    item_tokens = list(dataset.field2id_token["item_id"][1:])
    if len(item_tokens) < num_candidates:
        raise ValueError(f"need {num_candidates} item tokens, got {len(item_tokens)}")

    random.seed(seed)
    pick = min(num_users, len(uid_tokens))
    chosen = random.sample(uid_tokens, pick)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for u in chosen:
            cands = random.sample(item_tokens, num_candidates)
            f.write(f"{u}\t{' '.join(cands)}\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--category", type=str, required=True)
    p.add_argument("--recbole-root", type=Path, default=Path("data/recbole_llmrank"))
    p.add_argument("--num-users", type=int, default=400)
    p.add_argument("--num-candidates", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cw = Path(__file__).resolve().parents[2]
    llm = cw.parent / "LLMRank" / "llmrank"
    os.chdir(cw)
    data_path = (cw / args.recbole_root).resolve()
    out_path = data_path / args.category / f"{args.category}.random"
    write_random_candidates_test_split(
        llmrank_root=llm,
        coursework_root=cw,
        dataset_name=args.category,
        data_path=data_path,
        out_path=out_path,
        num_users=args.num_users,
        num_candidates=args.num_candidates,
        seed=args.seed,
    )
    print(f"[OK] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
