#!/usr/bin/env python3
"""
Run the upstream **LLMRank Rank** model (RecBole + OpenAI API) on Amazon coursework data.

Prerequisites:
  - Sibling repo folder ``../LLMRank/llmrank`` (same layout as the official LLMRank project).
  - Processed coursework data under ``data/processed/<category>/`` (``seqrec_sequence.txt``, ``id2item.json``).
  - ``pip install -r requirements-llmrank-recbole.txt`` inside conda env ``llmrec``.
  - Copy ``configs/llmrank_recbole/openai_api.yaml.example`` to ``secrets/openai_api.yaml`` and fill ``api_key``.

This script **does not train** the Rank module (LLMRank is zero-shot). It runs RecBole
sequential evaluation with ``SelectedUserTrainer`` + OpenAI calls, as in ``LLMRank/llmrank/evaluate.py``.
"""

from __future__ import annotations

import argparse
import os
import sys
from logging import getLogger
from pathlib import Path


def _coursework_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _llmrank_src() -> Path:
    return Path(__file__).resolve().parents[2] / "LLMRank" / "llmrank"


def _run_evaluate(
    *,
    props: list[str],
    model_name: str,
    dataset_name: str,
    pretrained_file: str,
    config_dict: dict,
) -> dict:
    import torch
    from recbole.config import Config
    from recbole.data import data_preparation
    from recbole.data.dataset.sequential_dataset import SequentialDataset
    from recbole.utils import init_seed, init_logger, set_color
    from trainer import SelectedUserTrainer
    from utils import get_model

    model_class = get_model(model_name)
    config = Config(
        model=model_class,
        dataset=dataset_name,
        config_file_list=props,
        config_dict=config_dict,
    )
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = SequentialDataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = model_class(config, train_data.dataset).to(config["device"])
    if pretrained_file:
        checkpoint = torch.load(pretrained_file, map_location=config["device"])
        logger.info("Loading from %s", pretrained_file)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        model.load_other_parameter(checkpoint.get("other_parameter"))

    logger.info(model)
    trainer = SelectedUserTrainer(config, model, dataset)
    test_result = trainer.evaluate(
        test_data,
        load_best_model=False,
        show_progress=config["show_progress"],
    )
    logger.info(set_color("test result", "yellow") + f": {test_result}")
    return dict(test_result)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--category", type=str, required=True)
    p.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    p.add_argument("--recbole-root", type=Path, default=Path("data/recbole_llmrank"))
    p.add_argument("--prepare", action="store_true", help="Export .inter/.item and build .random candidates.")
    p.add_argument("--num-eval-users", type=int, default=400)
    p.add_argument("--num-candidates-file", type=int, default=100)
    p.add_argument("--recall-budget", type=int, default=20)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--pretrained", type=str, default="", help="Optional RecBole checkpoint for Rank (usually empty).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cw = _coursework_root()
    llm = _llmrank_src()
    if not llm.is_dir():
        print(f"ERROR: LLMRank source not found at {llm}")
        return 2

    os.chdir(cw)
    if args.prepare:
        from llmrank_recbole.export_atomic import export_category
        from llmrank_recbole.sample_random_test_users import write_random_candidates_test_split

        export_category(args.processed_root, args.category, args.recbole_root)
        data_path = (cw / args.recbole_root).resolve()
        out_random = data_path / args.category / f"{args.category}.random"
        write_random_candidates_test_split(
            llmrank_root=llm,
            coursework_root=cw,
            dataset_name=args.category,
            data_path=data_path,
            out_path=out_random,
            num_users=args.num_eval_users,
            num_candidates=args.num_candidates_file,
            seed=args.seed,
        )
        print(f"[prepare] wrote RecBole dataset under {data_path / args.category}")

    secrets = cw / "secrets" / "openai_api.yaml"
    if not secrets.exists():
        print(f"ERROR: missing {secrets} — copy from configs/llmrank_recbole/openai_api.yaml.example")
        return 2

    props_rank = llm / "props" / "Rank.yaml"
    props_ds = cw / "configs" / "llmrank_recbole" / "amazon_sequential.yaml"
    props_overall = llm / "props" / "overall.yaml"
    for path in (props_rank, props_ds, props_overall):
        if not path.exists():
            print(f"ERROR: missing config file {path}")
            return 2

    data_path = (cw / args.recbole_root).resolve()
    props = [
        str(props_rank),
        str(props_ds),
        str(secrets.resolve()),
        str(props_overall),
    ]

    os.chdir(llm)
    sys.path.insert(0, str(llm))
    sys.path.insert(0, str(cw))

    from model.rank import Rank as RankCls

    from llmrank_recbole.patch_rank_amazon import apply_patch

    apply_patch(RankCls)

    extra = {
        "data_path": str(data_path),
        "device": args.device,
        "seed": args.seed,
        "selected_user_suffix": "random",
        "recall_budget": int(args.recall_budget),
        "reproducibility": True,
    }

    print("Running LLMRank Rank evaluation (OpenAI API calls; this may take a long time and cost money).")
    print(f"  dataset={args.category}  data_path={data_path}")
    result = _run_evaluate(
        props=props,
        model_name="Rank",
        dataset_name=args.category,
        pretrained_file=args.pretrained,
        config_dict=extra,
    )
    print("Done:", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
