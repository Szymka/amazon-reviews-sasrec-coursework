"""Write ``<dataset>.random`` (user_token + tab + space-separated item tokens) without pretrained checkpoints."""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def parse_inter_tokens(inter_path: Path) -> tuple[list[str], list[str]]:
    users: set[str] = set()
    items: set[str] = set()
    with inter_path.open(encoding="utf-8") as f:
        header = f.readline()
        if "user_id" not in header:
            raise ValueError(f"unexpected header in {inter_path}: {header!r}")
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            users.add(parts[0])
            items.add(parts[1])
    user_list = sorted(users, key=lambda x: int(x))
    item_list = sorted(items, key=lambda x: int(x))
    return user_list, item_list


def write_random_candidates(
    inter_path: Path,
    out_path: Path,
    *,
    num_users: int,
    num_candidates: int,
    seed: int,
) -> None:
    random.seed(seed)
    users, items = parse_inter_tokens(inter_path)
    if not users or not items:
        raise ValueError("empty users or items from .inter")
    pick_u = min(num_users, len(users))
    chosen_users = random.sample(users, pick_u)
    item_pool = [i for i in items if int(i) > 0]
    if len(item_pool) < num_candidates:
        raise ValueError(f"need at least {num_candidates} distinct items, got {len(item_pool)}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for u in chosen_users:
            cands = random.sample(item_pool, num_candidates)
            f.write(f"{u}\t{' '.join(cands)}\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--category", type=str, required=True)
    p.add_argument("--recbole-root", type=Path, default=Path("data/recbole_llmrank"))
    p.add_argument("--num-users", type=int, default=500, help="How many users to evaluate (API cost).")
    p.add_argument("--num-candidates", type=int, default=100, help="Candidates per user in file (uses first recall_budget in LLMRank).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    inter_path = args.recbole_root / args.category / f"{args.category}.inter"
    out_path = args.recbole_root / args.category / f"{args.category}.random"
    write_random_candidates(
        inter_path,
        out_path,
        num_users=args.num_users,
        num_candidates=args.num_candidates,
        seed=args.seed,
    )
    print(f"[OK] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
