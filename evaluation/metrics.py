from __future__ import annotations

from collections.abc import Iterable

import torch


def _sanitize_ranking_logits(scores: torch.Tensor) -> torch.Tensor:
    """Convert NaN/±Inf logits to finite extremes so rankings stay well-defined."""
    return torch.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)


def _effective_k(scores: torch.Tensor, k: int) -> int:
    """Top-k cannot exceed the number of scored classes."""
    return int(min(k, scores.shape[1]))


def _ranks_of_targets(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """1-based rank by descending score; ties broken by strict `>` count."""
    scores = _sanitize_ranking_logits(scores)
    target_scores = scores.gather(1, targets.view(-1, 1)).squeeze(1)
    ranks = (scores > target_scores.unsqueeze(1)).sum(dim=1) + 1
    bad = torch.isnan(target_scores) | torch.isinf(target_scores)
    if bad.any():
        ranks = ranks.masked_fill(bad, scores.size(1) + 1)
    return ranks


def ndcg_at_k(scores: torch.Tensor, targets: torch.Tensor, k: int = 10) -> float:
    scores = scores.detach().cpu()
    targets = targets.detach().cpu().long()
    scores = _sanitize_ranking_logits(scores)
    if scores.ndim != 2:
        raise ValueError("scores must be 2D (batch, num_classes).")
    ranks = _ranks_of_targets(scores, targets)
    gains = torch.where(
        ranks <= k,
        1.0 / torch.log2(ranks.double() + 1),
        torch.zeros_like(ranks, dtype=torch.double),
    )
    return float(gains.mean().item())


def hit_rate_at_k(scores: torch.Tensor, targets: torch.Tensor, k: int = 10) -> float:
    scores = scores.detach().cpu()
    scores = _sanitize_ranking_logits(scores)
    targets = targets.detach().cpu().long()
    ek = _effective_k(scores, k)
    topk = scores.topk(ek, dim=1).indices
    hits = (topk == targets.view(-1, 1)).any(dim=1)
    return float(hits.float().mean().item())


def mrr_at_k(scores: torch.Tensor, targets: torch.Tensor, k: int = 10) -> float:
    scores = scores.detach().cpu()
    scores = _sanitize_ranking_logits(scores)
    targets = targets.detach().cpu().long()
    ranks = _ranks_of_targets(scores, targets)
    reciprocal = torch.where(
        ranks <= k,
        1.0 / ranks.double(),
        torch.zeros_like(ranks, dtype=torch.double),
    )
    return float(reciprocal.mean().item())


def precision_at_k(scores: torch.Tensor, targets: torch.Tensor, k: int = 10) -> float:
    scores = scores.detach().cpu()
    scores = _sanitize_ranking_logits(scores)
    targets = targets.detach().cpu().long()
    ek = _effective_k(scores, k)
    topk = scores.topk(ek, dim=1).indices
    hits = (topk == targets.view(-1, 1)).any(dim=1)
    return float((hits.float().sum() / (len(targets) * ek)).item())


def evaluate(scores: torch.Tensor, targets: torch.Tensor, k: int = 10) -> dict[str, float]:
    return {
        "ndcg": ndcg_at_k(scores, targets, k),
        "hit_rate": hit_rate_at_k(scores, targets, k),
        "recall": hit_rate_at_k(scores, targets, k),
        "mrr": mrr_at_k(scores, targets, k),
        "precision": precision_at_k(scores, targets, k),
    }


def evaluate_batches(
    batch_iter: Iterable[tuple[torch.Tensor, torch.Tensor]],
    k: int = 10,
) -> dict[str, float]:
    """
    Same metrics as ``evaluate`` but over many small batches without concatenating
    ``scores`` into one (N, num_classes) tensor — avoids RAM blowups when N is large.
    """
    total_n = 0
    sum_ndcg = 0.0
    sum_hr = 0.0
    sum_mrr = 0.0
    sum_hit_users = 0.0
    ek_ref: int | None = None

    for scores, targets in batch_iter:
        scores = scores.detach().cpu()
        targets = targets.detach().cpu().long()
        if scores.ndim != 2:
            raise ValueError("scores must be 2D (batch, num_classes).")
        b = int(scores.size(0))
        if b == 0:
            continue
        ek = _effective_k(scores, k)
        ek_ref = ek
        total_n += b
        sum_ndcg += ndcg_at_k(scores, targets, k) * b
        sum_hr += hit_rate_at_k(scores, targets, k) * b
        sum_mrr += mrr_at_k(scores, targets, k) * b
        scores_s = _sanitize_ranking_logits(scores)
        topk_idx = scores_s.topk(ek, dim=1).indices
        hits = (topk_idx == targets.view(-1, 1)).any(dim=1)
        sum_hit_users += float(hits.float().sum().item())

    if total_n == 0:
        return {"ndcg": 0.0, "hit_rate": 0.0, "recall": 0.0, "mrr": 0.0, "precision": 0.0}
    ek = ek_ref if ek_ref is not None else min(k, 1)
    return {
        "ndcg": sum_ndcg / total_n,
        "hit_rate": sum_hr / total_n,
        "recall": sum_hr / total_n,
        "mrr": sum_mrr / total_n,
        "precision": sum_hit_users / (total_n * ek),
    }
