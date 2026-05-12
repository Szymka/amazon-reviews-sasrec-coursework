from __future__ import annotations

import torch


def _effective_k(scores: torch.Tensor, k: int) -> int:
    """Top-k cannot exceed the number of scored classes."""
    return int(min(k, scores.shape[1]))


def _ranks_of_targets(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """1-based rank by descending score; ties broken by strict `>` count."""
    target_scores = scores.gather(1, targets.view(-1, 1)).squeeze(1)
    return (scores > target_scores.unsqueeze(1)).sum(dim=1) + 1


def ndcg_at_k(scores: torch.Tensor, targets: torch.Tensor, k: int = 10) -> float:
    scores = scores.detach().cpu()
    targets = targets.detach().cpu().long()
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
    targets = targets.detach().cpu().long()
    ek = _effective_k(scores, k)
    topk = scores.topk(ek, dim=1).indices
    hits = (topk == targets.view(-1, 1)).any(dim=1)
    return float(hits.float().mean().item())


def mrr_at_k(scores: torch.Tensor, targets: torch.Tensor, k: int = 10) -> float:
    scores = scores.detach().cpu()
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
