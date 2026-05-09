from __future__ import annotations

import torch


def hit_rate_at_k(predicted: torch.Tensor, target: torch.Tensor, k: int = 10) -> float:
    batch_size = target.size(0)
    actual_k = min(k, predicted.size(1))
    _, topk_indices = torch.topk(predicted, actual_k, dim=1)
    
    hit = 0
    for i in range(batch_size):
        if target[i] in topk_indices[i]:
            hit += 1
    
    return hit / batch_size


def ndcg_at_k(predicted: torch.Tensor, target: torch.Tensor, k: int = 10) -> float:
    batch_size = target.size(0)
    actual_k = min(k, predicted.size(1))
    _, topk_indices = torch.topk(predicted, actual_k, dim=1)
    
    ndcg_sum = 0.0
    for i in range(batch_size):
        rank = (topk_indices[i] == target[i]).nonzero(as_tuple=True)[0]
        if len(rank) > 0:
            rank_pos = rank[0].item() + 1
            ndcg_sum += 1.0 / torch.log2(torch.tensor(rank_pos + 1, dtype=torch.float32)).item()
    
    return ndcg_sum / batch_size


def recall_at_k(predicted: torch.Tensor, target: torch.Tensor, k: int = 10) -> float:
    batch_size = target.size(0)
    actual_k = min(k, predicted.size(1))
    _, topk_indices = torch.topk(predicted, actual_k, dim=1)
    
    recall = 0
    for i in range(batch_size):
        if target[i] in topk_indices[i]:
            recall += 1
    
    return recall / batch_size


def mrr_at_k(predicted: torch.Tensor, target: torch.Tensor, k: int = 10) -> float:
    batch_size = target.size(0)
    actual_k = min(k, predicted.size(1))
    _, topk_indices = torch.topk(predicted, actual_k, dim=1)
    
    mrr_sum = 0.0
    for i in range(batch_size):
        rank = (topk_indices[i] == target[i]).nonzero(as_tuple=True)[0]
        if len(rank) > 0:
            rank_pos = rank[0].item() + 1
            mrr_sum += 1.0 / rank_pos
    
    return mrr_sum / batch_size


def precision_at_k(predicted: torch.Tensor, target: torch.Tensor, k: int = 10) -> float:
    batch_size = target.size(0)
    actual_k = min(k, predicted.size(1))
    _, topk_indices = torch.topk(predicted, actual_k, dim=1)
    
    precision_sum = 0.0
    for i in range(batch_size):
        if target[i] in topk_indices[i]:
            precision_sum += 1.0 / actual_k
    
    return precision_sum / batch_size


def evaluate(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k: int = 10,
) -> dict[str, float]:
    return {
        'hit_rate': hit_rate_at_k(logits, targets, k),
        'ndcg': ndcg_at_k(logits, targets, k),
        'recall': recall_at_k(logits, targets, k),
        'mrr': mrr_at_k(logits, targets, k),
        'precision': precision_at_k(logits, targets, k),
    }
