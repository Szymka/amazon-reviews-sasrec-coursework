"""
Standalone sequential backbone used in the LLMRank pipeline (candidate generator).

Upstream LLMRank wraps RecBole ``SASRec`` and adds ``predict_on_subsets`` in
``LLMRank/llmrank/model/sasrec.py``. This file provides a **PyTorch-only** implementation
that reads Amazon coursework TSV tensors via ``models.llmrank.dataset`` — no RecBole.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def sequence_lengths(input_ids: torch.Tensor, padding_id: int) -> torch.Tensor:
    """Non-padding token counts per row (minimum 1)."""
    mask = input_ids != padding_id
    return mask.sum(dim=1).clamp(min=1)


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units: int, dropout_rate: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.dropout2(
            self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2)))))
        )
        outputs = outputs.transpose(-1, -2)
        return outputs + inputs


class StableMHA(nn.Module):
    """
    Multi-head causal attention. Uses finite sentinel masks (~ -1e4) instead of -inf
    to avoid softmax NaNs on CUDA.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout_rate: float) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.zeros_(self.o_proj.bias)

    def forward(
        self,
        query_src: torch.Tensor,
        kv_src: torch.Tensor,
        attn_bool_ll: torch.Tensor,
        key_padding_bl: torch.Tensor,
    ) -> torch.Tensor:
        l_len, batch, _e = query_src.shape
        h, dh = self.num_heads, self.head_dim
        neg = query_src.new_tensor(-1e4)

        q = self.q_proj(query_src)
        k = self.k_proj(kv_src)
        v = self.v_proj(kv_src)

        def to_heads(t: torch.Tensor) -> torch.Tensor:
            x = t.permute(1, 0, 2).contiguous().view(batch, l_len, h, dh)
            return x.transpose(1, 2)

        qh, kh, vh = to_heads(q), to_heads(k), to_heads(v)
        attn_logits = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(dh)

        attn_logits = attn_logits.masked_fill(attn_bool_ll.view(1, 1, l_len, l_len), neg)
        attn_logits = attn_logits.masked_fill(key_padding_bl[:, None, None, :], neg)

        probs = attn_logits.softmax(dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0)
        probs = self.dropout(probs)
        ctx = torch.matmul(probs, vh)
        ctx = ctx.transpose(1, 2).contiguous().view(batch, l_len, self.embed_dim)
        ctx = ctx.transpose(0, 1).contiguous()
        return self.dropout(self.o_proj(ctx))


class LLMRankSequentialModel(nn.Module):
    """
    Causal Transformer over item ids for coursework batches:
    ``input_ids`` (B, maxlen) left-padded with ``padding_id``, ``target_id`` for loss.
    """

    def __init__(
        self,
        num_items: int,
        maxlen: int,
        hidden_units: int,
        num_blocks: int = 2,
        num_heads: int = 2,
        dropout_rate: float = 0.2,
        padding_id: int = 0,
    ) -> None:
        super().__init__()
        if num_items <= 0 or maxlen <= 0 or hidden_units <= 0:
            raise ValueError("num_items, maxlen, hidden_units must be positive.")
        if hidden_units % num_heads != 0:
            raise ValueError("hidden_units must be divisible by num_heads.")
        self.num_items = int(num_items)
        self.maxlen = int(maxlen)
        self.hidden_units = int(hidden_units)
        self.num_blocks = int(num_blocks)
        self.num_heads = int(num_heads)
        self.dropout_rate = float(dropout_rate)
        self.padding_id = int(padding_id)
        self.vocab_size = self.num_items + 1

        self.item_embedding = nn.Embedding(self.vocab_size, self.hidden_units, padding_idx=self.padding_id)
        self.position_embedding = nn.Embedding(self.maxlen, self.hidden_units)
        self.dropout = nn.Dropout(p=self.dropout_rate)

        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(self.hidden_units, eps=1e-5)

        for _ in range(self.num_blocks):
            self.attention_layernorms.append(nn.LayerNorm(self.hidden_units, eps=1e-5))
            self.attention_layers.append(
                StableMHA(self.hidden_units, self.num_heads, self.dropout_rate),
            )
            self.forward_layernorms.append(nn.LayerNorm(self.hidden_units, eps=1e-5))
            self.forward_layers.append(PointWiseFeedForward(self.hidden_units, self.dropout_rate))

        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        if self.padding_id is not None:
            with torch.no_grad():
                self.item_embedding.weight[self.padding_id].zero_()

    def _log2feats(self, seq: torch.Tensor) -> torch.Tensor:
        timeline_mask = seq.eq(self.padding_id)
        timeline_mask_eff = timeline_mask
        empty_seq = timeline_mask.all(dim=1)
        if empty_seq.any():
            timeline_mask_eff = timeline_mask.clone()
            timeline_mask_eff[empty_seq, -1] = False

        embs = self.item_embedding(seq) * math.sqrt(self.hidden_units)
        positions = torch.arange(seq.size(1), device=seq.device, dtype=torch.long).unsqueeze(0).expand_as(seq)
        positions = positions.masked_fill(timeline_mask_eff, 0)
        seqs = self.dropout(embs + self.position_embedding(positions))
        seqs = seqs.masked_fill(timeline_mask_eff.unsqueeze(-1), 0.0)

        tl = seqs.size(1)
        attn_mask_bool = torch.triu(torch.ones(tl, tl, device=seq.device, dtype=torch.bool), diagonal=1)

        for i in range(self.num_blocks):
            x = seqs.transpose(0, 1)
            normed = self.attention_layernorms[i](x)
            attn_out = self.attention_layers[i](normed, x, attn_mask_bool, timeline_mask_eff)
            seqs = (x + attn_out).transpose(0, 1)
            seqs = seqs.masked_fill(timeline_mask_eff.unsqueeze(-1), 0.0)

            normed = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](normed)
            seqs = seqs.masked_fill(timeline_mask_eff.unsqueeze(-1), 0.0)

        return self.last_layernorm(seqs)

    def encode(self, input_ids: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        _ = lengths
        feats = self._log2feats(input_ids)
        return feats[:, -1, :]

    def logits_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.matmul(hidden, self.item_embedding.weight.t())

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.encode(input_ids, lengths)
        return self.logits_from_hidden(hidden)

    def predict(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits = self.forward(input_ids)
        logits = logits.clone()
        logits[:, self.padding_id] = -1e9
        logits = torch.nan_to_num(logits, nan=-1e9)
        return logits

    @torch.no_grad()
    def predict_on_subsets(self, input_ids: torch.Tensor, idxs: torch.Tensor) -> torch.Tensor:
        """
        Score only candidate item ids (columns of ``idxs``), matching upstream
        ``LLMRank/llmrank/model/sasrec.py`` behavior for subset ranking.

        Args:
            input_ids: (B, L) padded item history.
            idxs: (B, C) int64 candidate internal item ids in ``1 .. num_items``.

        Returns:
            (B, vocab_size) tensor with ``-1e4`` on non-candidate positions (padding column masked).
        """
        device = input_ids.device
        idxs_t = idxs.to(device=device, dtype=torch.long)
        candidate_item_emb = self.item_embedding(idxs_t)
        seq_output = self.encode(input_ids)
        candidate_scores = (seq_output.unsqueeze(1) * candidate_item_emb).sum(dim=-1)
        scores = torch.full(
            (input_ids.shape[0], self.vocab_size),
            -1e4,
            device=device,
            dtype=candidate_scores.dtype,
        )
        scores.scatter_(1, idxs_t, candidate_scores)
        scores[:, self.padding_id] = -1e4
        return scores


def build_llmrank_model(
    num_items: int,
    maxlen: int,
    hidden_units: int,
    dropout_rate: float,
    padding_id: int,
    num_blocks: int = 2,
    num_heads: int = 2,
) -> nn.Module:
    return LLMRankSequentialModel(
        num_items=num_items,
        maxlen=maxlen,
        hidden_units=hidden_units,
        num_blocks=num_blocks,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        padding_id=padding_id,
    )


__all__ = [
    "LLMRankSequentialModel",
    "PointWiseFeedForward",
    "StableMHA",
    "build_llmrank_model",
    "sequence_lengths",
]
