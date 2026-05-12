from __future__ import annotations

import torch
import torch.nn as nn
class GRUSeqRec(nn.Module):
    """GRU-based sequential recommender (GRU4Rec-style), tied output item embeddings."""

    def __init__(
        self,
        num_items: int,
        maxlen: int,
        hidden_units: int,
        gru_num_layers: int = 1,
        dropout_rate: float = 0.2,
        padding_id: int = 0,
    ) -> None:
        super().__init__()
        if num_items <= 0 or maxlen <= 0 or hidden_units <= 0:
            raise ValueError("num_items, maxlen, hidden_units must be positive.")
        self.num_items = int(num_items)
        self.maxlen = int(maxlen)
        self.hidden_units = int(hidden_units)
        self.gru_num_layers = int(gru_num_layers)
        self.padding_id = int(padding_id)
        self.vocab_size = self.num_items + 1
        self.item_embedding = nn.Embedding(self.vocab_size, self.hidden_units, padding_idx=self.padding_id)
        self.gru = nn.GRU(
            input_size=self.hidden_units,
            hidden_size=self.hidden_units,
            num_layers=self.gru_num_layers,
            batch_first=True,
            dropout=dropout_rate if self.gru_num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(float(dropout_rate))
        self.out_norm = nn.LayerNorm(self.hidden_units)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)
        if self.padding_id is not None:
            with torch.no_grad():
                self.item_embedding.weight[self.padding_id].zero_()
        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    @staticmethod
    def sequence_lengths(input_ids: torch.Tensor, padding_id: int) -> torch.Tensor:
        mask = input_ids != padding_id
        return mask.sum(dim=1).clamp(min=1)

    def encode(self, input_ids: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if lengths is None:
            lengths = self.sequence_lengths(input_ids, self.padding_id)
        emb = self.dropout(self.item_embedding(input_ids))
        packed = nn.utils.rnn.pack_padded_sequence(
            emb,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, h_n = self.gru(packed)
        return self.out_norm(h_n[-1])

    def logits_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.matmul(hidden, self.item_embedding.weight.t())

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.encode(input_ids, lengths)
        logits = self.logits_from_hidden(hidden)
        return logits

    def predict(self, input_ids: torch.Tensor) -> torch.Tensor:
        lengths = self.sequence_lengths(input_ids, self.padding_id)
        logits = self.forward(input_ids, lengths)
        logits = logits.clone()
        logits[:, self.padding_id] = -1e9
        return logits


class SASRec(GRUSeqRec):
    """Backward-compatible name: implementation is GRU-based (non-transformer)."""

    def __init__(
        self,
        num_items: int,
        maxlen: int,
        hidden_units: int,
        num_blocks: int = 2,
        num_heads: int = 2,
        dropout_rate: float = 0.2,
        padding_id: int = 0,
        gru_num_layers: int | None = None,
    ) -> None:
        layers = int(gru_num_layers) if gru_num_layers is not None else max(1, int(num_blocks))
        super().__init__(
            num_items=num_items,
            maxlen=maxlen,
            hidden_units=hidden_units,
            gru_num_layers=layers,
            dropout_rate=dropout_rate,
            padding_id=padding_id,
        )
