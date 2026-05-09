from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, maxlen: int, hidden_units: int) -> None:
        super().__init__()
        self.maxlen = maxlen
        self.hidden_units = hidden_units
        
        position = torch.arange(maxlen).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_units, 2) * (-torch.log(torch.tensor(10000.0)) / hidden_units))
        
        pe = torch.zeros(maxlen, hidden_units)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_units: int, num_heads: int, dropout_rate: float) -> None:
        super().__init__()
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        
        self.q_proj = nn.Linear(hidden_units, hidden_units)
        self.k_proj = nn.Linear(hidden_units, hidden_units)
        self.v_proj = nn.Linear(hidden_units, hidden_units)
        self.out_proj = nn.Linear(hidden_units, hidden_units)
        
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, hidden_units = x.size()
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_units)
        output = self.out_proj(output)
        
        return output


class PointWiseFFN(nn.Module):
    def __init__(self, hidden_units: int, dropout_rate: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_units, hidden_units * 4)
        self.fc2 = nn.Linear(hidden_units * 4, hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.gelu = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, hidden_units: int, num_heads: int, dropout_rate: float) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(hidden_units, num_heads, dropout_rate)
        self.ffn = PointWiseFFN(hidden_units, dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_units)
        self.norm2 = nn.LayerNorm(hidden_units)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.attn(x, mask)
        x = self.dropout1(x)
        x = residual + x
        
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout2(x)
        x = residual + x
        
        return x


class SASRec(nn.Module):
    def __init__(
        self,
        num_items: int,
        maxlen: int = 50,
        hidden_units: int = 64,
        num_blocks: int = 2,
        num_heads: int = 2,
        dropout_rate: float = 0.2,
        padding_id: int = 0,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.maxlen = maxlen
        self.hidden_units = hidden_units
        self.padding_id = padding_id
        
        self.item_emb = nn.Embedding(num_items + 1, hidden_units, padding_idx=padding_id)
        self.pos_emb = PositionalEncoding(maxlen, hidden_units)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_units, num_heads, dropout_rate)
            for _ in range(num_blocks)
        ])
        
        self.norm = nn.LayerNorm(hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = (input_ids != self.padding_id).unsqueeze(1).repeat(1, input_ids.size(1), 1)
        mask = torch.tril(mask)
        
        x = self.item_emb(input_ids)
        x = self.pos_emb(x)
        x = self.dropout(x)
        
        for block in self.blocks:
            x = block(x, mask)
        
        x = self.norm(x)
        return x
    
    def predict(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.forward(input_ids)
        last_hidden = x[:, -1, :]
        logits = torch.matmul(last_hidden, self.item_emb.weight.T)
        return logits
