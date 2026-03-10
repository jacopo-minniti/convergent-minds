from __future__ import annotations

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, hidden_states: torch.Tensor, brain_context: torch.Tensor) -> torch.Tensor:
        queries = self.query_proj(hidden_states)
        keys = self.key_proj(brain_context)
        values = self.value_proj(brain_context)
        output, _ = self.attn(queries, keys, values, need_weights=False)
        return output
