from __future__ import annotations

import torch
import torch.nn as nn


class PrefixFusion(nn.Module):
    """Prepends brain latents to text embeddings."""

    def forward(
        self,
        brain_latents: torch.Tensor,
        text_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ):
        fused = torch.cat([brain_latents, text_embeds], dim=1)
        if attention_mask is None:
            return fused
        prefix_mask = torch.ones(
            attention_mask.size(0),
            brain_latents.size(1),
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        fused_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        return fused, fused_mask


class PrefixPromptFusion(nn.Module):
    def __init__(self, num_prefix_tokens: int, embed_dim: int, brain_dim: int | None = None):
        super().__init__()
        self.num_prefix_tokens = num_prefix_tokens
        self.embed_dim = embed_dim
        self.brain_dim = brain_dim or embed_dim
        self.cond_proj = nn.Linear(self.brain_dim, embed_dim)
        self.prefix_offsets = nn.Parameter(torch.zeros(num_prefix_tokens, embed_dim))

    def forward(
        self,
        brain_latents: torch.Tensor,
        text_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if brain_latents.dim() == 3:
            pooled = brain_latents.mean(dim=1)
        else:
            pooled = brain_latents

        prefix_base = self.cond_proj(pooled).unsqueeze(1)
        prefix = prefix_base + self.prefix_offsets.unsqueeze(0)
        fused = torch.cat([prefix, text_embeds], dim=1)

        if attention_mask is None:
            return fused, None

        prefix_mask = torch.ones(
            attention_mask.size(0),
            self.num_prefix_tokens,
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        fused_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        return fused, fused_mask
