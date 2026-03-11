from __future__ import annotations

import torch
import torch.nn as nn

class TemporalEncoder(nn.Module):
    """
    Phase 1: Brain Encoding and Pre-Alignment.
    Projects reduced BOLD features (e.g., from PCA) to the LLM's hidden dimension
    and adds learnable positional embeddings to preserve the chronological order of 
    hemodynamic frames.
    """
    def __init__(self, input_dim: int = 1000, embed_dim: int = 4096, num_frames: int = 4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.positional_embeddings = nn.Parameter(torch.randn(1, num_frames, embed_dim))

    def forward(self, brain_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            brain_tensor: Tensor of shape (batch, num_frames, input_dim)
        Returns:
            Tensor of shape (batch, num_frames, embed_dim)
        """
        # (batch, num_frames, embed_dim)
        hidden = self.mlp(brain_tensor)
        # Add temporal positional embeddings
        return hidden + self.positional_embeddings
