from __future__ import annotations

import torch
import torch.nn as nn

from convminds.data.primitives import BrainTensor


class SpatialAttentionEncoder(nn.Module):
    """Encodes fMRI signals using spatial coordinate cross-attention."""

    def __init__(self, num_queries: int = 128, query_dim: int = 4096, use_coords: bool = True):
        super().__init__()
        self.query_dim = query_dim
        # Use smaller initialization for queries to match regression target scales
        self.queries = nn.Parameter(torch.randn(1, num_queries, query_dim) * 0.01)
        self.coord_proj = nn.Linear(3, query_dim) if use_coords else None
        self.val_proj = nn.Linear(1, query_dim)
        self.attention = nn.MultiheadAttention(embed_dim=query_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(query_dim)

    def forward(self, brain_tensor: BrainTensor) -> torch.Tensor:
        signal = brain_tensor.signal
        if signal.dim() != 3:
            raise ValueError("BrainTensor.signal must be shaped (B, T, N_voxels).")

        batch_size, _, num_voxels = signal.shape
        queries = self.queries.expand(batch_size, -1, -1)

        coords = brain_tensor.coords
        if coords.dim() == 2:
            coords = coords.unsqueeze(0).expand(batch_size, -1, -1)
        
        if self.coord_proj is None:
            keys = torch.zeros(batch_size, num_voxels, self.query_dim, device=signal.device, dtype=signal.dtype)
            if torch.any(coords != 0):
                import logging
                logging.getLogger(__name__).warning("Queries enabled but coord_proj is None! Coordinates are being ignored.")
        else:
            keys = self.coord_proj(coords)

        # Baseline: Average across time (Beta maps have T=1 anyway)
        signal_mean = signal.mean(dim=1).unsqueeze(-1)
        values = self.val_proj(signal_mean)
        
        # Logging to confirm data flow (once per model instance or periodically)
        if not hasattr(self, "_logged_shapes"):
            import logging
            log = logging.getLogger(__name__)
            log.info(f"SpatialAttention Forward: Q={queries.shape}, K={keys.shape}, V={values.shape}")
            self._logged_shapes = True
            
        key_padding_mask = brain_tensor.padding_mask
        if key_padding_mask is not None and key_padding_mask.dim() == 1:
            key_padding_mask = key_padding_mask.unsqueeze(0)
            
        latents, _ = self.attention(query=queries, key=keys, value=values, key_padding_mask=key_padding_mask)
        return self.norm(latents)


