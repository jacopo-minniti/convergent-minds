from __future__ import annotations

import torch
import torch.nn as nn

from convminds.data.primitives import BrainTensor
from convminds.nn.encoders import SpatialAttentionEncoder
from convminds.nn.fusion import PrefixFusion
from convminds.nn.wrappers import ResidualInjector


def test_spatial_attention_shapes():
    brain = BrainTensor(
        signal=torch.randn(2, 3, 4),
        coords=torch.zeros(4, 3),
    )
    encoder = SpatialAttentionEncoder(num_queries=5, query_dim=8, use_coords=True)
    out = encoder(brain)
    assert out.shape == (2, 5, 8)


def test_prefix_fusion_mask():
    fusion = PrefixFusion()
    brain_latents = torch.randn(2, 3, 4)
    text_embeds = torch.randn(2, 5, 4)
    attention_mask = torch.ones(2, 5, dtype=torch.long)
    fused, fused_mask = fusion(brain_latents, text_embeds, attention_mask)
    assert fused.shape == (2, 8, 4)
    assert fused_mask.shape == (2, 8)


def test_residual_injector_forward():
    class DummyBase(nn.Module):
        def forward(self, hidden_states, **kwargs):
            return hidden_states + 1.0

    class DummyIntervention(nn.Module):
        def forward(self, hidden_states, brain_context):
            return torch.zeros_like(hidden_states)

    base = DummyBase()
    intervention = DummyIntervention()
    injector = ResidualInjector(base, intervention, kwarg_name="brain_context")

    hidden = torch.zeros(2, 3, 4)
    brain_context = torch.zeros(2, 2, 4)
    out = injector(hidden, brain_context=brain_context)
    assert torch.allclose(out, hidden + 1.0)
