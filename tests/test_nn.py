from __future__ import annotations

import torch
import torch.nn as nn

from convminds.data.primitives import BrainTensor
from convminds.nn.encoders import SpatialAttentionEncoder, TemporalEncoder
from convminds.nn.fusion import PrefixFusion, CrossAttentionFusion
from convminds.nn.wrappers import ResidualInjector, SteerInjector


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


def test_temporal_encoder_shapes():
    # (batch, num_frames, input_dim)
    brain_tensor = torch.randn(2, 4, 10)
    encoder = TemporalEncoder(input_dim=10, embed_dim=16, num_frames=4)
    out = encoder(brain_tensor)
    assert out.shape == (2, 4, 16)


def test_cross_attention_fusion_shapes():
    fusion = CrossAttentionFusion(embed_dim=16)
    hidden = torch.randn(2, 5, 16)
    brain_context = torch.randn(2, 4, 16)
    out = fusion(hidden, brain_context)
    assert out.shape == (2, 5, 16)


def test_steer_injector_penalty():
    class DummyLayer(nn.Module):
        def forward(self, x, **kwargs):
            return x

    class DummyMove(nn.Module):
        def forward(self, x, context):
            return torch.ones_like(x) * 2.0

    injector = SteerInjector(DummyLayer(), DummyMove())
    hidden = torch.zeros(1, 1, 4)
    brain = torch.zeros(1, 1, 4)
    
    out = injector(hidden, brain_context=brain)
    # Norm squared of [2, 2, 2, 2] is 4*2^2 = 16
    assert injector.last_penalty is not None
    assert torch.allclose(injector.last_penalty, torch.tensor([16.0]))
    assert torch.allclose(out, torch.ones_like(hidden) * 2.0)
