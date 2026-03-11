from __future__ import annotations

import types

import torch
import torch.nn as nn

import convminds as cm
from convminds.data.primitives import BrainTensor
from convminds.nn.encoders import SpatialAttentionEncoder
from convminds.nn.fusion import PrefixFusion


class DummyBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, hidden_states, **kwargs):
        return self.proj(hidden_states)


class DummyLM(nn.Module):
    def __init__(self, *, vocab: int = 11, dim: int = 8, layers: int = 2):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.model = types.SimpleNamespace(
            layers=nn.ModuleList([DummyBlock(dim) for _ in range(layers)])
        )

    def get_input_embeddings(self):
        return self.emb

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, brain_context=None, **kwargs):
        if inputs_embeds is None:
            hidden = self.emb(input_ids)
        else:
            hidden = inputs_embeds
        for layer in self.model.layers:
            hidden = layer(hidden, brain_context=brain_context)
        logits = torch.zeros(hidden.size(0), hidden.size(1), self.emb.num_embeddings, device=hidden.device)
        return types.SimpleNamespace(logits=logits)

    def generate(self, inputs_embeds=None, attention_mask=None, **kwargs):
        batch = inputs_embeds.size(0)
        return torch.zeros(batch, 1, dtype=torch.long)


def _patch_transformers(monkeypatch):
    import transformers

    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: DummyLM(),
    )


def test_prompt_conditioned_lm_forward(monkeypatch):
    _patch_transformers(monkeypatch)
    encoder = SpatialAttentionEncoder(num_queries=2, query_dim=8, use_coords=True)
    fusion = PrefixFusion()
    model = cm.models.PromptConditionedLM(llm_id="gpt2", encoder=encoder, fusion=fusion)

    brain = BrainTensor(signal=torch.randn(2, 1, 4), coords=torch.zeros(4, 3))
    input_ids = torch.randint(0, 10, (2, 3))
    output = model(brain, text_input_ids=input_ids)
    assert output.logits.shape == (2, 5, 11)


def test_deep_steered_lm_forward(monkeypatch):
    _patch_transformers(monkeypatch)
    model = cm.models.DeepSteeredLM(llm_id="gpt2", encoder_out_dim=8, injection_layer=0, num_queries=2)

    brain = BrainTensor(signal=torch.randn(2, 1, 4), coords=torch.zeros(4, 3))
    input_ids = torch.randint(0, 10, (2, 3))
    output = model(brain, text_input_ids=input_ids)
    assert output.logits.shape == (2, 3, 11)
