from __future__ import annotations

import torch

from convminds.models.base import BrainLanguageModel
from convminds.nn.encoders import SpatialAttentionEncoder
from convminds.nn.fusion import CrossAttentionFusion
from convminds.nn.wrappers import ResidualInjector


class DeepSteeredLM(BrainLanguageModel):
    def __init__(self, llm_id: str, encoder_out_dim: int, injection_layer: int, num_queries: int = 128):
        super().__init__()
        self.encoder = SpatialAttentionEncoder(num_queries=num_queries, query_dim=encoder_out_dim, use_coords=True)
        self.cross_attn = CrossAttentionFusion(embed_dim=encoder_out_dim)

        from transformers import AutoModelForCausalLM

        self.llm = AutoModelForCausalLM.from_pretrained(llm_id)

        layers = self._resolve_layers(self.llm)
        if injection_layer < 0 or injection_layer >= len(layers):
            raise ValueError("injection_layer index is out of range for the selected model.")

        original_layer = layers[injection_layer]
        layers[injection_layer] = ResidualInjector(
            base_layer=original_layer,
            intervention_module=self.cross_attn,
            kwarg_name="brain_context",
        )

        self.freeze_base_model()

    def forward(self, brain_tensor, text_input_ids, attention_mask=None, **kwargs):
        brain_latents = self.encoder(brain_tensor)
        return self.llm(
            input_ids=text_input_ids,
            attention_mask=attention_mask,
            brain_context=brain_latents,
            **kwargs,
        )

    def _resolve_layers(self, llm):
        if hasattr(llm, "model") and hasattr(llm.model, "layers"):
            return llm.model.layers
        if hasattr(llm, "model") and hasattr(llm.model, "decoder") and hasattr(llm.model.decoder, "layers"):
            return llm.model.decoder.layers
        if hasattr(llm, "transformer") and hasattr(llm.transformer, "h"):
            return llm.transformer.h
        if hasattr(llm, "gpt_neox") and hasattr(llm.gpt_neox, "layers"):
            return llm.gpt_neox.layers
        raise ValueError("Unable to locate transformer layers for injection.")
