from __future__ import annotations

import torch

from convminds.models.base import BrainLanguageModel
from convminds.nn.encoders import TemporalEncoder
from convminds.nn.fusion import CrossAttentionFusion
from convminds.nn.wrappers import SteerInjector


class ResidualSteerLM(BrainLanguageModel):
    """
    Residual Steering LM.
    Combines Phase 1 (TemporalEncoder), Phase 2 (CrossAttentionFusion),
    and Phase 3 (SteerInjector into residual stream).
    """
    def __init__(
        self, 
        llm_id: str, 
        encoder_in_dim: int = 1000, 
        injection_layer: int = 16, 
        num_frames: int = 4
    ):
        super().__init__()
        
        from transformers import AutoModelForCausalLM
        self.llm = AutoModelForCausalLM.from_pretrained(llm_id)
        
        embed_dim = self.llm.config.hidden_size
        
        # Phase 1: Encoder
        self.encoder = TemporalEncoder(input_dim=encoder_in_dim, embed_dim=embed_dim, num_frames=num_frames)
        
        # Phase 2: Cross Attention (The NeuroModule)
        self.cross_attn = CrossAttentionFusion(embed_dim=embed_dim)
        self.injection_layer = injection_layer
        
        # Phase 3: SteerInjector with Norm Penalty Tracking
        layers = self._resolve_layers(self.llm)
        if injection_layer < 0 or injection_layer >= len(layers):
            raise ValueError(f"injection_layer {injection_layer} is out of range.")
            
        original_layer = layers[injection_layer]
        layers[injection_layer] = SteerInjector(
            base_layer=original_layer,
            intervention_module=self.cross_attn,
            kwarg_name="brain_context",
        )
        
        self.injector = layers[injection_layer]
        self.freeze_base_model()

    def forward(self, brain_tensor, text_input_ids, attention_mask=None, **kwargs):
        # Phase 1 projection to latents
        brain_latents = self.encoder(brain_tensor)
        
        # Base LLM forward pass, routing latents to the targeted injection layer (Phase 2 & 3)
        return self.llm(
            input_ids=text_input_ids,
            attention_mask=attention_mask,
            brain_context=brain_latents,
            **kwargs,
        )
        
    def get_penalty(self) -> torch.Tensor | None:
        """Retrieve the latest norm penalty computed by the SteerInjector."""
        return self.injector.last_penalty

    def _resolve_layers(self, llm):
        if hasattr(llm, "model") and hasattr(llm.model, "layers"):
            return llm.model.layers
        if hasattr(llm, "base_model") and hasattr(llm.base_model, "model") and hasattr(llm.base_model.model, "layers"):
            return llm.base_model.model.layers
        if hasattr(llm, "model") and hasattr(llm.model, "decoder") and hasattr(llm.model.decoder, "layers"):
            return llm.model.decoder.layers
        if hasattr(llm, "transformer") and hasattr(llm.transformer, "h"):
            return llm.transformer.h
        if hasattr(llm, "gpt_neox") and hasattr(llm.gpt_neox, "layers"):
            return llm.gpt_neox.layers
        raise ValueError("Unable to locate transformer layers for injection.")
