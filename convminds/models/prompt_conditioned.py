from __future__ import annotations

import torch

from convminds.models.base import BrainLanguageModel


class PromptConditionedLM(BrainLanguageModel):
    """
    A generic architecture that encodes brain data and prepends it to an LLM.
    """

    def __init__(self, llm_id: str, encoder, fusion):
        super().__init__()
        self.encoder = encoder
        self.fusion = fusion
        from transformers import AutoModelForCausalLM

        self.llm = AutoModelForCausalLM.from_pretrained(llm_id)
        self.freeze_base_model()

    def forward(
        self,
        brain_tensor,
        text_input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        brain_latents = self.encoder(brain_tensor)
        text_embeds = self.llm.get_input_embeddings()(text_input_ids)
        fused_embeddings, fused_mask = self._apply_fusion(brain_latents, text_embeds, attention_mask)
        return self.llm(inputs_embeds=fused_embeddings, attention_mask=fused_mask, **kwargs)

    def generate(self, brain_tensor, text_input_ids: torch.Tensor | None = None, attention_mask=None, **kwargs):
        brain_latents = self.encoder(brain_tensor)
        if text_input_ids is None:
            inputs_embeds = brain_latents
            fused_mask = attention_mask
        else:
            text_embeds = self.llm.get_input_embeddings()(text_input_ids)
            inputs_embeds, fused_mask = self._apply_fusion(brain_latents, text_embeds, attention_mask)
        return self.llm.generate(inputs_embeds=inputs_embeds, attention_mask=fused_mask, **kwargs)

    def _apply_fusion(self, brain_latents, text_embeds, attention_mask):
        if attention_mask is None:
            result = self.fusion(brain_latents, text_embeds)
        else:
            result = self.fusion(brain_latents, text_embeds, attention_mask)
        if isinstance(result, tuple):
            return result
        return result, attention_mask
