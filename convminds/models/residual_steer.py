from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from convminds.models.base import BrainLanguageModel
from convminds.models.brain_steer import BrainSteerAdapter

class ResidualSteerLM(BrainLanguageModel):
    """
    Residual Steering LM following the updated architecture specification.
    Optimized for VRAM efficiency and autograd stability.
    """
    def __init__(
        self, 
        llm_id: str = "gpt2", 
        brain_dim: int = 1000, 
        injection_layer: int = 6, 
        n_frames: int = 4
    ):
        super().__init__()
        
        self.llm = AutoModelForCausalLM.from_pretrained(llm_id)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.llm_dim = self.llm.config.hidden_size
        self.injection_layer = injection_layer
        
        self.adapter = BrainSteerAdapter(
            brain_dim=brain_dim, 
            llm_dim=self.llm_dim, 
            n_frames=n_frames
        )
        
        self.freeze_base_model()

    def _get_pre_injection_state(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Runs the LLM up to the injection layer. 
        Used by both Phase 1 and Phase 2 to avoid redundant code.
        """
        device = input_ids.device
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        transformer = self.llm.transformer
        
        extended_attention_mask = transformer.get_extended_attention_mask(
            attention_mask, input_ids.size(), device
        )
        
        # Safer position IDs that handle batching/padding natively
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        
        hidden_states = transformer.wte(input_ids) + transformer.wpe(position_ids)
        hidden_states = transformer.drop(hidden_states)
        
        for i in range(self.injection_layer):
            layer_outputs = transformer.h[i](
                hidden_states, 
                attention_mask=extended_attention_mask
            )
            hidden_states = layer_outputs[0]
            
        return hidden_states, extended_attention_mask

    def get_h_at_layer(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Phase 1 Extraction: Optimized to stop computing at injection_layer.
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            
        # no_grad because Phase 1 target extraction doesn't train the LLM
        with torch.no_grad():
            hidden_states, _ = self._get_pre_injection_state(input_ids)
            
        return hidden_states

    def forward(self, brain_batch: torch.Tensor, input_ids: torch.Tensor, **kwargs):
        """Standard forward pass with steering injection."""
        return self.forward_steered(input_ids, brain_batch)

    def forward_steered(self, input_ids: torch.Tensor, brain_batch: torch.Tensor):
        """Phase 2: Main Training (Cross-Entropy & Injection)."""
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            
        # 1. First Half (Wrapped in no_grad to save 50% activation VRAM!)
        with torch.no_grad():
            hidden_states, extended_attention_mask = self._get_pre_injection_state(input_ids)
            
        # 2. Intercept & Adapter Forward
        H_query = hidden_states[:, -1:, :]
        v_steer = self.adapter(brain_batch, H_query)
        
        # 3. Autograd-Safe Injection (Split, Mutate, Concat)
        front_context = hidden_states[:, :-1, :]
        last_token_steered = hidden_states[:, -1:, :] + v_steer
        
        # Because v_steer requires_grad, steered_hidden_states will now automatically require_grad
        steered_hidden_states = torch.cat([front_context, last_token_steered], dim=1)
        
        # 4. Second Half of the LLM
        transformer = self.llm.transformer
        for i in range(self.injection_layer, len(transformer.h)):
            layer_outputs = transformer.h[i](
                steered_hidden_states, 
                attention_mask=extended_attention_mask
            )
            steered_hidden_states = layer_outputs[0]
            
        steered_hidden_states = transformer.ln_f(steered_hidden_states)
        logits = self.llm.lm_head(steered_hidden_states)
        
        return logits, v_steer