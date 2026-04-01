from __future__ import annotations

import torch
import torch.nn as nn
from convminds.models.base import BrainLanguageModel
from convminds.models.brain_steer import BrainSteerAdapter

class ResidualSteerLM(BrainLanguageModel):
    """
    Residual Steering LM following the updated architecture specification.
    Supports extraction of hidden states and direct injection of steering vectors.
    """
    def __init__(
        self, 
        llm_id: str = "gpt2", 
        brain_dim: int = 1000, 
        injection_layer: int = 6, 
        n_frames: int = 4
    ):
        super().__init__()
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.llm = AutoModelForCausalLM.from_pretrained(llm_id)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.llm_dim = self.llm.config.hidden_size
        self.injection_layer = injection_layer
        
        # New adapter from specification
        self.adapter = BrainSteerAdapter(
            brain_dim=brain_dim, 
            llm_dim=self.llm_dim, 
            n_frames=n_frames
        )
        
        self.freeze_base_model()

    def get_h_at_layer(self, input_ids: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Extract hidden states at a specific layer for a given input.
        Used to calculate H_query and H_target.
        """
        # Use the built-in HF mechanism to extract hidden states.
        # This is more robust than manual layer iteration.
        outputs = self.llm.transformer(input_ids, output_hidden_states=True)
        # index 0: embeddings, index 1: after layer 0, ..., index 6: after layer 5.
        return outputs.hidden_states[layer_idx]

    def forward(self, brain_batch: torch.Tensor, input_ids: torch.Tensor, **kwargs):
        """Standard forward pass with steering injection."""
        return self.forward_steered(input_ids, brain_batch)

    def forward_steered(self, input_ids: torch.Tensor, brain_batch: torch.Tensor):
        """
        Implementation of Phase 2: Main Training (Cross-Entropy & Injection).
        Follows Steps A-D of the specification.
        """
        # Step A: The First Half of the LLM (up to injection_layer)
        # Using output_hidden_states ensures internal logic (positions, masks) is handled.
        outputs = self.llm.transformer(input_ids, output_hidden_states=True)
        H_L6 = outputs.hidden_states[self.injection_layer]
        
        # Step B: The Intercept & Adapter Forward
        # Grab state of the last token
        H_query = H_L6[:, -1:, :] 
        v_steer = self.adapter(brain_batch, H_query)
        
        # Step C: The Injection
        # Add steering vector to the final token
        H_L6_steered = H_L6.clone()
        H_L6_steered[:, -1:, :] = H_L6_steered[:, -1:, :] + v_steer 
        
        # Step D: The Second Half of the LLM
        hidden_states = H_L6_steered
        transformer = self.llm.transformer
        for i in range(self.injection_layer, len(transformer.h)):
            # Layers from injection_layer onwards
            hidden_states = transformer.h[i](hidden_states)[0]
            
        hidden_states = transformer.ln_f(hidden_states)
        logits = self.llm.lm_head(hidden_states)
        
        return logits, v_steer

    def _resolve_layers(self, llm):
        if hasattr(llm, "transformer") and hasattr(llm.transformer, "h"):
            return llm.transformer.h
        raise ValueError("Only GPT-2 style models are currently supported by this implementation.")
