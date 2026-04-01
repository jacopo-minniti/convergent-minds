from __future__ import annotations

import torch
import torch.nn as nn
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from convminds.models.base import BrainLanguageModel

logger = logging.getLogger(__name__)

class BrainSteerAdapter(nn.Module):
    """
    Adapter that maps brain activity (4 TRs) to a steering vector 
    via cross-attention with the LLM's current hidden state.
    """
    def __init__(self, brain_dim=1000, llm_dim=768, num_heads=12, n_frames=4):
        super().__init__()
        self.brain_dim = brain_dim
        self.llm_dim = llm_dim
        self.n_frames = n_frames
        
        self.pos_embed = nn.Parameter(torch.randn(1, n_frames, brain_dim) * 0.02)
        
        self.W_K = nn.Linear(brain_dim, llm_dim)
        self.W_V = nn.Linear(brain_dim, llm_dim)
        self.W_Q = nn.Linear(llm_dim, llm_dim)
        
        self.attn = nn.MultiheadAttention(embed_dim=llm_dim, num_heads=num_heads, batch_first=True)
        
        self.mlp = nn.Sequential(
            nn.Linear(llm_dim, llm_dim * 4),
            nn.GELU(),
            nn.Linear(llm_dim * 4, llm_dim)
        )

    def forward(self, B, H_query):
        B = B + self.pos_embed
        K = self.W_K(B)
        V = self.W_V(B)
        Q = self.W_Q(H_query)
        A, _ = self.attn(query=Q, key=K, value=V)
        return self.mlp(A)

class ResidualSteerLM(BrainLanguageModel):
    """
    Residual Steering LM.
    Optimized for VRAM efficiency and autograd stability using PyTorch Hooks.
    """
    def __init__(self, llm_id: str = "gpt2", brain_dim: int = 1000, injection_layer: int = 6, n_frames: int = 4):
        super().__init__()
        
        self.llm = AutoModelForCausalLM.from_pretrained(llm_id)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.llm_dim = self.llm.config.hidden_size
        self.injection_layer = injection_layer
        
        self.adapter = BrainSteerAdapter(brain_dim=brain_dim, llm_dim=self.llm_dim, n_frames=n_frames)
        
        self.freeze_base_model() # This tells Autograd to NOT store activations for the base LLM

    def get_h_at_layer(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Phase 1 Extraction: Uses HF's native output_hidden_states to avoid manual loops.
        """
        with torch.no_grad():
            outputs = self.llm(input_ids=input_ids, output_hidden_states=True, **kwargs)
            # hidden_states is a tuple: (embeddings, layer_0, layer_1, ...)
            # Indexing with self.injection_layer gets the exact output state we need.
            return outputs.hidden_states[self.injection_layer]

    def forward(self, brain_batch: torch.Tensor, input_ids: torch.Tensor, **kwargs):
        return self.forward_steered(input_ids, brain_batch)

    def forward_steered(self, input_ids: torch.Tensor, brain_batch: torch.Tensor, **kwargs):
        """Phase 2: Main Training using PyTorch Forward Hooks."""
        v_steer_cache = []

        def steering_hook(module, inputs, output):
            # 0. Robust check: is it a tuple or a raw tensor?
            is_tuple = isinstance(output, tuple)
            hidden_states = output[0] if is_tuple else output
            
            # 1. Extract Query
            H_query = hidden_states[:, -1:, :]
            
            # 2. Compute Steering Vector
            v_steer = self.adapter(brain_batch, H_query)
            v_steer_cache.append(v_steer)
            
            # 3. Autograd-Safe Injection
            front_context = hidden_states[:, :-1, :]
            last_token_steered = hidden_states[:, -1:, :] + v_steer
            steered_hidden_states = torch.cat([front_context, last_token_steered], dim=1)
            
            # 4. Return modified state back to the LLM in the exact format it expects
            if is_tuple:
                return (steered_hidden_states,) + output[1:]
            else:
                return steered_hidden_states

        # Register hook on the layer BEFORE the injection point (0-indexed)
        target_layer = self.llm.transformer.h[self.injection_layer - 1]
        handle = target_layer.register_forward_hook(steering_hook)
        
        try:
            # The base model is frozen, so PyTorch only tracks gradients for the hook + adapter
            outputs = self.llm(input_ids=input_ids, **kwargs)
        finally:
            handle.remove() # Always clean up the hook
            
        return outputs.logits, v_steer_cache[0] if v_steer_cache else None