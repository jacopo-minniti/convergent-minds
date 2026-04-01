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
        
        # Trainable positional embeddings for the temporal window
        self.pos_embed = nn.Parameter(torch.randn(1, n_frames, brain_dim) * 0.02)
        
        # Projections for Cross-Attention
        self.W_K = nn.Linear(brain_dim, llm_dim)
        self.W_V = nn.Linear(brain_dim, llm_dim)
        self.W_Q = nn.Linear(llm_dim, llm_dim)
        
        self.attn = nn.MultiheadAttention(embed_dim=llm_dim, num_heads=num_heads, batch_first=True)
        
        # Final projection head
        self.mlp = nn.Sequential(
            nn.Linear(llm_dim, llm_dim * 4),
            nn.GELU(),
            nn.Linear(llm_dim * 4, llm_dim)
        )

    def forward(self, B, H_query):
        """
        B: Brain Window [Batch, n_frames, brain_dim]
        H_query: Current hidden state of the token being steered [Batch, 1, llm_dim]
        """
        B = B + self.pos_embed
        K = self.W_K(B)
        V = self.W_V(B)
        Q = self.W_Q(H_query)
        
        # Cross-Attend: LLM Query attends to Brain Key/Values
        A, _ = self.attn(query=Q, key=K, value=V)
        return self.mlp(A)

class ResidualSteerLM(BrainLanguageModel):
    """
    Residual Steering LM.
    Uses PyTorch Forward Hooks for localized latent injection and
    native HF APIs for activation extraction.
    """
    def __init__(self, llm_id: str = "gpt2", brain_dim: int = 1000, injection_layer: int = 6, n_frames: int = 4):
        super().__init__()
        
        self.llm = AutoModelForCausalLM.from_pretrained(llm_id)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_id)
        
        # Standardize tokenizer configuration
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left" # Essential for consistent indexing [:, -1, :]
            
        self.llm_dim = self.llm.config.hidden_size
        self.injection_layer = injection_layer
        
        # Brain-to-LLM Adapter
        self.adapter = BrainSteerAdapter(brain_dim=brain_dim, llm_dim=self.llm_dim, n_frames=n_frames)
        
        # Freeze LLM weights to ensure only the adapter is trained
        self.freeze_base_model()

    def get_h_at_layer(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Extract hidden states at the injection layer using HF native API.
        """
        with torch.no_grad():
            outputs = self.llm(input_ids=input_ids, output_hidden_states=True, **kwargs)
            # Tuple indexing: layer 0 is embeddings, layer N is transformer output
            return outputs.hidden_states[self.injection_layer]

    def forward(self, brain_batch: torch.Tensor, input_ids: torch.Tensor, **kwargs):
        """Standard interface for training and inference."""
        return self.forward_steered(input_ids, brain_batch, **kwargs)

    def forward_steered(self, input_ids: torch.Tensor, brain_batch: torch.Tensor, **kwargs):
        """
        Execute forward pass with localized residual injection using hooks.
        """
        v_steer_cache = []

        def steering_hook(module, inputs, output):
            # Capture and modify the transformer block output
            is_tuple = isinstance(output, tuple)
            hidden_states = output[0] if is_tuple else output
            
            # Query the adapter using the hidden state of the last token
            H_query = hidden_states[:, -1:, :]
            v_steer = self.adapter(brain_batch, H_query)
            v_steer_cache.append(v_steer)
            
            # Perform additive injection only on the steered position
            front_context = hidden_states[:, :-1, :]
            last_token_steered = hidden_states[:, -1:, :] + v_steer
            steered_hidden_states = torch.cat([front_context, last_token_steered], dim=1)
            
            if is_tuple:
                return (steered_hidden_states,) + output[1:]
            return steered_hidden_states

        # Target the block BEFORE the injection point
        target_layer = self.llm.transformer.h[self.injection_layer - 1]
        handle = target_layer.register_forward_hook(steering_hook)
        
        try:
            outputs = self.llm(input_ids=input_ids, **kwargs)
        finally:
            handle.remove() # Immediate cleanup to prevent state leakage
            
        return outputs.logits, v_steer_cache[0] if v_steer_cache else None

    def generate_steered(
        self, 
        input_ids: torch.Tensor, 
        brain_batch: torch.Tensor, 
        max_new_tokens: int = 15,
        **kwargs
    ) -> torch.Tensor:
        """
        Produce a sequence of tokens with persistent residual steering.
        The brain vector is calculated once and applied to all generated tokens.
        """
        # 1. Pre-calculate the steering vector from the initial context
        with torch.no_grad():
            H_query = self.get_h_at_layer(input_ids, **kwargs)[:, -1:, :]
            v_steer = self.adapter(brain_batch, H_query)

        # 2. Setup hook to inject this vector into every generation step
        def persistent_steering_hook(module, inputs, output):
            is_tuple = isinstance(output, tuple)
            hidden_states = output[0] if is_tuple else output
            
            # Additive steering (only on the last token of the current tensor)
            front_context = hidden_states[:, :-1, :]
            last_token_steered = hidden_states[:, -1:, :] + v_steer
            steered_hidden_states = torch.cat([front_context, last_token_steered], dim=1)
            
            if is_tuple:
                return (steered_hidden_states,) + output[1:]
            return steered_hidden_states

        target_layer = self.llm.transformer.h[self.injection_layer - 1]
        handle = target_layer.register_forward_hook(persistent_steering_hook)

        # 3. Use standard HF generation
        try:
            generated = self.llm.generate(
                input_ids=input_ids, 
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs
            )
        finally:
            handle.remove()

        return generated

    def freeze_base_model(self):
        """Ensure only adapter parameters are optimized."""
        for param in self.llm.parameters():
            param.requires_grad = False
        logger.info(f"Base LLM '{self.llm.config._name_or_path}' frozen.")