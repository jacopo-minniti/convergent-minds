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
        
        # 1. Positional Embedding
        self.pos_embed = nn.Parameter(torch.randn(1, n_frames, brain_dim) * 0.02)
        
        # 2. Projections
        self.W_K = nn.Linear(brain_dim, llm_dim)
        self.W_V = nn.Linear(brain_dim, llm_dim)
        self.W_Q = nn.Linear(llm_dim, llm_dim)
        
        # 3. Cross-Attention
        self.attn = nn.MultiheadAttention(
            embed_dim=llm_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        # 4. Transformer MLP (Output Head)
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
        A, weights = self.attn(query=Q, key=K, value=V)
        v_steer = self.mlp(A)
        return v_steer

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
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        transformer = self.llm.transformer
        
        logger.info(f"Input processing: ids={input_ids.shape}, mask={attention_mask.shape}")
        
        # Manually create the 4D broadcastable attention mask for GPT-2
        # Shape: [batch, 1, 1, seq]
        mask_4d = attention_mask[:, None, None, :]
        mask_4d = mask_4d.to(dtype=self.llm.dtype)
        mask_4d = (1.0 - mask_4d) * torch.finfo(self.llm.dtype).min
        
        logger.info(f"Manual 4D Mask shape: {mask_4d.shape}")
        
        # Safer position IDs that handle batching/padding natively
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        
        hidden_states = transformer.wte(input_ids) + transformer.wpe(position_ids)
        hidden_states = transformer.drop(hidden_states)
        
        logger.info(f"Embeddings complete: hidden_states={hidden_states.shape}")
        
        for i in range(self.injection_layer):
            logger.info(f"Layer {i} input: hidden_states={hidden_states.shape}")
            layer_outputs = transformer.h[i](
                hidden_states, 
                attention_mask=mask_4d
            )
            hidden_states = layer_outputs[0]
            
            # Robust check: if batch dimension is lost, we cannot proceed safely
            if hidden_states.shape[0] != batch_size:
                 logger.error(f"FATAL: Layer {i} dropped batch dimension. Expected {batch_size}, got {hidden_states.shape}")
                 # Force restore only if total elements match exactly (e.g. if it was just squeezed)
                 if hidden_states.numel() == batch_size * seq_len * self.llm_dim:
                     hidden_states = hidden_states.view(batch_size, seq_len, self.llm_dim)
                 else:
                     raise RuntimeError(f"Layer {i} corrupted hidden_states. Expected {batch_size*seq_len*768} elements, but got {hidden_states.numel()}")
            
        return hidden_states, extended_attention_mask

    def get_h_at_layer(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Phase 1 Extraction: Optimized to stop computing at injection_layer.
        """
        logger.info(f"get_h_at_layer input_ids: {input_ids.shape}, dtype: {input_ids.dtype}")
        
        # Critical Check: Are we accidentally passing floats?
        if torch.is_floating_point(input_ids):
             logger.error(f"CRITICAL: get_h_at_layer received floating point input! Shape: {input_ids.shape}. Usually means you are passing embeddings instead of token IDs.")
             raise TypeError(f"get_h_at_layer expects integer token IDs, got {input_ids.dtype}")
        
        # Robustly force input_ids to be 2D [batch_size, sequence_length]
        if input_ids.dim() == 0:
            logger.warning(f"Forcing 0D input_ids to 2D (1, 1). Shape: {input_ids.shape}")
            input_ids = input_ids.view(1, 1)
        elif input_ids.dim() == 1:
            logger.warning(f"Forcing 1D input_ids to 2D (1, seq). Shape: {input_ids.shape}")
            input_ids = input_ids.unsqueeze(0)
        elif input_ids.dim() > 2:
            logger.warning(f"Flattening >2D input_ids to 2D (collapsed_batches, seq). Shape: {input_ids.shape}")
            input_ids = input_ids.view(-1, input_ids.size(-1))
            
        # no_grad because Phase 1 target extraction doesn't train the LLM
        with torch.no_grad():
            hidden_states, _ = self._get_pre_injection_state(input_ids)
            
        return hidden_states

    def forward(self, brain_batch: torch.Tensor, input_ids: torch.Tensor, **kwargs):
        """Standard forward pass with steering injection."""
        return self.forward_steered(input_ids, brain_batch)

    def forward_steered(self, input_ids: torch.Tensor, brain_batch: torch.Tensor):
        """Phase 2: Main Training (Cross-Entropy & Injection)."""
        logger.info(f"forward_steered input_ids: {input_ids.shape}, dtype: {input_ids.dtype}")
        
        if torch.is_floating_point(input_ids):
             logger.error(f"CRITICAL: forward_steered received floats! Shape: {input_ids.shape}")
             raise TypeError(f"forward_steered expects integer token IDs, got {input_ids.dtype}")

        # Robustly force input_ids to be 2D
        if input_ids.dim() == 0:
            logger.warning(f"Forcing 0D input_ids to 2D (1, 1) in forward_steered. Shape: {input_ids.shape}")
            input_ids = input_ids.view(1, 1)
        elif input_ids.dim() == 1:
            logger.warning(f"Forcing 1D input_ids to 2D (1, seq) in forward_steered. Shape: {input_ids.shape}")
            input_ids = input_ids.unsqueeze(0)
        elif input_ids.dim() > 2:
            logger.warning(f"Flattening >2D input_ids in forward_steered. Shape: {input_ids.shape}")
            input_ids = input_ids.view(-1, input_ids.size(-1))
            
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
            logger.info(f"Post-injection Layer {i} input: hidden_states={steered_hidden_states.shape}")
            layer_outputs = transformer.h[i](
                steered_hidden_states, 
                attention_mask=mask_4d
            )
            steered_hidden_states = layer_outputs[0]
            
            if steered_hidden_states.shape[0] != batch_size:
                 if steered_hidden_states.numel() == batch_size * seq_len * self.llm_dim:
                     steered_hidden_states = steered_hidden_states.view(batch_size, seq_len, self.llm_dim)
                 else:
                     raise RuntimeError(f"Post-injection Layer {i} corrupted hidden_states.")
            
        steered_hidden_states = transformer.ln_f(steered_hidden_states)
        logits = self.llm.lm_head(steered_hidden_states)
        
        return logits, v_steer