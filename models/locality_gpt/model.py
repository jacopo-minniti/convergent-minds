import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from brainscore.model_helpers.huggingface import HuggingfaceSubject
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig



class LocalityGPT2Attention(GPT2Attention):
    """
    GPT2Attention layer modified to apply an exponential decay to attention scores
    based on the distance between tokens, enforcing a locality bias.
    """
    def __init__(self, config, is_cross_attention=False, layer_idx=None, decay_rate=1.0):
        """
        Initializes the attention layer.
        
        Args:
            config: The model configuration.
            is_cross_attention (bool): Whether the layer is used for cross-attention.
            layer_idx (int): The index of the layer.
            decay_rate (float): The rate of exponential decay.
                                A positive value creates a locality bias.
                                A negative value creates an anti-locality bias.
                                A value of 0.0 removes the bias.
                                Defaults to 1.0.
        """
        super().__init__(config, is_cross_attention=is_cross_attention, layer_idx=layer_idx)
        self.decay_rate = decay_rate

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / (float(value.size(-1)) ** 0.5)

        # --- START: Configurable Exponential Decay ---
        query_length, key_length = query.size(-2), key.size(-2)

        # Apply decay only during self-attention and if decay_rate is non-zero.
        if query_length == key_length and self.decay_rate != 0:
            seq_len = query_length
            
            # Create indices for i (query position) and j (key position)
            i_indices = torch.arange(seq_len, device=query.device, dtype=torch.long).unsqueeze(1)
            j_indices = torch.arange(seq_len, device=query.device, dtype=torch.long).unsqueeze(0)
            
            # Create a matrix of distances: |i - j|
            distance_matrix = torch.abs(i_indices - j_indices)
            
            # Apply exponential decay: exp(-decay_rate * distance)
            decay_matrix = torch.exp(-self.decay_rate * distance_matrix.to(attn_weights.dtype))
            
            # Broadcast decay_matrix to match attn_weights dimensions
            attn_weights = attn_weights * decay_matrix.unsqueeze(0).unsqueeze(0)
        
        # --- END: Configurable Exponential Decay ---

        if not self.is_cross_attention:
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].bool()
            attn_weights = torch.where(causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.Softmax(dim=-1)(attn_weights)
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights


class LocalityGPT2(HuggingfaceSubject):
    def __init__(self, model_id, region_layer_mapping, untrained=False, decay_rate=1.0, **kwargs):
        if untrained:
            config = AutoConfig.from_pretrained(model_id)
            # SDPA/flash attention cannot return attention weights; force eager for analysis tools.
            setattr(config, "attn_implementation", "eager")
            model = AutoModelForCausalLM.from_config(config)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="eager")

        # Replace standard attention layers with our locality-biased version
        for i, layer in enumerate(model.transformer.h):
            layer.attn = LocalityGPT2Attention(model.config, layer_idx=i, decay_rate=decay_rate)

        tokenizer = AutoTokenizer.from_pretrained(model_id, truncation_side='left')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        super().__init__(model_id=model_id,
                         region_layer_mapping=region_layer_mapping,
                         model=model,
                         tokenizer=tokenizer,
                         **kwargs)
