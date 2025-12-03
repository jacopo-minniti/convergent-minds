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
        # Force eager attention so our custom _attn is always used (avoids SDPA/flash paths).
        self.attn_implementation = "eager"

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        cache_position=None,
        use_cache=False,
        output_attentions=False,
        **kwargs,
    ):
        """
        Override to bypass SDPA/flash dispatch and always use our custom `_attn`
        (which applies the decay mask). Cross-attention falls back to the
        parent implementation to keep that pathway intact.
        """
        if encoder_hidden_states is not None or self.is_cross_attention:
            return super().forward(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=past_key_value,
                cache_position=cache_position,
                use_cache=use_cache,
                output_attentions=output_attentions,
                **kwargs,
            )

        layer_past = layer_past if layer_past is not None else past_key_value
        # Self-attention path with enforced eager attention
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        attn_output, attn_weights = self._attn(
            query, key, value, attention_mask, head_mask
        )

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        # Align return signature with GPT2Block expectations: (attn_output, attn_weights_or_none)
        outputs = (attn_output, attn_weights if output_attentions else None)
        return outputs

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """Copy of GPT2Attention._split_heads to avoid missing attribute across HF versions."""
        batch_size, seq_length, hidden_size = tensor.size()
        tensor = tensor.view(batch_size, seq_length, num_heads, attn_head_size)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq, head_dim)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """Copy of GPT2Attention._merge_heads to avoid missing attribute across HF versions."""
        batch_size, num_heads_, seq_length, head_dim = tensor.size()
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        return tensor.view(batch_size, seq_length, num_heads_ * head_dim)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        print(f"DEBUG: _attn called with decay_rate={self.decay_rate}")
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
            # We want to penalize distant tokens.
            # Standard attention: softmax(Q K^T / sqrt(d))
            # Local attention: softmax(Q K^T / sqrt(d) - decay * distance)
            # This is equivalent to multiplying probabilities by exp(-decay * distance)
            
            decay_penalty = self.decay_rate * distance_matrix.to(attn_weights.dtype)
            
            # Broadcast penalty to match attn_weights dimensions
            attn_weights = attn_weights - decay_penalty.unsqueeze(0).unsqueeze(0)
        
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
    def __init__(self, model_id, region_layer_mapping, untrained=False, decay_rate=1.0, device=None, **kwargs):
        if untrained:
            config = AutoConfig.from_pretrained(model_id)
            # SDPA/flash attention cannot return attention weights; force eager for analysis tools.
            setattr(config, "attn_implementation", "eager")
            setattr(config, "_attn_implementation", "eager")
            model = AutoModelForCausalLM.from_config(config)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="eager")

        # Ensure the model config itself reflects eager attention
        if hasattr(model, "config"):
            setattr(model.config, "attn_implementation", "eager")
            setattr(model.config, "_attn_implementation", "eager")

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
                         device=device,
                         **kwargs)
