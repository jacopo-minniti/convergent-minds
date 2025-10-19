import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from brainscore_language.model_helpers.huggingface import HuggingfaceSubject
from transformers import AutoModelForCausalLM

class ModifiedGPT2Attention(GPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__(config, is_cross_attention=is_cross_attention, layer_idx=layer_idx)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / (float(value.size(-1)) ** 0.5)

        # Apply exponential decay to attention scores
        seq_len = query.size(-2)
        decay = torch.exp(-torch.arange(seq_len, device=attn_weights.device, dtype=attn_weights.dtype)).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        attn_weights = attn_weights * decay

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].bool()
            attn_weights = torch.where(causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.Softmax(dim=-1)(attn_weights)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights


class ModifiedGPT2(HuggingfaceSubject):
    def __init__(self, model_id, region_layer_mapping):
        model = AutoModelForCausalLM.from_pretrained(model_id)
        for i, layer in enumerate(model.transformer.h):
            layer.attn = ModifiedGPT2Attention(model.config)
        super().__init__(model_id=model_id,
                         region_layer_mapping=region_layer_mapping,
                         model=model)