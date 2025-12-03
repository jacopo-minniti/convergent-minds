import torch
import torch.nn as nn
from brainscore.model_helpers.huggingface import HuggingfaceSubject
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

class HierarchicalGPT2(HuggingfaceSubject):
    def __init__(self, model_id, region_layer_mapping, depth, untrained=False, device=None, **kwargs):
        """
        HierarchicalGPT2 model that runs only the first `depth` transformer blocks.
        
        Args:
            model_id (str): The model identifier (e.g., 'gpt2').
            region_layer_mapping (dict): Mapping from regions to layer names.
            depth (int): The number of transformer blocks to keep.
            untrained (bool): Whether to use an untrained model.
            device (str): Device to use.
            **kwargs: Additional arguments for HuggingfaceSubject.
        """
        if untrained:
            config = AutoConfig.from_pretrained(model_id)
            # Ensure we don't load pretrained weights
            model = AutoModelForCausalLM.from_config(config)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id)

        # Truncate the model to the specified depth
        if depth < len(model.transformer.h):
            model.transformer.h = model.transformer.h[:depth]
            # Update config to reflect the new number of layers
            model.config.n_layer = depth
            model.config.num_hidden_layers = depth
        elif depth > len(model.transformer.h):
            raise ValueError(f"Requested depth {depth} exceeds model depth {len(model.transformer.h)}")

        # Filter region_layer_mapping to only include layers that exist in the truncated model
        # The layer names are usually 'transformer.h.N'
        # We need to ensure we don't request layers that don't exist
        
        # Helper to check if a layer name is within the depth
        def is_valid_layer(layer_name):
            parts = layer_name.split('.')
            # Assuming format 'transformer.h.N'
            if len(parts) >= 3 and parts[0] == 'transformer' and parts[1] == 'h':
                try:
                    idx = int(parts[2])
                    return idx < depth
                except ValueError:
                    return False
            return False

        # We might need to adjust the mapping if the user provided specific layers
        # For now, we assume the caller handles the mapping or we just pass it through
        # and let BrainScore handle errors if a layer is missing. 
        # However, for the specific use case of "last layer at depth d", the caller should provide the correct layer.
        
        tokenizer = AutoTokenizer.from_pretrained(model_id, truncation_side='left')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        super().__init__(model_id=model_id,
                         region_layer_mapping=region_layer_mapping,
                         model=model,
                         tokenizer=tokenizer,
                         device=device,
                         **kwargs)
