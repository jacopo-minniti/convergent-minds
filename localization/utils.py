import torch 
import pickle as pkl
from collections import OrderedDict

def write_pickle(path, data):
    """Write data to a pickle file."""
    with open(path, 'wb') as f:
        pkl.dump(data, f)

def read_pickle(path):
    """Read data from a pickle file."""
    with open(path, 'rb') as f:
        data = pkl.load(f)
    return data 

def _get_layer(module, layer_name: str) -> torch.nn.Module:
    """Recursively get a submodule from a module by layer name."""
    SUBMODULE_SEPARATOR = '.'
    for part in layer_name.split(SUBMODULE_SEPARATOR):
        module = module._modules.get(part)
        assert module is not None, f"No submodule found for layer {layer_name}, at part {part}"
    return module
    
def _register_hook(layer: torch.nn.Module,
                    key: str,
                    target_dict: dict):
    """Register a forward hook to a layer to store its output."""
    def hook_function(_layer: torch.nn.Module, _input, output: torch.Tensor, key=key):
        # The output of some layers is a tuple, we are interested in the first element
        output = output[0] if isinstance(output, (tuple, list)) else output
        target_dict[key] = output

    hook = layer.register_forward_hook(hook_function)
    return hook

def setup_hooks(model, layer_names):
    """Set up forward hooks for specified layers in the model."""
    hooks = []
    layer_representations = OrderedDict()

    for layer_name in layer_names:
        layer = _get_layer(model, layer_name)
        hook = _register_hook(layer, key=layer_name,
                                target_dict=layer_representations)
        hooks.append(hook)

    return hooks, layer_representations

def get_layer_names(model_name):
    """Get the layer names for a given model."""
    if "gpt2" in model_name:
        # gpt2 has 12 layers
        num_blocks = 12
        # Layer names for gpt2 follow this pattern
        return [f'transformer.h.{block}.{layer_desc}' 
            for block in range(num_blocks) 
            for layer_desc in ['ln_1', 'attn', 'ln_2', 'mlp']
        ]
    else:
        raise ValueError(f"{model_name} not supported currently!")