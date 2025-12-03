from brainscore.model_helpers.huggingface import get_layer_names
from transformers import AutoConfig

model_id = "gpt2"
layers = get_layer_names(model_id)
print(f"Layers for {model_id}:")
for l in layers:
    print(l)

config = AutoConfig.from_pretrained(model_id)
print(f"Config n_layer: {config.n_layer}")
