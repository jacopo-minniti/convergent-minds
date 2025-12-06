from transformers import AutoModelForCausalLM
import torch

model_id = "bert-base-uncased"
try:
    model = AutoModelForCausalLM.from_pretrained(model_id)
    print("Model class:", type(model).__name__)
    print("Top level modules:", list(model._modules.keys()))
    
    if hasattr(model, 'bert'):
        print("bert modules:", list(model.bert._modules.keys()))
        if hasattr(model.bert, 'encoder'):
            print("bert.encoder modules:", list(model.bert.encoder._modules.keys()))
except Exception as e:
    print(e)
