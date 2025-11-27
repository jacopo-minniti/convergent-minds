
import torch
from models.locality_gpt.model import LocalityGPT2, LocalityGPT2Attention
from transformers import AutoConfig

def investigate_model():
    print("Initializing model for investigation...")
    dummy_mapping = {'dummy_region': 0}
    subject = LocalityGPT2("gpt2", region_layer_mapping=dummy_mapping, untrained=True, decay_rate=2.0)
    
    model = subject.model
    print(f"Model type: {type(model)}")
    print(f"Config attn_implementation: {getattr(model.config, 'attn_implementation', 'Not Set')}")
    
    first_layer = model.transformer.h[0]
    attn_layer = first_layer.attn
    print(f"Layer 0 attn type: {type(attn_layer)}")
    print(f"Is instance of LocalityGPT2Attention: {isinstance(attn_layer, LocalityGPT2Attention)}")
    
    if isinstance(attn_layer, LocalityGPT2Attention):
        print(f"Decay rate: {attn_layer.decay_rate}")
    
    print("\nRunning forward pass with output_attentions=True...")
    input_ids = torch.tensor([[1, 2, 3, 4]])
    try:
        outputs = model(input_ids, output_attentions=True)
        print("Forward pass completed.")
        if outputs.attentions is None:
            print("outputs.attentions is None")
        else:
            print(f"outputs.attentions is a {type(outputs.attentions)}")
            if len(outputs.attentions) > 0:
                print(f"First element type: {type(outputs.attentions[0])}")
                if outputs.attentions[0] is None:
                     print("First element is None!")
            else:
                print("attentions tuple is empty")
                
    except Exception as e:
        print(f"Forward pass failed: {e}")

if __name__ == "__main__":
    investigate_model()
