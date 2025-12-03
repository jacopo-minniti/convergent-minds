import torch
import numpy as np
from models.locality_gpt.model import LocalityGPT2

def test_decay_impact():
    model_id = "gpt2"
    dummy_mapping = {'dummy_region': 0}
    
    # Create two models with different decay rates
    print("Initializing model with decay_rate=0.1...")
    model_01 = LocalityGPT2(model_id, dummy_mapping, untrained=True, decay_rate=0.1)
    
    print("Initializing model with decay_rate=1.0...")
    model_10 = LocalityGPT2(model_id, dummy_mapping, untrained=True, decay_rate=1.0)
    
    # Ensure they have the same weights for non-attention layers (and even attention layers if we want to be strict, 
    # but since they are untrained, they are random. 
    # However, to compare outputs, we should probably force them to have the SAME weights.)
    
    # Let's copy weights from model_01 to model_10
    print("Copying weights from model_01 to model_10...")
    model_10.model.load_state_dict(model_01.model.state_dict())
    
    # Verify decay rates are still different
    print(f"Model 0.1 decay rate (layer 0): {model_01.model.transformer.h[0].attn.decay_rate}")
    print(f"Model 1.0 decay rate (layer 0): {model_10.model.transformer.h[0].attn.decay_rate}")
    
    # Run forward pass
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    
    print("Running forward pass for model 0.1...")
    with torch.no_grad():
        output_01 = model_01.model(input_ids).logits
        
    print("Running forward pass for model 1.0...")
    with torch.no_grad():
        output_10 = model_10.model(input_ids).logits
        
    # Compare outputs
    diff = torch.abs(output_01 - output_10).max().item()
    print(f"Max difference between outputs: {diff}")
    
    if diff < 1e-6:
        print("FAIL: Outputs are identical despite different decay rates!")
    else:
        print("SUCCESS: Outputs are different.")

if __name__ == "__main__":
    test_decay_impact()
