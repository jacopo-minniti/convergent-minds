
import torch
from models.locality_gpt.model import LocalityGPT2
from transformers import AutoConfig

def test_generation_locality():
    """
    Test to check if locality bias is applied during token-by-token generation.
    """
    print("Initializing model...")
    dummy_mapping = {'dummy_region': 0}
    subject = LocalityGPT2("gpt2", region_layer_mapping=dummy_mapping, untrained=True, decay_rate=2.0)
    subject.model.eval()

    # Create a dummy input
    input_ids = torch.tensor([[1, 2, 3, 4]])
    
    # 1. Standard forward pass (Prompt processing)
    # query_len = 4, key_len = 4. Decay SHOULD be applied.
    print("\n--- Prompt Processing (Forward Pass) ---")
    outputs = subject.model(input_ids, output_attentions=True)
    attn_matrix = outputs.attentions[0] # Layer 0
    # Check if decay is applied (last token attending to first token should be very low)
    # Distance = 3. Decay = exp(-2 * 3) = exp(-6) ~= 0.002
    last_token_attn = attn_matrix[0, 0, 3, :]
    print(f"Attention weights for last token (pos 3): {last_token_attn.detach().numpy()}")
    
    # 2. Generation Step (Next token)
    # query_len = 1, key_len = 5 (4 context + 1 new). 
    # If logic is `if query_len == key_len`, Decay will be SKIPPED.
    print("\n--- Generation Step (Next Token) ---")
    past_key_values = outputs.past_key_values
    next_token = torch.tensor([[5]])
    
    outputs_gen = subject.model(next_token, past_key_values=past_key_values, output_attentions=True)
    attn_gen = outputs_gen.attentions[0]
    # attn_gen shape: [batch, head, q_len, k_len] -> [1, 12, 1, 5]
    
    # Check attention of the new token (pos 4) to the first token (pos 0).
    # Distance = 4. Decay = exp(-2 * 4) = exp(-8) ~= 0.0003
    # If decay is skipped, it will look like standard attention (likely higher).
    new_token_attn = attn_gen[0, 0, 0, :]
    print(f"Attention weights for new token (pos 4): {new_token_attn.detach().numpy()}")
    
    # Let's verify if logic held
    # In model.py: if query_length == key_length: apply decay
    # Here q_len=1, k_len=5.
    
    # We can inspect the code behavior by adding a print in the model.py temporarily or just inferring from output.
    # But since I am in the shell, I can just run this and see the values.

if __name__ == "__main__":
    test_generation_locality()
