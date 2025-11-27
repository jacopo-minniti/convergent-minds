import argparse
import torch
import numpy as np
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from scipy.stats import entropy

def main():
    parser = argparse.ArgumentParser(description="Calculate useful statistics for a model")
    parser.add_argument("--model", default="gpt2", help="Model identifier")
    parser.add_argument("--untrained", action="store_true", help="Use untrained model")
    parser.add_argument("--text", default="The quick brown fox jumps over the lazy dog.", help="Text to analyze")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    args = parser.parse_args()

    print(f"Loading model: {args.model} (Untrained: {args.untrained})")
    device = args.device
    
    if os.path.isdir(args.model):
        model_path = args.model
    else:
        model_path = args.model

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.untrained:
        config = AutoConfig.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_config(config)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path)

    model.to(device)
    model.eval()

    # Model Parameter Stats
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n--- Model Statistics ---")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Model Config: {model.config}")

    # Attention Entropy Stats
    print(f"\n--- Attention Statistics (on sample text) ---")
    inputs = tokenizer(args.text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    if outputs.attentions:
        # Average entropy across all layers and heads
        all_entropies = []
        for layer_idx, layer_attn in enumerate(outputs.attentions):
            # layer_attn: (batch, heads, seq, seq)
            attn = layer_attn[0].cpu().numpy()
            # Entropy per head per token
            # We want to see how "peaked" the attention is
            # entropy over the last dim (keys)
            ent = entropy(attn, axis=-1) # (heads, seq)
            mean_ent = np.mean(ent)
            all_entropies.append(mean_ent)
            print(f"Layer {layer_idx} Mean Entropy: {mean_ent:.4f}")
        
        print(f"Overall Mean Entropy: {np.mean(all_entropies):.4f}")
    else:
        print("Model did not return attentions.")

if __name__ == "__main__":
    main()
