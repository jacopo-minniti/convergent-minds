import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from scipy.stats import entropy

def main():
    parser = argparse.ArgumentParser(description="Generate attention plots for a model")
    parser.add_argument("--model", default="gpt2", help="Model identifier")
    parser.add_argument("--untrained", action="store_true", help="Use untrained model")
    parser.add_argument("--text", default="The quick brown fox jumps over the lazy dog.", help="Text to analyze")
    parser.add_argument("--output_dir", default="results", help="Directory to save plots")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    args = parser.parse_args()

    print(f"Loading model: {args.model} (Untrained: {args.untrained})")
    device = args.device
    
    # Load Model and Tokenizer
    # Handle LocalityGPT logic if needed, but for now assume standard HF or path
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

    print(f"Processing text: {args.text}")
    inputs = tokenizer(args.text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    if not outputs.attentions:
        print("Model did not return attentions.")
        return

    # Get last layer attention
    last_layer_attn = outputs.attentions[-1][0].cpu().numpy() # (num_heads, seq_len, seq_len)
    avg_attn = np.mean(last_layer_attn, axis=0) # (seq_len, seq_len)
    
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    
    # Calculate Entropy
    attn_entropy = entropy(avg_attn, axis=1)
    mean_entropy = np.mean(attn_entropy)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Plot Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_attn, xticklabels=tokens, yticklabels=tokens, cmap="viridis")
    plt.title(f"Avg Attention (Last Layer)\nMean Entropy: {mean_entropy:.4f}")
    plt.xlabel("Key")
    plt.ylabel("Query")
    plt.tight_layout()
    save_path = os.path.join(args.output_dir, "attention_heatmap.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved heatmap to {save_path}")

    # Plot Entropy
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(tokens)), attn_entropy)
    plt.xticks(range(len(tokens)), tokens, rotation=45, ha="right")
    plt.title("Attention Entropy per Token")
    plt.ylabel("Entropy")
    plt.tight_layout()
    save_path_entropy = os.path.join(args.output_dir, "attention_entropy.png")
    plt.savefig(save_path_entropy)
    plt.close()
    print(f"Saved entropy plot to {save_path_entropy}")

if __name__ == "__main__":
    main()
