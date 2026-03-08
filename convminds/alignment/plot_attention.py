import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy.stats import entropy
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def _force_eager_attn(model_or_config):
    if hasattr(model_or_config, "attn_implementation"):
        try:
            model_or_config.attn_implementation = "eager"
        except Exception:
            pass


def _capture_last_attention(model, inputs):
    blocks = None
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        blocks = model.transformer.h
    elif hasattr(model, "h"):
        blocks = model.h
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        blocks = model.model.layers

    if blocks is None:
        return None

    captured = {}

    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            if len(output) >= 3:
                captured["attn"] = output[2]
            elif len(output) == 2 and output[1] is not None:
                captured["attn"] = output[1]

    last_layer = blocks[-1]
    target_module = getattr(last_layer, "attn", getattr(last_layer, "self_attn", None))
    if target_module is None:
        return None

    handle = target_module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        handle.remove()

    return captured.get("attn")


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

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.untrained:
        config = AutoConfig.from_pretrained(args.model)
        _force_eager_attn(config)
        model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, attn_implementation="eager")

    _force_eager_attn(model.config)
    model.to(device)
    model.eval()

    current_impl = getattr(model.config, "attn_implementation", "eager")
    model.config.output_attentions = current_impl != "sdpa"

    print(f"Processing text: {args.text}")
    inputs = tokenizer(args.text, return_tensors="pt").to(device)

    last_layer_attn = None
    with torch.no_grad():
        if model.config.output_attentions:
            outputs = model(**inputs)
            if outputs.attentions and outputs.attentions[-1] is not None:
                last_layer_attn = outputs.attentions[-1][0]

    if last_layer_attn is None:
        print("Standard output_attentions failed or disabled. Attempting hook capture...")
        last_layer_attn = _capture_last_attention(model, inputs)

    if last_layer_attn is None:
        print("Error: Could not capture attention weights.")
        return

    if isinstance(last_layer_attn, torch.Tensor):
        last_layer_attn = last_layer_attn.detach().cpu().numpy()
    if len(last_layer_attn.shape) == 4:
        last_layer_attn = last_layer_attn[0]

    avg_attn = np.mean(last_layer_attn, axis=0)

    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    tokens = [t.replace("Ġ", " ") for t in tokens]

    avg_attn = np.maximum(avg_attn, 1e-9)
    attn_entropy = entropy(avg_attn, axis=1)
    mean_entropy = np.mean(attn_entropy)

    os.makedirs(args.output_dir, exist_ok=True)

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
