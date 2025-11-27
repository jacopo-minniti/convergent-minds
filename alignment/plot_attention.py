import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from scipy.stats import entropy
from brainscore.model_helpers.huggingface import get_layer_names
from brainscore import ArtificialSubject

from models.locality_gpt.model import LocalityGPT2


def _force_eager_attn(model_or_config):
    """
    Set attn_implementation to 'eager' if the attribute exists.
    Some HF versions default to 'sdpa' which does not support returning attentions.
    """
    if hasattr(model_or_config, "attn_implementation"):
        try:
            model_or_config.attn_implementation = "eager"
        except Exception:
            pass


def _capture_last_attention(model, inputs):
    """
    Fallback for models that do not populate `outputs.attentions` (e.g. custom wrappers
    or models forced into SDPA mode).
    Registers a forward hook on the last attention layer to grab its weights.
    """
    # Attempt to locate the last transformer block. 
    # GPT-2 usually uses model.transformer.h
    blocks = None
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        blocks = model.transformer.h
    elif hasattr(model, "h"): # Sometimes directly on model
        blocks = model.h
    elif hasattr(model, "model") and hasattr(model.model, "layers"): # Llama/Mistral style
        blocks = model.model.layers
    
    if blocks is None:
        return None

    captured = {}

    def hook(_module, _inputs, output):
        # GPT2Attention returns (attn_output, present, attn_weights) when output_attentions=True
        # Some implementations return just one tensor.
        if isinstance(output, tuple):
            if len(output) >= 3:
                captured["attn"] = output[2]
            elif len(output) == 2 and output[1] is not None:
                captured["attn"] = output[1]
        elif torch.is_tensor(output):
             # If the layer outputs just the tensor, we might need a different strategy,
             # but usually HF layers return tuples.
             pass

    # Register hook on the last layer's attention mechanism
    # Note: Structure depends on model architecture. GPT2 is .attn
    last_layer = blocks[-1]
    target_module = getattr(last_layer, "attn", getattr(last_layer, "self_attn", None))
    
    if target_module is None:
        return None

    handle = target_module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            # We don't ask for output_attentions here because the hook captures internal state
            model(**inputs)
    except Exception as e:
        print(f"Hook capture failed: {e}")
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
    parser.add_argument("--decay_rate", type=float, default=1.0, help="Decay rate for LocalityGPT2")
    args = parser.parse_args()

    print(f"Loading model: {args.model} (Untrained: {args.untrained})")
    device = args.device
    
    model = None
    tokenizer = None

    if "locality_gpt" in args.model:
        if LocalityGPT2 is None:
            raise ImportError("Could not import LocalityGPT2. Make sure models.locality_gpt is accessible.")
        
        print("Initializing LocalityGPT2...")
        base_model_id = "gpt2"
        config = AutoConfig.from_pretrained(base_model_id)
        
        # NOTE: LocalityGPT2 likely initializes the model internally. 
        # If LocalityGPT2 source code does not explicitly set attn_implementation="eager",
        # it might default to "sdpa" on newer PyTorch versions.
        # Since we can't easily patch the class from here, we will handle the SDPA check later.
        
        hidden_dim = getattr(config, "n_embd", getattr(config, "hidden_size", 768))
        localizer_kwargs = {
            'top_k': 256,
            'batch_size': 16,
            'hidden_dim': hidden_dim
        }
        
        layer_names = get_layer_names(base_model_id)
        subject = LocalityGPT2(
            model_id=base_model_id,
            region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: layer_names},
            untrained=args.untrained,
            use_localizer=False, 
            localizer_kwargs=localizer_kwargs, 
            decay_rate=args.decay_rate
        )
        model = subject.model
        tokenizer = subject.tokenizer
        _force_eager_attn(model.config)

    else:
        # Load Standard Model and Tokenizer
        if os.path.isdir(args.model):
            model_path = args.model
        else:
            model_path = args.model

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # ==============================================================================
        # CRITICAL FIX: Force 'eager' implementation.
        # SDPA (Flash Attn) does not support returning attention weights.
        # ==============================================================================
        if args.untrained:
            config = AutoConfig.from_pretrained(model_path)
            _force_eager_attn(config)
            model = AutoModelForCausalLM.from_config(config)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                attn_implementation="eager"
            )

    _force_eager_attn(model.config)
    model.to(device)
    model.eval()

    # Safely attempt to enable output_attentions
    current_impl = getattr(model.config, "attn_implementation", "eager")
    if current_impl == "sdpa":
        print("WARNING: Model is using 'sdpa' (Flash Attention). `output_attentions=True` is not supported.")
        print("Will attempt to capture attention via hooks, but this may fail if the kernel is fused.")
        model.config.output_attentions = False
    else:
        model.config.output_attentions = True

    print(f"Processing text: {args.text}")
    inputs = tokenizer(args.text, return_tensors="pt").to(device)
    
    last_layer_attn = None
    
    # Forward pass
    with torch.no_grad():
        if model.config.output_attentions:
            outputs = model(**inputs)
            if outputs.attentions and outputs.attentions[-1] is not None:
                last_layer_attn = outputs.attentions[-1][0]
        else:
            # If output_attentions is False (or SDPA forced), try the fallback hook
            pass

    # If we didn't get attentions from the standard output, try the fallback hook
    if last_layer_attn is None:
        print("Standard output_attentions failed or disabled. Attempting hook capture...")
        last_layer_attn = _capture_last_attention(model, inputs)

    if last_layer_attn is None:
        print("Error: Could not capture attention weights.")
        if current_impl == "sdpa":
            print("Tip: If using LocalityGPT2, edit the class to load the base model with `attn_implementation='eager'`.")
        return

    # Process Attention Weights
    # Ensure it's on CPU and numpy
    if isinstance(last_layer_attn, torch.Tensor):
        last_layer_attn = last_layer_attn.detach().cpu().numpy() 
    
    # Expected shape: (num_heads, seq_len, seq_len)
    if len(last_layer_attn.shape) == 4: 
        # Sometimes (batch, heads, seq, seq)
        last_layer_attn = last_layer_attn[0]
        
    avg_attn = np.mean(last_layer_attn, axis=0) # (seq_len, seq_len)
    
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    # Clean up GPT2 tokens (replace Ġ with space)
    tokens = [t.replace('Ġ', ' ') for t in tokens]
    
    # Calculate Entropy
    # Add epsilon to avoid log(0)
    avg_attn = np.maximum(avg_attn, 1e-9)
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
