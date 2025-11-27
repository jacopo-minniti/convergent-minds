import argparse
import torch
import numpy as np
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from scipy.stats import entropy
from brainscore.model_helpers.huggingface import get_layer_names
from brainscore import ArtificialSubject

# Try to import LocalityGPT2
try:
    from models.locality_gpt.model import LocalityGPT2
except ImportError:
    LocalityGPT2 = None

def main():
    parser = argparse.ArgumentParser(description="Calculate useful statistics for a model")
    parser.add_argument("--model", default="gpt2", help="Model identifier")
    parser.add_argument("--untrained", action="store_true", help="Use untrained model")
    parser.add_argument("--text", default="The quick brown fox jumps over the lazy dog.", help="Text to analyze")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--decay_rate", type=float, default=1.0, help="Decay rate for LocalityGPT2")
    args = parser.parse_args()

    print(f"Loading model: {args.model} (Untrained: {args.untrained})")
    device = args.device
    
    model = None
    
    if "locality_gpt" in args.model:
        if LocalityGPT2 is None:
            raise ImportError("Could not import LocalityGPT2. Make sure models.locality_gpt is accessible.")
        
        print("Initializing LocalityGPT2...")
        base_model_id = "gpt2"
        config = AutoConfig.from_pretrained(base_model_id)
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
        # Tokenizer is also available as subject.tokenizer but we need model for stats
        tokenizer = subject.tokenizer
    else:
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
