import os
import argparse
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from brainscore import score, load_benchmark, ArtificialSubject
from brainscore.model_helpers.huggingface import get_layer_names
from models.hierarchical_gpt import HierarchicalGPT2
import datetime
import sys

def main():
    parser = argparse.ArgumentParser(description="Hierarchical Alignment Analysis")
    parser.add_argument("--model", default="gpt2", help="Base model identifier")
    parser.add_argument("--untrained", action="store_true", help="Use an untrained version of the model")
    parser.add_argument("--localize", action="store_true", help="Perform localization before scoring")
    parser.add_argument("--num-units", type=int, default=256, help="Number of units to select during localization")
    parser.add_argument("--benchmark", default="Pereira2018.243sentences-partialr2", help="Benchmark identifier")
    parser.add_argument("--device", default="cuda", help="Device to use (cpu, cuda)")
    parser.add_argument("--save_path", default="data/scores/hierarchical_alignment", help="Directory to save results")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for localization")
    parser.add_argument("--depths", type=int, nargs='+', default=[1, 2, 3, 4, 6, 8, 12], help="Depths to evaluate")
    args = parser.parse_args()

    # Device setup
    device = args.device
    if device.isdigit():
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    elif device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'

    print(f"Output directory: {args.save_path}")
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Load benchmark once
    print(f"Loading benchmark: {args.benchmark}")
    benchmark = load_benchmark(args.benchmark)

    results_by_depth = []

    for depth in args.depths:
        print(f"\n=== Running Depth {depth} ===")
        
        # We run multiple seeds per depth to get error bars? 
        # The user said "For each depth... we re-run...". 
        # Usually we want error bars. `main.py` runs 5 seeds.
        # Let's run 5 seeds per depth.
        seeds = [0, 1, 2, 3, 4]
        depth_scores = []
        
        for seed in seeds:
            print(f"  Seed {seed}")
            # Set seed
            import random
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            
            # Instantiate Model
            # We need to map the "language_system" to the last layer of the truncated model.
            # The layers are 0-indexed. So if depth is d, the last layer is d-1.
            # Layer name format: transformer.h.{i}
            last_layer_idx = depth - 1
            layer_name = f"transformer.h.{last_layer_idx}"
            
            # Localizer kwargs
            # We need hidden_dim. We can get it from config.
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(args.model)
            hidden_dim = getattr(config, "n_embd", getattr(config, "hidden_size", 768))
            
            localizer_kwargs = {
                'top_k': args.num_units,
                'batch_size': args.batch_size,
                'hidden_dim': hidden_dim
            }
            
            subject = HierarchicalGPT2(
                model_id=args.model,
                region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: layer_name},
                depth=depth,
                untrained=args.untrained,
                use_localizer=args.localize,
                localizer_kwargs=localizer_kwargs,
                device=device
            )
            
            # Score
            score_result = score(subject, benchmark)
            
            # Extract normalized partial R2
            # Assuming the benchmark returns a Score object with 'objective_normalized_alignment_score' or similar
            # The user mentioned "normalized partial R²".
            # In `main.py`, it extracts `objective_normalized_alignment_score` and `original_normalized_alignment_score`.
            # But the main score returned by `score()` for `Pereira2018.243sentences-partialr2` IS the partial R2 (or related).
            # Let's look at `main.py` again.
            # `avg_score = np.mean([float(r.values) ...])`
            # And `avg_score` is saved as `explained_variance.partial`.
            # So the primary value of the score object is what we want.
            # But we should also check if it's normalized.
            # The user said "plot this normalized partial R²".
            # Usually BrainScore scores are normalized by ceiling.
            # Let's assume the primary value is what we want, or check for a specific attribute if needed.
            # `main.py` saves `avg_score` as `partial`.
            
            val = float(score_result.values) if score_result.values.size == 1 else np.mean(score_result.values)
            depth_scores.append(val)
            print(f"    Score: {val}")
            
        # Average over seeds
        avg_score = np.mean(depth_scores)
        std_score = np.std(depth_scores)
        results_by_depth.append({
            "depth": depth,
            "score_mean": avg_score,
            "score_std": std_score,
            "raw_scores": depth_scores
        })
        print(f"  Depth {depth} Average Score: {avg_score:.4f} +/- {std_score:.4f}")

    # Save results
    results_df = pd.DataFrame(results_by_depth)
    csv_path = os.path.join(args.save_path, "hierarchical_scores.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(results_df["depth"], results_df["score_mean"], yerr=results_df["score_std"], fmt='-o', capsize=5)
    plt.xlabel("Depth (Number of Transformer Blocks)")
    plt.ylabel("Normalized Partial R²")
    plt.title(f"Hierarchical Alignment: {args.model} ({'Untrained' if args.untrained else 'Trained'})")
    plt.grid(True)
    plt.xticks(args.depths)
    
    plot_path = os.path.join(args.save_path, "depth_vs_alignment.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

    # Save info.json
    info = {
        "model": args.model,
        "benchmark": args.benchmark,
        "untrained": args.untrained,
        "localize": args.localize,
        "num_units": args.num_units,
        "depths": args.depths,
        "timestamp": datetime.datetime.now().isoformat(),
        "command": " ".join(sys.argv),
        "results": results_by_depth
    }
    import json
    with open(os.path.join(args.save_path, "info.json"), 'w') as f:
        json.dump(info, f, indent=4)

if __name__ == "__main__":
    main()
