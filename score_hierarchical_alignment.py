import os
import argparse
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import sys
# Force local import of brainscore by adding the current directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from brainscore import score, load_benchmark, ArtificialSubject
from brainscore.model_helpers.huggingface import get_layer_names
from models.hierarchical_gpt import HierarchicalGPT2

def main():
    import logging
    logging.basicConfig(level=logging.INFO)
    
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
    parser.add_argument("--topic_wise_cv", action="store_true", help="Use topic-wise cross-validation (GroupKFold)")
    parser.add_argument("--no_topic_wise_cv", dest="topic_wise_cv", action="store_false", help="Use random cross-validation (KFold)")
    parser.add_argument("--use_surprisal", action="store_true", help="Add aggregated surprisal as a feature")
    parser.set_defaults(topic_wise_cv=True)
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
    if hasattr(benchmark, 'topic_wise_cv'):
        print(f"Setting topic_wise_cv to {args.topic_wise_cv}")
        benchmark.topic_wise_cv = args.topic_wise_cv

    if args.use_surprisal:
        if hasattr(benchmark, 'use_surprisal'):
            print("Enabling use_surprisal in benchmark")
            benchmark.use_surprisal = True
        else:
            print("Warning: Benchmark does not support use_surprisal")

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
                region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: [layer_name]},
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
            
            # Extract scores
            raw_partial_score = float(score_result.values) if score_result.values.size == 1 else np.mean(score_result.values)
            normalized_partial_score = score_result.attrs.get('normalized_partial_r2')

            if normalized_partial_score is None:
                print("Warning: normalized_partial_r2 not found or None. Validation might differ.")
                partial_score = raw_partial_score # Fallback, though ideally we want normalized
            else:
                partial_score = normalized_partial_score
            
            llm_score = score_result.attrs.get('original_normalized_alignment_score')
            # If attributes are not preserved or different, we might need to look at raw attributes
            if llm_score is None:
                # Try fallback or just use 0 if not available (shouldn't happen with correct benchmark)
                llm_score = 0.0
                print("Warning: original_normalized_alignment_score not found in result attributes.")
            
            depth_scores.append({
                "partial": partial_score,
                "partial_raw": raw_partial_score,
                "llm": llm_score
            })
            print(f"    Normalized Partial Score: {partial_score:.4f}, Raw Partial Score: {raw_partial_score:.4f}, LLM Score: {llm_score:.4f}")
            
        # Average over seeds
        partials = [d['partial'] for d in depth_scores]
        llms = [d['llm'] for d in depth_scores]
        
        avg_partial = np.mean(partials)
        std_partial = np.std(partials)
        
        avg_llm = np.mean(llms)
        std_llm = np.std(llms)
        
        results_by_depth.append({
            "depth": depth,
            "partial_mean": avg_partial,
            "partial_std": std_partial,
            "llm_mean": avg_llm,
            "llm_std": std_llm,
            "raw_scores": depth_scores
        })
        print(f"  Depth {depth} Avg Partial: {avg_partial:.4f} +/- {std_partial:.4f}")
        print(f"  Depth {depth} Avg LLM: {avg_llm:.4f} +/- {std_llm:.4f}")

    # Save results
    results_df = pd.DataFrame(results_by_depth)
    csv_path = os.path.join(args.save_path, "hierarchical_scores.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    # Plot 1: Normalized Partial R2
    plt.figure(figsize=(10, 6))
    plt.errorbar(results_df["depth"], results_df["partial_mean"], yerr=results_df["partial_std"], fmt='-o', capsize=5, label='Partial R²')
    plt.xlabel("Depth (Number of Transformer Blocks)")
    plt.ylabel("Normalized Partial R²")
    plt.title(f"Hierarchical Alignment: {args.model} ({'Untrained' if args.untrained else 'Trained'})\nNormalized Partial R²")
    plt.grid(True)
    plt.xticks(args.depths)
    plt.legend()
    
    plot_path = os.path.join(args.save_path, "depth_vs_partial_r2.png")
    plt.savefig(plot_path)
    print(f"Partial R2 Plot saved to {plot_path}")
    plt.close()

    # Plot 2: Normalized LLM-only R2
    plt.figure(figsize=(10, 6))
    plt.errorbar(results_df["depth"], results_df["llm_mean"], yerr=results_df["llm_std"], fmt='-s', capsize=5, color='orange', label='LLM-Only R²')
    plt.xlabel("Depth (Number of Transformer Blocks)")
    plt.ylabel("Normalized LLM-Only R²")
    plt.title(f"Hierarchical Alignment: {args.model} ({'Untrained' if args.untrained else 'Trained'})\nNormalized LLM-Only R² (No Objective Features)")
    plt.grid(True)
    plt.xticks(args.depths)
    plt.legend()
    
    plot_path_llm = os.path.join(args.save_path, "depth_vs_llm_r2.png")
    plt.savefig(plot_path_llm)
    print(f"LLM-Only Plot saved to {plot_path_llm}")
    plt.close()

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
