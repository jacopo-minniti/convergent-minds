
import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    parser = argparse.ArgumentParser(description="Plot Alignment Score vs Decay Rate")
    parser.add_argument("--data_dir", default="data/scores/localized_512", help="Path to scores directory")
    parser.add_argument("--output_dir", default="figures", help="Directory to save the plot")
    parser.add_argument("--untrained_only", action="store_true", default=True, help="Filter for untrained models only")
    args = parser.parse_args()

    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print(f"Searching for info.json files in {args.data_dir}...")
    
    # Recursive search for info.json
    pattern = os.path.join(args.data_dir, "**", "info.json")
    files = glob.glob(pattern, recursive=True)
    
    print(f"Found {len(files)} info.json files.")
    
    data = []
    
    for fpath in files:
        try:
            with open(fpath, 'r') as f:
                info = json.load(f)
            
            # Filter for untrained if requested
            is_untrained = info.get("untrained", False)
            # Some old runs might store it in args
            if "args" in info and "untrained" in info["args"]:
                is_untrained = info["args"]["untrained"]
                
            if args.untrained_only and not is_untrained:
                continue
                
            # Extract Decay Rate
            decay_rate = info.get("decay_rate")
            if decay_rate is None:
                # Try to parse from args
                if "args" in info and "decay_rate" in info["args"]:
                    decay_rate = info["args"]["decay_rate"]
            
            # If still None, check if it's locality_gpt model
            model = info.get("model", "")
            if "locality_gpt" not in model and decay_rate is None:
                # Likely standard GPT2, effectively decay 0.0 but let's skip to be safe 
                # or treat as baseline
                continue
                
            if decay_rate is None:
                 continue
                 
            # Extract Score
            # We want 'explained_variance_normalized' -> 'partial'
            try:
                scores = info.get("scores", {})
                ev_norm = scores.get("explained_variance_normalized", {})
                score = ev_norm.get("partial")
            except:
                score = None
                
            if score is not None:
                data.append({
                    "decay_rate": float(decay_rate),
                    "alignment_score": float(score),
                    "model": model,
                    "path": fpath
                })
                
        except Exception as e:
            print(f"Error reading {fpath}: {e}")

    if not data:
        print("No valid data found matching criteria.")
        return

    df = pd.DataFrame(data)
    
    print("Data Summary:")
    print(df.groupby("decay_rate")["alignment_score"].describe())

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Plot with error bars (estimator='mean', errorbar='se' is default for lineplot)
    # We use lineplot to show the trend
    sns.lineplot(data=df, x="decay_rate", y="alignment_score", marker="o", errorbar='sd')
    
    plt.title("Alignment Score vs Attention Decay Rate (Untrained)")
    plt.xlabel("Decay Rate (Negative=Anti-Local, Positive=Local)")
    plt.ylabel("Normalized Partial R2")
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label="Standard GPT2 (Approx)")
    
    # Add annotation for Locality interpretation
    plt.text(df["decay_rate"].min(), df["alignment_score"].min(), 
             " <- Long Context Integration", ha='left', va='bottom', fontsize=9, color='green')
    plt.text(df["decay_rate"].max(), df["alignment_score"].min(), 
             "Short Context Integration -> ", ha='right', va='bottom', fontsize=9, color='red')

    save_path = os.path.join(args.output_dir, "decay_alignment.png")
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    main()
