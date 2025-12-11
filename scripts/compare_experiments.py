import argparse
import json
import scipy.stats as stats
import numpy as np
import sys
import os


def load_scores(dir_path):
    # Try loading detailed scores first
    detailed_path = os.path.join(dir_path, "detailed_scores.json")
    if os.path.exists(detailed_path):
        try:
            with open(detailed_path, 'r') as f:
                data = json.load(f)
            return data, 'detailed'
        except Exception as e:
            print(f"Warning: Found detailed_scores.json at {detailed_path} but failed to load: {e}")

    # Fallback to info.json (aggregated per seed)
    info_path = os.path.join(dir_path, "info.json")
    if os.path.exists(info_path):
        try:
            with open(info_path, 'r') as f:
                data = json.load(f)
            if 'scores' in data and 'scores_per_seed' in data['scores']:
               return data['scores']['scores_per_seed'], 'aggregated'
        except Exception as e:
            print(f"Error loading info.json at {info_path}: {e}")
            
    print(f"Error: Could not load valid scores from {dir_path}")
    return None, None

def main():
    parser = argparse.ArgumentParser(description="Compare two experiments using statistical significance tests.")
    parser.add_argument("path_a", help="Path to experiment directory A")
    parser.add_argument("path_b", help="Path to experiment directory B")
    parser.add_argument("--metric", default="partial", help="Metric to compare (partial, objective_corr, llm_corr). Default: partial")
    args = parser.parse_args()
    
    # Paths might be the directory or the info.json file. Normalize to directory.
    dir_a = os.path.dirname(args.path_a) if args.path_a.endswith('.json') else args.path_a
    dir_b = os.path.dirname(args.path_b) if args.path_b.endswith('.json') else args.path_b

    scores_a, type_a = load_scores(dir_a)
    scores_b, type_b = load_scores(dir_b)
    
    if scores_a is None or scores_b is None:
        sys.exit(1)
        
    print(f"Loaded Experiment A: {type_a} data")
    print(f"Loaded Experiment B: {type_b} data")

    # Helper to flatten list of lists
    def flatten(lob):
        if not lob: return []
        if isinstance(lob[0], list):
            return [item for sublist in lob for item in sublist]
        return lob

    # Extract data based on type
    def get_data(scores, data_type, metric):
        raw_pooled = [] # All samples (folds * seeds)
        means_seed = [] # One mean per seed
        
        if data_type == 'detailed':
            # Map metric names for detailed_scores.json keys
            key_map = {
                "partial": "partial",
                "objective_corr": "objective_corr",
                "llm_corr": "llm_corr",
                "objective_var": "objective_var",
                "llm_only_var": "llm_only_var",
                 "joint_var": "joint_var"
            }
            key = key_map.get(metric, metric)
            
            if key not in scores['raw_scores']:
                 print(f"Metric {metric} not found in detailed scores.")
                 sys.exit(1)
            
            # raw_scores[key] is a list of lists (seeds -> folds)
            list_of_lists = scores['raw_scores'][key]
            
            # Flatten for pooled analysis
            raw_pooled = flatten(list_of_lists)
            
            # Calculate means per seed for conservative analysis
            # Handle cases where inner list might be empty or None
            # Filter out None/empty first
            valid_lists = [l for l in list_of_lists if l]
            means_seed = [float(np.mean(l)) for l in valid_lists]

        elif data_type == 'aggregated':
            # Map metric names for info.json structure
            # structure: scores_per_seed -> category -> key -> list of values
            # logic from original script
            if metric == "partial":
                vals = scores['explained_variance']['partial']
            elif metric == "objective_corr":
                vals = scores['correlation']['objective']
            elif metric == "llm_corr":
                vals = scores['correlation']['llm']
            else:
                # Try to find it generically or fail
                 print(f"Metric {metric} not supported for legacy aggregated files.")
                 sys.exit(1)
            
            means_seed = [x for x in vals if x is not None]
            raw_pooled = means_seed # For aggregated, pooled is just the seeds
            
        return np.array(raw_pooled), np.array(means_seed)

    pool_a, seed_a = get_data(scores_a, type_a, args.metric)
    pool_b, seed_b = get_data(scores_b, type_b, args.metric)
    
    print(f"\nAnalyzing Metric: {args.metric}")
    print(f"{'-'*40}")
    
    # 1. Conservative Test (Seed-level means)
    print(f"Test 1: Conservative (N = Number of Seeds)")
    print(f"A: Mean={np.mean(seed_a):.4f}, Std={np.std(seed_a):.4f}, N={len(seed_a)}")
    print(f"B: Mean={np.mean(seed_b):.4f}, Std={np.std(seed_b):.4f}, N={len(seed_b)}")
    
    if len(seed_a) < 2 or len(seed_b) < 2:
        print("-> Not enough seeds for t-test.")
    else:
        # Paired if same N, else Welch
        if len(seed_a) == len(seed_b):
            stat, p = stats.ttest_rel(seed_a, seed_b)
            test_name = "Paired t-test"
        else:
            stat, p = stats.ttest_ind(seed_a, seed_b, equal_var=False)
            test_name = "Welch's t-test"
        print(f"-> {test_name}: t={stat:.4f}, p={p:.4e} {'*' if p<0.05 else ''}")

    print(f"{'-'*40}")

    # 2. Fine-grained / Pooled Test (All valid CV folds)
    # Only if we actually have more data than seeds
    if len(pool_a) > len(seed_a) or len(pool_b) > len(seed_b):
        print(f"Test 2: Fine-grained / Pooled (N = Total Folds)")
        print(f"A: Mean={np.mean(pool_a):.4f}, Std={np.std(pool_a):.4f}, N={len(pool_a)}")
        print(f"B: Mean={np.mean(pool_b):.4f}, Std={np.std(pool_b):.4f}, N={len(pool_b)}")
        
        # We generally treat these as independent samples unless we strictly track fold-to-fold identity,
        # which is harder to guarantee across arbitrary experiments.
        # But if they are from the same seeds and folds, technically they are paired-ish.
        # However, simple Welch's t-test on the pool is a common approach for "significance given all data".
        # CAUTION: This inflates p-values if samples within a seed are correlated. 
        # But this is what the user requested ("take into account both seed runs AND the actual you know many samples").
        
        stat, p = stats.ttest_ind(pool_a, pool_b, equal_var=False)
        print(f"-> Welch's t-test: t={stat:.4f}, p={p:.4e} {'*' if p<0.05 else ''}")
    else:
        print("detailed_scores.json not available or no extra internal folds found.")
        print("Skipping fine-grained test.")

if __name__ == "__main__":
    main()

