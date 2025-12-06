import argparse
import json
import scipy.stats as stats
import numpy as np
import sys
import os

def load_scores(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Check if scores_per_seed exists
    if 'scores_per_seed' not in data['scores']:
        print(f"Error: {json_path} does not contain 'scores_per_seed'. Please re-run the experiment with the updated script.")
        return None
        
    return data['scores']['scores_per_seed']

def main():
    parser = argparse.ArgumentParser(description="Compare two experiments using statistical significance tests.")
    parser.add_argument("path_a", help="Path to info.json for Experiment A")
    parser.add_argument("path_b", help="Path to info.json for Experiment B")
    parser.add_argument("--metric", default="partial", help="Metric to compare (partial, objective_corr, llm_corr). Default: partial")
    args = parser.parse_args()
    
    scores_a = load_scores(args.path_a)
    scores_b = load_scores(args.path_b)
    
    if scores_a is None or scores_b is None:
        sys.exit(1)
        
    # Extract the requested metric data
    # Structure: scores_per_seed -> category -> key -> list of values
    def get_data(scores, metric):
        if metric == "partial":
            return scores['explained_variance']['partial']
        elif metric == "objective_corr":
            return scores['correlation']['objective']
        elif metric == "llm_corr":
            return scores['correlation']['llm']
        else:
            print(f"Unknown metric: {metric}")
            sys.exit(1)

    data_a = get_data(scores_a, args.metric)
    data_b = get_data(scores_b, args.metric)
    
    # Filter out Nones just in case
    data_a = [x for x in data_a if x is not None]
    data_b = [x for x in data_b if x is not None]
    
    if len(data_a) < 2 or len(data_b) < 2:
        print("Not enough data points for t-test.")
        sys.exit(1)
        
    print(f"\nComparing Metric: {args.metric}")
    print(f"Experiment A: {os.path.basename(os.path.dirname(args.path_a))}")
    print(f"Mean: {np.mean(data_a):.4f}, Std: {np.std(data_a):.4f}, N: {len(data_a)}")
    
    print(f"Experiment B: {os.path.basename(os.path.dirname(args.path_b))}")
    print(f"Mean: {np.mean(data_b):.4f}, Std: {np.std(data_b):.4f}, N: {len(data_b)}")
    
    # Paired t-test if possible (and desired - usually better if seeds are aligned)
    # Assuming seeds are [0, 1, 2, 3, 4] for both
    if len(data_a) == len(data_b):
        stat, p_val = stats.ttest_rel(data_a, data_b)
        test_type = "Paired t-test"
    else:
        stat, p_val = stats.ttest_ind(data_a, data_b, equal_var=False) # Welch's t-test
        test_type = "Independent t-test (Welch's)"
        
    print(f"\n{test_type}:")
    print(f"Statistic: {stat:.4f}")
    print(f"P-value: {p_val:.4e}")
    
    if p_val < 0.05:
        print("Result: SIGNIFICANT difference (p < 0.05)")
    else:
        print("Result: NO significant difference (p >= 0.05)")

if __name__ == "__main__":
    main()
