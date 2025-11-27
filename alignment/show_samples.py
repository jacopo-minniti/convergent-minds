import argparse
import pandas as pd
import os
import sys

# Add project root to sys.path to allow importing brainscore
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brainscore import load_benchmark

def main():
    parser = argparse.ArgumentParser(description="Show samples from the benchmark")
    parser.add_argument("--benchmark", default="Pereira2018.384sentences-linear", help="Benchmark identifier")
    parser.add_argument("--input_dir", default=None, help="Directory containing benchmark_examples.csv (optional)")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to show")
    args = parser.parse_args()

    df = None
    
    # Try to load from input_dir if provided
    if args.input_dir:
        csv_path = os.path.join(args.input_dir, "benchmark_examples.csv")
        if os.path.exists(csv_path):
            print(f"Loading samples from {csv_path}")
            df = pd.read_csv(csv_path)
        else:
            print(f"File not found: {csv_path}")

    # Fallback to loading from BrainScore
    if df is None:
        print(f"Loading benchmark: {args.benchmark}")
        try:
            benchmark = load_benchmark(args.benchmark)
            if hasattr(benchmark, 'stimulus_set'):
                df = benchmark.stimulus_set
            elif hasattr(benchmark, '_stimulus_set'):
                df = benchmark._stimulus_set
            else:
                print("Could not find stimulus_set in benchmark.")
        except Exception as e:
            print(f"Failed to load benchmark: {e}")

    if df is not None:
        print(f"\n--- Benchmark Samples ({args.benchmark}) ---")
        # Check for common columns
        cols_to_show = []
        for col in ['sentence', 'stimulus', 'word', 'text']:
            if col in df.columns:
                cols_to_show.append(col)
        
        if not cols_to_show:
            cols_to_show = df.columns[:5] # Fallback to first 5 columns

        print(df[cols_to_show].head(args.num_samples))
        print(f"\nTotal samples: {len(df)}")
    else:
        print("No samples available.")

if __name__ == "__main__":
    main()
