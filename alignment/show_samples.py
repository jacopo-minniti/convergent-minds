import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from brainscore import load_benchmark



def main():
    parser = argparse.ArgumentParser(description="Show first stimuli from the benchmark")
    parser.add_argument("--benchmark", default="Pereira2018.384sentences-linear")
    parser.add_argument("--num_samples", type=int, default=20)
    args = parser.parse_args()

    # Load benchmark
    print(f"Loading benchmark: {args.benchmark}")
    benchmark = load_benchmark(args.benchmark)

    # Extract the stimulus coordinate from benchmark.data (NeuroidAssembly)
    assembly = benchmark.data
    stimuli = assembly['stimulus'].values

    # Print first N strings
    print("\n--- Stimuli ---")
    for i, stim in enumerate(stimuli[:args.num_samples]):
        print(f"{i}: {stim}")

    print(f"\nTotal stimuli: {len(stimuli)}")

if __name__ == "__main__":
    main()
