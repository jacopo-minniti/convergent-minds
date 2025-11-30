import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from brainscore import load_benchmark



def main():
    parser = argparse.ArgumentParser(description="Show benchmark stimuli grouped by passage/topic")
    parser.add_argument("--benchmark", default="Pereira2018.384sentences-linear")
    parser.add_argument("--max-passages", type=int, default=5,
                        help="Maximum number of passages/topics to display")
    parser.add_argument("--max-sentences", type=int, default=None,
                        help="Maximum sentences per passage to display (default: all)")
    args = parser.parse_args()

    # Load benchmark
    print(f"Loading benchmark: {args.benchmark}")
    benchmark = load_benchmark(args.benchmark)

    # Extract the stimulus coordinate from benchmark.data (NeuroidAssembly)
    assembly = benchmark.data
    stimuli = assembly['stimulus'].values
    passage_labels = assembly['passage_label'].values if 'passage_label' in assembly.coords else None
    passage_categories = assembly['passage_category'].values if 'passage_category' in assembly.coords else None

    # Group sentences by passage label while preserving order
    grouped = {}
    for i, stim in enumerate(stimuli):
        passage = passage_labels[i] if passage_labels is not None else "UNKNOWN_PASSAGE"
        grouped.setdefault(passage, []).append((i, stim))

    total_passages = len(grouped)
    print(f"\n--- Stimuli grouped by passage/topic (showing up to {args.max_passages}) ---")
    for passage_idx, (passage, items) in enumerate(grouped.items()):
        if passage_idx >= args.max_passages:
            break
        category = passage_categories[items[0][0]] if passage_categories is not None else None
        header_extra = f" | category: {category}" if category is not None else ""
        print(f"\nPassage {passage_idx} ({passage}){header_extra}:")
        for sent_idx, (global_idx, sentence) in enumerate(items):
            if args.max_sentences is not None and sent_idx >= args.max_sentences:
                break
            print(f"  [{global_idx}] {sentence}")

    print(f"\nTotal passages: {total_passages}")
    print(f"Total stimuli: {len(stimuli)}")

if __name__ == "__main__":
    main()
