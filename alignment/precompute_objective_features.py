import os
import argparse
import numpy as np
import logging
from brainscore.benchmarks import load_benchmark
from alignment.objective_features import compute_objective_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Precompute objective features for a benchmark")
    parser.add_argument("--benchmark", type=str, required=True, help="Benchmark identifier (e.g. Pereira2018.243sentences-linear)")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to save the output .npy file")
    args = parser.parse_args()

    logger.info(f"Loading benchmark: {args.benchmark}")
    # Use .data as per benchmark implementation
    assembly = benchmark.data
    
    # Extract sentences and metadata
    sentences = assembly["stimulus"].values
    stimulus_ids = assembly["stimulus_id"].values
    passage_labels = assembly["passage_label"].values
    
    # We need sentence indices within passage.
    # Pereira assembly usually has 'sentence_num' or we can derive it.
    if "sentence_num" in assembly.coords:
        sentence_indices = assembly["sentence_num"].values
    else:
        logger.warning("'sentence_num' not found in coords, attempting to derive from passage_label grouping.")
        sentence_indices = []
        current_passage = None
        current_idx = 0
        for p in passage_labels:
            if p != current_passage:
                current_passage = p
                current_idx = 0
            current_idx += 1
            sentence_indices.append(current_idx)
        sentence_indices = np.array(sentence_indices)

    
    # Compute sentence counts per passage
    from collections import Counter
    passage_counts = Counter(passage_labels)
    
    logger.info(f"Computing objective features for {len(sentences)} sentences...")
    X_obj = compute_objective_features(
        sentences=sentences.tolist(),
        passage_ids=passage_labels.tolist(),
        sentence_indices=sentence_indices.tolist(),
        sentence_counts_per_passage=passage_counts
    )
    
    logger.info(f"X_obj shape: {X_obj.shape}")
    
    # Save as .npz to include stimulus_ids
    os.makedirs(args.output_dir, exist_ok=True)
    name_part = args.benchmark.lower().replace("pereira2018.", "pereira2018_").replace("sentences-linear", "")
    output_filename = f"{name_part}_obj.npz"
    output_path = os.path.join(args.output_dir, output_filename)
    
    np.savez(output_path, X_obj=X_obj, stimulus_ids=stimulus_ids)
    logger.info(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
