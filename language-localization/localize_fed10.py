import os
import argparse
import numpy as np
from scipy.stats import ttest_ind
from tqdm import tqdm

from utils import read_pickle, write_pickle

def localize_language_units(
    sent_activations,
    non_word_activations,
    num_units,
    ):
    """Localize language-selective units by comparing activations from sentences and non-words."""
    
    layer_names = list(sent_activations.keys())
    if not layer_names:
        return {}
        
    # Get the hidden dimension from the first layer's activations
    hidden_dim = next(iter(sent_activations.values())).shape[1]
    t_values_matrix = np.zeros((len(layer_names), hidden_dim))

    # --- T-test --- 
    print("> Performing t-tests for all units in all layers...")
    for i, layer_name in enumerate(tqdm(layer_names)):
        
        # Activations for sentences (X) and non-words (Y)
        X = sent_activations[layer_name]
        Y = non_word_activations[layer_name]

        # Perform independent t-test
        # We use 'greater' because we expect sentence activations to be higher for language-selective units
        t_stat, _ = ttest_ind(X, Y, axis=0, alternative='greater', equal_var=False)
        t_values_matrix[i] = np.nan_to_num(t_stat) # Replace NaNs with 0

    # --- Unit Selection ---
    print(f"> Selecting top {num_units} units from the entire model...")
    # Ensure we don't request more units than available
    if num_units > t_values_matrix.size:
        print(f"Warning: Requested {num_units} units, but only {t_values_matrix.size} are available. Selecting all units.")
        num_units = t_values_matrix.size
        
    # Find the indices of the top N t-statistics from the flattened matrix
    top_n_flat_indices = np.argsort(t_values_matrix.flatten())[-num_units:]
    
    # Convert flat indices back to 2D indices (layer_idx, unit_idx)
    layer_indices, unit_indices = np.unravel_index(top_n_flat_indices, t_values_matrix.shape)
    
    # Create a dictionary to store the language unit masks for each layer
    lang_unit_masks = {layer_name: [] for layer_name in layer_names}
    for layer_idx, unit_idx in zip(layer_indices, unit_indices):
        layer_name = layer_names[layer_idx]
        lang_unit_masks[layer_name].append(unit_idx)
        
    # Convert lists to numpy arrays
    for layer_name in lang_unit_masks:
        lang_unit_masks[layer_name] = np.array(lang_unit_masks[layer_name], dtype=int)
        
    return lang_unit_masks

if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Parameters for language localization')
    parser.add_argument('--model-name', type=str, default="gpt2", help='HuggingFace model name')
    parser.add_argument('--dataset-name', type=str, default="fedorenko10", help='Dataset name')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--embed-agg', type=str, default="last-token", help='Aggregation method for embeddings')
    parser.add_argument('--num-units', type=int, default=4096, help='Number of language units to select')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files')
    args = parser.parse_args()

    # --- Path Definitions ---
    # Path to the extracted representations file
    reps_path = f"language-localization/reps_model={args.model_name}_dataset={args.dataset_name}_pretrained=True_agg={args.embed_agg}_seed={args.seed}.pkl"
    # Path to save the resulting language unit mask
    mask_path = f"language-localization/l-mask_model={args.model_name}_dataset={args.dataset_name}_pretrained=True_agg={args.embed_agg}_nunits={args.num_units}_seed={args.seed}.pkl"

    # --- File Existence Checks ---
    if not os.path.exists(reps_path):
        print(f"> Activations file not found, skipping localization: {reps_path}")
        exit()

    if os.path.exists(mask_path) and not args.overwrite:
        print(f"> Already Exists: {mask_path}")
        exit()

    # --- Localization ---
    print(f"> Loading activations from: {reps_path}")
    activations = read_pickle(reps_path)

    sent_activations = activations["sentences"]
    non_word_activations = activations["non-words"]

    lang_unit_masks = localize_language_units(
        sent_activations=sent_activations,
        non_word_activations=non_word_activations,
        num_units=args.num_units,
    )

    # --- Saving Results ---
    print(f"> Saving language unit mask to: {mask_path}")
    write_pickle(mask_path, lang_unit_masks)