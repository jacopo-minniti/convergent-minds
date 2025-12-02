import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

def linear_partial_r2(
    X_obj: np.ndarray,
    X_llm: np.ndarray,
    y: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    alpha_grid: np.ndarray = np.logspace(0, 5, 6),
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute ΔR² brain alignment score using multi-output ridge and the given CV splits.

    Args:
        X_obj: Objective features (n_samples, n_features_obj)
        X_llm: LLM features (n_samples, n_features_llm)
        y: Neuroid responses (n_samples, n_neuroids)
        splits: List of (train_indices, test_indices)
        alpha_grid: Alphas for RidgeCV. Defaults to stronger regularization [1, 10, ..., 100000].

    Returns:
        alignment_score: scalar float (final ΔR²)
        diagnostics: dict with raw per-neuroid and per-split R²_baseline, R²_combined, ΔR².
    """
    
    # Storage for diagnostics
    r2_baseline_splits = []
    r2_combined_splits = []
    delta_r2_splits = []
    
    # Debugging: track alphas
    alphas_baseline = []
    alphas_llm = []
    
    print(f"linear_partial_r2 input shapes: X_obj={X_obj.shape}, X_llm={X_llm.shape}, y={y.shape}")
    print(f"X_obj stats: mean={np.mean(X_obj):.4f}, std={np.std(X_obj):.4f}, min={np.min(X_obj):.4f}, max={np.max(X_obj):.4f}")
    print(f"X_llm stats: mean={np.mean(X_llm):.4f}, std={np.std(X_llm):.4f}, min={np.min(X_llm):.4f}, max={np.max(X_llm):.4f}")
    print(f"y stats: mean={np.mean(y):.4f}, std={np.std(y):.4f}, min={np.min(y):.4f}, max={np.max(y):.4f}")
    
    if np.isnan(X_obj).any():
        print("X_obj contains NaNs!")
    if np.isnan(X_llm).any():
        print("X_llm contains NaNs!")
    if np.isnan(y).any():
        print("y contains NaNs!")
    
    for split_idx, (train_idx, test_idx) in enumerate(tqdm(splits, desc="Partial R2 CV")):
        # ... (rest of loop) ...
        
        if split_idx == 0:
            print(f"Split 0: Median R2 Baseline = {np.median(r2_baseline):.4f}")
            print(f"Split 0: Median R2 Combined = {np.median(r2_combined):.4f}")
            print(f"Split 0: Median Delta R2 = {np.median(delta_r2):.4f}")
            print(f"Split 0: Max R2 Combined = {np.max(r2_combined):.4f}")

    # ... (aggregation) ...

    # Log alpha stats
    if alphas_baseline:
        avg_alpha_b = np.mean(alphas_baseline)
        print(f"Average Baseline Alpha: {avg_alpha_b}")
    if alphas_llm:
        avg_alpha_l = np.mean(alphas_llm)
        print(f"Average LLM Alpha: {avg_alpha_l}")

    diagnostics = {
        "r2_baseline_per_split_neuroid": r2_baseline_splits,
        "r2_combined_per_split_neuroid": r2_combined_splits,
        "delta_r2_per_split_neuroid": delta_r2_splits,
        "score_per_split": score_per_split,
        "objective_explained_variance": objective_explained_variance,
        "obj_llm_explained_variance": obj_llm_explained_variance,
        "alphas_baseline": alphas_baseline,
        "alphas_llm": alphas_llm
    }
    
    return float(alignment_score), diagnostics
