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
        # 4.1 Normalization of features
        # Standardize using the training partition only to avoid leakage.
        scaler_obj = StandardScaler()
        X_obj_train = scaler_obj.fit_transform(X_obj[train_idx])
        X_obj_test = scaler_obj.transform(X_obj[test_idx])

        scaler_llm = StandardScaler()
        X_llm_train = scaler_llm.fit_transform(X_llm[train_idx])
        X_llm_test = scaler_llm.transform(X_llm[test_idx])

        # Build baseline design matrix
        Xb_train = X_obj_train
        Xb_test  = X_obj_test
        
        y_train = y[train_idx]
        y_test  = y[test_idx]
        
        # Fit baseline
        model_baseline = RidgeCV(alphas=alpha_grid)
        model_baseline.fit(Xb_train, y_train)
        y_pred_baseline_train = model_baseline.predict(Xb_train)
        y_pred_baseline_test = model_baseline.predict(Xb_test)
        
        if hasattr(model_baseline, 'alpha_'):
            alphas_baseline.append(model_baseline.alpha_)
        
        # Fit LLM on residuals from baseline (i.e., partial out objective features)
        residual_train = y_train - y_pred_baseline_train
        model_llm = RidgeCV(alphas=alpha_grid)
        model_llm.fit(X_llm_train, residual_train)
        residual_pred_test = model_llm.predict(X_llm_test)
        
        if hasattr(model_llm, 'alpha_'):
            alphas_llm.append(model_llm.alpha_)
        
        # Combined prediction = baseline + LLM residual prediction
        y_pred_combined = y_pred_baseline_test + residual_pred_test
        
        # 4.3 R² computation per neuroid
        # Compute R² manually to match the spec:
        # SST_j = sum_i (y_test_j[i] - mean(y_train_j))²
        # Note: mean is from training data!
        
        y_train_mean = np.mean(y_train, axis=0)
        
        # Baseline R²
        sse_baseline = np.sum((y_test - y_pred_baseline_test)**2, axis=0)
        sst_baseline = np.sum((y_test - y_train_mean)**2, axis=0)
        # Avoid division by zero
        sst_baseline[sst_baseline == 0] = 1e-10
        r2_baseline = 1 - sse_baseline / sst_baseline
        
        # Combined R²
        sse_combined = np.sum((y_test - y_pred_combined)**2, axis=0)
        sst_combined = np.sum((y_test - y_train_mean)**2, axis=0) # Same SST
        sst_combined[sst_combined == 0] = 1e-10
        r2_combined = 1 - sse_combined / sst_combined
        
        # ΔR²
        delta_r2 = r2_combined - r2_baseline
        
        r2_baseline_splits.append(r2_baseline)
        r2_combined_splits.append(r2_combined)
        delta_r2_splits.append(delta_r2)
        
        if split_idx == 0:
            print(f"Split 0: Median R2 Baseline = {np.median(r2_baseline):.4f}")
            print(f"Split 0: Median R2 Combined = {np.median(r2_combined):.4f}")
            print(f"Split 0: Median Delta R2 = {np.median(delta_r2):.4f}")
            print(f"Split 0: Max R2 Combined = {np.max(r2_combined):.4f}")
        
    # 4.4 Aggregation
    # Median across neuroids per split
    score_per_split = [np.median(d) for d in delta_r2_splits]
    
    # Mean across splits
    alignment_score = np.mean(score_per_split)
    
    # Calculate aggregated variances
    # Median across neuroids per split, then mean across splits
    r2_baseline_per_split = [np.median(d) for d in r2_baseline_splits]
    objective_explained_variance = np.mean(r2_baseline_per_split)
    
    r2_combined_per_split = [np.median(d) for d in r2_combined_splits]
    obj_llm_explained_variance = np.mean(r2_combined_per_split)
    
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
