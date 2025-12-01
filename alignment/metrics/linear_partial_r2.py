import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from typing import List, Tuple, Dict, Any

def linear_partial_r2(
    X_obj: np.ndarray,
    X_llm: np.ndarray,
    y: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    alpha_grid: np.ndarray = np.logspace(-3, 3, 7),
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute ΔR² brain alignment score using multi-output ridge and the given CV splits.

    Args:
        X_obj: Objective features (n_samples, n_features_obj)
        X_llm: LLM features (n_samples, n_features_llm)
        y: Neuroid responses (n_samples, n_neuroids)
        splits: List of (train_indices, test_indices)
        alpha_grid: Alphas for RidgeCV

    Returns:
        alignment_score: scalar float (final ΔR²)
        diagnostics: dict with raw per-neuroid and per-split R²_baseline, R²_combined, ΔR².
    """
    
    # 4.1 Normalization of features
    # Standardize X_obj and X_llm once on the full dataset before splitting.
    scaler_obj = StandardScaler()
    X_obj_scaled = scaler_obj.fit_transform(X_obj)
    
    scaler_llm = StandardScaler()
    X_llm_scaled = scaler_llm.fit_transform(X_llm)
    
    # Build baseline and combined design matrices
    X_baseline = X_obj_scaled
    X_combined = np.concatenate([X_obj_scaled, X_llm_scaled], axis=1)
    
    # Storage for diagnostics
    r2_baseline_splits = []
    r2_combined_splits = []
    delta_r2_splits = []
    
    for split_idx, (train_idx, test_idx) in enumerate(splits):
        # 4.2 Cross-validation and ridge fitting
        Xb_train = X_baseline[train_idx]
        Xb_test  = X_baseline[test_idx]
        Xc_train = X_combined[train_idx]
        Xc_test  = X_combined[test_idx]
        
        y_train = y[train_idx]
        y_test  = y[test_idx]
        
        # Fit baseline
        model_baseline = RidgeCV(alphas=alpha_grid)
        model_baseline.fit(Xb_train, y_train)
        y_pred_baseline = model_baseline.predict(Xb_test)
        
        # Fit combined
        model_combined = RidgeCV(alphas=alpha_grid)
        model_combined.fit(Xc_train, y_train)
        y_pred_combined = model_combined.predict(Xc_test)
        
        # 4.3 R² computation per neuroid
        # Compute R² manually to match the spec:
        # SST_j = sum_i (y_test_j[i] - mean(y_train_j))²
        # Note: mean is from training data!
        
        y_train_mean = np.mean(y_train, axis=0)
        
        # Baseline R²
        sse_baseline = np.sum((y_test - y_pred_baseline)**2, axis=0)
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
    
    diagnostics = {
        "r2_baseline_per_split_neuroid": r2_baseline_splits,
        "r2_combined_per_split_neuroid": r2_combined_splits,
        "delta_r2_per_split_neuroid": delta_r2_splits,
        "score_per_split": score_per_split,
        "objective_explained_variance": objective_explained_variance,
        "obj_llm_explained_variance": obj_llm_explained_variance
    }
    
    return float(alignment_score), diagnostics
