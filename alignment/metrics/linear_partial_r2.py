import numpy as np
from sklearn.linear_model import RidgeCV, Ridge
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
    alpha_grid: np.ndarray = np.logspace(-1, 5, 7),
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
    r2_llm_splits = [] # New: LLM-only R2
    delta_r2_splits = []
    # Training splits storage
    r2_baseline_train_splits = []
    r2_combined_train_splits = []
    delta_r2_train_splits = []
    pearson_r_splits = [] # New: for LLM-only Pearson r
    pearson_r_objective_splits = [] # New: for Objective-only Pearson r
    
    # Debugging: track alphas
    alphas_baseline = []
    alphas_llm = []
    alphas_combined = []
    
    logger.info(f"linear_partial_r2 input shapes: X_obj={X_obj.shape}, X_llm={X_llm.shape}, y={y.shape}")
    logger.info(f"X_obj stats: mean={np.mean(X_obj):.4f}, std={np.std(X_obj):.4f}, min={np.min(X_obj):.4f}, max={np.max(X_obj):.4f}")
    logger.info(f"X_llm stats: mean={np.mean(X_llm):.4f}, std={np.std(X_llm):.4f}, min={np.min(X_llm):.4f}, max={np.max(X_llm):.4f}")
    logger.info(f"y stats: mean={np.mean(y):.4f}, std={np.std(y):.4f}, min={np.min(y):.4f}, max={np.max(y):.4f}")
    
    if np.isnan(X_obj).any():
        logger.error("X_obj contains NaNs!")
    if np.isnan(X_llm).any():
        logger.error("X_llm contains NaNs!")
    if np.isnan(y).any():
        logger.error("y contains NaNs!")
    
    for split_idx, (train_idx, test_idx) in enumerate(tqdm(splits, desc="Partial R2 CV")):
        # 1. Data splitting
        X_obj_train, X_obj_test = X_obj[train_idx], X_obj[test_idx]
        X_llm_train, X_llm_test = X_llm[train_idx], X_llm[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 2. Scaling
        scaler_obj = StandardScaler()
        X_obj_train_scaled = scaler_obj.fit_transform(X_obj_train)
        X_obj_test_scaled = scaler_obj.transform(X_obj_test)

        scaler_llm = StandardScaler()
        X_llm_train_scaled = scaler_llm.fit_transform(X_llm_train)
        X_llm_test_scaled = scaler_llm.transform(X_llm_test)
        
        # Combined features for the full model
        X_combined_train_scaled = np.concatenate((X_obj_train_scaled, X_llm_train_scaled), axis=1)
        X_combined_test_scaled = np.concatenate((X_obj_test_scaled, X_llm_test_scaled), axis=1)

        # 3. Model training and prediction
        # 3.1 Baseline model (objective features only)
        model_baseline = RidgeCV(alphas=alpha_grid)
        model_baseline.fit(X_obj_train_scaled, y_train)
        y_pred_baseline_test = model_baseline.predict(X_obj_test_scaled)
        if hasattr(model_baseline, 'alpha_'):
            alphas_baseline.append(model_baseline.alpha_)

        # 3.2 Combined model (objective + LLM features)
        model_combined = RidgeCV(alphas=alpha_grid)
        model_combined.fit(X_combined_train_scaled, y_train)
        y_pred_combined_test = model_combined.predict(X_combined_test_scaled)
        if hasattr(model_combined, 'alpha_'):
            alphas_combined.append(model_combined.alpha_)
            best_alpha_combined = model_combined.alpha_
        else:
            best_alpha_combined = 1.0 # Fallback should not happen with RidgeCV

        # Use Joint Model predictions for combined performance
        y_pred_combined = y_pred_combined_test
        
        # --- New: LLM Only Correlation ---
        model_llm_only = RidgeCV(alphas=alpha_grid)
        model_llm_only.fit(X_llm_train_scaled, y_train)
        y_pred_llm_only = model_llm_only.predict(X_llm_test_scaled)
        if hasattr(model_llm_only, 'alpha_'):
            alphas_llm.append(model_llm_only.alpha_)
        
        # Pearson r per neuroid
        # Centering
        y_test_c = y_test - np.mean(y_test, axis=0)
        y_pred_c = y_pred_llm_only - np.mean(y_pred_llm_only, axis=0)
        
        # Norms
        y_test_norm = np.sqrt(np.sum(y_test_c**2, axis=0))
        y_pred_norm = np.sqrt(np.sum(y_pred_c**2, axis=0))
        
        # Avoid div by zero
        y_test_norm[y_test_norm == 0] = 1e-10
        y_pred_norm[y_pred_norm == 0] = 1e-10
        
        pearson_r = np.sum(y_test_c * y_pred_c, axis=0) / (y_test_norm * y_pred_norm)
        pearson_r_splits.append(pearson_r)
        
        # --- New: LLM Only R2 ---
        sse_llm = np.sum((y_test - y_pred_llm_only)**2, axis=0)
        # sst_test is already calculated below (using test mean)
        # We need to ensure we use the same SST for consistency.
        # Let's move SST calculation up or duplicate it for clarity.
        
        # Compute SST using Test Mean (Standard R2)
        y_test_mean = np.mean(y_test, axis=0)
        sst_test = np.sum((y_test - y_test_mean)**2, axis=0)
        sst_test[sst_test == 0] = 1e-10
        
        r2_llm = 1 - sse_llm / sst_test
        # We need a storage for this
        # But wait, local variable scope.
        # I'll initializing lists at the top in a separate change or just append to diagnostics at the end
        # forcing me to edit the top too.
        # For now, let's keep it local and append to a new list 'r2_llm_splits' which I will define via multi-replace or just assume I edit top next.
        # I'll assume I will add `r2_llm_splits = []` at the top in the next step.
        # Just putting the calc here.
        # ---------------------------------
        
        # --- New: Objective Only Correlation ---
        # Centering
        y_pred_baseline_c = y_pred_baseline_test - np.mean(y_pred_baseline_test, axis=0)
        y_pred_baseline_norm = np.sqrt(np.sum(y_pred_baseline_c**2, axis=0))
        y_pred_baseline_norm[y_pred_baseline_norm == 0] = 1e-10
        
        pearson_r_obj = np.sum(y_test_c * y_pred_baseline_c, axis=0) / (y_test_norm * y_pred_baseline_norm)
        pearson_r_objective_splits.append(pearson_r_obj)
        # ---------------------------------------

        # 4.3 R² computation per neuroid
        # SST is now calculated above (sst_test)
        
        # Baseline R² (Test)
        sse_baseline = np.sum((y_test - y_pred_baseline_test)**2, axis=0)
        # sst_test calculated above
        r2_baseline = 1 - sse_baseline / sst_test
        
        # Combined R² (Test)
        sse_combined = np.sum((y_test - y_pred_combined)**2, axis=0)
        r2_combined = 1 - sse_combined / sst_test
        
        # ΔR²
        delta_r2 = r2_combined - r2_baseline

        
        # [DEBUG] Training R²
        # Baseline Train
        y_train_mean = np.mean(y_train, axis=0)
        y_pred_baseline_train = model_baseline.predict(X_obj_train_scaled)
        sse_baseline_train = np.sum((y_train - y_pred_baseline_train)**2, axis=0)
        sst_train = np.sum((y_train - y_train_mean)**2, axis=0)
        sst_train[sst_train == 0] = 1e-10
        r2_baseline_train = 1 - sse_baseline_train / sst_train
        r2_baseline_train_splits.append(r2_baseline_train)
        
        # Combined Train
        y_pred_combined_train = model_combined.predict(X_combined_train_scaled)
        sse_combined_train = np.sum((y_train - y_pred_combined_train)**2, axis=0)
        r2_combined_train = 1 - sse_combined_train / sst_train
        r2_combined_train_splits.append(r2_combined_train)

        delta_r2_train_splits.append(r2_combined_train - r2_baseline_train)
        
        # [DEBUG] Feature Correlation (Objective vs LLM)
        # We can check the correlation between the predictions of the two models on the test set
        # to see if they are redundant.
        # y_pred_baseline_test vs y_pred_llm_only
        
        # We need y_pred_llm_only from above (it is calculated).
        # Centered already available as y_pred_c (LLM) and y_pred_baseline_c (Objective)
        # Note: y_pred_c was centered using its own mean, y_pred_baseline_c using its own mean.
        
        # Norms
        # y_pred_norm (LLM) already calc
        # y_pred_baseline_norm (Obj) already calc
        
        feature_corr = np.sum(y_pred_baseline_c * y_pred_c, axis=0) / (y_pred_baseline_norm * y_pred_norm)
        
        feature_corr = np.sum(y_pred_baseline_c * y_pred_c, axis=0) / (y_pred_baseline_norm * y_pred_norm)
        
        r2_baseline_splits.append(r2_baseline)
        r2_combined_splits.append(r2_combined)
        r2_llm_splits.append(r2_llm)
        delta_r2_splits.append(delta_r2)
        
        if split_idx == 0:
            logger.info(f"Split 0: Median R2 Baseline (Test) = {np.median(r2_baseline):.4f}")
            logger.info(f"Split 0: Median R2 LLM-Only (Test) = {np.median(r2_llm):.4f}")
            logger.info(f"Split 0: Median R2 Combined (Test) = {np.median(r2_combined):.4f}")
            logger.info(f"Split 0: Median Delta R2 (Test) = {np.median(delta_r2):.4f}")
            
            logger.info(f"Split 0: Median R2 Baseline (Train) = {np.median(r2_baseline_train):.4f}")
            logger.info(f"Split 0: Median R2 Combined (Train) = {np.median(r2_combined_train):.4f}")
            logger.info(f"Split 0: Median Delta R2 (Train) = {np.median(r2_combined_train - r2_baseline_train):.4f}")
            
            logger.info(f"Split 0: Max R2 Combined (Test) = {np.max(r2_combined):.4f}")
            logger.info(f"Split 0: Median LLM-Only Pearson r = {np.median(pearson_r):.4f}")
            logger.info(f"Split 0: Median Objective Pearson r = {np.median(pearson_r_obj):.4f}")
            logger.info(f"Split 0: Median Feature Prediction Correlation = {np.median(feature_corr):.4f}")
            logger.info(f"Split 0: Median SST (Test, using Test Mean) = {np.median(sst_test):.4f}")
            
            # [DEBUG] Scale/Bias Check
            logger.info(f"Split 0: y_test Mean={np.mean(y_test):.4f}, Std={np.std(y_test):.4f}")
            logger.info(f"Split 0: Pred LLM Mean={np.mean(y_pred_llm_only):.4f}, Std={np.std(y_pred_llm_only):.4f}")
            logger.info(f"Split 0: Pred Obj Mean={np.mean(y_pred_baseline_test):.4f}, Std={np.std(y_pred_baseline_test):.4f}")

    # 4.4 Aggregation
    # Median across neuroids per split
    score_per_split = [np.median(d) for d in delta_r2_splits]


    # Mean across splits
    alignment_score = np.mean(score_per_split)
    
    # LLM Only Correlation Aggregation
    pearson_r_per_split = [np.median(d) for d in pearson_r_splits]
    original_alignment_score = np.mean(pearson_r_per_split)
    
    # Objective Only Correlation Aggregation
    pearson_r_obj_per_split = [np.median(d) for d in pearson_r_objective_splits]
    objective_alignment_score = np.mean(pearson_r_obj_per_split)
    
    # Calculate aggregated variances
    # Median across neuroids per split, then mean across splits
    r2_baseline_per_split = [np.median(d) for d in r2_baseline_splits]
    objective_explained_variance = np.mean(r2_baseline_per_split)
    
    r2_combined_per_split = [np.median(d) for d in r2_combined_splits]
    obj_llm_explained_variance = np.mean(r2_combined_per_split)

    r2_llm_per_split = [np.median(d) for d in r2_llm_splits]
    llm_explained_variance = np.mean(r2_llm_per_split)
    
    # Log alpha stats
    if alphas_baseline:
        avg_alpha_b = np.mean(alphas_baseline)
        logger.info(f"Average Baseline Alpha: {avg_alpha_b}")
    if alphas_llm:
        avg_alpha_l = np.mean(alphas_llm)
        logger.info(f"Average LLM Alpha: {avg_alpha_l}")
    if alphas_combined:
        avg_alpha_c = np.mean(alphas_combined)
        logger.info(f"Average Combined Alpha: {avg_alpha_c}")

    diagnostics = {
        "r2_baseline_per_split_neuroid": r2_baseline_splits,
        "r2_combined_per_split_neuroid": r2_combined_splits,
        "r2_baseline_train_per_split_neuroid": r2_baseline_train_splits,
        "r2_combined_train_per_split_neuroid": r2_combined_train_splits,
        "delta_r2_train_per_split_neuroid": delta_r2_train_splits,
        "delta_r2_per_split_neuroid": delta_r2_splits,
        "pearson_r_splits": pearson_r_splits, # List of arrays
        "pearson_r_objective_splits": pearson_r_objective_splits,
        "score_per_split": score_per_split,
        "objective_explained_variance": objective_explained_variance,
        "obj_llm_explained_variance": obj_llm_explained_variance,
        "llm_explained_variance": llm_explained_variance,
        "original_alignment_score": original_alignment_score,
        "objective_alignment_score": objective_alignment_score,
        "alphas_baseline": alphas_baseline,
        "alphas_llm": alphas_llm,
        "alphas_combined": alphas_combined
    }
    
    return float(alignment_score), diagnostics
