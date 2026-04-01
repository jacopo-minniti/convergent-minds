import torch
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error

def correlation(y_true: torch.Tensor | np.ndarray, y_pred: torch.Tensor | np.ndarray) -> float:
    """
    Computes the average Pearson correlation coefficient between true and predicted latents.
    Supports both Tensor and NumPy inputs.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
        
    # Flatten if multi-dimensional to get a global correlation or compute per-feature mean
    if y_true.ndim > 1:
        corrs = []
        for i in range(y_true.shape[1]):
            # Simple check to avoid constant signals (std=0) resulting in NaN
            if np.std(y_true[:, i]) > 1e-9 and np.std(y_pred[:, i]) > 1e-9:
                r, _ = pearsonr(y_true[:, i], y_pred[:, i])
                corrs.append(r)
        return float(np.mean(corrs)) if corrs else 0.0
    
    r, _ = pearsonr(y_true, y_pred)
    return float(r)

def r2(y_true: torch.Tensor | np.ndarray, y_pred: torch.Tensor | np.ndarray) -> float:
    """Computes R-squared (Coefficient of Determination)."""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    return float(r2_score(y_true, y_pred))

def mse(y_true: torch.Tensor | np.ndarray, y_pred: torch.Tensor | np.ndarray) -> float:
    """Computes Mean Squared Error."""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    return float(mean_squared_error(y_true, y_pred))
