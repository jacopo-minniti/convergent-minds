from __future__ import annotations

import numpy as np


def R2(predictions, targets) -> float:
    preds = np.asarray(predictions, dtype=float)
    refs = np.asarray(targets, dtype=float)
    
    if preds.ndim > 2:
        preds = preds.reshape(-1, preds.shape[-1])
        refs = refs.reshape(-1, refs.shape[-1])
        
    residual = np.sum((refs - preds) ** 2, axis=0)
    total = np.sum((refs - refs.mean(axis=0, keepdims=True)) ** 2, axis=0)
    
    valid = total > 0
    r2_features = np.zeros_like(total)
    r2_features[valid] = 1.0 - (residual[valid] / total[valid])
    
    if not np.any(valid):
        return 0.0
        
    return float(np.mean(r2_features))
