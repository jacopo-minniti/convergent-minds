from __future__ import annotations

import numpy as np


def PairwiseRetrieval(predictions, targets, metric: str = "cosine") -> float:
    preds = np.asarray(predictions, dtype=float)
    refs = np.asarray(targets, dtype=float)
    if preds.shape != refs.shape:
        raise ValueError("Predictions and targets must share the same shape.")
        
    def _normalize(values: np.ndarray) -> np.ndarray:
        denom = np.linalg.norm(values, axis=1, keepdims=True)
        denom = np.where(denom == 0, 1.0, denom)
        return values / denom

    preds = _normalize(preds)
    refs = _normalize(refs)
    similarity = preds @ refs.T
    correct = np.argmax(similarity, axis=1) == np.arange(similarity.shape[0])
    return float(np.mean(correct))
