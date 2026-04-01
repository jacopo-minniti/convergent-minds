from __future__ import annotations
import numpy as np

def pairwise_retrieval(predictions, targets) -> float:
    """
    Computes top-k retrieval accuracy for a batch (B, B).
    Measures how often the correct target is the closest match to the predicted vector.
    """
    preds = np.asarray(predictions, dtype=float)
    refs = np.asarray(targets, dtype=float)
    
    if preds.shape != refs.shape:
        raise ValueError("Predictions and targets must share the same shape.")
        
    def _normalize(values: np.ndarray) -> np.ndarray:
        denom = np.linalg.norm(values, axis=1, keepdims=True)
        denom = np.where(denom == 0, 1.0, denom)
        return values / denom

    # Standard cosine similarity check
    preds = _normalize(preds)
    refs = _normalize(refs)
    similarity = preds @ refs.T 
    
    # Correct if the diagonal contains the max value per row
    correct = np.argmax(similarity, axis=1) == np.arange(similarity.shape[0])
    return float(np.mean(correct))
