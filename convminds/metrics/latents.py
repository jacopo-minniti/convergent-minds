from __future__ import annotations

import numpy as np


class PairwiseRetrieval:
    def __init__(self, metric: str = "cosine"):
        self.metric = metric

    def compute(self, predictions, targets) -> float:
        preds = np.asarray(predictions, dtype=float)
        refs = np.asarray(targets, dtype=float)
        if preds.shape != refs.shape:
            raise ValueError("Predictions and targets must share the same shape.")
        preds = self._normalize(preds)
        refs = self._normalize(refs)
        similarity = preds @ refs.T
        correct = np.argmax(similarity, axis=1) == np.arange(similarity.shape[0])
        return float(np.mean(correct))

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        denom = np.linalg.norm(values, axis=1, keepdims=True)
        denom = np.where(denom == 0, 1.0, denom)
        return values / denom
