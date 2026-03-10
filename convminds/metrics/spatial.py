from __future__ import annotations

import numpy as np


class R2:
    def compute(self, predictions, targets) -> float:
        preds = np.asarray(predictions, dtype=float)
        refs = np.asarray(targets, dtype=float)
        residual = np.sum((refs - preds) ** 2)
        total = np.sum((refs - refs.mean()) ** 2)
        if total == 0:
            return 0.0
        return 1.0 - (residual / total)
