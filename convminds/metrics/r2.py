from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class DecoderScore:
    value: float
    per_output: np.ndarray
    sse: np.ndarray
    sst: np.ndarray
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "per_output": self.per_output.tolist(),
            "sse": self.sse.tolist(),
            "sst": self.sst.tolist(),
            "metadata": dict(self.metadata),
        }

    def __str__(self) -> str:
        return f"R^2 = {self.value:.6f}"


def linear_r2(decoder, brain_test: Any, llm_test: Any) -> DecoderScore:
    target = np.asarray(llm_test, dtype=float)
    if target.ndim != 2:
        raise ValueError("linear_r2 expects a 2D LLM target matrix.")

    prediction = np.asarray(decoder.predict(brain_test), dtype=float)
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction shape {prediction.shape} does not match target shape {target.shape}."
        )

    target_mean = target.mean(axis=0)
    sst = ((target - target_mean) ** 2).sum(axis=0)
    sst = np.where(sst == 0.0, 1e-12, sst)
    sse = ((target - prediction) ** 2).sum(axis=0)
    per_output = 1.0 - (sse / sst)
    return DecoderScore(
        value=float(np.mean(per_output)),
        per_output=per_output,
        sse=sse,
        sst=sst,
        metadata={"metric": "linear_r2"},
    )
