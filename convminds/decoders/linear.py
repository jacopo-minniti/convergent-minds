from __future__ import annotations

from typing import Any

import numpy as np

from convminds.decoders.base import Decoder


class LinearDecoder(Decoder):
    def __init__(
        self,
        *,
        loss: str = "mse",
        l2_penalty: float = 0.0,
        fit_intercept: bool = True,
    ) -> None:
        if loss != "mse":
            raise ValueError("LinearDecoder currently supports only loss='mse'.")
        self.loss = loss
        self.l2_penalty = float(l2_penalty)
        self.fit_intercept = fit_intercept
        self.coef_: np.ndarray | None = None
        self.intercept_: np.ndarray | None = None

    def decoder_config(self) -> dict[str, Any]:
        return {
            "decoder": "linear",
            "loss": self.loss,
            "l2_penalty": self.l2_penalty,
            "fit_intercept": self.fit_intercept,
        }

    def reset(self):
        self.coef_ = None
        self.intercept_ = None
        return self

    def train(self, brain_train: Any, llm_train: Any):
        x = np.asarray(brain_train, dtype=float)
        y = np.asarray(llm_train, dtype=float)
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError("LinearDecoder expects 2D train matrices.")
        if x.shape[0] != y.shape[0]:
            raise ValueError("brain_train and llm_train must have the same number of rows.")

        if self.fit_intercept:
            x_mean = x.mean(axis=0, keepdims=True)
            y_mean = y.mean(axis=0, keepdims=True)
            x_centered = x - x_mean
            y_centered = y - y_mean
        else:
            x_mean = np.zeros((1, x.shape[1]), dtype=float)
            y_mean = np.zeros((1, y.shape[1]), dtype=float)
            x_centered = x
            y_centered = y

        xtx = x_centered.T @ x_centered
        if self.l2_penalty > 0:
            xtx = xtx + self.l2_penalty * np.eye(xtx.shape[0], dtype=float)

        xty = x_centered.T @ y_centered
        try:
            coef = np.linalg.solve(xtx, xty)
        except np.linalg.LinAlgError:
            coef = np.linalg.pinv(xtx) @ xty

        intercept = (y_mean - x_mean @ coef).reshape(-1)
        self.coef_ = coef
        self.intercept_ = intercept
        return self

    def predict(self, brain_values: Any) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("LinearDecoder.predict called before train.")
        x = np.asarray(brain_values, dtype=float)
        if x.ndim != 2:
            raise ValueError("LinearDecoder expects a 2D matrix for prediction.")
        return x @ self.coef_ + self.intercept_
