from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Decoder(ABC):
    @abstractmethod
    def decoder_config(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def train(self, brain_train: Any, llm_train: Any):
        raise NotImplementedError

    @abstractmethod
    def predict(self, brain_values: Any) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError
