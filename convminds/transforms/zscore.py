from __future__ import annotations

from typing import Iterable, Sequence

import torch

from convminds.data.primitives import BrainTensor
from convminds.transforms.base import StatefulTransform


class ZScore(StatefulTransform):
    def __init__(self, dim: str | Sequence[int] = "time", eps: float = 1e-6):
        self.dim = dim
        self.eps = eps
        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None

    def fit(self, brain: BrainTensor) -> "ZScore":
        dims = self._resolve_dims(brain.signal)
        mean = brain.signal.mean(dim=dims, keepdim=True)
        std = brain.signal.std(dim=dims, keepdim=True).clamp_min(self.eps)
        self._mean = mean.detach()
        self._std = std.detach()
        return self

    def __call__(self, brain: BrainTensor) -> BrainTensor:
        if self._mean is None or self._std is None:
            raise RuntimeError("ZScore must be fit before calling.")
        mean = self._mean.to(device=brain.signal.device, dtype=brain.signal.dtype)
        std = self._std.to(device=brain.signal.device, dtype=brain.signal.dtype)
        signal = (brain.signal - mean) / std
        return BrainTensor(signal=signal, coords=brain.coords, rois=brain.rois)

    def _resolve_dims(self, signal: torch.Tensor) -> tuple[int, ...]:
        if isinstance(self.dim, str):
            mapping = {"batch": 0, "time": 1, "voxels": 2, "all": (0, 1, 2)}
            dims = mapping.get(self.dim)
            if dims is None:
                raise ValueError(f"Unknown dim value: {self.dim}")
            if isinstance(dims, tuple):
                return dims
            return (dims,)
        return tuple(self.dim)
