from __future__ import annotations

from typing import Optional

import torch

from convminds.data.primitives import BrainTensor
from convminds.transforms.base import StatefulTransform


class VoxelPCA(StatefulTransform):
    def __init__(self, n_components: int, whiten: bool = False, random_state: int | None = 0):
        if n_components <= 0:
            raise ValueError("n_components must be positive.")
        self.n_components = n_components
        self.whiten = whiten
        self.random_state = random_state
        self._pca = None

    def fit(self, brain: BrainTensor) -> "VoxelPCA":
        from sklearn.decomposition import PCA

        data = brain.signal.detach().cpu().numpy()
        flat = data.reshape(-1, data.shape[-1])
        self._pca = PCA(
            n_components=self.n_components,
            whiten=self.whiten,
            random_state=self.random_state,
        )
        self._pca.fit(flat)
        return self

    def __call__(self, brain: BrainTensor) -> BrainTensor:
        if self._pca is None:
            raise RuntimeError("VoxelPCA must be fit before calling.")
        data = brain.signal.detach().cpu().numpy()
        flat = data.reshape(-1, data.shape[-1])
        transformed = self._pca.transform(flat)
        new_signal = torch.as_tensor(
            transformed,
            device=brain.signal.device,
            dtype=brain.signal.dtype,
        ).reshape(brain.signal.shape[0], brain.signal.shape[1], -1)
        coords = torch.zeros(
            (new_signal.shape[-1], 3),
            device=brain.coords.device,
            dtype=brain.coords.dtype,
        )
        return BrainTensor(signal=new_signal, coords=coords, rois={})
