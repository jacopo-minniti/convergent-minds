from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from convminds.data.primitives import BrainTensor
from convminds.transforms.base import StatefulTransform


class VoxelPCA(StatefulTransform):
    def __init__(
        self, 
        n_components: int, 
        whiten: bool = False, 
        random_state: int | None = 0,
        cache_path: str | Path | None = None
    ):
        if n_components <= 0:
            raise ValueError("n_components must be positive.")
        self.n_components = n_components
        self.whiten = whiten
        self.random_state = random_state
        self.cache_path = Path(cache_path) if cache_path else None
        self._pca = None

    def fit(self, brain: BrainTensor | np.ndarray) -> "VoxelPCA":
        from sklearn.decomposition import PCA
        import joblib

        # Handle both BrainTensor and raw numpy arrays
        if hasattr(brain, "signal"):
            data = brain.signal.detach().cpu().numpy()
            flat = data.reshape(-1, data.shape[-1])
        else:
            flat = brain.reshape(-1, brain.shape[-1])

        # Try loading from cache
        if self.cache_path and self.cache_path.exists():
            import logging
            logging.getLogger(__name__).info(f"Loading PCA model from cache: {self.cache_path}")
            self._pca = joblib.load(self.cache_path)
            return self

        self._pca = PCA(
            n_components=self.n_components,
            whiten=self.whiten,
            random_state=self.random_state,
            svd_solver="randomized" if self.n_components < 0.8 * min(flat.shape) else "auto"
        )
        
        import logging
        logging.getLogger(__name__).info(f"Fitting PCA ({self.n_components} components) on {flat.shape} matrix...")
        self._pca.fit(flat)

        # Save to cache
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self._pca, self.cache_path)
            logging.getLogger(__name__).info(f"Saved PCA model to cache: {self.cache_path}")
            
        return self

    def __call__(self, brain: BrainTensor) -> BrainTensor:
        if self._pca is None:
            raise RuntimeError("VoxelPCA must be fit before calling.")
        
        data = brain.signal.detach().cpu().numpy()
        # Keep track of original shape except for the voxel dimension
        orig_shape = brain.signal.shape
        flat = data.reshape(-1, orig_shape[-1])
        
        transformed = self._pca.transform(flat)
        
        # Reshape back to (Batch, Sequence, Components)
        new_shape = list(orig_shape[:-1]) + [self.n_components]
        new_signal = torch.as_tensor(
            transformed,
            device=brain.signal.device,
            dtype=brain.signal.dtype,
        ).reshape(*new_shape)
        
        # PCA loses spatial coordinates
        coords = torch.zeros(
            (self.n_components, 3),
            device=brain.coords.device,
            dtype=brain.coords.dtype,
        )
        return BrainTensor(signal=new_signal, coords=coords, rois={})
