from __future__ import annotations

import torch
import numpy as np
from typing import Sequence
from convminds.data.primitives import BrainTensor
from convminds.transforms.base import StatelessTransform
from convminds.data.alignment import lanczos_interp2d


class LanczosInterpolate(StatelessTransform):
    """
    General Lanczos interpolation transform.
    Resamples a signal from its original timestamps to a new set of timestamps.
    
    This is typically used to downsample event-level stimulus features to fMRI TR times.
    """
    def __init__(
        self,
        *,
        old_times: np.ndarray | Sequence[float],
        new_times: np.ndarray | Sequence[float],
        window: int = 3,
        cutoff_mult: float = 1.0,
        rectify: bool = False,
    ):
        """
        Args:
            old_times: Original timestamps for each sample in the input.
            new_times: Target timestamps (e.g. TR times).
            window: Lanczos window size (default 3).
            cutoff_mult: Cutoff frequency multiplier.
            rectify: Whether to return split positive/negative components.
        """
        self.old_times = np.asarray(old_times)
        self.new_times = np.asarray(new_times)
        self.window = window
        self.cutoff_mult = cutoff_mult
        self.rectify = rectify

    def __call__(self, brain: BrainTensor) -> BrainTensor:
        # Convert signal to numpy for lanczos_interp2d
        # data: (T, F)
        data = brain.signal.cpu().numpy()
        
        # Interpolate
        new_data = lanczos_interp2d(
            data,
            self.old_times,
            self.new_times,
            window=self.window,
            cutoff_mult=self.cutoff_mult,
            rectify=self.rectify
        )
        
        # Convert back to torch
        new_signal = torch.as_tensor(new_data, dtype=brain.signal.dtype, device=brain.device)
        
        return BrainTensor(
            signal=new_signal,
            coords=brain.coords,
            rois=brain.rois,
            padding_mask=None, # Timestamps changed, so original mask is invalid
            category=brain.category,
        )
