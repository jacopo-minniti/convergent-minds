from __future__ import annotations

import torch

from convminds.data.primitives import BrainTensor
from convminds.transforms.base import StatelessTransform


class HRFWindow(StatelessTransform):
    def __init__(self, t: int = 4, pad: bool = False):
        if t <= 0:
            raise ValueError("HRFWindow t must be positive.")
        self.t = t
        self.pad = pad

    def __call__(self, brain: BrainTensor) -> BrainTensor:
        signal = brain.signal
        if signal.size(1) < self.t:
            if not self.pad:
                raise ValueError("HRFWindow received fewer timepoints than the window size.")
            pad_len = self.t - signal.size(1)
            padding = torch.zeros(
                signal.size(0),
                pad_len,
                signal.size(2),
                device=signal.device,
                dtype=signal.dtype,
            )
            signal = torch.cat([padding, signal], dim=1)
        window = signal[:, -self.t :, :]
        return BrainTensor(signal=window, coords=brain.coords, rois=brain.rois)
