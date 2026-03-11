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
        is_2d = signal.dim() == 2
        t_dim = 0 if is_2d else 1
        v_dim = 1 if is_2d else 2

        if signal.size(t_dim) < self.t:
            if not self.pad:
                raise ValueError("HRFWindow received fewer timepoints than the window size.")
            pad_len = self.t - signal.size(t_dim)
            if is_2d:
                padding = torch.zeros(
                    pad_len,
                    signal.size(v_dim),
                    device=signal.device,
                    dtype=signal.dtype,
                )
            else:
                padding = torch.zeros(
                    signal.size(0),
                    pad_len,
                    signal.size(v_dim),
                    device=signal.device,
                    dtype=signal.dtype,
                )
            signal = torch.cat([padding, signal], dim=t_dim)
            
        if is_2d:
            window = signal[-self.t :, :]
        else:
            window = signal[:, -self.t :, :]

        return BrainTensor(signal=window, coords=brain.coords, rois=brain.rois, padding_mask=brain.padding_mask)
