from __future__ import annotations

import torch
from typing import Sequence
from convminds.data.primitives import BrainTensor
from convminds.transforms.base import StatelessTransform


class FIRDelay(StatelessTransform):
    """
    Finite Impulse Response (FIR) delay transform.
    Shifts the signal by given TR delays and concatenates them along the feature dimension.
    
    If delays = [1, 2, 3], and signal has shape (T, F), 
    the output signal will have shape (T, F * len(delays)).
    
    The first 'max(delays)' timepoints will be padded with zeros for the shifted components.
    """
    def __init__(self, delays: Sequence[int]):
        """
        Args:
            delays: Sequence of integer TR delays (e.g. [1, 2, 3, 4]).
        """
        self.delays = sorted(delays)

    def __call__(self, brain: BrainTensor) -> BrainTensor:
        signal = brain.signal
        # Handle both (T, F) and (B, T, F)
        is_batched = signal.dim() == 3
        
        if not is_batched:
            # (T, F) -> (1, T, F)
            signal = signal.unsqueeze(0)
            
        batch_size, n_trs, n_features = signal.shape
        delayed_parts = []
        
        for d in self.delays:
            # Create a zero-padded delayed version
            # For delay d, the value at time t comes from time t-d
            part = torch.zeros_like(signal)
            if d == 0:
                part = signal
            elif d > 0:
                if d < n_trs:
                    part[:, d:, :] = signal[:, :-d, :]
            else: # Negative delay (future)
                d_abs = abs(d)
                if d_abs < n_trs:
                    part[:, :-d_abs, :] = signal[:, d_abs:, :]
            
            delayed_parts.append(part)
            
        new_signal = torch.cat(delayed_parts, dim=-1)
        
        if not is_batched:
            new_signal = new_signal.squeeze(0)
            
        return BrainTensor(
            signal=new_signal,
            coords=brain.coords,
            rois=brain.rois,
            padding_mask=brain.padding_mask,
            category=brain.category,
        )
