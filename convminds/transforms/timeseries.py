from __future__ import annotations

import torch
from convminds.data.primitives import BrainTensor
from convminds.transforms.base import StatelessTransform


class TrimTRs(StatelessTransform):
    """
    Stateless transform that trims fMRI TRs from the beginning and end of a sequence.
    
    This is common in naturalistic datasets (e.g. Huth) to remove transient activity 
    at story boundaries.
    """
    def __init__(self, start: int = 0, end: int = 0):
        """
        Args:
            start: Number of TRs to remove from the start.
            end: Number of TRs to remove from the end.
        """
        self.start = start
        self.end = end

    def __call__(self, brain: BrainTensor) -> BrainTensor:
        # signal: (B, T, V) or (T, V)
        original_signal = brain.signal
        n_trs = original_signal.shape[-2]
        
        if self.start + self.end >= n_trs:
            # Cannot trim more than we have
            return brain
            
        # Perform slicing
        # Use negative indexing for end to support 0 case
        end_idx = n_trs - self.end if self.end > 0 else n_trs
        
        if original_signal.dim() == 3:
            # (B, T, V)
            new_signal = original_signal[:, self.start:end_idx, :]
        else:
            # (T, V)
            new_signal = original_signal[self.start:end_idx, :]
            
        # Update padding mask if present
        new_padding_mask = None
        if brain.padding_mask is not None:
             # padding_mask can be (B, V) or (B, T, V)
             if brain.padding_mask.dim() == 3:
                 new_padding_mask = brain.padding_mask[:, self.start:end_idx, :]
             elif brain.padding_mask.dim() == 2:
                 # For (B, V) spatial mask, trimming TRs is a no-op for the mask.
                 pass
        
        return BrainTensor(
            signal=new_signal,
            coords=brain.coords,
            rois=brain.rois,
            padding_mask=new_padding_mask,
            category=brain.category,
        )


class RandomWindow(StatelessTransform):
    """
    Picks a random temporal window of a fixed size from the sequence.
    Useful for training on long fMRI stories.
    """
    def __init__(self, window_size: int):
        self.window_size = window_size

    def __call__(self, brain: BrainTensor) -> BrainTensor:
        signal = brain.signal
        n_trs = signal.shape[-2]
        
        if n_trs <= self.window_size:
            return brain
            
        start = torch.randint(0, n_trs - self.window_size + 1, (1,)).item()
        end = start + self.window_size
        
        if signal.dim() == 3:
            new_signal = signal[:, start:end, :]
        else:
            new_signal = signal[start:end, :]
            
        new_padding_mask = None
        if brain.padding_mask is not None:
             if brain.padding_mask.dim() == 3:
                 new_padding_mask = brain.padding_mask[:, start:end, :]
             else:
                 new_padding_mask = brain.padding_mask[..., start:end]
             
        return BrainTensor(
            signal=new_signal,
            coords=brain.coords,
            rois=brain.rois,
            padding_mask=new_padding_mask,
            category=brain.category,
        )


class SlidingWindow(StatelessTransform):
    """
    Picks a fixed temporal window from the sequence starting at a given offset.
    Useful for deterministic evaluation.
    """
    def __init__(self, window_size: int, offset: int = 0):
        self.window_size = window_size
        self.offset = offset

    def __call__(self, brain: BrainTensor) -> BrainTensor:
        signal = brain.signal
        n_trs = signal.shape[-2]
        
        start = min(max(0, self.offset), n_trs)
        end = min(start + self.window_size, n_trs)
        
        if signal.dim() == 3:
            new_signal = signal[:, start:end, :]
        else:
            new_signal = signal[start:end, :]
            
        new_padding_mask = None
        if brain.padding_mask is not None:
             if brain.padding_mask.dim() == 3:
                 new_padding_mask = brain.padding_mask[:, start:end, :]
             else:
                 new_padding_mask = brain.padding_mask[..., start:end]

        return BrainTensor(
            signal=new_signal,
            coords=brain.coords,
            rois=brain.rois,
            padding_mask=new_padding_mask,
            category=brain.category,
        )
