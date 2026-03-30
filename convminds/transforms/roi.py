from __future__ import annotations

import logging
import torch
from convminds.data.primitives import BrainTensor
from convminds.transforms.base import StatelessTransform

logger = logging.getLogger(__name__)


class ROIFilter(StatelessTransform):
    """
    Stateless transform that filters voxels based on a specific Region of Interest (ROI) mask.
    The mask is retrieved from the input BrainTensor's 'rois' dictionary.
    """
    def __init__(self, roi: str):
        """
        Args:
            roi: The key in the brain.rois dictionary to use for filtering.
        """
        self.roi = roi

    def __call__(self, brain: BrainTensor) -> BrainTensor:
        if self.roi not in brain.rois:
            available = list(brain.rois.keys())
            logger.warning(f"ROI '{self.roi}' not found in BrainTensor. Available: {available}. Returning original.")
            return brain
            
        mask = brain.rois[self.roi]
        
        # mask is usually (Voxels,) or same shape as signal's last dim
        # signal: (B, T, V) or (T, V) or (V,)
        original_signal = brain.signal
        
        # Apply mask
        # Assuming last dimension is voxels
        new_signal = original_signal[..., mask]
        
        # Also filter coordinates
        new_coords = brain.coords[mask]
        
        # Filter other ROI masks if they are still relevant, or just keep them for hierarchical filtering?
        # Typically we just want the new signal. 
        # But we can update all masks to match the new size.
        new_rois = {
            name: m[mask] for name, m in brain.rois.items()
        }
        
        return BrainTensor(
            signal=new_signal,
            coords=new_coords,
            rois=new_rois,
            padding_mask=brain.padding_mask,
            category=brain.category,
        )
