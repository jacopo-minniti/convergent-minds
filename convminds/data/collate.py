from __future__ import annotations

from typing import Dict, Iterable, List

import torch

from convminds.data.primitives import BrainTensor


def _ensure_batched(brain: BrainTensor) -> BrainTensor:
    if brain.signal.dim() == 2:
        signal = brain.signal.unsqueeze(0)
    else:
        signal = brain.signal

    coords = brain.coords
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)

    rois: Dict[str, torch.Tensor] = {}
    for name, mask in brain.rois.items():
        if mask.dim() == 1:
            rois[name] = mask.unsqueeze(0)
        else:
            rois[name] = mask

    padding_mask = brain.padding_mask
    if padding_mask is not None and padding_mask.dim() == 1:
        padding_mask = padding_mask.unsqueeze(0)

    return BrainTensor(signal=signal, coords=coords, rois=rois, padding_mask=padding_mask)


def collate_brains(brains: Iterable[BrainTensor]) -> BrainTensor:
    brains_list = [_ensure_batched(brain) for brain in brains]
    if not brains_list:
        raise ValueError("collate_brains requires at least one BrainTensor.")

    # Find max dimensions across all samples in the batch
    max_trs = max(brain.signal.shape[-2] for brain in brains_list)
    max_voxels = max(brain.signal.shape[-1] for brain in brains_list)
    
    batch_signals: List[torch.Tensor] = []
    batch_coords: List[torch.Tensor] = []
    batch_padding: List[torch.Tensor] = []

    roi_keys: List[str] = sorted({key for brain in brains_list for key in brain.rois.keys()})
    batch_rois: Dict[str, List[torch.Tensor]] = {key: [] for key in roi_keys}

    for brain in brains_list:
        signal = brain.signal # (B, T, V)
        coords = brain.coords # (B, V, 3) or (V, 3)
        
        num_trs = signal.shape[-2]
        num_voxels = signal.shape[-1]

        # 1. Pad Signal and Coordinates
        if num_trs < max_trs or num_voxels < max_voxels:
            pad_v = max_voxels - num_voxels
            pad_t = max_trs - num_trs
            # Padding for (B, T, V) is (pad_v_left, pad_v_right, pad_t_left, pad_t_right)
            signal = torch.nn.functional.pad(signal, (0, pad_v, 0, pad_t))
            
            # Pad coords (B, V, 3) or (V, 3) along the V dimension (second to last)
            if coords.dim() == 3:
                coords = torch.nn.functional.pad(coords, (0, 0, 0, pad_v))
            else:
                coords = torch.nn.functional.pad(coords, (0, 0, 0, pad_v))

        batch_signals.append(signal)
        batch_coords.append(coords)

        # 2. Update Padding Mask
        # We need a temporal mask for padded TRs and a spatial mask for padded voxels
        # Final padding_mask shape: (B, T, V) to cover both dimensions?
        # Or (B, T) for time and (B, V) for space?
        # Current BrainTensor.padding_mask is usually (B, V) or (B, T, V).
        # We'll use (B, T, V) if any padding occurs to be safe.
        mask = torch.zeros(signal.shape[0], max_trs, max_voxels, device=signal.device, dtype=torch.bool)
        
        # Mark temporal padding
        if num_trs < max_trs:
            mask[:, num_trs:, :] = True
        
        # Mark spatial padding
        if num_voxels < max_voxels:
            mask[:, :, num_voxels:] = True
            
        # Incorporate existing mask if present
        if brain.padding_mask is not None:
            existing = brain.padding_mask
            # Handle both (B, V) and (B, T, V)
            if existing.dim() == 2:
                # (B, V) -> (B, 1, V) and expand
                existing = existing.unsqueeze(1).expand(-1, num_trs, -1)
            
            # Pad existing mask to match (B, max_trs, max_voxels)
            if num_trs < max_trs or num_voxels < max_voxels:
                existing = torch.nn.functional.pad(existing, (0, max_voxels - num_voxels, 0, max_trs - num_trs))
            
            mask = mask | existing.to(dtype=torch.bool)
            
        batch_padding.append(mask)

        # 3. Handle ROI Masks
        for key in roi_keys:
            roi_mask = brain.rois.get(key)
            if roi_mask is None:
                roi_mask = torch.zeros(signal.shape[0], num_voxels, device=signal.device, dtype=torch.bool)
            
            if roi_mask.dim() == 1:
                roi_mask = roi_mask.unsqueeze(0)
            
            # Pad ROI mask (B, V) along V dimension
            if num_voxels < max_voxels:
                roi_mask = torch.nn.functional.pad(roi_mask, (0, max_voxels - num_voxels))
            
            batch_rois[key].append(roi_mask)

    signal = torch.cat(batch_signals, dim=0)
    coords = torch.cat(batch_coords, dim=0)
    padding_mask = torch.cat(batch_padding, dim=0)
    rois = {key: torch.cat(masks, dim=0) for key, masks in batch_rois.items()}
    return BrainTensor(signal=signal, coords=coords, rois=rois, padding_mask=padding_mask)
