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

    max_voxels = max(brain.signal.shape[-1] for brain in brains_list)
    batch_signals: List[torch.Tensor] = []
    batch_coords: List[torch.Tensor] = []
    batch_padding: List[torch.Tensor] = []

    roi_keys: List[str] = sorted({key for brain in brains_list for key in brain.rois.keys()})
    batch_rois: Dict[str, List[torch.Tensor]] = {key: [] for key in roi_keys}

    for brain in brains_list:
        signal = brain.signal
        coords = brain.coords
        num_voxels = signal.shape[-1]

        if num_voxels < max_voxels:
            pad_width = max_voxels - num_voxels
            signal = torch.nn.functional.pad(signal, (0, pad_width))
            coords = torch.nn.functional.pad(coords, (0, 0, 0, pad_width))

        batch_signals.append(signal)
        batch_coords.append(coords)

        padding_mask = torch.zeros(signal.shape[0], max_voxels, device=signal.device, dtype=torch.bool)
        if num_voxels < max_voxels:
            padding_mask[:, num_voxels:] = True
        if brain.padding_mask is not None:
            existing = brain.padding_mask
            if existing.dim() == 1:
                existing = existing.unsqueeze(0)
            if existing.shape[1] != max_voxels:
                padded = torch.zeros(signal.shape[0], max_voxels, device=signal.device, dtype=torch.bool)
                padded[:, : existing.shape[1]] = existing
                existing = padded
            padding_mask = padding_mask | existing.to(dtype=torch.bool)
        batch_padding.append(padding_mask)

        for key in roi_keys:
            mask = brain.rois.get(key)
            if mask is None:
                mask = torch.zeros(signal.shape[0], num_voxels, device=signal.device, dtype=torch.bool)
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            if num_voxels < max_voxels:
                mask = torch.nn.functional.pad(mask, (0, max_voxels - num_voxels))
            batch_rois[key].append(mask)

    signal = torch.cat(batch_signals, dim=0)
    coords = torch.cat(batch_coords, dim=0)
    padding_mask = torch.cat(batch_padding, dim=0)
    rois = {key: torch.cat(masks, dim=0) for key, masks in batch_rois.items()}
    return BrainTensor(signal=signal, coords=coords, rois=rois, padding_mask=padding_mask)
