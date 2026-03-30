from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch
from convminds.data.types import DataCategory


@dataclass
class BrainTensor:
    signal: torch.Tensor
    coords: torch.Tensor
    rois: Dict[str, torch.Tensor] = field(default_factory=dict)
    padding_mask: torch.Tensor | None = None
    category: DataCategory | None = None

    def to(self, *args, **kwargs) -> "BrainTensor":
        return BrainTensor(
            signal=self.signal.to(*args, **kwargs),
            coords=self.coords.to(*args, **kwargs),
            rois={name: mask.to(*args, **kwargs) for name, mask in self.rois.items()},
            padding_mask=self.padding_mask.to(*args, **kwargs) if self.padding_mask is not None else None,
            category=self.category,
        )

    @property
    def device(self) -> torch.device:
        return self.signal.device
