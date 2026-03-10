from __future__ import annotations

from typing import Iterable, Sequence

from convminds.data.primitives import BrainTensor


class StatelessTransform:
    def __call__(self, brain: BrainTensor) -> BrainTensor:
        return brain


class StatefulTransform:
    def fit(self, brain: BrainTensor) -> "StatefulTransform":
        raise NotImplementedError

    def __call__(self, brain: BrainTensor) -> BrainTensor:
        return brain


class Compose:
    def __init__(self, transforms: Sequence):
        self.transforms = list(transforms)

    def __call__(self, brain: BrainTensor) -> BrainTensor:
        output = brain
        for transform in self.transforms:
            output = transform(output)
        return output
