from __future__ import annotations

import torch
from torch.utils.data import Dataset

import convminds as cm
from convminds.data.collate import collate_brains
from convminds.data.primitives import BrainTensor


class DummyBrainDataset(Dataset):
    def __init__(self, *, count: int = 6, time: int = 2, voxels: int = 4):
        self.items: list[BrainTensor] = []
        for index in range(count):
            signal = torch.arange(time * voxels, dtype=torch.float32).reshape(time, voxels) + index
            coords = torch.zeros(voxels, 3, dtype=torch.float32)
            self.items.append(BrainTensor(signal=signal, coords=coords))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> BrainTensor:
        return self.items[index]


def test_collate_brains_padding():
    brain_a = BrainTensor(signal=torch.zeros(1, 3), coords=torch.zeros(3, 3))
    brain_b = BrainTensor(signal=torch.zeros(1, 5), coords=torch.zeros(5, 3))
    batch = collate_brains([brain_a, brain_b])
    assert batch.signal.shape == (2, 1, 5)
    assert batch.padding_mask is not None
    assert batch.padding_mask.shape == (2, 5)
    assert torch.all(batch.padding_mask[0, 3:])
    assert torch.all(batch.padding_mask[1] == 0)


def test_brain_datamodule_transforms():
    dataset = DummyBrainDataset()
    datamodule = cm.data.BrainDataModule(
        dataset=dataset,
        stateless_transforms=[cm.transforms.HRFWindow(t=1)],
        stateful_transforms=[cm.transforms.ZScore(dim="time")],
        batch_size=2,
    )
    datamodule.setup()
    batch = next(iter(datamodule.train_dataloader()))
    assert isinstance(batch, BrainTensor)
    assert batch.signal.dim() == 3
    assert not torch.isnan(batch.signal).any()
