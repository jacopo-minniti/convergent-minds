from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import torch
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, random_split

logger = logging.getLogger(__name__)
from torch.utils.data._utils.collate import default_collate

from convminds.data.collate import collate_brains
from convminds.data.primitives import BrainTensor


def _ensure_batched(brain: BrainTensor) -> BrainTensor:
    if brain.signal.dim() == 2:
        return BrainTensor(brain.signal.unsqueeze(0), brain.coords, brain.rois, padding_mask=brain.padding_mask)
    return brain


def _stack_brain_tensors(brains: Sequence[BrainTensor]) -> BrainTensor:
    if not brains:
        raise ValueError("No BrainTensor entries to stack.")
    flattened = []
    for brain in brains:
        batched = _ensure_batched(brain)
        if batched.signal.size(0) == 1:
            flattened.append(
                BrainTensor(
                    signal=batched.signal.squeeze(0),
                    coords=batched.coords.squeeze(0) if batched.coords.dim() == 3 else batched.coords,
                    rois={name: mask.squeeze(0) if mask.dim() == 2 else mask for name, mask in batched.rois.items()},
                    padding_mask=batched.padding_mask.squeeze(0) if batched.padding_mask is not None else None,
                )
            )
        else:
            for index in range(batched.signal.size(0)):
                coords = batched.coords[index] if batched.coords.dim() == 3 else batched.coords
                rois = {
                    name: mask[index] if mask.dim() == 2 else mask for name, mask in batched.rois.items()
                }
                padding_mask = batched.padding_mask[index] if batched.padding_mask is not None else None
                flattened.append(
                    BrainTensor(
                        signal=batched.signal[index],
                        coords=coords,
                        rois=rois,
                        padding_mask=padding_mask,
                    )
                )
    return collate_brains(flattened)


def _map_brain(item, fn):
    if isinstance(item, BrainTensor):
        return fn(item)
    if isinstance(item, dict):
        if "brain_tensor" in item and isinstance(item["brain_tensor"], BrainTensor):
            mapped = dict(item)
            mapped["brain_tensor"] = fn(item["brain_tensor"])
            return {key: _map_brain(value, fn) for key, value in mapped.items()}
        if "signal" in item and "coords" in item:
            mapped = dict(item)
            brain_tensor = BrainTensor(
                signal=mapped.pop("signal"),
                coords=mapped.pop("coords"),
                rois=mapped.pop("rois", {}),
            )
            mapped["brain_tensor"] = fn(brain_tensor)
            return {key: _map_brain(value, fn) for key, value in mapped.items()}
        return {key: _map_brain(value, fn) for key, value in item.items()}
    if isinstance(item, (list, tuple)):
        mapped = [_map_brain(value, fn) for value in item]
        return type(item)(mapped)
    return item


def _find_brain(item) -> BrainTensor:
    if isinstance(item, BrainTensor):
        return item
    if isinstance(item, dict):
        for key in ("brain", "brain_tensor", "x"):
            value = item.get(key)
            if isinstance(value, BrainTensor):
                return value
        if "signal" in item and "coords" in item:
            return BrainTensor(signal=item["signal"], coords=item["coords"], rois=item.get("rois", {}))
        for value in item.values():
            if isinstance(value, BrainTensor):
                return value
    if isinstance(item, (list, tuple)):
        for value in item:
            if isinstance(value, BrainTensor):
                return value
    raise TypeError("Unable to locate a BrainTensor in batch item.")


def _collate(batch):
    if not batch:
        return batch
    first = batch[0]
    if isinstance(first, BrainTensor):
        return collate_brains(batch)
    if isinstance(first, dict):
        collated = {key: _collate([sample[key] for sample in batch]) for key in first}
        for value in collated.values():
            if isinstance(value, BrainTensor) and value.padding_mask is not None:
                collated.setdefault("brain_padding_mask", value.padding_mask)
                break
        return collated
    if isinstance(first, (list, tuple)):
        transposed = list(zip(*batch))
        collated = [_collate(list(items)) for items in transposed]
        return type(first)(collated)
    return default_collate(batch)


class _TransformedDataset(Dataset):
    def __init__(self, dataset: Dataset, transform_fn):
        self.dataset = dataset
        self.transform_fn = transform_fn

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        return _map_brain(item, self.transform_fn)


@dataclass
class BrainDataModule:
    dataset: Dataset | None = None
    benchmark: object | None = None
    human_subject: object | None = None
    artificial_subject: object | None = None
    stateless_transforms: Optional[Sequence] = None
    stateful_transforms: Optional[Sequence] = None
    batch_size: int = 32
    text_transform: object | None = None
    split_index: int = 0
    train_size: float = 0.8
    seed: int = 0
    num_workers: int = 0
    pin_memory: bool = False
    drop_last: bool = False

    def __post_init__(self) -> None:
        self.stateless_transforms = self._normalize_transforms(self.stateless_transforms)
        self.stateful_transforms = self._normalize_transforms(self.stateful_transforms)
        self._train_dataset: Optional[Dataset] = None
        self._test_dataset: Optional[Dataset] = None
        self._is_setup = False

    def _normalize_transforms(self, transforms):
        if transforms is None:
            return []
        if isinstance(transforms, (list, tuple)):
            return list(transforms)
        return [transforms]

    def setup(self) -> None:
        logger.info("Setting up BrainDataModule")
        if self.human_subject is not None:
            self._setup_from_subjects()
        else:
            logger.info("No human subject provided, splitting from provided dataset")
            train_dataset, test_dataset = self._split_dataset()
            self._fit_stateful_transforms(train_dataset)
            self._train_dataset = _TransformedDataset(train_dataset, self._apply_transforms)
            self._test_dataset = _TransformedDataset(test_dataset, self._apply_transforms)
        self._is_setup = True

    def train_dataloader(self) -> DataLoader:
        self._assert_setup()
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            collate_fn=_collate,
        )

    def test_dataloader(self) -> DataLoader:
        self._assert_setup()
        return DataLoader(
            self._test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            collate_fn=_collate,
        )

    def _assert_setup(self) -> None:
        if not self._is_setup:
            raise RuntimeError("BrainDataModule.setup() must be called before requesting dataloaders.")

    def _split_dataset(self) -> tuple[Dataset, Dataset]:
        dataset = self.dataset
        if dataset is None:
            raise ValueError("BrainDataModule requires a dataset when human_subject is not provided.")
        if hasattr(dataset, "get_splits"):
            splits = dataset.get_splits()
            return splits["train"], splits["test"]
        if isinstance(dataset, dict) and "train" in dataset and "test" in dataset:
            return dataset["train"], dataset["test"]
        if hasattr(dataset, "train") and hasattr(dataset, "test"):
            return dataset.train, dataset.test
        total = len(dataset)
        train_len = int(total * self.train_size)
        test_len = total - train_len
        generator = torch.Generator().manual_seed(self.seed)
        train_dataset, test_dataset = random_split(dataset, [train_len, test_len], generator=generator)
        return train_dataset, test_dataset

    def _apply_transforms(self, brain: BrainTensor, *, include_stateful: bool = True) -> BrainTensor:
        output = brain
        for transform in self.stateless_transforms:
            output = transform(output)
        if include_stateful:
            for transform in self.stateful_transforms:
                output = transform(output)
        return output

    def _fit_stateful_transforms(self, train_dataset: Dataset) -> None:
        if not self.stateful_transforms:
            return
        logger.info(f"Fitting {len(self.stateful_transforms)} stateful transforms")
        stateless_dataset = _TransformedDataset(
            train_dataset,
            lambda brain: self._apply_transforms(brain, include_stateful=False),
        )
        brain = self._stack_dataset(stateless_dataset)
        for transform in self.stateful_transforms:
            transform.fit(brain)
            brain = transform(brain)

    def _stack_dataset(self, dataset: Dataset) -> BrainTensor:
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=_collate,
        )
        brains: List[BrainTensor] = []
        for batch in tqdm(loader, desc="Stacking dataset", leave=False):
            brain = _ensure_batched(_find_brain(batch))
            brains.append(brain)
        return _stack_brain_tensors(brains)

    def _setup_from_subjects(self) -> None:
        benchmark = self.benchmark or self.dataset
        if benchmark is None:
            raise ValueError("BrainDataModule requires a benchmark when using human_subject.")

        if hasattr(self.human_subject, "record") and self.human_subject.recordings is None:
            logger.info(f"Retrieving recordings for human subject: {self.human_subject.identifier()}")
            self.human_subject.record(benchmark)
        
        # Log basic benchmark stats for debugging
        num_stimuli = len(benchmark.stimuli) if hasattr(benchmark, "stimuli") else "unknown"
        logger.info(f"Setup from subjects: {num_stimuli} stimuli in benchmark")
        if self.artificial_subject is not None and self.artificial_subject.recordings is None:
            logger.info(f"Retrieving activations for artificial subject: {self.artificial_subject.identifier()}")
            self.artificial_subject.record(benchmark)

        if self.human_subject.recordings is None:
            raise RuntimeError("HumanSubject did not populate recordings.")
        if self.split_index < 0 or self.split_index >= len(self.human_subject.recordings):
            raise ValueError("split_index is out of range for the recorded splits.")

        artificial_split = None
        if self.artificial_subject is not None:
            if self.artificial_subject.recordings is None:
                raise RuntimeError("ArtificialSubject did not populate recordings.")
            if len(self.artificial_subject.recordings) != len(self.human_subject.recordings):
                raise ValueError("Human and artificial subjects produced different numbers of splits.")
            artificial_split = self.artificial_subject.recordings[self.split_index]

        split_stimuli = self.human_subject.split_stimuli[self.split_index]
        human_split = self.human_subject.recordings[self.split_index]
        metadata = getattr(self.human_subject, "recording_metadata", {}) or {}
        coords = metadata.get("coords")
        rois = metadata.get("rois") or metadata.get("roi_masks")

        train_dataset = _SubjectSplitDataset(
            human_values=human_split["train"],
            stimuli=split_stimuli["train"],
            target_values=artificial_split["train"] if artificial_split else None,
            text_transform=self.text_transform,
            coords=coords,
            rois=rois,
        )
        test_dataset = _SubjectSplitDataset(
            human_values=human_split["test"],
            stimuli=split_stimuli["test"],
            target_values=artificial_split["test"] if artificial_split else None,
            text_transform=self.text_transform,
            coords=coords,
            rois=rois,
        )

        self._test_dataset = _TransformedDataset(test_dataset, self._apply_transforms)
        
        logger.info(f"DataModule Setup Complete: Train={len(train_dataset)}, Test={len(test_dataset)}")
        # Log target mapping info
        first_item = train_dataset[0]
        if "target_latents" in first_item:
            logger.info(f"Detected target_latents with shape: {first_item['target_latents'].shape}")
        
        brain = first_item["brain_tensor"]
        logger.info(f"First brain sample: signal={brain.signal.shape}, coords={brain.coords.shape}")
        if brain.coords.sum() == 0:
            logger.warning("BRAIN COORDINATES ARE ALL ZEROS! SpatialAttentionEncoder will have no spatial grounding.")
        else:
            logger.info(f"Brain coordinates verified: mean={brain.coords.mean(dim=0).tolist()}")


class _SubjectSplitDataset(Dataset):
    def __init__(
        self,
        *,
        human_values,
        stimuli,
        target_values=None,
        text_transform=None,
        coords=None,
        rois=None,
    ):
        self.human_values = human_values
        self.stimuli = stimuli
        self.target_values = target_values
        self.text_transform = text_transform
        self._coords = coords
        self._rois = rois or {}

    def __len__(self) -> int:
        return len(self.stimuli)

    def __getitem__(self, index):
        signal = torch.as_tensor(self.human_values[index], dtype=torch.float32)
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)
        if self._coords is None:
            self._coords = torch.zeros(signal.shape[-1], 3, dtype=signal.dtype)
        if not torch.is_tensor(self._coords):
            self._coords = torch.as_tensor(self._coords, dtype=signal.dtype)
        rois = self._rois
        if rois and not isinstance(next(iter(rois.values())), torch.Tensor):
            rois = {name: torch.as_tensor(mask, dtype=torch.bool) for name, mask in rois.items()}
        brain_tensor = BrainTensor(signal=signal, coords=self._coords, rois=rois)
        record = self.stimuli[index]
        payload = {
            "brain_tensor": brain_tensor,
            "text": record.text,
            "stimulus_id": record.stimulus_id,
            "metadata": dict(record.metadata),
        }
        if self.text_transform is not None:
            payload.update(self.text_transform(record.text))
        if self.target_values is not None:
            payload["target_latents"] = torch.as_tensor(self.target_values[index], dtype=torch.float32)
        return payload
