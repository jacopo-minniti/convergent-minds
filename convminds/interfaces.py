from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence, TYPE_CHECKING
if TYPE_CHECKING:
    from convminds.data.types import DataCategory


@dataclass(frozen=True)
class StimulusRecord:
    stimulus_id: str
    text: str
    topic: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stimulus_id": self.stimulus_id,
            "text": self.text,
            "topic": self.topic,
            "metadata": dict(self.metadata),
        }


@dataclass
class StimulusSet:
    records: list[StimulusRecord]

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def __getitem__(self, item):
        return self.records[item]

    def ids(self) -> list[str]:
        return [record.stimulus_id for record in self.records]

    def texts(self) -> list[str]:
        return [record.text for record in self.records]

    def topics(self) -> list[str | None]:
        return [record.topic for record in self.records]

    def subset(self, indices: Sequence[int]) -> "StimulusSet":
        return StimulusSet([self.records[index] for index in indices])

    def to_serializable(self) -> list[dict[str, Any]]:
        return [record.to_dict() for record in self.records]

    def __repr__(self) -> str:
        preview = ", ".join(
            f"{record.stimulus_id}:{record.topic or 'no-topic'}:{record.text[:32]!r}"
            for record in self.records[:5]
        )
        if len(self.records) > 5:
            preview += ", ..."
        return f"StimulusSet(n={len(self.records)}, records=[{preview}])"


@dataclass(frozen=True)
class SplitConfig:
    cv: int = 1
    topic_splitting: bool = True
    train_size: float = 0.9
    random_state: int = 42

    def to_dict(self) -> dict[str, Any]:
        return {
            "cv": self.cv,
            "topic_splitting": self.topic_splitting,
            "train_size": self.train_size,
            "random_state": self.random_state,
        }


@dataclass(frozen=True)
class SplitPlan:
    index: int
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    train_stimulus_ids: tuple[str, ...]
    test_stimulus_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "train_indices": list(self.train_indices),
            "test_indices": list(self.test_indices),
            "train_stimulus_ids": list(self.train_stimulus_ids),
            "test_stimulus_ids": list(self.test_stimulus_ids),
        }


@dataclass
class RecordedSplit:
    index: int
    train: Any
    test: Any
    train_stimuli: StimulusSet
    test_stimuli: StimulusSet
    metadata: dict[str, Any] = field(default_factory=dict)
    category: DataCategory | None = None

    def to_cache_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "train": self.train,
            "test": self.test,
            "train_stimuli": self.train_stimuli.to_serializable(),
            "test_stimuli": self.test_stimuli.to_serializable(),
            "metadata": dict(self.metadata),
            "category": self.category,
        }


@dataclass
class HumanRecordingData:
    values: Any
    stimulus_ids: list[str]
    feature_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    category: DataCategory | None = None


@dataclass
class HumanRecordingSource(ABC):
    identifier: str
    storage_mode: str
    recording_type: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @abstractmethod
    def load_stimuli(self, benchmark: "Benchmark") -> StimulusSet:
        raise NotImplementedError

    @abstractmethod
    def load_recordings(
        self,
        benchmark: "Benchmark",
        selector: Mapping[str, Any] | None = None,
    ) -> HumanRecordingData:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "storage_mode": self.storage_mode,
            "recording_type": self.recording_type,
            "metadata": dict(self.metadata),
        }


class Benchmark(ABC):
    identifier: str
    stimuli: StimulusSet
    split_config: SplitConfig
    human_recording_source: HumanRecordingSource

    @abstractmethod
    def benchmark_config(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def build_split_plan(self) -> list[SplitPlan]:
        raise NotImplementedError

    @abstractmethod
    def align_values(
        self,
        values: Any,
        stimulus_ids: Sequence[str] | None = None,
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def prepare_recorded_splits(
        self,
        values: Any,
        stimulus_ids: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> list[RecordedSplit]:
        raise NotImplementedError

    @abstractmethod
    def load_human_recordings(
        self,
        selector: Mapping[str, Any] | None = None,
    ) -> HumanRecordingData:
        raise NotImplementedError

    @abstractmethod
    def iter_context_groups(self) -> Iterable[list[int]]:
        raise NotImplementedError
