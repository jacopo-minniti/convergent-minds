from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

from convminds.interfaces import RecordedSplit


class Subject(ABC):
    def __init__(self) -> None:
        self.recordings: list[dict[str, Any]] | None = None
        self.recorded_splits: list[RecordedSplit] | None = None
        self.split_stimuli: list[dict[str, Any]] | None = None
        self.feature_ids: list[str] | None = None
        self.recording_metadata: dict[str, Any] = {}

    @abstractmethod
    def identifier(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def subject_config(self) -> dict[str, Any]:
        raise NotImplementedError

    def reset_recordings(self) -> None:
        self.recordings = None
        self.recorded_splits = None
        self.split_stimuli = None
        self.feature_ids = None
        self.recording_metadata = {}

    def _apply_recorded_splits(
        self,
        recorded_splits: list[RecordedSplit],
        *,
        feature_ids: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.recorded_splits = recorded_splits
        self.recordings = [{"train": split.train, "test": split.test} for split in recorded_splits]
        self.split_stimuli = [{"train": split.train_stimuli, "test": split.test_stimuli} for split in recorded_splits]
        self.feature_ids = list(feature_ids) if feature_ids is not None else None
        self.recording_metadata = dict(metadata or {})

    def _cache_payload(self) -> dict[str, Any]:
        return {
            "recorded_splits": self.recorded_splits,
            "feature_ids": self.feature_ids,
            "recording_metadata": self.recording_metadata,
        }

    def _load_cache_payload(self, payload: dict[str, Any]) -> None:
        self._apply_recorded_splits(
            payload["recorded_splits"],
            feature_ids=payload.get("feature_ids"),
            metadata=payload.get("recording_metadata"),
        )

    def _store_recordings(
        self,
        benchmark,
        values: Any,
        *,
        stimulus_ids: Sequence[str] | None = None,
        feature_ids: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> list[RecordedSplit]:
        recorded_splits = benchmark.prepare_recorded_splits(values, stimulus_ids=stimulus_ids, metadata=metadata)
        self._apply_recorded_splits(recorded_splits, feature_ids=feature_ids, metadata=metadata)
        return recorded_splits


class ArtificialSubject(Subject, ABC):
    @abstractmethod
    def record(self, benchmark, **kwargs):
        raise NotImplementedError
