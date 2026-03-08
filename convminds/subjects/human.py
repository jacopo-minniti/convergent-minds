from __future__ import annotations

from typing import Any, Mapping

from convminds.cache import load_cache, save_cache
from convminds.subjects.base import Subject


class HumanSubject(Subject):
    def __init__(
        self,
        *,
        identifier: str = "human-subject",
        subject_ids: list[str] | None = None,
        roi_filters: list[str] | None = None,
        atlas_filters: list[str] | None = None,
        settings: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self._identifier = identifier
        self.subject_ids = list(subject_ids or [])
        self.roi_filters = list(roi_filters or [])
        self.atlas_filters = list(atlas_filters or [])
        self.settings = dict(settings or {})

    def identifier(self) -> str:
        return self._identifier

    def selector(self) -> dict[str, Any]:
        selector = dict(self.settings)
        if self.subject_ids:
            selector["subject"] = list(self.subject_ids)
        if self.roi_filters:
            selector["roi"] = list(self.roi_filters)
        if self.atlas_filters:
            selector["atlas"] = list(self.atlas_filters)
        return selector

    def subject_config(self) -> dict[str, Any]:
        return {
            "kind": "human",
            "identifier": self.identifier(),
            "selector": self.selector(),
        }

    def record(self, benchmark, *, force: bool = False):
        cache_config = {
            "kind": "human-recording",
            "subject": self.subject_config(),
            "benchmark": benchmark.benchmark_config(),
        }
        if not force:
            cached = load_cache("human", config=cache_config)
            if cached is not None:
                self._load_cache_payload(cached)
                return self

        recording = benchmark.load_human_recordings(selector=self.selector())
        metadata = {
            "subject_identifier": self.identifier(),
            "benchmark_identifier": benchmark.identifier,
            **recording.metadata,
        }
        self._store_recordings(
            benchmark,
            recording.values,
            stimulus_ids=recording.stimulus_ids,
            feature_ids=recording.feature_ids,
            metadata=metadata,
        )
        save_cache("human", config=cache_config, payload=self._cache_payload())
        return self
