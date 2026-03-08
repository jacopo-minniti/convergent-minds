from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from convminds._splits import build_split_plan as compute_split_plan
from convminds.cache import load_cache, save_cache
from convminds.interfaces import Benchmark, HumanRecordingData, HumanRecordingSource, RecordedSplit, SplitConfig, SplitPlan, StimulusRecord, StimulusSet


def _stimulus_records_from_rows(rows: Sequence[Mapping[str, Any]]) -> StimulusSet:
    records = []
    for row in rows:
        records.append(
            StimulusRecord(
                stimulus_id=str(row["stimulus_id"]),
                text=str(row["text"]),
                topic=str(row["topic"]) if row.get("topic") is not None else None,
                metadata=dict(row.get("metadata", {})),
            )
        )
    return StimulusSet(records=records)


class BaseBenchmark(Benchmark):
    def __init__(
        self,
        *,
        identifier: str,
        stimuli: StimulusSet,
        split_config: SplitConfig,
        human_recording_source: HumanRecordingSource,
    ) -> None:
        self.identifier = identifier
        self.stimuli = stimuli
        self.split_config = split_config
        self.human_recording_source = human_recording_source

    def benchmark_config(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "split_config": self.split_config.to_dict(),
            "human_recording_source": self.human_recording_source.describe(),
        }

    def build_split_plan(self) -> list[SplitPlan]:
        config = {"kind": "split-plan", "benchmark": self.benchmark_config()}
        cached = load_cache("benchmarks", config=config)
        if cached is not None:
            return [
                SplitPlan(
                    index=int(item["index"]),
                    train_indices=tuple(item["train_indices"]),
                    test_indices=tuple(item["test_indices"]),
                    train_stimulus_ids=tuple(item["train_stimulus_ids"]),
                    test_stimulus_ids=tuple(item["test_stimulus_ids"]),
                )
                for item in cached
            ]

        split_plan = compute_split_plan(self.stimuli, self.split_config)
        save_cache("benchmarks", config=config, payload=[plan.to_dict() for plan in split_plan])
        return split_plan

    def align_values(
        self,
        values: Any,
        stimulus_ids: Sequence[str] | None = None,
    ) -> np.ndarray:
        matrix = np.asarray(values, dtype=float)
        if matrix.ndim != 2:
            raise ValueError(f"Expected a 2D matrix of recordings, got shape {matrix.shape}.")

        benchmark_ids = self.stimuli.ids()
        if stimulus_ids is None:
            if matrix.shape[0] != len(benchmark_ids):
                raise ValueError(
                    f"Expected {len(benchmark_ids)} rows in benchmark order, got {matrix.shape[0]} rows."
                )
            return matrix

        if len(stimulus_ids) != matrix.shape[0]:
            raise ValueError("The number of stimulus ids must match the number of rows in the values matrix.")

        id_to_row = {str(stimulus_id): index for index, stimulus_id in enumerate(stimulus_ids)}
        missing = [stimulus_id for stimulus_id in benchmark_ids if stimulus_id not in id_to_row]
        if missing:
            raise ValueError(f"Missing recordings for benchmark stimulus ids: {missing[:5]}")
        ordered_rows = [matrix[id_to_row[stimulus_id]] for stimulus_id in benchmark_ids]
        return np.asarray(ordered_rows, dtype=float)

    def prepare_recorded_splits(
        self,
        values: Any,
        stimulus_ids: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> list[RecordedSplit]:
        aligned = self.align_values(values, stimulus_ids=stimulus_ids)
        metadata = dict(metadata or {})
        splits = []
        for split in self.build_split_plan():
            train_indices = list(split.train_indices)
            test_indices = list(split.test_indices)
            splits.append(
                RecordedSplit(
                    index=split.index,
                    train=np.asarray(aligned[train_indices], dtype=float),
                    test=np.asarray(aligned[test_indices], dtype=float),
                    train_stimuli=self.stimuli.subset(train_indices),
                    test_stimuli=self.stimuli.subset(test_indices),
                    metadata=metadata,
                )
            )
        return splits

    def load_human_recordings(
        self,
        selector: Mapping[str, Any] | None = None,
    ) -> HumanRecordingData:
        return self.human_recording_source.load_recordings(self, selector=selector)

    def iter_context_groups(self) -> list[list[int]]:
        ordered_groups: list[list[int]] = []
        group_indices: dict[str, list[int]] = {}
        for index, record in enumerate(self.stimuli):
            group_key = record.topic or f"stimulus-{index}"
            if group_key not in group_indices:
                group_indices[group_key] = []
                ordered_groups.append(group_indices[group_key])
            group_indices[group_key].append(index)
        return ordered_groups


class _InMemoryHumanRecordingSource(HumanRecordingSource):
    def __init__(
        self,
        *,
        identifier: str,
        storage_mode: str,
        recording_type: str,
        metadata: dict[str, Any],
        stimuli: StimulusSet,
        values: np.ndarray,
        feature_ids: list[str],
    ) -> None:
        super().__init__(
            identifier=identifier,
            storage_mode=storage_mode,
            recording_type=recording_type,
            metadata=metadata,
        )
        self.stimuli = stimuli
        self.values = values
        self.feature_ids = feature_ids

    def load_stimuli(self, benchmark: Benchmark) -> StimulusSet:
        return self.stimuli

    def load_recordings(
        self,
        benchmark: Benchmark,
        selector: Mapping[str, Any] | None = None,
    ) -> HumanRecordingData:
        metadata = dict(self.metadata)
        if selector:
            metadata["selector"] = dict(selector)
        return HumanRecordingData(
            values=np.asarray(self.values, dtype=float),
            stimulus_ids=self.stimuli.ids(),
            feature_ids=list(self.feature_ids),
            metadata=metadata,
        )


class InMemoryBenchmark(BaseBenchmark):
    def __init__(
        self,
        identifier: str,
        stimuli: StimulusSet | Sequence[Mapping[str, Any]],
        human_values: np.ndarray,
        feature_ids: Sequence[str] | None = None,
        *,
        cv: int = 1,
        topic_splitting: bool = True,
        train_size: float = 0.9,
        random_state: int = 42,
        storage_mode: str = "in_memory",
        recording_type: str = "synthetic",
    ):
        stimulus_set = stimuli if isinstance(stimuli, StimulusSet) else _stimulus_records_from_rows(stimuli)
        values = np.asarray(human_values, dtype=float)
        if values.ndim != 2:
            raise ValueError("human_values must be a 2D matrix.")
        if values.shape[0] != len(stimulus_set):
            raise ValueError("human_values row count must match the number of stimuli.")
        if feature_ids is None:
            feature_ids = [f"feature-{index}" for index in range(values.shape[1])]

        source = _InMemoryHumanRecordingSource(
            identifier=f"{identifier}.source",
            storage_mode=storage_mode,
            recording_type=recording_type,
            metadata={"kind": "in_memory"},
            stimuli=stimulus_set,
            values=values,
            feature_ids=list(feature_ids),
        )
        super().__init__(
            identifier=identifier,
            stimuli=stimulus_set,
            split_config=SplitConfig(cv=cv, topic_splitting=topic_splitting, train_size=train_size, random_state=random_state),
            human_recording_source=source,
        )
