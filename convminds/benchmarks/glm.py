from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from convminds.benchmarks.base import BaseBenchmark
from convminds.interfaces import Benchmark, HumanRecordingData, HumanRecordingSource, SplitConfig, StimulusRecord, StimulusSet


def _safe_coord(assembly: Any, coord: str, index: int, default: Any = None) -> Any:
    if coord not in getattr(assembly, "coords", {}):
        return default
    value = assembly[coord].values[index]
    return value.item() if hasattr(value, "item") else value


def _coerce_identifier(value: Any) -> str:
    if hasattr(value, "item"):
        value = value.item()
    return str(value)


class PereiraGLMSource(HumanRecordingSource):
    def __init__(self, experiment: str):
        super().__init__(
            identifier="pereira2018.glm",
            storage_mode="glm",
            recording_type="fMRI",
            metadata={"dataset": "pereira2018", "experiment": experiment},
        )
        self.experiment = experiment
        self._assembly = None

    def _load_assembly(self):
        if self._assembly is not None:
            return self._assembly

        from convminds.brainscore.data.pereira2018 import load_pereira2018_language

        assembly = load_pereira2018_language()
        assembly = assembly.sel(experiment=self.experiment)
        if hasattr(assembly, "dropna"):
            assembly = assembly.dropna("neuroid")
        self._assembly = assembly
        return assembly

    def load_stimuli(self, benchmark: Benchmark) -> StimulusSet:
        assembly = self._load_assembly()
        sentence_counts: dict[str, int] = {}
        records = []
        for index in range(assembly.sizes["presentation"]):
            topic = _safe_coord(assembly, "passage_label", index, default=None)
            topic_value = str(topic) if topic is not None else None
            if topic_value is not None:
                sentence_counts[topic_value] = sentence_counts.get(topic_value, 0) + 1
            metadata = {
                "passage_category": _safe_coord(assembly, "passage_category", index, default=None),
                "stimulus_num": _safe_coord(assembly, "stimulus_num", index, default=index),
                "sentence_index": sentence_counts.get(topic_value, 1),
            }
            records.append(
                StimulusRecord(
                    stimulus_id=_coerce_identifier(_safe_coord(assembly, "stimulus_id", index)),
                    text=str(_safe_coord(assembly, "stimulus", index)),
                    topic=topic_value,
                    metadata=metadata,
                )
            )
        return StimulusSet(records=records)

    def load_recordings(
        self,
        benchmark: Benchmark,
        selector: Mapping[str, Any] | None = None,
    ) -> HumanRecordingData:
        assembly = self._load_assembly()
        selector = dict(selector or {})
        for coord_name, selected_values in selector.items():
            if coord_name not in assembly.coords:
                continue
            if not isinstance(selected_values, (list, tuple, set)):
                selected_values = [selected_values]
            mask = np.isin(assembly[coord_name].values, list(selected_values))
            assembly = assembly.sel(neuroid=mask) if "neuroid" in assembly[coord_name].dims else assembly.sel(presentation=mask)

        feature_ids = [_coerce_identifier(value) for value in assembly["neuroid_id"].values]
        stimulus_ids = [_coerce_identifier(value) for value in assembly["stimulus_id"].values]
        metadata = {
            "storage_mode": self.storage_mode,
            "recording_type": self.recording_type,
            "selector": selector,
            "neuroid_count": len(feature_ids),
        }
        return HumanRecordingData(
            values=np.asarray(assembly.values, dtype=float),
            stimulus_ids=stimulus_ids,
            feature_ids=feature_ids,
            metadata=metadata,
        )


class GLMBenchmark(BaseBenchmark):
    def __init__(
        self,
        dataset_name: str,
        *,
        experiment: str,
        cv: int = 1,
        topic_splitting: bool = True,
        train_size: float = 0.9,
        random_state: int = 42,
    ):
        if dataset_name.lower() != "pereira2018":
            raise ValueError("Only the 'pereira2018' GLM benchmark is implemented right now.")
        if not experiment:
            raise ValueError("GLMBenchmark('pereira2018') requires an explicit experiment value.")

        source = PereiraGLMSource(experiment=experiment)
        stimuli = source.load_stimuli(self)
        identifier = f"{dataset_name.lower()}.{experiment}.glm"
        super().__init__(
            identifier=identifier,
            stimuli=stimuli,
            split_config=SplitConfig(cv=cv, topic_splitting=topic_splitting, train_size=train_size, random_state=random_state),
            human_recording_source=source,
        )
        self.dataset_name = dataset_name.lower()
        self.experiment = experiment

    def benchmark_config(self) -> dict[str, Any]:
        config = super().benchmark_config()
        config.update(
            {
                "dataset_name": self.dataset_name,
                "experiment": self.experiment,
                "storage_mode": self.human_recording_source.storage_mode,
            }
        )
        return config
