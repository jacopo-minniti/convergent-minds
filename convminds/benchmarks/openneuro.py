from __future__ import annotations

import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from convminds.benchmarks.base import BaseBenchmark
from convminds.cache import convminds_home, load_cache, read_pickle, save_cache, write_pickle, file_signature
from convminds.data.alignment import build_word_aligned_story, align_tokens_to_trs
from convminds.data.cleaning import flatten_nifti, apply_pca
from convminds.data.events import load_events_tsv
from convminds.interfaces import HumanRecordingData, HumanRecordingSource, SplitConfig, StimulusRecord, StimulusSet

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpenNeuroSpec:
    name: str
    dataset_id: str
    tr: float
    description: str


HUTH_SPEC = OpenNeuroSpec(
    name="huth",
    dataset_id="ds003020",
    tr=2.0,
    description="fMRI responses to auditory narrative stories (Huth et al)."
)

NARRATIVES_SPEC = OpenNeuroSpec(
    name="narratives",
    dataset_id="ds002345",
    tr=1.5,
    description="fMRI responses to auditory narrative stories (Narratives dataset)."
)


class WordAlignedRecordingSource(HumanRecordingSource):
    """
    General Token-Level (Word-Aligned) Recording Source.
    Used for datasets where fMRI signals follow a temporal progression (e.g. Huth, Narratives).
    """
    def __init__(
        self,
        *,
        identifier: str,
        dataset_name: str,
        processed_path: str | Path,
        tr: float,
        alignment_window: int = 4,
        reduce: str = "mean",
        ensure_fn=None,
        use_cache: bool = True,
    ) -> None:
        metadata = {
            "dataset": dataset_name,
            "processed_path": str(processed_path),
            "tr": tr,
            "alignment_window": alignment_window,
            "reduce": reduce,
        }
        super().__init__(
            identifier=identifier,
            storage_mode="word_aligned",
            recording_type="fmri",
            metadata=metadata,
        )
        self.dataset_name = dataset_name
        self.processed_path = Path(processed_path).expanduser()
        self.tr = tr
        self.alignment_window = alignment_window
        self.reduce = reduce
        self.ensure_fn = ensure_fn
        self.use_cache = use_cache

    def load_stimuli(self, benchmark) -> StimulusSet:
        stimuli, _, _, _ = self._load_payload()
        return stimuli

    def load_recordings(
        self,
        benchmark,
        selector: Mapping[str, Any] | None = None,
    ) -> HumanRecordingData:
        stimuli, values, feature_ids, metadata = self._load_payload()
        from convminds.data.types import DataCategory
        return HumanRecordingData(
            values=values,
            stimulus_ids=stimuli.ids(),
            feature_ids=feature_ids,
            metadata=metadata,
            category=DataCategory.TOKEN_LEVEL,
        )

    def _load_payload(self) -> tuple[StimulusSet, np.ndarray, list[str], dict[str, Any]]:
        if not self.processed_path.exists() and self.ensure_fn is not None:
            self.ensure_fn()
            
        config = {
            "kind": "word-aligned",
            "dataset": self.dataset_name,
            "signature": file_signature(self.processed_path),
            "alignment_window": self.alignment_window,
            "reduce": self.reduce,
        }
        if self.use_cache:
            cached = load_cache("datasets", config=config)
            if cached is not None:
                from convminds.benchmarks.base import _stimulus_set_from_serialized
                stimuli = _stimulus_set_from_serialized(cached["stimuli"])
                return stimuli, np.asarray(cached["values"]), list(cached["feature_ids"]), dict(cached["metadata"])

        payload = read_pickle(self.processed_path)
        # Extract logic simplified for brevity (can be expanded later)
        # ... 
        # (Assuming existing payload structure is maintained for now)
        # ...
        return payload["stimuli"], payload["values"], payload["feature_ids"], payload["metadata"]


class OpenNeuroBenchmark(BaseBenchmark):
    @staticmethod
    def download_raw(dataset_id: str, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = ["aws", "s3", "sync", "--no-sign-request", f"s3://openneuro.org/{dataset_id}", str(output_dir)]
        subprocess.run(cmd, check=False)


def prepare_openneuro_processed(
    raw_dir: Path,
    output_path: Path,
    tr: float,
    **kwargs
) -> Path:
    """Explicit preliminary processing step for OpenNeuro time-series data."""
    logger.info(f"Applying preliminary transformation: Token Alignment & PCA on {raw_dir}")
    # ... (Actual implementation logic would go here, using alignment.py and cleaning.py)
    return output_path
