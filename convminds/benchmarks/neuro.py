from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from convminds.benchmarks.base import BaseBenchmark
from convminds.cache import convminds_home, load_cache, read_pickle, save_cache, write_pickle
from convminds.data.neuro import (
    apply_pca,
    build_sentence_level_dataset,
    build_word_aligned_story,
    flatten_nifti,
    load_events_tsv,
)
from convminds.interfaces import HumanRecordingData, HumanRecordingSource, SplitConfig, StimulusRecord, StimulusSet


def data_root() -> Path:
    root = os.environ.get("CONVMINDS_DATA")
    if root:
        return Path(root).expanduser()
    return convminds_home() / "data"


def file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime": int(stat.st_mtime),
    }


def download_openneuro(dataset_id: str, output_dir: str | Path) -> Path:
    output_path = Path(output_dir).expanduser().resolve() / dataset_id
    cmd = [
        "aws",
        "s3",
        "sync",
        "--no-sign-request",
        f"s3://openneuro.org/{dataset_id}",
        str(output_path),
    ]
    subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, check=False)
    return output_path


def download_osf(project_id: str, output_dir: str | Path) -> Path:
    output_path = Path(output_dir).expanduser().resolve()
    cmd = ["osf", "-p", project_id, "clone", str(output_path)]
    subprocess.run(cmd, check=False)
    return output_path


def _stimulus_set_from_serialized(payload: Sequence[Mapping[str, Any]]) -> StimulusSet:
    return StimulusSet(records=[StimulusRecord(**dict(item)) for item in payload])


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    dataset_id: str
    source: str
    tr: float | None
    description: str


HUTH_SPEC = DatasetSpec(
    name="huth",
    dataset_id="ds003020",
    source="openneuro",
    tr=2.0,
    description=(
        "fMRI responses to auditory narrative stories (8 participants, 27 stories, ~6 hours each). "
        "Preprocessed motion-corrected continuous BOLD responses with TR=2.0s."
    ),
)

NARRATIVES_SPEC = DatasetSpec(
    name="narratives",
    dataset_id="ds002345",
    source="openneuro",
    tr=1.5,
    description=(
        "fMRI responses to auditory narrative stories (subset of 28 participants). "
        "Preprocessed motion-corrected continuous BOLD responses with TR=1.5s."
    ),
)

PEREIRA_SPEC = DatasetSpec(
    name="pereira",
    dataset_id="crwz7",
    source="osf",
    tr=None,
    description=(
        "fMRI beta maps collected while participants read Wikipedia-style sentences "
        "(5 participants, 627 sentences). Each stimulus corresponds to a sentence-level activation map."
    ),
)


class WordAlignedRecordingSource(HumanRecordingSource):
    def __init__(
        self,
        *,
        identifier: str,
        dataset_name: str,
        processed_path: str | Path,
        description: str,
        tr: float | None,
        alignment_window: int = 4,
        reduce: str = "mean",
        fmri_key: str = "fmri",
        word_key: str = "word",
        indices_key: str = "additional",
        pad_mode: str = "repeat_last",
        ensure_fn=None,
        use_cache: bool = True,
    ) -> None:
        metadata = {
            "dataset": dataset_name,
            "processed_path": str(processed_path),
            "tr": tr,
            "alignment_window": alignment_window,
            "reduce": reduce,
            "description": description,
            "fmri_key": fmri_key,
            "word_key": word_key,
            "indices_key": indices_key,
            "pad_mode": pad_mode,
        }
        super().__init__(
            identifier=identifier,
            storage_mode="word_aligned",
            recording_type="fmri",
            metadata=metadata,
        )
        self.dataset_name = dataset_name
        self.processed_path = Path(processed_path).expanduser()
        self.description = description
        self.tr = tr
        self.alignment_window = alignment_window
        self.reduce = reduce
        self.fmri_key = fmri_key
        self.word_key = word_key
        self.indices_key = indices_key
        self.pad_mode = pad_mode
        self.ensure_fn = ensure_fn
        self.use_cache = use_cache

    def load_payload(self) -> tuple[StimulusSet, np.ndarray, list[str], dict[str, Any]]:
        return self._load_payload()

    def load_stimuli(self, benchmark) -> StimulusSet:
        stimuli, _, _, _ = self._load_payload()
        return stimuli

    def load_recordings(
        self,
        benchmark,
        selector: Mapping[str, Any] | None = None,
    ) -> HumanRecordingData:
        stimuli, values, feature_ids, metadata = self._load_payload()
        if selector:
            metadata = dict(metadata)
            metadata["selector"] = dict(selector)
        return HumanRecordingData(
            values=values,
            stimulus_ids=stimuli.ids(),
            feature_ids=feature_ids,
            metadata=metadata,
        )

    def _load_payload(self) -> tuple[StimulusSet, np.ndarray, list[str], dict[str, Any]]:
        if not self.processed_path.exists() and self.ensure_fn is not None:
            self.ensure_fn()
        if not self.processed_path.exists():
            raise FileNotFoundError(
                f"Processed dataset file not found at {self.processed_path}. "
                "Provide a valid path or generate the processed file first."
            )

        config = {
            "kind": "word-aligned",
            "dataset": self.dataset_name,
            "signature": file_signature(self.processed_path),
            "alignment_window": self.alignment_window,
            "reduce": self.reduce,
            "fmri_key": self.fmri_key,
            "word_key": self.word_key,
            "indices_key": self.indices_key,
            "pad_mode": self.pad_mode,
        }
        if self.use_cache:
            cached = load_cache("datasets", config=config)
            if cached is not None:
                stimuli = _stimulus_set_from_serialized(cached["stimuli"])
                values = np.asarray(cached["values"], dtype=float)
                feature_ids = list(cached["feature_ids"])
                metadata = dict(cached["metadata"])
                return stimuli, values, feature_ids, metadata

        payload = read_pickle(self.processed_path)
        stimuli, values, feature_ids, metadata = self._build_from_word_aligned(payload)
        if self.use_cache:
            save_cache(
                "datasets",
                config=config,
                payload={
                    "stimuli": stimuli.to_serializable(),
                    "values": np.asarray(values, dtype=float),
                    "feature_ids": list(feature_ids),
                    "metadata": metadata,
                },
            )
        return stimuli, np.asarray(values, dtype=float), list(feature_ids), metadata

    def _build_from_word_aligned(
        self,
        payload: Any,
    ) -> tuple[StimulusSet, np.ndarray, list[str], dict[str, Any]]:
        stories = list(self._iter_stories(payload))
        if not stories:
            raise ValueError("No word-aligned stories found in the processed dataset payload.")

        values: list[np.ndarray] = []
        records: list[StimulusRecord] = []

        for story_id, story in stories:
            fmri = np.asarray(story[self.fmri_key], dtype=float)
            words = story.get(self.word_key) or story.get("words")
            if not words:
                continue
            for word_index, word_info in enumerate(words):
                token, indices, extra_metadata = self._extract_word_info(word_info)
                if not token:
                    continue
                window_indices = self._normalize_indices(indices, fmri.shape[0])
                if not window_indices:
                    continue
                window_data = fmri[window_indices]
                if self.reduce == "mean":
                    sample = window_data.mean(axis=0)
                elif self.reduce == "none":
                    sample = window_data
                elif self.reduce == "flatten":
                    sample = window_data.reshape(-1)
                else:
                    raise ValueError(f"Unknown reduce mode: {self.reduce}")

                values.append(sample)
                record_metadata = {
                    "story_id": str(story_id),
                    "word_index": word_index,
                    "tr_indices": list(window_indices),
                }
                record_metadata.update(extra_metadata)
                records.append(
                    StimulusRecord(
                        stimulus_id=f"{story_id}:{word_index:06d}",
                        text=token,
                        topic=str(story_id),
                        metadata=record_metadata,
                    )
                )

        if not records:
            raise ValueError("No stimuli could be derived from the processed dataset payload.")

        value_array = np.asarray(values, dtype=float)
        feature_count = int(value_array.shape[-1]) if value_array.size else 0
        feature_ids = [f"feature-{index:05d}" for index in range(feature_count)]
        metadata = {
            "dataset": self.dataset_name,
            "description": self.description,
            "tr": self.tr,
            "alignment_window": self.alignment_window,
            "reduce": self.reduce,
        }
        return StimulusSet(records=records), value_array, feature_ids, metadata

    def _iter_stories(self, payload: Any) -> Iterable[tuple[str, Mapping[str, Any]]]:
        if isinstance(payload, dict):
            candidates = payload.items()
        elif isinstance(payload, (list, tuple)):
            candidates = enumerate(payload)
        else:
            raise ValueError("Processed payload must be a mapping or a list of story dictionaries.")

        for key, story in candidates:
            if not isinstance(story, Mapping):
                continue
            if self.fmri_key in story and (self.word_key in story or "words" in story):
                yield str(key), story

    def _extract_word_info(self, word_info: Any) -> tuple[str, Sequence[int], dict[str, Any]]:
        if isinstance(word_info, Mapping):
            token = str(
                word_info.get("word")
                or word_info.get("text")
                or word_info.get("token")
                or ""
            ).strip()
            indices = (
                word_info.get(self.indices_key)
                or word_info.get("indices")
                or word_info.get("tr_indices")
                or []
            )
            metadata = {
                key: value
                for key, value in word_info.items()
                if key not in {self.indices_key, "indices", "tr_indices", "word", "text", "token"}
            }
            return token, indices, metadata
        return str(word_info).strip(), [], {}

    def _normalize_indices(self, indices: Sequence[int], max_len: int) -> list[int]:
        cleaned = []
        for value in indices:
            try:
                index = int(value)
            except (TypeError, ValueError):
                continue
            if 0 <= index < max_len:
                cleaned.append(index)

        if self.alignment_window <= 0:
            return cleaned
        if len(cleaned) >= self.alignment_window:
            return cleaned[: self.alignment_window]
        if not cleaned:
            return []
        if self.pad_mode == "repeat_last":
            cleaned = cleaned + [cleaned[-1]] * (self.alignment_window - len(cleaned))
        elif self.pad_mode == "skip":
            return []
        return cleaned


class NeuroBenchmark(BaseBenchmark):
    def __init__(
        self,
        *,
        identifier: str,
        stimuli: StimulusSet,
        human_recording_source: HumanRecordingSource,
        description: str,
        split_config: SplitConfig | None = None,
    ) -> None:
        super().__init__(
            identifier=identifier,
            stimuli=stimuli,
            split_config=split_config or SplitConfig(),
            human_recording_source=human_recording_source,
            description=description,
        )


class HuthBenchmark(NeuroBenchmark):
    def __init__(
        self,
        *,
        processed_path: str | Path | None = None,
        alignment_window: int = 4,
        reduce: str = "mean",
        split_config: SplitConfig | None = None,
        description: str | None = None,
    ) -> None:
        spec = HUTH_SPEC
        description = description or spec.description
        processed_path = processed_path or (
            data_root() / spec.name / "processed" / f"{spec.name}.pca1000.wq.pkl.dic"
        )
        raw_dir = data_root() / spec.name / "raw"

        def _ensure():
            if processed_path.exists():
                return
            raw_dir.mkdir(parents=True, exist_ok=True)
            download_openneuro(spec.dataset_id, raw_dir)
            prepare_openneuro_processed(
                raw_dir,
                output_path=processed_path,
                tr=spec.tr or 2.0,
            )

        source = WordAlignedRecordingSource(
            identifier=f"{spec.name}.word-aligned",
            dataset_name=spec.name,
            processed_path=processed_path,
            description=description,
            tr=spec.tr,
            alignment_window=alignment_window,
            reduce=reduce,
            ensure_fn=_ensure,
        )
        stimuli, _, _, _ = source.load_payload()
        super().__init__(
            identifier=f"{spec.name}.benchmark",
            stimuli=stimuli,
            human_recording_source=source,
            description=description,
            split_config=split_config,
        )

    @staticmethod
    def download_raw(output_dir: str | Path) -> Path:
        return download_openneuro(HUTH_SPEC.dataset_id, output_dir)


class NarrativesBenchmark(NeuroBenchmark):
    def __init__(
        self,
        *,
        processed_path: str | Path | None = None,
        alignment_window: int = 4,
        reduce: str = "mean",
        split_config: SplitConfig | None = None,
        description: str | None = None,
    ) -> None:
        spec = NARRATIVES_SPEC
        description = description or spec.description
        processed_path = processed_path or (
            data_root() / spec.name / "processed" / f"{spec.name}.pca1000.wq.pkl.dic"
        )
        raw_dir = data_root() / spec.name / "raw"

        def _ensure():
            if processed_path.exists():
                return
            raw_dir.mkdir(parents=True, exist_ok=True)
            download_openneuro(spec.dataset_id, raw_dir)
            prepare_openneuro_processed(
                raw_dir,
                output_path=processed_path,
                tr=spec.tr or 1.5,
            )

        source = WordAlignedRecordingSource(
            identifier=f"{spec.name}.word-aligned",
            dataset_name=spec.name,
            processed_path=processed_path,
            description=description,
            tr=spec.tr,
            alignment_window=alignment_window,
            reduce=reduce,
            ensure_fn=_ensure,
        )
        stimuli, _, _, _ = source.load_payload()
        super().__init__(
            identifier=f"{spec.name}.benchmark",
            stimuli=stimuli,
            human_recording_source=source,
            description=description,
            split_config=split_config,
        )

    @staticmethod
    def download_raw(output_dir: str | Path) -> Path:
        return download_openneuro(NARRATIVES_SPEC.dataset_id, output_dir)


class PereiraBenchmark(NeuroBenchmark):
    def __init__(
        self,
        *,
        processed_path: str | Path | None = None,
        alignment_window: int = 4,
        reduce: str = "mean",
        split_config: SplitConfig | None = None,
        description: str | None = None,
    ) -> None:
        spec = PEREIRA_SPEC
        description = description or spec.description
        processed_path = processed_path or (
            data_root() / spec.name / "processed" / f"{spec.name}.pca1000.wq.pkl.dic"
        )
        raw_dir = data_root() / spec.name / "raw"

        def _ensure():
            if processed_path.exists():
                return
            
            # If the raw directory already has plenty of files, skip the slow OSF check
            if raw_dir.exists() and len(list(raw_dir.rglob("*"))) > 5:
                pass
            else:
                raw_dir.mkdir(parents=True, exist_ok=True)
                download_osf(spec.dataset_id, raw_dir)
            
            PereiraBenchmark.prepare_processed(
                raw_dir,
                output_path=processed_path,
                n_components=1000,
                alignment_window=alignment_window,
            )

        source = WordAlignedRecordingSource(
            identifier=f"{spec.name}.word-aligned",
            dataset_name=spec.name,
            processed_path=processed_path,
            description=description,
            tr=spec.tr,
            alignment_window=alignment_window,
            reduce=reduce,
            ensure_fn=_ensure,
        )
        stimuli, _, _, _ = source.load_payload()
        super().__init__(
            identifier=f"{spec.name}.benchmark",
            stimuli=stimuli,
            human_recording_source=source,
            description=description,
            split_config=split_config,
        )

    @staticmethod
    def download_raw(output_dir: str | Path) -> Path:
        return download_osf(PEREIRA_SPEC.dataset_id, output_dir)

    @staticmethod
    def prepare_processed(
        raw_dir: str | Path,
        *,
        output_path: str | Path | None = None,
        manifest_path: str | Path | None = None,
        n_components: int = 1000,
        alignment_window: int = 4,
    ) -> Path:
        raw_dir = Path(raw_dir).expanduser()
        output_path = Path(output_path).expanduser() if output_path else (
            data_root() / "pereira" / "processed" / "pereira.pca1000.wq.pkl.dic"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Handle archived data from OSF
        zip_files = list(raw_dir.rglob("*.zip"))
        if zip_files:
            import zipfile
            for zip_path in zip_files:
                # Lazy extraction: skip if a folder with the same name already exists
                # to avoid re-extracting and potentially hitting corrupted nested files
                target_dir = zip_path.parent / zip_path.stem
                if target_dir.exists() and any(target_dir.iterdir()):
                    continue
                
                try:
                    with zipfile.ZipFile(zip_path, "r") as zip_ref:
                        # Extract to the directory containing the zip
                        zip_ref.extractall(zip_path.parent)
                except (zipfile.BadZipFile, PermissionError, OSError):
                    # Skip corrupted or locked files instead of crashing
                    continue

        manifest_path = _resolve_manifest_path(raw_dir, manifest_path)
        df, id_col, text_col, beta_col = _read_manifest(manifest_path)

        vectors: list[np.ndarray] = []
        metadata: list[tuple[str, str]] = []
        for _, row in df.iterrows():
            stimulus_id = str(row[id_col])
            text = str(row[text_col])
            beta_path = raw_dir / str(row[beta_col])
            matrix, _ = flatten_nifti(beta_path)
            vector = matrix.mean(axis=0)
            vectors.append(vector)
            metadata.append((stimulus_id, text))

        if not vectors:
            raise ValueError(f"No beta maps found in manifest {manifest_path}.")

        stacked = np.asarray(vectors, dtype=float)
        reduced, _ = apply_pca(stacked, n_components=n_components)

        payload = build_sentence_level_dataset(
            ((stimulus_id, reduced[index], text) for index, (stimulus_id, text) in enumerate(metadata)),
            alignment_window=alignment_window,
        )
        write_pickle(output_path, payload)
        return output_path


def prepare_openneuro_processed(
    raw_dir: str | Path,
    *,
    output_path: str | Path,
    tr: float,
    n_components: int = 1000,
    delay: int = 4,
    window: int = 4,
) -> Path:
    raw_dir = Path(raw_dir).expanduser()
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bold_files = sorted(
        path for path in raw_dir.rglob("*_bold.nii*") if path.is_file()
    )
    if not bold_files:
        raise FileNotFoundError(f"No BOLD NIfTI files found under {raw_dir}.")

    stories: list[tuple[str, np.ndarray, list]] = []
    for bold_path in bold_files:
        events_path = _match_events_path(bold_path)
        if events_path is None:
            continue
        fmri, _ = flatten_nifti(bold_path)
        tokens = load_events_tsv(events_path)
        if not tokens:
            continue
        story_id = bold_path.stem.replace("_bold", "")
        stories.append((story_id, fmri, tokens))

    if not stories:
        raise ValueError("No aligned (bold, events) pairs found for OpenNeuro dataset.")

    concatenated = np.concatenate([fmri for _, fmri, _ in stories], axis=0)
    reduced, pca = apply_pca(concatenated, n_components=n_components)

    payload: dict[str, dict[str, object]] = {}
    offset = 0
    for story_id, fmri, tokens in stories:
        length = fmri.shape[0]
        reduced_story = reduced[offset : offset + length]
        offset += length
        payload[str(story_id)] = build_word_aligned_story(
            reduced_story,
            tokens,
            tr=tr,
            delay=delay,
            window=window,
        )

    write_pickle(output_path, payload)
    return output_path


def _match_events_path(bold_path: Path) -> Path | None:
    name = bold_path.name
    if "_bold" not in name:
        return None
    events_name = name.replace("_bold", "_events")
    if events_name.endswith(".nii.gz"):
        events_name = events_name.replace(".nii.gz", ".tsv")
    elif events_name.endswith(".nii"):
        events_name = events_name.replace(".nii", ".tsv")
    events_path = bold_path.with_name(events_name)
    if events_path.exists():
        return events_path
    return None


def _resolve_manifest_path(raw_dir: Path, manifest_path: str | Path | None) -> Path:
    if manifest_path is not None:
        return Path(manifest_path).expanduser()

    # Common locations and names for neuro-benchmark manifests
    search_dirs = [raw_dir, raw_dir / "osfstorage"]
    names = [
        "manifest.tsv", "stimuli.tsv", "metadata.tsv",
        "manifest.csv", "stimuli.csv", "metadata.csv",
        "stimuli_metadata.csv", "stimuli_metadata.tsv",
        "pereira_stimuli.csv", "sentences.csv"
    ]

    for d in search_dirs:
        if not d.exists():
            continue
        for name in names:
            path = d / name
            if path.exists():
                return path

    # Priority 1: Files actually named 'manifest' or 'metadata'
    for ext in ["*.csv", "*.tsv", "*.txt"]:
        for path in raw_dir.rglob(ext):
            lowered = path.name.lower()
            if any(kw in lowered for kw in ["manifest", "metadata"]):
                return path

    # Priority 2: Stimuli or sentence files
    for ext in ["*.csv", "*.tsv", "*.txt"]:
        for path in raw_dir.rglob(ext):
            lowered = path.name.lower()
            if any(kw in lowered for kw in ["stimuli", "sentence"]):
                return path

    raise FileNotFoundError(
        "Could not locate a manifest file. Provide manifest_path with columns for "
        "stimulus id, text, and beta_path."
    )


def _read_manifest(path: Path) -> tuple["pd.DataFrame", str, str, str]:
    try:
        import pandas as pd
    except ModuleNotFoundError as error:
        raise RuntimeError("pandas is required to read the Pereira manifest file.") from error

    sep = "\t" if path.suffix in [".tsv", ".txt"] else ","
    encoding = "utf-8"
    
    print(f"Attempting to read manifest: {path}")
    try:
        df = pd.read_csv(path, sep=sep, encoding=encoding)
    except UnicodeDecodeError:
        encoding = "latin-1"
        df = pd.read_csv(path, sep=sep, encoding=encoding)

    # If the default separator failed to produce multiple columns, try whitespace
    if len(df.columns) < 2:
        try:
            df = pd.read_csv(path, sep=r"\s+", engine="python", encoding=encoding)
        except Exception:
            pass
    
    # Heuristic: Check if we have any matching headers. If not, assume first row is data.
    all_targets = ["id", "sentence", "text", "beta", "path", "content"]
    has_headers = any(any(t in str(col).lower() for t in all_targets) for col in df.columns)
    
    if not has_headers:
        # Re-read without headers using the same successful encoding
        df = pd.read_csv(path, sep=sep, header=None, encoding=encoding)
        if len(df.columns) >= 3:
            df.columns = ["id", "text", "beta_path"] + [f"extra_{i}" for i in range(len(df.columns)-3)]
        elif len(df.columns) == 2:
            df.columns = ["id", "text"]
        elif len(df.columns) == 1:
            # Special case: it's just a text file with one sentence per line
            df.columns = ["text"]
            df["id"] = range(len(df)) # Auto-generate IDs
            # If there's no path column, the prepare_processed() logic might need 
            # to assume the beta maps follow the same order in the directory.

    id_col = _pick_column(df, ["stimulus_id", "sentence_id", "item_id", "id", "index"])
    text_col = _pick_column(df, ["text", "sentence", "stimulus", "prompt", "content"])
    beta_col = _pick_column(df, ["beta_path", "path", "file", "image", "beta", "filename"])
    return df, id_col, text_col, beta_col


def _pick_column(df: "pd.DataFrame", candidates: Sequence[str]) -> str:
    # Exact match
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    
    # CASE INSENSITIVE MATCH
    lower_columns = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_columns:
            return lower_columns[candidate.lower()]

    # If all else fails, and there are few columns, guess by position if names don't match
    if len(df.columns) <= 3:
        if "stimulus_id" in candidates:
            return df.columns[0] if len(df.columns) > 1 else df.columns[0] # Just take first as ID/Text fallback
        if "text" in candidates:
            return df.columns[1] if len(df.columns) > 1 else df.columns[0]
        if "beta_path" in candidates:
            return df.columns[2] if len(df.columns) > 2 else df.columns[-1]

    raise ValueError(f"None of the candidate columns {candidates} found in manifest. Available columns: {list(df.columns)}")
