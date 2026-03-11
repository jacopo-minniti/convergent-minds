from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import re


def _require_nibabel():
    try:
        import nibabel as nib
    except ModuleNotFoundError as error:
        raise RuntimeError("nibabel is required for NIfTI loading. Install it via pip.") from error
    return nib


def _require_pandas():
    try:
        import pandas as pd
    except ModuleNotFoundError as error:
        raise RuntimeError("pandas is required for TSV parsing. Install it via pip.") from error
    return pd


@dataclass(frozen=True)
class TokenEvent:
    text: str
    onset: float
    duration: float | None = None
    metadata: dict[str, object] | None = None


def load_events_tsv(
    path: str | Path,
    *,
    text_columns: Sequence[str] = ("word", "trial_type", "token", "text"),
    onset_column: str = "onset",
    duration_column: str = "duration",
) -> list[TokenEvent]:
    pd = _require_pandas()
    df = pd.read_csv(Path(path).expanduser(), sep="\t")

    text_column = None
    for candidate in text_columns:
        if candidate in df.columns:
            text_column = candidate
            break
    if text_column is None:
        raise ValueError(f"None of the text columns {text_columns} found in {path}.")
    if onset_column not in df.columns:
        raise ValueError(f"Expected onset column '{onset_column}' in {path}.")

    events: list[TokenEvent] = []
    for _, row in df.iterrows():
        text = str(row[text_column]).strip()
        if not text or text.lower() == "nan":
            continue
        onset = float(row[onset_column])
        duration = None
        if duration_column in df.columns:
            value = row[duration_column]
            if value == value:
                duration = float(value)
        metadata = {key: row[key] for key in df.columns if key not in {text_column, onset_column, duration_column}}
        events.append(TokenEvent(text=text, onset=onset, duration=duration, metadata=metadata))
    return events


def flatten_nifti(
    image: str | Path | np.ndarray,
    *,
    mask: str | Path | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(image, np.ndarray):
        data = image
    else:
        nib = _require_nibabel()
        img = nib.load(str(Path(image).expanduser()))
        data = np.asarray(img.get_fdata())

    if data.ndim == 3:
        data = data[..., np.newaxis]
    if data.ndim != 4:
        raise ValueError(f"Expected 3D or 4D fMRI data, got shape {data.shape}.")

    if mask is None:
        mask_data = np.any(data != 0, axis=3)
    elif isinstance(mask, np.ndarray):
        mask_data = mask.astype(bool)
    else:
        nib = _require_nibabel()
        mask_img = nib.load(str(Path(mask).expanduser()))
        mask_data = np.asarray(mask_img.get_fdata()) > 0

    coords = np.column_stack(np.where(mask_data))
    voxel_series = data[mask_data]
    matrix = voxel_series.T
    return matrix.astype(float), coords.astype(float)


def align_tokens_to_trs(
    tokens: Sequence[TokenEvent],
    *,
    tr: float,
    num_trs: int,
    delay: int = 4,
    window: int = 4,
    rounding: str = "floor",
    pad_mode: str = "repeat_last",
) -> list[list[int]]:
    if tr <= 0:
        raise ValueError("TR must be positive.")
    if window <= 0:
        raise ValueError("Window must be positive.")

    aligned: list[list[int]] = []
    for token in tokens:
        ratio = token.onset / tr
        if rounding == "round":
            base_index = int(round(ratio))
        elif rounding == "ceil":
            base_index = int(np.ceil(ratio))
        else:
            base_index = int(np.floor(ratio))

        start = base_index + delay
        indices = list(range(start, start + window))
        indices = [idx for idx in indices if 0 <= idx < num_trs]
        if len(indices) < window:
            if pad_mode == "repeat_last" and indices:
                indices = indices + [indices[-1]] * (window - len(indices))
            elif pad_mode == "skip":
                indices = []
        aligned.append(indices)
    return aligned


def build_word_aligned_story(
    fmri: np.ndarray,
    tokens: Sequence[TokenEvent],
    *,
    tr: float,
    delay: int = 4,
    window: int = 4,
    rounding: str = "floor",
    pad_mode: str = "repeat_last",
) -> dict[str, object]:
    num_trs = fmri.shape[0]
    alignment = align_tokens_to_trs(
        tokens,
        tr=tr,
        num_trs=num_trs,
        delay=delay,
        window=window,
        rounding=rounding,
        pad_mode=pad_mode,
    )
    word_nodes: list[dict[str, object]] = []
    for token, indices in zip(tokens, alignment):
        if not indices:
            continue
        node = {
            "word": token.text,
            "additional": indices,
        }
        if token.duration is not None:
            node["duration"] = token.duration
        if token.metadata:
            node.update(token.metadata)
        word_nodes.append(node)
    return {"fmri": fmri, "word": word_nodes}


def build_word_aligned_dataset(
    stories: Iterable[tuple[str, np.ndarray, Sequence[TokenEvent]]],
    *,
    tr: float,
    delay: int = 4,
    window: int = 4,
    rounding: str = "floor",
    pad_mode: str = "repeat_last",
) -> dict[str, dict[str, object]]:
    payload: dict[str, dict[str, object]] = {}
    for story_id, fmri, tokens in stories:
        payload[str(story_id)] = build_word_aligned_story(
            fmri,
            tokens,
            tr=tr,
            delay=delay,
            window=window,
            rounding=rounding,
            pad_mode=pad_mode,
        )
    return payload


def simple_tokenize(text: str) -> list[str]:
    cleaned = text.strip()
    if not cleaned:
        return []
    tokens = re.findall(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]", cleaned)
    return [token for token in tokens if token.strip()]


def build_sentence_level_story(
    fmri_vector: np.ndarray,
    text: str,
    *,
    alignment_window: int = 1,
    tokens: Sequence[str] | None = None,
) -> dict[str, object]:
    if alignment_window <= 0:
        raise ValueError("alignment_window must be positive.")
    fmri = np.asarray(fmri_vector, dtype=float).reshape(1, -1)
    token_list = list(tokens) if tokens is not None else simple_tokenize(text)
    indices = [0] * alignment_window
    word_nodes = [{"word": token, "additional": indices} for token in token_list]
    return {"fmri": fmri, "word": word_nodes}


def build_sentence_level_dataset(
    samples: Iterable[tuple[str, np.ndarray, str]],
    *,
    alignment_window: int = 1,
) -> dict[str, dict[str, object]]:
    payload: dict[str, dict[str, object]] = {}
    for stimulus_id, fmri_vector, text in samples:
        payload[str(stimulus_id)] = build_sentence_level_story(
            fmri_vector,
            text,
            alignment_window=alignment_window,
        )
    return payload


def apply_pca(
    data: np.ndarray,
    *,
    n_components: int = 1000,
    whiten: bool = False,
    random_state: int | None = 0,
) -> tuple[np.ndarray, object]:
    from sklearn.decomposition import PCA

    if data.ndim != 2:
        raise ValueError("apply_pca expects a 2D matrix (time, features).")
    pca = PCA(n_components=n_components, whiten=whiten, random_state=random_state)
    transformed = pca.fit_transform(data)
    return transformed.astype(float), pca
