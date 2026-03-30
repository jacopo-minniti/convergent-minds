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


def _require_scipy():
    try:
        import scipy.io as sio
    except ModuleNotFoundError as error:
        raise RuntimeError("scipy is required for MATLAB (.mat) loading. Install it via pip.") from error
    return sio


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
        path = Path(image).expanduser()
        if path.suffix == ".mat":
            return load_mat_brain_data(path)
        nib = _require_nibabel()
        img = nib.load(str(path))
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


def load_mat_brain_data(
    path: str | Path,
    *,
    atlas_key: str | None = None,
    pool_rois: bool = False,
    enforce_shape: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    sio = _require_scipy()
    path = Path(path).expanduser()
    mat = sio.loadmat(str(path))
    
    # Check for Pereira 2018 standard format (examples + meta)
    if "examples" in mat and "meta" in mat:
        examples = mat["examples"] # format (stimuli, voxels)
        meta = mat["meta"][0, 0] # access first element of 1x1 struct array
        
        # 1. Resolve Initial indices and coordinates
        if "indicesIn3D" in meta.dtype.names:
            dims = meta["dimensions"].flatten() if "dimensions" in meta.dtype.names else [0, 0, 0]
            indices = meta["indicesIn3D"].flatten() - 1 # MATLAB is 1-indexed
            try:
                coords = np.column_stack(np.unravel_index(indices, dims))
            except Exception:
                coords = np.column_stack([np.arange(examples.shape[1]), np.zeros(examples.shape[1]), np.zeros(examples.shape[1])])
        else:
            n_voxels = examples.shape[1]
            coords = np.column_stack([np.arange(n_voxels), np.zeros(n_voxels), np.zeros(n_voxels)])

        # 2. Handle Atlas Filtering and ROI Pooling
        if atlas_key and atlas_key in meta.dtype.names:
            roi_ids = meta[atlas_key].flatten()
            
            if pool_rois:
                # Average voxels within each unique ROI (excluding 0 which is typically non-ROI)
                valid_roi_ids = np.unique(roi_ids[roi_ids > 0])
                pooled_examples = []
                pooled_coords = []
                for rid in valid_roi_ids:
                    roi_indices = np.where(roi_ids == rid)[0]
                    # Average activations
                    pooled_examples.append(examples[:, roi_indices].mean(axis=1))
                    # Average coordinates (centroid)
                    pooled_coords.append(coords[roi_indices].mean(axis=0))
                
                if pooled_examples:
                    examples = np.column_stack(pooled_examples)
                    coords = np.asarray(pooled_coords)
            else:
                # Standard voxel subsetting
                indices = np.where(roi_ids > 0)[0]
                examples = examples[:, indices]
                coords = coords[indices]
        
        # 3. Shape Enforcement (e.g. for multi-subject stacking if not pooled or if pool sizes differ)
        if enforce_shape is not None:
            examples = align_brain_vectors(examples, target_len=enforce_shape)
            if coords.shape[0] < enforce_shape:
                pad_len = enforce_shape - coords.shape[0]
                coords = np.pad(coords, ((0, pad_len), (0, 0)), mode='constant')
            elif coords.shape[0] > enforce_shape:
                coords = coords[:enforce_shape]

        return examples.astype(float), coords.astype(float)

    raise ValueError(f"MAT file {path} does not match any recognized Pereira/Neuro brain data format. "
                     f"Expected keys 'examples' and 'meta'. Found: {list(mat.keys())}")


def align_brain_vectors(data: np.ndarray, target_len: int) -> np.ndarray:
    """Ensures a brain activation matrix has a consistent number of voxels."""
    n_stim, n_voxels = data.shape
    if n_voxels == target_len:
        return data
    
    if n_voxels > target_len:
        return data[:, :target_len]
    else:
        # Pad with zeros
        padding = np.zeros((n_stim, target_len - n_voxels))
        return np.concatenate([data, padding], axis=1)


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
