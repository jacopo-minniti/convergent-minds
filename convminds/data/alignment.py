from __future__ import annotations

from typing import Iterable, Sequence
import numpy as np
import re
from convminds.data.events import TokenEvent


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
    """Aligns temporal token onsets to fMRI Repetition Times (TR)."""
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
    """Wraps TR-aligned tokens into a 'story' payload for TOKEN_LEVEL datasets."""
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


def build_sentence_level_story(
    fmri_vector: np.ndarray,
    text: str,
    *,
    alignment_window: int = 1,
) -> dict[str, object]:
    """
    Wraps single-vector fMRI data into a 'story' payload for STIMULUS_LEVEL datasets.
    Each whole-sentence activation is treated as a single token to maintain the generic data structure.
    """
    fmri = np.asarray(fmri_vector, dtype=float).reshape(1, -1)
    indices = [0] * alignment_window
    word_nodes = [{"word": text.strip(), "additional": indices}]
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


def build_word_aligned_dataset(
    stories: Iterable[tuple[str, np.ndarray, Sequence[TokenEvent]]],
    **,
    tr: float,
    **kwargs
) -> dict[str, dict[str, object]]:
    payload: dict[str, dict[str, object]] = {}
    for story_id, fmri, tokens in stories:
        payload[str(story_id)] = build_word_aligned_story(
            fmri,
            tokens,
            tr=tr,
            **kwargs
        )
    return payload


def simple_tokenize(text: str) -> list[str]:
    cleaned = text.strip()
    if not cleaned:
        return []
    tokens = re.findall(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]", cleaned)
    return [token for token in tokens if token.strip()]
