from __future__ import annotations

import numpy as np
import torch

from convminds.data.neuro import (
    TokenEvent,
    align_tokens_to_trs,
    apply_pca,
    build_word_aligned_story,
    flatten_nifti,
)
from convminds.data.primitives import BrainTensor
from convminds.transforms import HRFWindow, VoxelPCA, ZScore


def test_align_tokens_to_trs_basic():
    tokens = [
        TokenEvent(text="hello", onset=0.0),
        TokenEvent(text="world", onset=3.9),
    ]
    indices = align_tokens_to_trs(tokens, tr=1.0, num_trs=10, delay=4, window=4, rounding="floor")
    assert indices[0] == [4, 5, 6, 7]
    assert indices[1] == [7, 8, 9, 9]


def test_build_word_aligned_story():
    fmri = np.arange(30, dtype=float).reshape(10, 3)
    tokens = [TokenEvent(text="hi", onset=0.0)]
    story = build_word_aligned_story(fmri, tokens, tr=1.0, delay=1, window=2)
    assert story["word"][0]["additional"] == [1, 2]
    assert np.allclose(story["fmri"], fmri)


def test_apply_pca_reduces_dimension():
    data = np.random.randn(6, 12)
    reduced, pca = apply_pca(data, n_components=3)
    assert reduced.shape == (6, 3)
    assert hasattr(pca, "components_")


def test_flatten_nifti_numpy_input():
    data = np.zeros((2, 2, 1, 4), dtype=float)
    data[0, 0, 0, :] = 1.0
    matrix, coords = flatten_nifti(data)
    assert matrix.shape == (4, 1)
    assert coords.shape == (1, 3)


def test_transforms_pipeline():
    signal = torch.randn(1, 4, 8)
    coords = torch.zeros(8, 3)
    brain = BrainTensor(signal=signal, coords=coords)

    zscore = ZScore(dim="time")
    zscore.fit(brain)
    normalized = zscore(brain)
    assert normalized.signal.shape == signal.shape

    pca = VoxelPCA(n_components=4)
    pca.fit(brain)
    reduced = pca(brain)
    assert reduced.signal.shape[-1] == 4

    window = HRFWindow(t=2)
    windowed = window(brain)
    assert windowed.signal.shape[1] == 2
