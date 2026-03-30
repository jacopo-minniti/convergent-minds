from __future__ import annotations

from pathlib import Path
import numpy as np
from convminds.data.io import _require_nibabel


def flatten_nifti(
    image: str | Path | np.ndarray,
    *,
    mask: str | Path | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Flattens 4D NIfTI brain data into 2D (time, voxels) using a mask."""
    if isinstance(image, np.ndarray):
        data = image
    else:
        path = Path(image).expanduser()
        # Note: MATLAB loading delegated to dataset-specific sources
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


def align_brain_vectors(data: np.ndarray, target_len: int) -> np.ndarray:
    """Ensures a brain activation matrix has a consistent number of voxels through padding/truncation."""
    n_stim, n_voxels = data.shape
    if n_voxels == target_len:
        return data
    
    if n_voxels > target_len:
        return data[:, :target_len]
    else:
        # Pad with zeros
        padding = np.zeros((n_stim, target_len - n_voxels))
        return np.concatenate([data, padding], axis=1)


def apply_pca(
    data: np.ndarray,
    *,
    n_components: int = 1000,
    whiten: bool = False,
    random_state: int | None = 0,
) -> tuple[np.ndarray, object]:
    """Explicit dimensionality reduction via PCA."""
    from sklearn.decomposition import PCA

    if data.ndim != 2:
        raise ValueError("apply_pca expects a 2D matrix (time, features).")
    pca = PCA(n_components=n_components, whiten=whiten, random_state=random_state)
    transformed = pca.fit_transform(data)
    return transformed.astype(float), pca
