from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import scipy.io as sio

from convminds.data.types import DataCategory
from convminds.interfaces import Benchmark, HumanRecordingData, HumanRecordingSource, StimulusSet
from convminds.cache import file_signature, load_cache, save_cache, read_pickle, write_pickle

logger = logging.getLogger(__name__)


def load_mat_brain_data(
    path: str | Path,
    *,
    atlas_key: str | None = None,
    pool_rois: bool = False,
    enforce_shape: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Lower-level utility for loading brain data from Pereira-style .mat files.
    This was moved to 'sources' to keep dataset-specific loading logic contained.
    """
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"MAT file not found: {path}")

    mat = sio.loadmat(str(path))
    
    # Standard Pereira format (examples + meta)
    if "examples" in mat and "meta" in mat:
        examples = mat["examples"] # (stimuli, voxels)
        meta = mat["meta"][0, 0]
        
        # 1. Resolve coordinates
        if "indicesIn3D" in meta.dtype.names:
            dims = meta["dimensions"].flatten() if "dimensions" in meta.dtype.names else [0, 0, 0]
            indices = meta["indicesIn3D"].flatten() - 1 
            try:
                coords = np.column_stack(np.unravel_index(indices, dims))
            except Exception:
                coords = np.column_stack([np.arange(examples.shape[1]), np.zeros(examples.shape[1]), np.zeros(examples.shape[1])])
        else:
            n_voxels = examples.shape[1]
            coords = np.column_stack([np.arange(n_voxels), np.zeros(n_voxels), np.zeros(n_voxels)])

        # 2. Handle ROI Pooling
        if atlas_key and atlas_key in meta.dtype.names:
            roi_ids = meta[atlas_key].flatten()
            
            if pool_rois:
                valid_roi_ids = np.unique(roi_ids[roi_ids > 0])
                logger.info(f"Applying preliminary transform: ROI Pooling ({len(valid_roi_ids)} ROIs) on {path.name}")
                pooled_examples = []
                pooled_coords = []
                for rid in valid_roi_ids:
                    roi_indices = np.where(roi_ids == rid)[0]
                    pooled_examples.append(examples[:, roi_indices].mean(axis=1))
                    pooled_coords.append(coords[roi_indices].mean(axis=0))
                
                if pooled_examples:
                    examples = np.column_stack(pooled_examples)
                    coords = np.asarray(pooled_coords)
            else:
                indices = np.where(roi_ids > 0)[0]
                examples = examples[:, indices]
                coords = coords[indices]
        
        if enforce_shape is not None:
            # Note: This is an explicit data alignment step
            n_stim, n_voxels = examples.shape
            if n_voxels != enforce_shape:
                logger.info(f"Preliminary aligning brain vectors: {n_voxels} -> {enforce_shape}")
                if n_voxels > enforce_shape:
                    examples = examples[:, :enforce_shape]
                    coords = coords[:enforce_shape]
                else:
                    padding = np.zeros((n_stim, enforce_shape - n_voxels))
                    examples = np.concatenate([examples, padding], axis=1)
                    pad_len = enforce_shape - coords.shape[0]
                    coords = np.pad(coords, ((0, pad_len), (0, 0)), mode='constant')

        return examples.astype(float), coords.astype(float)

    raise ValueError(f"MAT file {path.name} does not match the Pereira format (missing 'examples' or 'meta').")


class PereiraRecordingSource(HumanRecordingSource):
    """
    Dataset-specific loader for Pereira 2018.
    Encapsulates downloading, extraction, and preliminary ROI pooling.
    """
    def __init__(
        self,
        *,
        processed_path: str | Path,
        ensure_fn: Any | None = None,
        use_cache: bool = True,
    ) -> None:
        super().__init__(
            identifier="pereira-source",
            storage_mode="disk",
            recording_type="fmri-beta",
            metadata={"kind": "pereira-2018"},
        )
        self.processed_path = Path(processed_path).expanduser()
        self.ensure_fn = ensure_fn
        self.use_cache = use_cache

    def load_stimuli(self, benchmark: Benchmark) -> StimulusSet:
        # Stimuli are usually recovered from the same processed file as brain data
        return benchmark.stimuli

    def load_recordings(
        self,
        benchmark: Benchmark,
        selector: Mapping[str, Any] | None = None,
    ) -> HumanRecordingData:
        if not self.processed_path.exists() and self.ensure_fn is not None:
            logger.info("Pereira processed data missing. Running preliminary preparation...")
            self.ensure_fn()
            
        # Check cache for aligned data
        config = {
            "kind": "pereira-source",
            "path": str(self.processed_path),
            "signature": file_signature(self.processed_path),
        }
        if self.use_cache:
            cached = load_cache("datasets", config=config)
            if cached is not None:
                return HumanRecordingData(
                    values=np.asarray(cached["values"], dtype=float),
                    stimulus_ids=list(cached["stimulus_ids"]),
                    feature_ids=list(cached["feature_ids"]),
                    metadata=dict(cached["metadata"]),
                    category=DataCategory.STIMULUS_LEVEL,
                )

        payload = read_pickle(self.processed_path)
        # Extract from new dict-wrapped format containing coords
        if isinstance(payload, dict) and "payload" in payload:
            stim_payload = payload["payload"]
            top_metadata = {k: v for k, v in payload.items() if k != "payload"}
        else:
            stim_payload = payload
            top_metadata = {}

        # Reconstruct HumanRecordingData
        # (For Pereira, each key in stim_payload is a unique Subject:Stimulus ID)
        stim_ids = sorted(stim_payload.keys())
        values = []
        for sid in stim_ids:
            # stim_payload[sid] is a 'story' dict: {'fmri': matrix, 'word': nodes}
            values.append(stim_payload[sid]["fmri"][0])
            
        values_arr = np.asarray(values, dtype=float)
        feature_ids = [f"feature-{i}" for i in range(values_arr.shape[1])]
        
        if self.use_cache:
            save_cache("datasets", config=config, payload={
                "values": values_arr,
                "stimulus_ids": stim_ids,
                "feature_ids": feature_ids,
                "metadata": top_metadata,
            })

        return HumanRecordingData(
            values=values_arr,
            stimulus_ids=stim_ids,
            feature_ids=feature_ids,
            metadata=top_metadata,
            category=DataCategory.STIMULUS_LEVEL,
        )
