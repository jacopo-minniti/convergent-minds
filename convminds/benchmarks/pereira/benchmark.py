from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from convminds.benchmarks.base import BaseBenchmark
from convminds.data.sources.pereira import PereiraRecordingSource, load_mat_brain_data
from convminds.data.alignment import build_sentence_level_dataset
from convminds.interfaces import SplitConfig, StimulusRecord, StimulusSet
from convminds.cache import write_pickle

logger = logging.getLogger(__name__)


class PereiraBenchmark(BaseBenchmark):
    """
    Benchmark interface for the Pereira 2018 (Sentences) dataset.
    
    This benchmark aggregates brain responses to individual sentences.
    - Experiments 1, 2, and 3: fMRI beta images for sentences.
    - Default split: stimulus-balanced training/test split.
    """
    def __init__(
        self,
        *,
        raw_dir: str | Path | None = None,
        processed_path: str | Path | None = None,
        atlas_key: str = "roiMultimaskGordon",
        pool_rois: bool = True,
        enforce_shape: int | None = None,
        alignment_window: int = 4,
        reduce: str = "mean",
    ) -> None:
        raw_dir = Path(raw_dir or ".cache/data/pereira/raw").expanduser()
        processed_path = Path(processed_path or f".cache/data/pereira/processed/pereira.{atlas_key}.wq.pkl.dic").expanduser()
        
        self.raw_dir = raw_dir
        self.processed_path = processed_path
        
        # 1. Preliminary Transformation: Extract and Aligned Brain & Text data
        source = PereiraRecordingSource(
            processed_path=processed_path,
            ensure_fn=lambda: self.ensure_data(atlas_key=atlas_key, pool_rois=pool_rois, enforce_shape=enforce_shape, alignment_window=alignment_window),
        )
        
        # 2. Stimuli recovery (Pereira has unique sentence identifiers)
        # We need a first pass to get stimulus list for splitting before loading data
        if not processed_path.exists():
            self.ensure_data(atlas_key=atlas_key, pool_rois=pool_rois, enforce_shape=enforce_shape, alignment_window=alignment_window)
        
        from convminds.cache import read_pickle
        payload = read_pickle(processed_path)
        stim_payload = payload["payload"] if isinstance(payload, dict) and "payload" in payload else payload
        
        records = []
        for unique_id, story in stim_payload.items():
            # word-node: [{'word': text, 'additional': [0, 0, 0, 0]}]
            text = story["word"][0]["word"]
            subject_id, stim_id = unique_id.split(":", 1)
            # The topic is the original stimulus identifier (ensures we group cross-subject responses to the same sentence)
            records.append(StimulusRecord(stimulus_id=unique_id, text=text, topic=stim_id))
        
        stimuli = StimulusSet(records=records)
        
        super().__init__(
            identifier="pereira.benchmark",
            stimuli=stimuli,
            split_config=SplitConfig(cv=1, topic_splitting=True, train_size=0.9, random_state=42),
            human_recording_source=source,
            description="Pereira 2018 sentence-level fMRI benchmark (Experiment 1, 2, 3)."
        )

    def ensure_data(self, **kwargs):
        """Preliminary step: Prepare raw Pereira data into a standardized cache."""
        raw_dir = Path(self.raw_dir)
        output_path = Path(self.processed_path)
        
        if not raw_dir.exists() or not any(raw_dir.iterdir()):
            logger.info("Pereira raw data not found. Commencing automated download...")
            raw_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. Materials & Stimuli
            materials = {
                "IARPA_expt1_stim_images.zip": "https://www.dropbox.com/s/4bi7e7rds8thg11/IARPA_expt1_stim_images.zip?dl=1",
                "IARPA_expt1_stim_sents.zip": "https://www.dropbox.com/s/eughbv6dgrk05yf/IARPA_expt1_stim_sents.zip?dl=1",
                "IARPA_expt2_stim.zip": "https://www.dropbox.com/s/s4ulonvo4y1lj1x/IARPA_expt2_stim.zip?dl=1",
                "IARPA_expt3_stim.zip": "https://www.dropbox.com/s/kjs6s09w2py93o5/IARPA_expt3_stim.zip?dl=1",
                "DMN_overlap_n60.zip": "https://www.dropbox.com/s/obbx7mgmzd8t3h2/DMN_overlap_n60.zip?dl=1",
                "voxelInform_FractionMNI.nii": "https://www.dropbox.com/s/gmiubb0p73ushm3/voxelInform_FractionMNI.nii?dl=1",
            }
            import subprocess
            for name, url in materials.items():
                dest = raw_dir / name
                if not dest.exists():
                    logger.info(f"Downloading {name}...")
                    subprocess.run(["curl", "-L", "-o", str(dest), url], check=True)

            # 2. Subject activations (several GBs)
            subjects_url = "https://www.dropbox.com/sh/5z1ikn8osaao57w/AABg9LIlJfEOrQgZN7Sj7WHRa?dl=1"
            subjects_zip = raw_dir / "pereira_subjects_all.zip"
            if not subjects_zip.exists():
                logger.info("Downloading all subject brain data (~several GBs)...")
                subprocess.run(["curl", "-L", "-o", str(subjects_zip), subjects_url], check=True)

            # 3. Extraction
            import zipfile, tarfile
            logger.info("Extracting archives...")
            for zip_path in raw_dir.glob("*.zip"):
                with zipfile.ZipFile(zip_path, "r") as ref:
                    ref.extractall(raw_dir)
            for tar_path in raw_dir.rglob("*.tar"):
                with tarfile.open(tar_path, "r") as ref:
                    ref.extractall(raw_dir)
            
        self.prepare_processed(raw_dir, output_path=output_path, **kwargs)

    @staticmethod
    def prepare_processed(
        raw_dir: str | Path,
        *,
        output_path: str | Path | None = None,
        atlas_key: str | None = "roiMultimaskGordon",
        pool_rois: bool = True,
        enforce_shape: int | None = None,
        alignment_window: int = 4,
    ) -> Path:
        """
        Processes and aligns Pereira raw data (MATLAB/NIfTI).
        This is a PRELIMINARY step that is dataset-specific.
        """
        raw_dir = Path(raw_dir).expanduser()
        output_path = Path(output_path or ".cache/data/pereira/processed/pereira.processed.pkl.dic").expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Preparing preliminary processed data for Pereira: {output_path.name}...")
        
        mat_files = sorted(list(raw_dir.rglob("examples_*.mat")))
        if not mat_files:
            raise FileNotFoundError(f"No Pereira MAT data files found in {raw_dir}")

        manifest_path = next(raw_dir.rglob("sentence.csv"), None)
        if not manifest_path:
             raise FileNotFoundError("Could not locate Pereira stimulus manifest (sentence.csv).")
        
        df = pd.read_csv(manifest_path)
        
        vectors = []
        metadata = []
        coords = None

        for mat_path in mat_files:
            subject_id = str(mat_path.parent.name)
            matrix, current_coords = load_mat_brain_data(mat_path, atlas_key=atlas_key, pool_rois=pool_rois, enforce_shape=enforce_shape)
            if coords is None:
                coords = current_coords
            
            import scipy.io as sio
            mat = sio.loadmat(str(mat_path))
            id_keys = ["keyConcept", "labelsSentences", "keySentences", "labelsConcept"]
            ids = None
            for k in id_keys:
                if k in mat:
                    ids = mat[k]
                    break
            
            if ids is None: continue
            ids = [str(id[0][0] if isinstance(id[0], (list, np.ndarray)) else id[0]) for id in ids]
            
            for i, stim_id in enumerate(ids):
                # Match to manifest
                match = df[df["sentence_id"].astype(str).str.contains(stim_id)]
                if not match.empty:
                    text = str(match.iloc[0]["sentence"])
                    unique_id = f"{subject_id}:{stim_id}"
                    vectors.append(matrix[i])
                    metadata.append((unique_id, text))

        stacked = np.asarray(vectors, dtype=float)
        payload = build_sentence_level_dataset(
            ((stim_id, stacked[idx], stim_text) for idx, (stim_id, stim_text) in enumerate(metadata)),
            alignment_window=alignment_window,
        )
        
        full_payload = {
            "payload": payload,
            "id": "pereira",
            "atlas": atlas_key,
            "coords": coords,
        }
        write_pickle(output_path, full_payload)
        return output_path
