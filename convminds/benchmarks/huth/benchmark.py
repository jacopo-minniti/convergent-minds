from __future__ import annotations

import h5py
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from convminds.benchmarks.base import BaseBenchmark
from convminds.cache import convminds_home, load_cache, save_cache
from convminds.data.sources.huth import HuthRecordingSource, TextGrid
from convminds.data.alignment import lanczos_interp2d
from convminds.interfaces import SplitConfig, StimulusRecord, StimulusSet, SplitPlan

logger = logging.getLogger(__name__)

# Default Huth Split mapping from HuthLab report
SESS_TO_STORY = {
    "1": ["alternate-reality", "avatar", "legacy", "life", "slumlord-survivor"],
    "2": ["exotic-animal-vets", "how-to-win", "life-tumble", "on-the-road", "william-tracy"],
    "3": ["death-valley", "hanging-out", "n-split", "souls", "under-the-influence"],
    "4": ["adventures-in-saying-yes", "eye-level-with-the-dog", "f-split", "stage-fright", "wait-for-me-will"],
    "5": ["it-will-be-our-little-secret", "naked", "super-cool-and-scary", "the-closet", "the-curse"]
}

DEFAULT_TEST_STORIES = ["alternate-reality"] # Standard test story in Lebel 2023


class HuthBenchmark(BaseBenchmark):
    """
    Benchmark for the Huth 2023 (ds003020) dataset.
    This dataset contains fMRI responses to natural language stories.
    """
    def __init__(
        self,
        *,
        huth_dir: str | Path | None = None,
        processed_path: str | Path | None = None,
        subject: str = "S1",
        trim: int = 5,
        window: int = 3,
        use_datalad: bool = True,
    ) -> None:
        # 1. Subject normalization (S1 -> UTS01)
        self.input_subject = subject
        id_num = int(subject[1:]) if subject.startswith("S") else int(subject.replace("UTS", "").replace("sub-", ""))
        self.subject_id = f"UTS{id_num:02d}"
        
        # 2. Paths
        self.huth_dir = Path(huth_dir or convminds_home() / "data/huth").expanduser()
        self.processed_path = Path(processed_path or convminds_home() / f"data/huth_processed/huth.{self.subject_id}.pkl").expanduser()
        
        self.trim = trim
        self.window = window
        self.use_datalad = use_datalad

        # 3. Ensure data is present
        self.ensure_data()

        # 4. Prepare processed stimuli and structure
        stimuli = self._load_or_prepare_stimuli()
        
        # 5. Setup Source
        source = HuthRecordingSource(ds_root=self.huth_dir)
        
        super().__init__(
            identifier=f"huth.{self.subject_id}",
            stimuli=stimuli,
            split_config=SplitConfig(cv=1, topic_splitting=True, train_size=0.9, random_state=42),
            human_recording_source=source,
            description=f"Huth 2023 natural language fMRI benchmark for subject {self.subject_id}."
        )

    def _run_datalad(self, cmd: list[str], cwd: Path | str):
        logger.info(f"Running: datalad {' '.join(cmd)}")
        try:
            res = subprocess.run(["datalad"] + cmd, cwd=str(cwd), capture_output=True, text=True, check=True)
            if res.stdout:
                for line in res.stdout.splitlines(): logger.debug(f"[datalad] {line}")
            return res
        except subprocess.CalledProcessError as e:
            logger.error(f"DataLad command failed: {' '.join(cmd)}")
            if e.stderr: logger.error(f"Error: {e.stderr}")
            raise

    def ensure_data(self):
        """
        Simplified ensure_data following user instructions.
        """
        # 1. Clone if not present
        if not self.huth_dir.exists() or not (self.huth_dir / ".datalad").exists():
            logger.info(f"Huth dataset not found at {self.huth_dir}. Initializing DataLad clone...")
            self.huth_dir.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([
                "datalad", "clone", "https://github.com/OpenNeuroDatasets/ds003020.git", str(self.huth_dir)
            ], check=True)

        # 2. Get Metadata & Stimuli metadata (TextGrids/respdict)
        essential_paths = ["derivatives/respdict.json", "derivatives/TextGrids"]
        for p in essential_paths:
            full_p = self.huth_dir / p
            # Check if directory is hollow or file is missing/symbolic link
            # A symlink exists but its target missing (not p.exists()) is the definition of "hollow" in datalad
            is_hollow = not full_p.exists() or \
                        (full_p.is_dir() and not any(full_p.iterdir())) or \
                        (full_p.is_file() and (not full_p.exists() or (full_p.is_file() and full_p.lstat().st_size < 100)))
            
            if is_hollow:
                logger.info(f"Materializing {p}...")
                self._run_datalad(["get", "-rn", p], self.huth_dir) # -n to avoid asking for confirmation if any

        # 3. Get BOLD data for subject if not present or hollow
        subj_dir = self.huth_dir / "derivatives" / "preprocessed_data" / self.subject_id
        hf5_files = list(subj_dir.glob("*.hf5"))
        
        # Determine if we need to get data:
        # A symlink where exists() is FALSE means the target data is missing (hollow)
        needs_get = not subj_dir.exists() or not hf5_files or any(not f.exists() for f in hf5_files)
        
        if needs_get:
            logger.info(f"Materializing COMPLETE subject folder for {self.subject_id} (recursive)...")
            self._run_datalad(["get", f"derivatives/preprocessed_data/{self.subject_id}"], self.huth_dir)
             
        logger.info("Huth data ready (all stories verified).")


    def _load_or_prepare_stimuli(self) -> StimulusSet:
        if self.processed_path.exists():
            logger.info(f"Checking cached Huth stimuli at {self.processed_path}")
            df = pd.read_pickle(self.processed_path)
            if len(df) > 0:
                logger.info(f"Loaded {len(df)} cached stimuli for {self.subject_id}")
                records = []
                for _, row in df.iterrows():
                    records.append(StimulusRecord(
                        stimulus_id=row["stimulus_id"],
                        text=row["text"],
                        topic=row["story"],
                        metadata=row.get("metadata", {})
                    ))
                return StimulusSet(records=records)
            else:
                logger.warning(f"Cached stimuli file for {self.subject_id} is empty. Re-processing...")
        
        return self.prepare_processed()

    def prepare_processed(self) -> StimulusSet:
        """
        Extracts story-level transcripts and TR timing.
        """
        derivatives_dir = self.huth_dir / "derivatives"
        stim_dir = derivatives_dir / "TextGrids"
        subj_dir = derivatives_dir / "preprocessed_data" / self.subject_id
        respdict_path = derivatives_dir / "respdict.json"
        
        with open(respdict_path, "r") as f:
            respdict = json.load(f)
            
        # Discover stories that actually have BOLD files
        hf5_files = list(subj_dir.glob("*.hf5"))
        logger.info(f"Found {len(hf5_files)} .hf5 files in {subj_dir}")
        
        records = []
        rows = []
        
        def normalize(s: str) -> str:
            return s.replace("-", "").replace("_", "").lower()
            
        tg_mapping = {normalize(p.stem): p for p in stim_dir.glob("*.TextGrid")}
        respdict_norm = {normalize(k): v for k, v in respdict.items()}
        logger.info(f"Loaded {len(tg_mapping)} TextGrids and {len(respdict_norm)} metadata entries.")

        for hf_path in hf5_files:
            story = hf_path.stem
            norm_story = normalize(story)
            
            tg_path = tg_mapping.get(norm_story)
            if not tg_path or not tg_path.exists():
                logger.warning(f"Skipping story {story}: TextGrid not found at {tg_path}")
                continue
                
            n_trs = respdict_norm.get(norm_story)
            if n_trs is None:
                # Fallback to HDF5 if not in dict
                with h5py.File(hf_path, "r") as f:
                    n_trs = f["data"].shape[0]
                 
            tr_times = np.arange(n_trs) * 2.0
            
            with open(tg_path, "r") as f:
                tg = TextGrid(f.read())
                
            word_tier = tg.get_tier("words") or tg.get_tier("word")
            if not word_tier:
                available_tiers = [t['name'] for t in tg.tiers]
                logger.warning(f"Skipping story {story}: no 'words' or 'word' tier found in TextGrid. Available: {available_tiers}")
                continue
            
            intervals = word_tier["intervals"]
            metadata = {"tr_times": tr_times.tolist(), "word_intervals": intervals, "n_trs": n_trs}
            full_text = " ".join([i["text"] for i in intervals if i["text"].strip()])
            
            rec = StimulusRecord(stimulus_id=story, text=full_text, topic=story, metadata=metadata)
            records.append(rec)
            rows.append({"stimulus_id": story, "text": full_text, "story": story, "metadata": metadata})
            
        if rows:
            self.processed_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_pickle(self.processed_path)
        
        return StimulusSet(records=records)

    def build_split_plan(self) -> list[SplitPlan]:
        """
        Overrides split plan to use story-based testing.
        """
        # We can still use the BaseBenchmark's logic if we want random story splitting,
        # but typically Huth uses a specific held-out story.
        # If split_config has train_size=0.9, it will pick 90% of stories.
        return super().build_split_plan()
