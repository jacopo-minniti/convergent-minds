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
        raw_dir: str | Path | None = None,
        processed_path: str | Path | None = None,
        subject: str = "S1",
        trim: int = 5,
        window: int = 3,
        use_datalad: bool = True,
    ) -> None:
        # 1. Subject normalization (S1 -> UTS01)
        self.raw_subject = subject
        id_num = int(subject[1:]) if subject.startswith("S") else int(subject.replace("UTS", "").replace("sub-", ""))
        self.subject_id = f"UTS{id_num:02d}"
        
        # 2. Paths
        self.raw_dir = Path(raw_dir or convminds_home() / "data/huth").expanduser()
        self.processed_path = Path(processed_path or convminds_home() / f"data/huth_processed/huth.{self.subject_id}.pkl").expanduser()
        
        self.trim = trim
        self.window = window
        self.use_datalad = use_datalad

        # 3. Ensure data is present
        self.ensure_data()

        # 4. Prepare processed stimuli and structure
        stimuli = self._load_or_prepare_stimuli()
        
        # 5. Setup Source
        source = HuthRecordingSource(ds_root=self.raw_dir)
        
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
        Uses DataLad to fetch the ds003020 dataset if it doesn't exist.
        Follows user instructions to download only necessary subject data.
        """
        # 1. Clone if not present
        if not self.raw_dir.exists() or not (self.raw_dir / ".datalad").exists():
            logger.info(f"Huth dataset not found at {self.raw_dir}. Initializing DataLad clone from OpenNeuro...")
            self.raw_dir.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([
                "datalad", "clone", "https://github.com/OpenNeuroDatasets/ds003020.git", str(self.raw_dir)
            ], check=True)

        # 2. Get Metadata (json files are already there but symlinks)
        # Actually OpenNeuro clones contain json files, but we should make sure respdict.json is there
        respdict_path = self.raw_dir / "derivatives/respdict.json"
        if not respdict_path.exists() or respdict_path.stat().st_size < 1000: # Symlinks are small
            logger.info("Materializing Huth metadata (respdict.json)...")
            self._run_datalad(["get", "derivatives/respdict.json"], self.raw_dir)

        # 3. Get Stimuli (TextGrids)
        textgrids_dir = self.raw_dir / "derivatives/TextGrids"
        # Check if dir is empty or contains only broken symlinks
        if not textgrids_dir.exists() or not any(f.is_file() and f.stat().st_size > 100 for f in textgrids_dir.glob("*.TextGrid")):
            logger.info("Materializing Huth stimuli (TextGrids)...")
            self._run_datalad(["get", "derivatives/TextGrids"], self.raw_dir)

        # 4. Get BOLD data for the specific subject
        subj_data_dir = self.raw_dir / "derivatives/preprocessed_data" / self.subject_id
        # The user mentioned {01} goes until {09}. The repo might have sub-UTS01 or just UTS01.
        # User explicitly said "derivatives/preprocessed_data/UTS01"
        if not subj_data_dir.exists() or not any(f.is_file() and f.stat().st_size > 1e6 for f in subj_data_dir.glob("*.hf5")):
            logger.info(f"Materializing Huth BOLD data for subject: {self.subject_id}...")
            # Try plain UTSXX first
            rel_path = f"derivatives/preprocessed_data/{self.subject_id}"
            try:
                self._run_datalad(["get", rel_path], self.raw_dir)
            except Exception:
                # Try sub-UTSXX if it fails
                logger.info(f"Retrying with sub- prefix for {self.subject_id}...")
                self._run_datalad(["get", f"derivatives/preprocessed_data/sub-{self.subject_id}"], self.raw_dir)
             
        logger.info("Huth data environment ready.")


    def _load_or_prepare_stimuli(self) -> StimulusSet:
        if self.processed_path.exists():
            logger.info(f"Loading cached Huth stimuli from {self.processed_path}")
            df = pd.read_pickle(self.processed_path)
            records = []
            for _, row in df.iterrows():
                records.append(StimulusRecord(
                    stimulus_id=row["stimulus_id"],
                    text=row["text"],
                    topic=row["story"],
                    metadata=row.get("metadata", {})
                ))
            return StimulusSet(records=records)

        return self.prepare_processed()

    def prepare_processed(self) -> StimulusSet:
        """
        Extracts story-level transcripts and TR timing.
        Returns one StimulusRecord per story.
        """
        derivatives_dir = self.raw_dir / "derivatives"
        stim_dir = derivatives_dir / "TextGrids"
        
        # 1. Resolve subject directory (could be UTS01 or sub-UTS01)
        subj_dir = derivatives_dir / "preprocessed_data" / self.subject_id
        if not subj_dir.exists():
            subj_dir = derivatives_dir / "preprocessed_data" / f"sub-{self.subject_id}"
            
        if not subj_dir.exists():
             raise FileNotFoundError(f"Subject folder not found: {self.subject_id}. Did datalad get fail?")
        
        respdict_path = derivatives_dir / "respdict.json"
        with open(respdict_path, "r") as f:
            respdict = json.load(f)
            
        # 2. Discover stories from the HDF5 files in the subject folder
        hf5_files = list(subj_dir.glob("*.hf5"))
        if not hf5_files:
             logger.warning(f"No BOLD HDF5 files found for subject {self.subject_id} in {subj_dir}")
             
        records = []
        rows = []
        
        for hf_path in hf5_files:
            story = hf_path.stem
            # Naming normalization to handle hyphen discrepancies (e.g. adventures-in-saying-yes vs adventuresinsayingyes)
            def normalize(s: str) -> str:
                return s.replace("-", "").replace("_", "").lower()
            
            norm_story = normalize(story)
            
            # Map TextGrids if not cached
            if not hasattr(self, "_tg_mapping"):
                 self._tg_mapping = {normalize(p.stem): p for p in stim_dir.glob("*.TextGrid")}
            
            tg_path = self._tg_mapping.get(norm_story)
            
            if not tg_path:
                logger.warning(f"TextGrid not found for story {story} (normalized: {norm_story})")
                continue
                
            # Story TR count from respdict (also normalized)
            if not hasattr(self, "_respdict_norm"):
                 self._respdict_norm = {normalize(k): v for k, v in respdict.items()}
            
            if norm_story not in self._respdict_norm:
                  logger.warning(f"Story {story} not found in respdict.json. Guessing length from HDF5.")
                  with h5py.File(hf_path, "r") as f:
                       n_trs = f["data"].shape[0]
            else:
                  n_trs = self._respdict_norm[norm_story]
                 
            tr_times = np.arange(n_trs) * 2.0
            
            # Skip if broken symlink (fetch failed)
            if not tg_path.exists():
                 logger.warning(f"Stimulus file not found (or broken symlink): {tg_path}")
                 continue

            with open(tg_path, "r") as f:
                content = f.read()
                tg = TextGrid(content)
                
            word_tier = tg.get_tier("words")
            if not word_tier:
                continue
            
            intervals = word_tier["intervals"]
            metadata = {
                "tr_times": tr_times.tolist(),
                "word_intervals": intervals,
                "n_trs": n_trs
            }
            
            full_text = " ".join([i["text"] for i in intervals if i["text"].strip()])
            
            rec = StimulusRecord(
                stimulus_id=story,
                text=full_text,
                topic=story,
                metadata=metadata
            )
            records.append(rec)
            rows.append({
                "stimulus_id": rec.stimulus_id,
                "text": rec.text,
                "story": story,
                "metadata": rec.metadata
            })
        # Cache the rows for faster reload
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
