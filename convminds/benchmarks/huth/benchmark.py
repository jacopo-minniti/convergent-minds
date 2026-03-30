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
        self.raw_dir = Path(raw_dir or convminds_home() / "data/huth/raw").expanduser()
        self.processed_path = Path(processed_path or convminds_home() / f"data/huth/processed/huth.{subject}.pkl").expanduser()
        
        # OpenNeuro naming: S1 -> UTS01 / sub-UTS01. We try both for robustness.
        self.raw_subject = subject
        id_num = int(subject[1:]) if subject.startswith("S") else int(subject.replace("UTS", "").replace("sub-", ""))
        self.subject_id = f"UTS{id_num:02d}"
        self.sub_prefix_id = f"sub-{self.subject_id}"
        
        # We will resolve the actual subject directory in ensure_data or prepare_processed
        self.subject = self.sub_prefix_id # Default
        
        self.trim = trim
        self.window = window
        self.use_datalad = use_datalad

        # 1. Ensure data is present
        self.ensure_data()

        # 2. Prepare processed stimuli and structure
        stimuli = self._load_or_prepare_stimuli()
        
        # 3. Setup Source
        source = HuthRecordingSource(ds_root=self.raw_dir)
        
        super().__init__(
            identifier=f"huth.{self.raw_subject}",
            stimuli=stimuli,
            split_config=SplitConfig(cv=1, topic_splitting=True, train_size=0.9, random_state=42),
            human_recording_source=source,
            description=f"Huth 2023 natural language fMRI benchmark for subject {self.raw_subject} ({self.subject})."
        )

    def ensure_data(self):
        """
        Uses DataLad to fetch the ds003020 dataset if it doesn't exist.
        """
        if not self.raw_dir.exists() or not (self.raw_dir / ".datalad").exists():
            logger.info(f"Huth dataset not found at {self.raw_dir}. Initializing DataLad clone...")
            self.raw_dir.parent.mkdir(parents=True, exist_ok=True)
            try:
                subprocess.run([
                    "datalad", "clone", "https://github.com/OpenNeuroDatasets/ds003020.git", str(self.raw_dir)
                ], check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to clone Huth dataset: {e}")
                raise

        # Ensure derivatives/preprocessed_data and derivatives/TextGrids are fetched
        # Note: OpenNeuro uses 'derivatives' (plural) and TextGrids
        derivatives_dir = self.raw_dir / "derivatives"
        if not (derivatives_dir / "preprocessed_data").exists() or not (derivatives_dir / "TextGrids").exists():
            logger.info("Fetching derivatives via DataLad...")
            
            # --- AGGRESSIVE RECOVERY LOGIC (CLUSTER-FRIENDLY) ---
            dataset_root = str(self.raw_dir)
            def run_git(args, check=False):
                return subprocess.run(["git"] + args, cwd=dataset_root, check=check, capture_output=True, text=True)
            
            def run_datalad(args, check=False):
                return subprocess.run(["datalad"] + args, cwd=dataset_root, check=check, capture_output=True, text=True)

            logger.info("Forcing S3 mirror connectivity...")
            # 1. Reset any 'annex-ignore' blocks
            run_git(["config", "remote.origin.annex-ignore", "false"])
            run_git(["config", "--unset", "remote.origin.annex-ignore"])
            
            # 2. Directly enable S3 mirrors (OpenNeuro's names)
            run_git(["annex", "enableremote", "s3-PUBLIC"])
            run_git(["annex", "enableremote", "s3-BACKUP"])
            
            # 3. Recursive 'get' for required data (executed from dataset root)
            try:
                # Ensure we have git identity else datalad fails (common on new clusters)
                res = run_git(["config", "--global", "user.email"])
                if not res.stdout.strip():
                    logger.info("Setting temporary git identity for DataLad...")
                    subprocess.run(["git", "config", "--global", "user.email", "convminds@google.com"], check=False)
                    subprocess.run(["git", "config", "--global", "user.name", "Convminds Bot"], check=False)

                # Fetch story metadata first (small files)
                logger.info("Fetching Huth metadata (TextGrids/respdict)...")
                run_datalad(["get", "-r", "derivatives/TextGrids"], check=True)
                run_datalad(["get", "derivatives/respdict.json"], check=True)
                
                # Fetch BOLD data for both possible candidate folders
                for cand in [self.sub_prefix_id, self.subject_id]:
                    logger.info(f"Fetching BOLD data for {cand}...")
                    # We try multiple sources explicitly if they didn't automount
                    p = f"derivatives/preprocessed_data/{cand}"
                    run_datalad(["get", "-r", p], check=False)
                    # If that failed, try forcing the source
                    if not (self.raw_dir / p).exists():
                         run_datalad(["get", "-r", "--source", "s3-PUBLIC", p], check=False)
                         run_datalad(["get", "-r", "--source", "s3-BACKUP", p], check=False)
            except subprocess.CalledProcessError as e:
                logger.warning(f"DataLad get attempt failed: {e}")
                if not (self.raw_dir / "derivatives/respdict.json").exists():
                     msg = f"Missing Huth metadata ({self.raw_dir / 'derivatives/respdict.json'}). Please verify internet access or run on a login node."
                     logger.error(msg)
                     raise FileNotFoundError(msg)

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
        
        # Opportunistically find correctly named subject directory
        def is_valid_dir(p: Path) -> bool:
            return p.exists() and any(f.exists() for f in p.glob("*.hf5"))
            
        subject_dir = derivatives_dir / "preprocessed_data" / self.sub_prefix_id
        if not is_valid_dir(subject_dir):
             alt_dir = derivatives_dir / "preprocessed_data" / self.subject_id
             if is_valid_dir(alt_dir):
                  subject_dir = alt_dir
                  self.subject = self.subject_id 
             elif alt_dir.exists():
                  # If alt exists but is empty/broken, still favor it over sub_prefix if sub_prefix was worse
                  subject_dir = alt_dir
                  self.subject = self.subject_id
        
        respdict_path = derivatives_dir / "respdict.json"
        
        with open(respdict_path, "r") as f:
            respdict = json.load(f)
            
        # Discover stories from the HDF5 files in the subject folder
        if not subject_dir.exists():
             raise FileNotFoundError(f"Subject folder not found: {subject_dir}")
             
        hf5_files = list(subject_dir.glob("*.hf5"))
        if not hf5_files:
             logger.warning(f"No BOLD HDF5 files found for subject {self.subject} in {subject_dir}")
             
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
