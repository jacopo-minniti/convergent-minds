from __future__ import annotations

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
        self.subject = subject
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
            identifier=f"huth.{subject}",
            stimuli=stimuli,
            split_config=SplitConfig(cv=1, topic_splitting=True, train_size=0.9, random_state=42),
            human_recording_source=source,
            description=f"Huth 2023 natural language fMRI benchmark for subject {subject}."
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

        # Ensure derivative/preprocessed_data and derivative/stimuli are fetched
        derivative_dir = self.raw_dir / "derivative"
        if not (derivative_dir / "preprocessed_data").exists() or not (derivative_dir / "stimuli").exists():
            logger.info("Fetching derivatives via DataLad...")
            try:
                # We fetch the specific subject and stimuli
                subprocess.run([
                    "datalad", "get", "-d", str(self.raw_dir), 
                    f"derivative/preprocessed_data/{self.subject}", 
                    "derivative/stimuli",
                    "derivative/respdict.json"
                ], check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to fetch Huth derivatives: {e}")
                # We don't raise here if some data exists, but ideally we should
                if not (self.raw_dir / "derivative/respdict.json").exists():
                     raise

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
        logger.info(f"Preparing Huth story-level stimuli for subject {self.subject}...")
        stim_dir = self.raw_dir / "derivative/stimuli"
        respdict_path = self.raw_dir / "derivative/respdict.json"
        
        with open(respdict_path, "r") as f:
            respdict = json.load(f)
            
        if self.subject not in respdict:
             raise ValueError(f"Subject {self.subject} not found in respdict.json")
             
        subject_stories = respdict[self.subject]
        records = []
        rows = []
        
        for story, tr_times in subject_stories.items():
            tg_path = stim_dir / f"{story}.TextGrid"
            if not tg_path.exists():
                logger.warning(f"TextGrid not found for story {story} at {tg_path}")
                continue
                
            with open(tg_path, "r") as f:
                content = f.read()
                tg = TextGrid(content)
                
            word_tier = tg.get_tier("words")
            if not word_tier:
                logger.warning(f"No 'words' tier found in {story}.TextGrid")
                continue
            
            # Extract word events
            intervals = word_tier["intervals"]
            
            # Metadata holds the timing for later alignment
            metadata = {
                "tr_times": tr_times,
                "word_intervals": intervals,
                "n_trs": len(tr_times)
            }
            
            # Standard 'text' is the whole transcript
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
