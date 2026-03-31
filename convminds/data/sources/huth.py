from __future__ import annotations

import h5py
import json
import logging
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from convminds.cache import load_cache, save_cache, file_signature
from convminds.interfaces import Benchmark, HumanRecordingData, HumanRecordingSource, StimulusRecord, StimulusSet
from convminds.data.types import DataCategory

logger = logging.getLogger(__name__)


class TextGrid:
    """
    Minimal TextGrid parser for Praat files.
    Focuses on IntervalTiers used in the Huth 2023 dataset.
    """
    def __init__(self, content: str):
        self.tiers = []
        self._parse(content)

    def _parse(self, content: str):
        # 1. Split into tier blocks (item [1]:, item [2]:, etc.)
        item_blocks = re.split(r'item\s*\[\s*\d+\s*\]\s*:?', content)
        
        for block in item_blocks[1:]:
            # Find the tier name (name = "word")
            name_match = re.search(r'name\s*=\s*"(.*?)"', block)
            if not name_match:
                continue
            
            tier_name = name_match.group(1).lower().strip()
            
            # Find starts of each interval [X] within this tier block
            starts = [m.start() for m in re.finditer(r'intervals\s*\[\s*\d+\s*\]', block)]
            if not starts:
                continue
            
            intervals = []
            for i in range(len(starts)):
                # Take everything from current start to the next start (or end of block)
                end = starts[i+1] if i + 1 < len(starts) else len(block)
                segment = block[starts[i]:end]
                
                xmin_m = re.search(r'xmin\s*[=:]\s*([\d\.\-\+e]+)', segment)
                xmax_m = re.search(r'xmax\s*[=:]\s*([\d\.\-\+e]+)', segment)
                text_m = re.search(r'text\s*[=:]\s*"(.*?)"', segment, re.DOTALL)
                
                if xmin_m and xmax_m and text_m:
                    intervals.append({
                        "xmin": float(xmin_m.group(1)),
                        "xmax": float(xmax_m.group(1)),
                        "text": text_m.group(1)
                    })
            
            if intervals:
                self.tiers.append({"name": tier_name, "intervals": intervals})
                logger.debug(f"Parsed tier {tier_name} with {len(intervals)} intervals.")
            else:
                logger.debug(f"Found tier {tier_name} but zero valid intervals were parsed.")
        
        if not self.tiers:
            logger.warning("No tiers found or parsed in TextGrid content.")

    def get_tier(self, name: str):
        name = name.lower()
        for tier in self.tiers:
            if tier["name"] == name:
                return tier
        return None


class HuthRecordingSource(HumanRecordingSource):
    """
    Dataset-specific loader for Huth 2023 (ds003020).
    Handles BOLD data from HDF5 and TextGrid alignment.
    """
    def __init__(
        self,
        *,
        ds_root: str | Path,
        use_cache: bool = True,
    ) -> None:
        super().__init__(
            identifier="huth-source",
            storage_mode="disk",
            recording_type="fmri-bold",
            metadata={"kind": "huth-2023-ds003020"},
        )
        self.ds_root = Path(ds_root).expanduser()
        self.use_cache = use_cache

    def load_stimuli(self, benchmark: Benchmark) -> StimulusSet:
        # Stimuli are usually reconstructed during the benchmark's ensure_data phase
        # or recovered from the TextGrids
        return benchmark.stimuli

    def load_recordings(
        self,
        benchmark: Benchmark,
        selector: Mapping[str, Any] | None = None,
    ) -> HumanRecordingData:
        """
        Simplified load_recordings mirroring the successful test script.
        """
        import torch
        subject_orig = selector.get("subject", "S1") if selector else "S1"
        id_num = int(subject_orig[1:]) if subject_orig.startswith("S") else int(subject_orig.replace("UTS", "").replace("sub-", ""))
        subject_id = f"UTS{id_num:02d}"
        
        subject_dir = self.ds_root / "derivatives" / "preprocessed_data" / subject_id
        
        if not benchmark.stimuli:
            logger.error(f"Benchmark stimuli list is empty for {subject_id}. Check TextGrids and hf5 files.")
            raise ValueError(f"No stimuli (stories) found for benchmark {benchmark.identifier} and subject {subject_id}")

        stories = [s.stimulus_id for s in benchmark.stimuli]
        logger.info(f"Attempting to load recordings for {len(stories)} stories from {subject_dir}")
        
        all_values = []
        all_story_ids = []
        all_rois = {}
        
        for i, story in enumerate(stories):
            logger.info(f"[{i+1}/{len(stories)}] Loading: {story}...")
            resp_path = subject_dir / f"{story}.hf5"
            
            if not resp_path.exists():
                logger.error(f"File missing after ensure_data: {resp_path}")
                continue
                
            if not resp_path.exists():
                logger.warning(f"File {story}.hf5 is hollow. This subject was not fully materialized in ensure_data.")
                continue
                
            try:
                with h5py.File(resp_path, "r") as hf:
                    data = hf["data"][:] # (TR, Voxels)
                    data = np.nan_to_num(data, copy=False)
                    all_values.append(data.astype(np.float32))
                    all_story_ids.append(story)
                    
                    if not all_rois:
                        for key in hf.keys():
                            if key.startswith("roi_"):
                                all_rois[key] = torch.as_tensor(hf[key][:], dtype=torch.bool)
            except Exception as e:
                logger.error(f"Failed to load {resp_path}: {e}")
                continue
        
        total_stories = len(all_values)
        if not all_values:
            raise FileNotFoundError(f"No valid BOLD data found for {subject_id} in {subject_dir}")
            
        logger.info(f"Loaded {total_stories} stories into memory. Finalizing data structure...")
            
        # Create feature IDs efficiently (cached)
        if not hasattr(self, "_feature_ids_cache") or len(self._feature_ids_cache) != all_values[0].shape[1]:
            n_voxels = all_values[0].shape[1]
            self._feature_ids_cache = [f"v{i}" for i in range(n_voxels)]
            
        return HumanRecordingData(
            values=all_values,
            stimulus_ids=all_story_ids,
            feature_ids=self._feature_ids_cache,
            metadata={"subject": subject_id, "rois": all_rois},
            category=DataCategory.TOKEN_LEVEL,
        )
