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
        # Extremely simplified parser for "ooTextFile" format
        # Finds items (tiers) and their intervals
        # Handle both "item [1]:" and "item [1]"
        items = re.split(r'item\s*\[\s*\d+\s*\]\s*:?', content)
        for item in items[1:]:
            tier_name_match = re.search(r'name\s*=\s*"(.*?)"', item)
            if not tier_name_match:
                continue
            
            tier_name = tier_name_match.group(1).lower()
            intervals = []
            
            # More robust interval regex: find everything between intervals[...] and the next one
            interval_blocks = re.findall(r'intervals\s*\[\s*\d+\s*\]\s*:?.*?(?=intervals\s*\[|$)', item, re.DOTALL)
            for block in interval_blocks:
                # Find xmin/xmax/text with or without quotes/equal signs
                xmin = re.search(r'xmin\s*[=:]\s*([\d\.]+)', block)
                xmax = re.search(r'xmax\s*[=:]\s*([\d\.]+)', block)
                text = re.search(r'text\s*[=:]\s*"(.*?)"', block)
                if xmin and xmax and text:
                    intervals.append({
                        "xmin": float(xmin.group(1)),
                        "xmax": float(xmax.group(1)),
                        "text": text.group(1)
                    })
            if intervals:
                self.tiers.append({"name": tier_name, "intervals": intervals})
        
        if not self.tiers:
            logger.debug("TextGrid parsing yielded no tiers.")

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
        
        for story in stories:
            resp_path = subject_dir / f"{story}.hf5"
            logger.debug(f"Checking {resp_path}")
            
            # Simple datalad get if file is hollow or missing
            if not resp_path.exists() or resp_path.stat().st_size < 1000:
                logger.info(f"Materializing {story}.hf5 via datalad...")
                rel_path = f"derivatives/preprocessed_data/{subject_id}/{story}.hf5"
                subprocess.run(["datalad", "get", rel_path], cwd=str(self.ds_root), capture_output=True)
                
            try:
                with h5py.File(resp_path, "r") as hf:
                    data = hf["data"][:] # (TR, Voxels)
                    all_values.append(data.astype(np.float32))
                    all_story_ids.append(story)
                    
                    for key in hf.keys():
                        if key.startswith("roi_"):
                            all_rois[key] = torch.as_tensor(hf[key][:], dtype=torch.bool)
            except Exception as e:
                logger.error(f"Failed to load {resp_path}: {e}")
                continue
        
        if not all_values:
            raise FileNotFoundError(f"No valid BOLD data found for {subject_id} in {subject_dir}")
            
        feature_ids = [f"voxel-{i}" for i in range(all_values[0].shape[1])]
        return HumanRecordingData(
            values=all_values,
            stimulus_ids=all_story_ids,
            feature_ids=feature_ids,
            metadata={"subject": subject_id, "rois": all_rois},
            category=DataCategory.TOKEN_LEVEL,
        )
