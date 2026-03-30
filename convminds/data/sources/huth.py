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
        items = re.split(r'item\s*\[\s*\d+\s*\]\s*:', content)
        for item in items[1:]:
            tier_name_match = re.search(r'name\s*=\s*"(.*)"', item)
            if not tier_name_match:
                continue
            
            tier_name = tier_name_match.group(1)
            intervals = []
            # Find interval blocks
            interval_blocks = re.findall(r'intervals\s*\[\s*\d+\s*\]\s*:.*?(?=intervals\s*\[|$)', item, re.DOTALL)
            for block in interval_blocks:
                xmin = re.search(r'xmin\s*=\s*([\d\.]+)', block)
                xmax = re.search(r'xmax\s*=\s*([\d\.]+)', block)
                text = re.search(r'text\s*=\s*"(.*)"', block)
                if xmin and xmax and text:
                    intervals.append({
                        "xmin": float(xmin.group(1)),
                        "xmax": float(xmax.group(1)),
                        "text": text.group(1)
                    })
            self.tiers.append({"name": tier_name, "intervals": intervals})

    def get_tier(self, name: str):
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
        Loads fMRI responses for the stories defined in the benchmark.
        """
        subject = selector.get("subject", "S1") if selector else "S1"
        if isinstance(subject, list):
            subject = subject[0]
            
        # 1. Resolve response files
        derivative_path = self.ds_root / "derivative"
        subject_dir = derivative_path / "preprocessed_data" / subject
        
        stories = [s.topic for s in benchmark.stimuli] # Topic holds the story name
        unique_stories = sorted(list(set(stories)))
        
        # 2. Check cache
        config = {
            "kind": "huth-source",
            "subject": subject,
            "stories": unique_stories,
            "ds_root": str(self.ds_root),
        }
        
        if self.use_cache:
            cached = load_cache("datasets", config=config)
            if cached is not None:
                logger.info(f"Loaded cached Huth recordings for {subject}")
                return HumanRecordingData(
                    values=np.asarray(cached["values"], dtype=float),
                    stimulus_ids=list(cached["stimulus_ids"]),
                    feature_ids=list(cached["feature_ids"]),
                    metadata=dict(cached["metadata"]),
                    category=DataCategory.TOKEN_LEVEL,
                )

        logger.info(f"Loading Huth BOLD data for {subject} from {subject_dir}")
        all_values = []
        all_story_ids = []
        all_rois = {}
        
        for story in unique_stories:
            resp_path = subject_dir / f"{story}.hf5"
            if not resp_path.exists():
                logger.warning(f"Response file not found: {resp_path}")
                continue
                
            with h5py.File(resp_path, "r") as hf:
                data = hf["data"][:] # (TR, Voxels)
                all_values.append(data)
                all_story_ids.append(story)
                
                # Extract all ROI masks (datasets starting with 'roi_')
                for key in hf.keys():
                    if key.startswith("roi_"):
                        mask = hf[key][:]
                        if key not in all_rois:
                            # Use torch.bool for ROI masks
                            all_rois[key] = torch.as_tensor(mask, dtype=torch.bool)
        
        if not all_values:
            raise FileNotFoundError(f"No BOLD data found for subject {subject} in {subject_dir}")
            
        feature_ids = [f"voxel-{i}" for i in range(all_values[0].shape[1])]
        
        recorded_data = HumanRecordingData(
            values=all_values, # List of arrays (stories)
            stimulus_ids=all_story_ids,
            feature_ids=feature_ids,
            metadata={"subject": subject, "stories": unique_stories, "rois": all_rois},
            category=DataCategory.TOKEN_LEVEL,
        )
        
        if self.use_cache:
            save_cache("datasets", config=config, payload={
                "values": stacked_values,
                "stimulus_ids": all_stim_ids,
                "feature_ids": feature_ids,
                "metadata": recorded_data.metadata,
            })
            
        return recorded_data
