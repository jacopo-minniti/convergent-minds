import torch
import numpy as np
from torch.utils.data import Dataset
from convminds.benchmarks import HuthBenchmark

class HuthAlignmentDataset(Dataset):
    """
    Standardized dataset for Brain-to-Language alignment using the Huth protocol.
    
    Windowing logic:
    - Target Words: TR X
    - Story Context: TR X-3 to X-1
    - Brain Input: BOLD TR X+1 to X+4 (capturing HRF lag)
    """
    def __init__(self, subject_ids=["S1"], split="train", pca_dim=1000):
        self.benchmark = HuthBenchmark()
        self.bold_data = {}
        self.story_metadata = {}
        self.all_samples = []
        
        # Load and preprocess (consistent with the HuthBenchmark format)
        for subj in subject_ids:
            recordings = self.benchmark.human_recording_source.load_recordings(
                subject_id=subj, compute_type="pca", n_components=pca_dim
            )
            self.bold_data[subj] = {}
            for record in recordings:
                story_name = record.metadata["story_name"]
                self.bold_data[subj][story_name] = record.data
                self.story_metadata[story_name] = record.metadata
                
                # Split logic: Huth has thousands of TRs, we use the last 10% for test
                actual_trs = record.data.shape[0]
                test_start = int(actual_trs * 0.9)
                
                tr_range = range(3, actual_trs - 4)
                if split == "train":
                    tr_range = range(3, test_start - 4)
                elif split == "test":
                    tr_range = range(test_start, actual_trs - 4)
                
                for x in tr_range:
                    self.all_samples.append((subj, story_name, x))

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, index):
        subj, story, x = self.all_samples[index]
        bold_window = self.bold_data[subj][story][x+1:x+5]
        
        metadata = self.story_metadata[story]
        tr_times = metadata["tr_times"]
        word_intervals = metadata["word_intervals"]
        
        def get_words(tr_start, tr_end):
            t_start = tr_times[tr_start]
            t_end = tr_times[tr_end+1] if tr_end+1 < len(tr_times) else tr_times[-1] + 2.0
            noise_tokens = {"sp", "uh", "um"}
            words = [i["text"].lower().strip() for i in word_intervals 
                     if i["xmin"] >= t_start and i["xmin"] < t_end]
            words = [w for w in words if w not in noise_tokens and len(w) > 0]
            return " ".join(words) if words else ""

        return {
            "bold": torch.from_numpy(bold_window).float(),
            "context": get_words(x-3, x-1),
            "target": get_words(x, x),
            "subject": subj,
            "story": story,
            "tr": x,
            "time_window": (tr_times[x], tr_times[x+1] if x+1 < len(tr_times) else tr_times[x]+2.0)
        }
