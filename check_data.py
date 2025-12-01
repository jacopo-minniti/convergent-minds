
from brainscore import load_dataset
import numpy as np

def check_pereira_data():
    data = load_dataset('Pereira2018.language')
    print(f"Data shape: {data.shape}")
    print(f"Coords: {data.coords}")
    
    stimulus_ids = data['stimulus_id'].values
    unique_ids = np.unique(stimulus_ids)
    print(f"Total stimulus_ids: {len(stimulus_ids)}")
    print(f"Unique stimulus_ids: {len(unique_ids)}")
    
    if len(stimulus_ids) != len(unique_ids):
        print("WARNING: Duplicate stimulus_ids found! Data contains repetitions.")
        # Check how many repetitions
        from collections import Counter
        counts = Counter(stimulus_ids)
        print(f"Common counts: {counts.most_common(5)}")
    else:
        print("No repetitions found. Stimulus_ids are unique.")

if __name__ == "__main__":
    check_pereira_data()
