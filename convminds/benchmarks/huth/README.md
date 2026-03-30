# Huth Dataset (ds003020) Benchmark

This benchmark provides a modular interface to the Huth 2023 (ds003020) natural language fMRI dataset. 
It supports automated downloading via DataLad, TextGrid alignment, and TR-level BOLD data loading.

## Dataset Reference
> LeBel, A., Wagner, L., Jain, S., Adibi, A., Gupta, A., Huth, A.G. (2023). 
> **A natural language fMRI dataset for voxel-wise encoding and decoding models.** 
> *Nature Scientific Data*. DOI: 10.1038/s41597-023-02437-z

## Structure
- **Raw Data**: Downloaded via `datalad clone https://github.com/OpenNeuroDatasets/ds003020.git`.
- **Derivatives**: Preprocessed BOLD data (`.hf5`) and TextGrid stimuli.
- **Alignment**: Words/Phonemes are aligned to fMRI TRs. 
- **TR Logic**: The benchmark trims 5 TRs from the start and end of each story (LeBel 2023 convention).

## Usage (Story-Level TOKEN_LEVEL)
```python
from convminds.benchmarks.huth.benchmark import HuthBenchmark
from convminds.data.datamodule import BrainDataModule
from convminds.transforms.timeseries import TrimTRs, RandomWindow
from convminds.transforms.fir import FIRDelay
from convminds.transforms.roi import ROIFilter

# 1. Initialize Benchmark (Stories as primary units)
benchmark = HuthBenchmark(subject="S1")

# 2. Define General Transform Pipeline
#    Note: Transforms are dataset-independent and applied in sequence.
transforms = [
    ROIFilter(roi="roi_earlyV"),      # Select specific voxels
    TrimTRs(start=5, end=5),         # Remove story boundaries
    FIRDelay(delays=[1, 2, 3, 4]),   # Add hemodynamic delays
    RandomWindow(window_size=20),    # Pick a 20-TR segment for training
]

# 3. Create DataModule
dm = BrainDataModule(
    benchmark=benchmark, 
    stateless_transforms=transforms,
    batch_size=4 # 4 stories per batch, padded temporally
)
dm.setup()

# 4. Access data
train_loader = dm.train_dataloader()
for batch in train_loader:
    # batch["brain_tensor"].signal shape: (4, 20, N_voxels * 4)
    print(f"Signal shape: {batch['brain_tensor'].signal.shape}")
    break
```

## Configuration
- `subject`: (default: "S1") Supports S1-S8.
- `trim`: Number of TRs to trim from start/end (default: 5).
- `window`: Lanczos window size (default: 3).
- `use_datalad`: (default: True) Set to False if data is already manually downloaded.

## Notes
- Ensure `datalad` and `git-annex` are installed on your system.
- The first run will take several minutes to download the raw repository and specific subject/stimuli derivatives.
- Processed stimuli (TextGrid-to-TR mapping) are cached in `~/.convminds/data/huth/processed/`.
