# Convergent Minds

![Convergent Minds Logo](logo.png)

Convergent Minds is a research codebase for brain alignment workflows where:
- benchmarks are data containers,
- metrics are selected independently,
- pipelines are composed explicitly in Python,
- components are imported directly via `convminds` (no registries).

## Setup

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install the project

```bash
pip install -e "."
```

### 3. Optional: install extra runtime dependencies used by analysis scripts

```bash
pip install torch transformers numpy pandas scipy tqdm xarray matplotlib seaborn
```

### 4. Optional: download Fedorenko2010 stimuli (for localization utilities)

```bash
wget "https://www.dropbox.com/sh/c9jhmsy4l9ly2xx/AACQ41zipSZFj9mFbDfJJ9c4a?e=1&dl=1" -O fedorenko10_stimuli.zip
mkdir -p data/fedorenko10_stimuli
unzip -o fedorenko10_stimuli.zip -d data/fedorenko10_stimuli
```

## Project Layout

```text
convergent-minds/
├── convminds/
│   ├── alignment/
│   ├── benchmarks/
│   ├── brainscore/        # internal legacy core modules (integrated under convminds)
│   ├── datasets/
│   ├── localization/
│   ├── metrics/
│   └── subjects/
├── data/
└── test.py
```

## Decoupled Pipeline Example

Run the root example:

```bash
python test.py --benchmark synthetic
```

This runs a full pipeline and prints:
- model R2 score,
- objective-feature subject R2 score,
- `delta_r2 = model_r2 - objective_r2`.

Minimal direct-import example:

```python
from convminds import metrics, get_benchmark_by_id, score_model_on_benchmark
from convminds.subjects import HFLLMSubject

benchmark = get_benchmark_by_id("Pereira2018.243sentences")
subject = HFLLMSubject("gpt2")
metric = metrics.linear_pearsonr()
score = score_model_on_benchmark(subject, benchmark, metric=metric)
```

## Notes

- `Pereira2018` benchmarks are data-only (`Pereira2018.243sentences`, `Pereira2018.384sentences`).
- Metric selection happens in the alignment pipeline (`convminds/alignment/pipeline.py`).
- `HFLLMSubject` provides a convenience abstraction over HuggingFace model IDs.
- `ObjectiveFeatureSubject` lets objective features be scored exactly like any other subject.
