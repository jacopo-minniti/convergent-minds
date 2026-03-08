# Convergent Minds

![Convergent Minds Logo](logo.png)

Convergent Minds now centers on a benchmark-driven decoding workflow:

- a `Benchmark` owns stimuli, human recording metadata, and split policy,
- a `HumanSubject` and an `HFArtificialSubject` both `record(benchmark)` into aligned train/test splits,
- a `Decoder` is trained from brain space to model space,
- scores and prepared artifacts can be cached under `CONVMINDS_HOME`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e "."
```

Optional runtime packages for the real Pereira/HuggingFace path:

```bash
pip install torch transformers brainio xarray
```

## Cache Layout

`CONVMINDS_HOME` defaults to `~/.convminds`.

- `benchmarks/`: cached split manifests
- `human/`: prepared human recordings
- `activations/`: cached artificial activations when opt-in saving is enabled
- `scores/`: saved score summaries and diagnostics

Human recordings and benchmark split manifests are saved by default. Artificial activations are always looked up in cache first, but they are saved only when `save_cache=True`. Scores are saved only when `save_score=True`.

## Core API

```python
from convminds import GLMBenchmark, HumanSubject, HFArtificialSubject, LinearDecoder
from convminds.metrics import linear_r2

benchmark = GLMBenchmark(
    "pereira2018",
    experiment="243sentences",
    cv=1,
    topic_splitting=True,
)

humans = HumanSubject()
model = HFArtificialSubject("openai/gpt2", trained=True)

print(model.neurons, humans.neurons)
print(benchmark.stimuli)

humans.record(benchmark)
model.record(benchmark, save_cache=False)

decoder = LinearDecoder(loss="mse", l2_penalty=0.01)
decoder.train(humans.neurons[0]["train"], model.neurons[0]["train"])

score = linear_r2(decoder, humans.neurons[0]["test"], model.neurons[0]["test"])
print(score)
```

Notes:

- Pereira requires an explicit subset. `GLMBenchmark("pereira2018")` without `experiment=` is invalid.
- `cv=1` produces one train/test split with a default `train_size=0.9`.
- Topic splitting keeps topics together and also prevents duplicate sentence text from leaking across train/test.

## Pipelines

`convminds.pipelines` contains prebuilt workflows. The first one is `run_basic_decoder_pipeline(...)`:

```python
from convminds import HumanSubject, HFArtificialSubject, GLMBenchmark, LinearDecoder
from convminds.pipelines import run_basic_decoder_pipeline

result = run_basic_decoder_pipeline(
    artificial_subject=HFArtificialSubject("openai/gpt2", trained=True),
    human_subject=HumanSubject(),
    benchmark=GLMBenchmark("pereira2018", experiment="243sentences"),
    decoder=LinearDecoder(loss="mse", l2_penalty=0.01),
    save_activations=False,
    save_score=True,
)

print(result)
```

If a score was saved, it can be displayed later with:

```python
import convminds

convminds.display_score(
    artificial_subject="openai/gpt2",
    benchmark="pereira2018.243sentences.glm",
)
```

## Local Smoke Test

`test.py` is a fully local synthetic example that does not need Pereira downloads or HuggingFace weights:

```bash
python3 test.py
python3 test.py --cv 3 --save-score
```

It exercises:

- benchmark-owned split planning,
- aligned human and artificial recordings,
- `LinearDecoder` training,
- `linear_r2` scoring,
- cached score display.

## Project Layout

```text
convergent-minds/
├── convminds/
│   ├── benchmarks/
│   ├── brainscore/
│   ├── cache.py
│   ├── decoders/
│   ├── metrics/
│   ├── pipelines/
│   └── subjects/
├── tests/
└── test.py
```
