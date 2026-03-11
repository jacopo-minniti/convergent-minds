<p align="center">
  <img src="logo.png" alt="Convergent Minds Logo" width="75%">
</p>

**Convergent Minds** (`convminds`) is a PyTorch-native framework for aligning biological brain activity with Large Language Models (LLMs). It provides standardized neuro-components, leakage-safe data flows, and end-to-end recipes to replicate state-of-the-art research in brain-to-text decoding.

## Core Philosophy: The Physics and the Engine
`convminds` separates **mathematical mechanisms** from **architectural routing**:
- **`cm.nn`**: Generic neural building blocks (Encoders, Adapters, Wrappers).
- **`cm.models`**: Assembled architectures that wire `cm.nn` blocks into LLM backbones.
- **`cm.recipes`**: High-level replications that match specific papers end-to-end.
- **`cm.data`**: Primitives that enforce biological constraints (e.g., preserving 3D coordinates and preventing data leakage).

---

## Installation

```bash
pip install convminds
# or
uv add convminds
```

---

## Quickstart: Building a Brain-to-Text Model

### 1. Data & Preprocessing
`convminds` uses `BrainDataModule` to ensure that stateful transforms (like PCA or Z-scoring) are fitted **only** on the training split, preventing the most common error in neuro-ML.

```python
import convminds as cm
import convminds.transforms as T

stimuli = [
    {"stimulus_id": "s-001", "text": "A cat sits on a mat.", "topic": "animals", "metadata": {}},
    {"stimulus_id": "s-002", "text": "The sky is bright blue.", "topic": "weather", "metadata": {}},
]
human_values = [[0.1, 0.2], [0.3, 0.4]]
benchmark = cm.benchmarks.InMemoryBenchmark(
    identifier="demo.synthetic",
    stimuli=stimuli,
    human_values=human_values,
)

human = cm.subjects.HumanSubject(subject_ids=["1"])
oracle = cm.subjects.HFArtificialSubject("microsoft/deberta-large", layers=[-1])

datamodule = cm.data.BrainDataModule(
    benchmark=benchmark,
    human_subject=human,
    artificial_subject=oracle,
    stateless_transforms=[T.HRFWindow(t=1)],
    stateful_transforms=[T.ZScore(dim="batch")],
    batch_size=32,
)
datamodule.setup() # Safely fits stateful transforms on train split
```

### 2. The Model (Assembling Lego Bricks)
Build architectures natively by combining generic blocks from `cm.nn`. 

```python
import torch.nn as nn
import convminds.nn as cnn
from convminds.models import PromptConditionedLM

# Mix and match SOTA components
encoder = cnn.encoders.SpatialAttentionEncoder(query_dim=4096, use_coords=True) # From MindLLM
fusion = cnn.fusion.PrefixFusion() # Standard prefix tuning

# Assemble into a model
model = PromptConditionedLM(
    llm_id="meta-llama/Llama-2-7b", 
    encoder=encoder, 
    fusion=fusion
)
```

### 3. Model Surgery (Native Interventions)
To perform deep-layer injection without "magic" strings, use explicit `cm.nn.wrappers`. This preserves HuggingFace's KV-caching and `.generate()` compatibility.

```python
# Wrap an existing LLM layer with a brain-steered Residual Injector
model.llm.model.layers[16] = cnn.wrappers.ResidualInjector(
    base_layer=model.llm.model.layers[16],
    intervention_module=nn.MultiheadAttention(embed_dim=4096, num_heads=8),
    kwarg_name="brain_context"
)
```

---

## Subjects: Human and Artificial
`convminds` introduces the **ArtificialSubject**—a frozen feature-extractor (CLIP, DeBERTa, GPT) that acts as a "ground truth" mind for alignment training. All activations are heavily cached under `CONVMINDS_HOME` (~/.convminds) to accelerate research.

```python
from convminds.subjects import HumanSubject, HFArtificialSubject

human = HumanSubject(subject_ids=["1"])
oracle = HFArtificialSubject(model_id="openai/clip-vit-large-patch14", layers=[-1])

datamodule = cm.data.BrainDataModule(
    benchmark=benchmark,
    human_subject=human,
    artificial_subject=oracle,
)
```

---

## Recipes: One-Liner Replications
Replicate top papers instantly using standardized `cm.recipes`. These classes encapsulate the exact dataset, model, and training curriculum described in the literature.

```python
from convminds.recipes import MindLLM

recipe = MindLLM(benchmark=benchmark, subject_id=1, llm_id="lmsys/vicuna-7b-v1.5")
recipe.train()
recipe.evaluate()
```

---

## Metrics
Metrics are decoupled and categorized by their mathematical domain:
- **`cm.metrics.spatial`**: Voxel-wise evaluation (`R2`).
- **`cm.metrics.latents`**: Representation evaluation (`PairwiseRetrieval`).
- **`cm.metrics.text`**: Language evaluation (`BLEU`, `BERTScore` placeholder).

---

## Cache Layout
`CONVMINDS_HOME` defaults to `~/.convminds`.
- `benchmarks/`: cached split manifests
- `activations/`: cached artificial activations (GPT/CLIP/DeBERTa)
- `scores/`: saved evaluation diagnostics
