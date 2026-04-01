from . import metrics
from .benchmarks import Benchmark, InMemoryBenchmark
from .subjects import HFArtificialSubject, HumanSubject
import random
import numpy as np
import torch

def set_seed(seed=0):
    """Ensure reproducibility by seeding all relevant libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

_LAZY_SUBMODULES = {
    "benchmarks",
    "subjects",
    "data",
    "transforms",
    "nn",
    "models",
    "trainers",
    "objectives",
    "pipelines",
    "cache",
}


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        import importlib

        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "metrics",
    "Benchmark",
    "InMemoryBenchmark",
    "HumanSubject",
    "HFArtificialSubject",
    "data",
    "transforms",
    "nn",
    "models",
    "trainers",
    "objectives",
    "pipelines",
    "benchmarks",
    "subjects",
    "cache",
    "set_seed",
]
