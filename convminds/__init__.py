from . import metrics
from .benchmarks import Benchmark, InMemoryBenchmark
from .subjects import HFArtificialSubject, HumanSubject

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
]
