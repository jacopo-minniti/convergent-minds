from typing import Dict, Any, Union, Callable
import pkgutil
import importlib
from pathlib import Path

from brainio.assemblies import DataAssembly
from brainscore_core.benchmarks import Benchmark
from brainscore_core.metrics import Score, Metric
from brainscore_core.plugin_management.conda_score import wrap_score
from brainscore.artificial_subject import ArtificialSubject

data_registry: Dict[str, Callable[[], Union[DataAssembly, Any]]] = {}
""" Pool of available data """

metric_registry: Dict[str, Callable[[], Metric]] = {}
""" Pool of available metrics """

benchmark_registry: Dict[str, Callable[[], Benchmark]] = {}
""" Pool of available benchmarks """

# Removed model_registry as requested

def load_dataset(identifier: str) -> Union[DataAssembly, Any]:
    if identifier not in data_registry:
        raise ValueError(f"Dataset '{identifier}' not found in registry.")
    return data_registry[identifier]()


def load_metric(identifier: str, *args, **kwargs) -> Metric:
    if identifier not in metric_registry:
        raise ValueError(f"Metric '{identifier}' not found in registry.")
    return metric_registry[identifier](*args, **kwargs)


def load_benchmark(identifier: str) -> Benchmark:
    if identifier not in benchmark_registry:
        raise ValueError(f"Benchmark '{identifier}' not found in registry.")
    return benchmark_registry[identifier]()

def score(model: ArtificialSubject, benchmark: Benchmark) -> Score:
    """
    Score the model on the benchmark.
    
    :param model: the model to test
    :param benchmark: the benchmark to test the model against
    :return: a Score of how brain-like the candidate model is under this benchmark.
    """
    score = benchmark(model)
    if hasattr(model, 'identifier'):
        score.attrs['model_identifier'] = model.identifier
    if hasattr(benchmark, 'identifier'):
        score.attrs['benchmark_identifier'] = benchmark.identifier
    return score

# Auto-import submodules to populate registries
def _import_submodules(package_name):
    package = importlib.import_module(package_name)
    if hasattr(package, '__path__'):
        for _, name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
            try:
                importlib.import_module(name)
            except ImportError as e:
                # Ignore import errors for now, or print warning
                pass

_import_submodules('brainscore.data')
_import_submodules('brainscore.metrics')
_import_submodules('brainscore.benchmarks')
