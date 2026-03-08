from abc import ABC, abstractmethod
from typing import Any


class Dataset(ABC):
    """Data source abstraction."""

    identifier: str

    @abstractmethod
    def load(self) -> Any:
        raise NotImplementedError


class Benchmark(ABC):
    """Data benchmark abstraction (data-only by design)."""

    identifier: str
    data: Any


class Metric(ABC):
    """Metric abstraction."""

    identifier: str

    @abstractmethod
    def __call__(self, prediction: Any, target: Any) -> Any:
        raise NotImplementedError
