from convminds.interfaces import Benchmark

from .base import InMemoryBenchmark
from .glm import GLMBenchmark

__all__ = ["Benchmark", "GLMBenchmark", "InMemoryBenchmark"]
