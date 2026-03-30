from convminds.interfaces import Benchmark

from .base import InMemoryBenchmark
from .pereira import PereiraBenchmark
from .openneuro import HuthBenchmark, NarrativesBenchmark

__all__ = [
    "Benchmark",
    "InMemoryBenchmark",
    "PereiraBenchmark",
    "HuthBenchmark",
    "NarrativesBenchmark",
]
