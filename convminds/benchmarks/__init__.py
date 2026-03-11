from convminds.interfaces import Benchmark

from .base import InMemoryBenchmark
from .neuro import HuthBenchmark, NarrativesBenchmark, PereiraBenchmark, NeuroBenchmark, WordAlignedRecordingSource

__all__ = [
    "Benchmark",
    "InMemoryBenchmark",
    "NeuroBenchmark",
    "WordAlignedRecordingSource",
    "HuthBenchmark",
    "NarrativesBenchmark",
    "PereiraBenchmark",
]
