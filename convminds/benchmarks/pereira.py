from __future__ import annotations

from convminds.benchmarks.glm import GLMBenchmark


def get_benchmark_by_id(identifier: str):
    normalized = identifier.strip()
    if normalized in {"Pereira2018.243sentences", "pereira2018.243sentences"}:
        return GLMBenchmark("pereira2018", experiment="243sentences")
    if normalized in {"Pereira2018.384sentences", "pereira2018.384sentences"}:
        return GLMBenchmark("pereira2018", experiment="384sentences")
    raise ValueError(f"Unknown benchmark id '{identifier}'.")
