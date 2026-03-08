from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PipelineResult:
    split_scores: list[Any]
    mean_score: float
    config: dict[str, Any]
    cache_path: str | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "mean_score": self.mean_score,
            "num_splits": len(self.split_scores),
            "split_scores": [float(score.value) for score in self.split_scores],
        }

    def __str__(self) -> str:
        return f"PipelineResult(mean_score={self.mean_score:.6f}, splits={len(self.split_scores)})"
