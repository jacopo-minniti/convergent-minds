from __future__ import annotations

import numpy as np

from convminds.cache import save_score_summary
from convminds.metrics import linear_r2
from convminds.pipelines.base import PipelineResult


def run_basic_decoder_pipeline(
    artificial_subject,
    human_subject,
    benchmark,
    decoder,
    *,
    metric=linear_r2,
    save_activations: bool = False,
    save_score: bool = False,
    force: bool = False,
):
    human_subject.record(benchmark, force=force)
    artificial_subject.record(benchmark, save_cache=save_activations, force=force)

    if human_subject.neurons is None or artificial_subject.neurons is None:
        raise RuntimeError("Both subjects must have recorded neurons before the pipeline can run.")
    if len(human_subject.neurons) != len(artificial_subject.neurons):
        raise ValueError("Human and artificial subjects produced a different number of splits.")

    split_scores = []
    for split_index in range(len(human_subject.neurons)):
        decoder.reset()
        decoder.train(human_subject.neurons[split_index]["train"], artificial_subject.neurons[split_index]["train"])
        score = metric(decoder, human_subject.neurons[split_index]["test"], artificial_subject.neurons[split_index]["test"])
        score.metadata.update({"split_index": split_index})
        split_scores.append(score)

    mean_score = float(np.mean([score.value for score in split_scores]))
    config = {
        "artificial_subject": artificial_subject.identifier(),
        "human_subject": human_subject.identifier(),
        "benchmark": benchmark.identifier,
        "decoder": decoder.decoder_config(),
        "metric": getattr(metric, "__name__", str(metric)),
    }

    cache_path = None
    if save_score:
        summary = {
            "mean_score": mean_score,
            "split_scores": [float(score.value) for score in split_scores],
            "metric": config["metric"],
        }
        diagnostics = {"splits": [score.to_dict() for score in split_scores]}
        cache_path = str(save_score_summary(config=config, summary=summary, diagnostics=diagnostics))

    return PipelineResult(
        split_scores=split_scores,
        mean_score=mean_score,
        config=config,
        cache_path=cache_path,
    )
