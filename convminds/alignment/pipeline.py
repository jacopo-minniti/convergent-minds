from typing import Any, Iterable

import numpy as np
import xarray as xr
from brainio.assemblies import DataAssembly
from brainscore_core.metrics import Score
from sklearn.model_selection import GroupKFold, KFold

from convminds.brainscore.artificial_subject import ArtificialSubject


def collect_model_activations(candidate: ArtificialSubject, benchmark_data: DataAssembly) -> DataAssembly:
    """Run a subject on benchmark stimuli and return a presentation x neuroid assembly."""
    candidate.start_neural_recording(
        recording_target=ArtificialSubject.RecordingTarget.language_system,
        recording_type=ArtificialSubject.RecordingType.fMRI,
    )

    stimuli = benchmark_data["stimulus"]
    passages = benchmark_data["passage_label"].values
    predictions = []

    for passage in sorted(set(passages)):
        passage_indexer = [stimulus_passage == passage for stimulus_passage in passages]
        passage_stimuli = stimuli[passage_indexer]
        output = candidate.digest_text(passage_stimuli.values)
        passage_predictions = output["neural"]
        passage_predictions["stimulus_id"] = "presentation", passage_stimuli["stimulus_id"].values
        predictions.append(passage_predictions)

    return xr.concat(predictions, dim="presentation")


def build_cv_splits(
    X: np.ndarray,
    y: np.ndarray,
    groups: Iterable[Any],
    topic_wise_cv: bool,
    n_splits: int = 10,
    random_state: int = 42,
):
    if topic_wise_cv:
        splitter = GroupKFold(n_splits=n_splits)
        return list(splitter.split(X, y, groups=np.asarray(list(groups))))

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(splitter.split(X, y))


def score_model_on_benchmark(
    candidate: ArtificialSubject,
    benchmark: Any,
    metric,
) -> Score:
    """Score a subject on a data-only benchmark using a provided metric callable/object."""
    data = benchmark.data
    predictions = collect_model_activations(candidate, data)
    score = metric(predictions, data)

    score.attrs["model_identifier"] = candidate.identifier()
    score.attrs["benchmark_identifier"] = benchmark.identifier
    score.attrs["metric_identifier"] = getattr(metric, "__name__", metric.__class__.__name__)
    return score
