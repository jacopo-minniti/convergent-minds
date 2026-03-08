from typing import Any

from brainio.assemblies import DataAssembly
from brainscore_core.metrics import Score

from convminds.brainscore.artificial_subject import ArtificialSubject


def load_dataset(identifier: str) -> Any:
    if identifier == "Pereira2018.language":
        from convminds.brainscore.data.pereira2018 import load_pereira2018_language

        return load_pereira2018_language()
    if identifier == "Pereira2018.auditory":
        from convminds.brainscore.data.pereira2018 import load_pereira2018_auditory

        return load_pereira2018_auditory()
    if identifier == "Fedorenko2010.localization":
        from convminds.brainscore.data.fedorenko2010_localization import load_data

        return load_data()
    if identifier == "Fedorenko2016.language":
        from convminds.brainscore.data.fedorenko2016 import load_fedorenko2016_language

        return load_fedorenko2016_language()
    if identifier == "Blank2014.fROI":
        from convminds.brainscore.data.blank2014 import load_blank2014_froi

        return load_blank2014_froi()
    if identifier == "Futrell2018":
        from convminds.brainscore.data.futrell2018 import load_assembly

        return load_assembly()
    if identifier == "Tuckute2024.language":
        from convminds.brainscore.data.tuckute2024 import load_tuckute2024_language

        return load_tuckute2024_language()
    if identifier == "wikitext-2/test":
        from convminds.brainscore.data.wikitext import wikitext2TestFromHuggingface

        return wikitext2TestFromHuggingface()

    raise ValueError(f"Dataset '{identifier}' not found.")


def load_metric(identifier: str, *args, **kwargs):
    if identifier == "accuracy":
        from convminds.brainscore.metrics.accuracy.metric import Accuracy

        return Accuracy(*args, **kwargs)
    if identifier == "pearsonr":
        from convminds.brainscore.metrics.pearson_correlation.metric import PearsonCorrelation

        return PearsonCorrelation(*args, **kwargs)
    if identifier == "linear_pearsonr":
        from convminds.brainscore.metrics.linear_predictivity.metric import linear_pearsonr

        return linear_pearsonr(*args, **kwargs)
    if identifier == "ridge_pearsonr":
        from convminds.brainscore.metrics.linear_predictivity.metric import ridge_pearsonr

        return ridge_pearsonr(*args, **kwargs)
    if identifier == "cka":
        from convminds.brainscore.metrics.cka.metric import CKACrossValidated

        return CKACrossValidated(*args, **kwargs)
    if identifier == "rdm":
        from convminds.brainscore.metrics.rdm.metric import RDMCrossValidated

        return RDMCrossValidated(*args, **kwargs)

    raise ValueError(f"Metric '{identifier}' not found.")


def load_benchmark(identifier: str):
    if identifier in {"Pereira2018.243sentences", "Pereira2018.243sentences-linear"}:
        from convminds.brainscore.benchmarks.pereira2018.benchmark import Pereira2018_243sentences

        return Pereira2018_243sentences()
    if identifier in {"Pereira2018.384sentences", "Pereira2018.384sentences-linear"}:
        from convminds.brainscore.benchmarks.pereira2018.benchmark import Pereira2018_384sentences

        return Pereira2018_384sentences()

    raise ValueError(f"Benchmark '{identifier}' not found.")


def score(model: ArtificialSubject, benchmark: Any) -> Score:
    if not callable(benchmark):
        raise TypeError(
            "This helper expects a callable benchmark. "
            "For data-only benchmarks, use alignment.pipeline.score_model_on_benchmark."
        )

    score_value = benchmark(model)
    if hasattr(model, "identifier"):
        score_value.attrs["model_identifier"] = model.identifier
    if hasattr(benchmark, "identifier"):
        score_value.attrs["benchmark_identifier"] = benchmark.identifier
    return score_value
