from dataclasses import dataclass

from brainio.assemblies import NeuroidAssembly

from convminds.brainscore import load_dataset
from convminds.brainscore.data.pereira2018 import BIBTEX


@dataclass
class Pereira2018Benchmark:
    """Data-only benchmark container decoupled from any metric implementation."""

    identifier: str
    experiment: str
    data: NeuroidAssembly
    bibtex: str = BIBTEX


def _load_data(experiment: str) -> NeuroidAssembly:
    data = load_dataset("Pereira2018.language")
    data = data.sel(experiment=experiment)
    data = data.dropna("neuroid")
    data.attrs["identifier"] = f"{data.identifier}.{experiment}"
    return data


def Pereira2018_243sentences() -> Pereira2018Benchmark:
    experiment = "243sentences"
    return Pereira2018Benchmark(
        identifier=f"Pereira2018.{experiment}",
        experiment=experiment,
        data=_load_data(experiment),
    )


def Pereira2018_384sentences() -> Pereira2018Benchmark:
    experiment = "384sentences"
    return Pereira2018Benchmark(
        identifier=f"Pereira2018.{experiment}",
        experiment=experiment,
        data=_load_data(experiment),
    )
