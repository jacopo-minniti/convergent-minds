from dataclasses import dataclass

from convminds.datasets import Pereira2018LanguageDataset
from convminds.interfaces import Benchmark


@dataclass
class PereiraSentencesBenchmark(Benchmark):
    experiment: str

    def __post_init__(self):
        self.identifier = f"Pereira2018.{self.experiment}"
        data = Pereira2018LanguageDataset().load()
        data = data.sel(experiment=self.experiment)
        data = data.dropna("neuroid")
        data.attrs["identifier"] = f"{data.identifier}.{self.experiment}"
        self.data = data


def get_benchmark_by_id(identifier: str):
    if identifier in {"Pereira2018.243sentences", "Pereira2018.243sentences-linear"}:
        return PereiraSentencesBenchmark("243sentences")
    if identifier in {"Pereira2018.384sentences", "Pereira2018.384sentences-linear"}:
        return PereiraSentencesBenchmark("384sentences")
    raise ValueError(f"Unknown benchmark id '{identifier}'.")
