from brainscore import benchmark_registry
from .benchmark import (Pereira2018_243sentences, Pereira2018_384sentences,
                        Pereira2018_243sentences_cka, Pereira2018_384sentences_cka,
                        Pereira2018_243sentences_partialr2, Pereira2018_384sentences_partialr2)

benchmark_registry['Pereira2018.243sentences-linear'] = Pereira2018_243sentences
benchmark_registry['Pereira2018.384sentences-linear'] = Pereira2018_384sentences
benchmark_registry['Pereira2018.243sentences-cka'] = Pereira2018_243sentences_cka
benchmark_registry['Pereira2018.384sentences-cka'] = Pereira2018_384sentences_cka
benchmark_registry['Pereira2018.243sentences-partialr2'] = Pereira2018_243sentences_partialr2
benchmark_registry['Pereira2018.384sentences-partialr2'] = Pereira2018_384sentences_partialr2
