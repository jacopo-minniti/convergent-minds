from dataclasses import dataclass

from convminds.interfaces import Dataset
from convminds.brainscore.data.pereira2018 import load_pereira2018_language, load_pereira2018_auditory


@dataclass
class Pereira2018LanguageDataset(Dataset):
    identifier: str = "Pereira2018.language"

    def load(self):
        return load_pereira2018_language()


@dataclass
class Pereira2018AuditoryDataset(Dataset):
    identifier: str = "Pereira2018.auditory"

    def load(self):
        return load_pereira2018_auditory()
