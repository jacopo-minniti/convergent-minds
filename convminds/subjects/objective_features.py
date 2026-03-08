from collections import Counter
from typing import List, Union

import numpy as np
from brainio.assemblies import NeuroidAssembly

from convminds.alignment.objective_features import compute_objective_features
from convminds.brainscore.artificial_subject import ArtificialSubject


class ObjectiveFeatureSubject(ArtificialSubject):
    """
    ArtificialSubject that emits handcrafted objective features as neural representations.
    This allows objective features to be scored with the same benchmark+metric machinery
    used for any other subject.
    """

    def identifier(self) -> str:
        return "objective-features"

    def start_behavioral_task(self, task: ArtificialSubject.Task):
        return None

    def start_neural_recording(self, recording_target: ArtificialSubject.RecordingTarget, recording_type: ArtificialSubject.RecordingType):
        return None

    def digest_text(self, text: Union[str, List[str]]):
        if isinstance(text, str):
            text = [text]

        sentence_indices = list(range(1, len(text) + 1))
        passage_ids = ["passage"] * len(text)
        sentence_counts = Counter(passage_ids)

        features = compute_objective_features(
            sentences=text,
            passage_ids=passage_ids,
            sentence_indices=sentence_indices,
            sentence_counts_per_passage=sentence_counts,
        )

        num_features = features.shape[1]
        assembly = NeuroidAssembly(
            features,
            coords={
                "stimulus": ("presentation", np.array(text, dtype=object)),
                "stimulus_id": ("presentation", np.arange(len(text), dtype=int)),
                "layer": ("neuroid", np.array(["objective"] * num_features, dtype=object)),
                "region": ("neuroid", np.array(["language_system"] * num_features, dtype=object)),
                "recording_type": ("neuroid", np.array(["fMRI"] * num_features, dtype=object)),
                "neuroid_id": ("neuroid", np.array([f"obj--{i}" for i in range(num_features)], dtype=object)),
            },
            dims=["presentation", "neuroid"],
        )

        return {"neural": assembly, "behavior": None}
