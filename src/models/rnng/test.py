import numpy as np
import pytest

from brainscore.artificial_subject import ArtificialSubject
from models.rnng import (
    get_rnn_tdg_ptb,
    get_rnn_slm_ptb,
    get_rnn_tdg_ptboanc,
    get_rnn_slm_ptboanc,
    get_rnn_tdg_ptboanc_1024,
    get_rnn_slm_ptboanc_1024,
)

# Map identifiers to factory functions
model_factories = {
    "rnn-tdg-ptb": get_rnn_tdg_ptb,
    "rnn-slm-ptb": get_rnn_slm_ptb, # Replaced rnn-lcg-ptb with rnn-slm-ptb based on __init__.py
    "rnn-tdg-ptboanc": get_rnn_tdg_ptboanc,
    "rnn-slm-ptboanc": get_rnn_slm_ptboanc, # Replaced rnn-lcg-ptboanc with rnn-slm-ptboanc
    "rnn-tdg-ptboanc-1024": get_rnn_tdg_ptboanc_1024,
    "rnn-slm-ptboanc-1024": get_rnn_slm_ptboanc_1024, # Replaced rnn-lcg-ptboanc-1024 with rnn-slm-ptboanc-1024
}

@pytest.mark.memory_intense
@pytest.mark.parametrize(
    "model_identifier, feature_size",
    [
        ("rnn-tdg-ptb", 512),
        ("rnn-slm-ptb", 512),
        ("rnn-tdg-ptboanc", 512),
        ("rnn-slm-ptboanc", 512),
        ("rnn-tdg-ptboanc-1024", 1024),
        ("rnn-slm-ptboanc-1024", 1024),
    ],
)
def test_neural(model_identifier, feature_size):
    model = model_factories[model_identifier]()
    text = ["the quick brown fox jumps over the lazy dog."]
    model.start_neural_recording(
        recording_target=ArtificialSubject.RecordingTarget.language_system,
        recording_type=ArtificialSubject.RecordingType.fMRI,
    )
    representations = model.digest_text(text)["neural"]
    assert len(representations["presentation"]) == 1
    np.testing.assert_array_equal(representations["stimulus"], text)
    assert len(representations["neuroid"]) == feature_size
