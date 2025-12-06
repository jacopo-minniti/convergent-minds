import numpy as np
import pytest
import torch
from brainscore.artificial_subject import ArtificialSubject
from models.bert import models

@pytest.mark.memory_intense
def test_neural():
    model_identifier = 'bert-base-uncased'
    model = models[model_identifier]()
    text = ['the quick brown fox', 'jumps over', 'the lazy dog']
    model.start_neural_recording(recording_target=ArtificialSubject.RecordingTarget.language_system,
                                 recording_type=ArtificialSubject.RecordingType.fMRI)
    representations = model.digest_text(text)['neural']
    
    # Check that we have 3 presentations
    assert len(representations['presentation']) == 3
    # Check that dimensions are correct: (presentation, neuroid)
    # BERT-base hidden size is 768
    assert representations.values.shape[1] == 768

    # Verify we are getting the [CLS] token (index 0)
    # We can inspect the model's internal hook behavior manually if needed, 
    # but functionally we check that output is not empty and has correct shape.
    assert not np.isnan(representations.values).any()
