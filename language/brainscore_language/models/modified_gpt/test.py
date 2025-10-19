import numpy as np
from brainscore_language import load_model
from brainscore_language.artificial_subject import ArtificialSubject

def test_next_word():
    model = load_model('modified-gpt2')
    text = ['the quick brown fox', 'jumps over', 'the lazy']
    model.start_behavioral_task(task=ArtificialSubject.Task.next_word)
    next_word_predictions = model.digest_text(text)['behavior']
    # not checking for specific words since the modification might change the output
    assert len(next_word_predictions) == 3