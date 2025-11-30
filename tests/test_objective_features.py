import pytest
import numpy as np
from alignment.objective_features import (
    simple_tokenize,
    compute_length_features,
    compute_position_features,
    compute_frequency_features,
    compute_objective_features
)

def test_simple_tokenize():
    sentence = "Hello, world! This is a test."
    tokens = simple_tokenize(sentence)
    assert tokens == ["hello", "world", "this", "is", "a", "test"]

def test_compute_length_features():
    sentence = "Hello world"
    tokens = ["hello", "world"]
    # 1. num_tokens = 2
    # 2. num_chars = 11
    # 3. avg_chars = 5.5
    # 4. content_words = 2 (hello, world - assuming not in stoplist or simple stoplist)
    # Stoplist check: 'hello' might not be in the simple list I defined? 
    # 'hello' is not in the list I pasted (it has 'i', 'me', etc.).
    # 'world' is not.
    
    feats = compute_length_features(tokens, sentence)
    assert len(feats) == 8
    assert feats[0] == 2.0
    assert feats[1] == 11.0
    assert feats[2] == 5.5
    # Check binary indicators
    assert feats[4] == 0.0 # has_digit
    assert feats[5] == 0.0 # has_quotes
    assert feats[6] == 0.0 # has_colon_semicolon
    assert feats[7] == 0.0 # has_special_punct

    sentence_complex = "It's 2023: a 'new' year!"
    tokens_complex = simple_tokenize(sentence_complex)
    feats_complex = compute_length_features(tokens_complex, sentence_complex)
    assert feats_complex[4] == 1.0 # has_digit
    assert feats_complex[5] == 1.0 # has_quotes
    assert feats_complex[6] == 1.0 # has_colon_semicolon
    
def test_compute_position_features():
    # idx=1, n=10
    feats = compute_position_features(1, 10)
    assert len(feats) == 5
    assert feats[0] == 1.0
    assert feats[1] == 0.1
    assert feats[2] == 1.0 # is_first
    assert feats[3] == 0.0 # is_last
    
    # idx=10, n=10
    feats = compute_position_features(10, 10)
    assert feats[2] == 0.0
    assert feats[3] == 1.0

def test_compute_frequency_features():
    tokens = ["the", "apple"]
    # 'the' is very frequent, 'apple' less so.
    feats = compute_frequency_features(tokens)
    assert len(feats) == 3
    # mean, std, ttr
    assert feats[2] == 1.0 # 2 unique / 2 total
    
    tokens_rep = ["the", "the"]
    feats_rep = compute_frequency_features(tokens_rep)
    assert feats_rep[2] == 0.5 # 1 unique / 2 total

def test_compute_objective_features_integration():
    sentences = ["Hello world", "Another test"]
    passage_ids = ["p1", "p1"]
    sentence_indices = [1, 2]
    counts = {"p1": 2}
    
    X_obj = compute_objective_features(sentences, passage_ids, sentence_indices, counts)
    assert X_obj.shape == (2, 16)
