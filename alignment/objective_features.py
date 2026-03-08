import re
import numpy as np
from typing import List, Hashable, Dict
from wordfreq import zipf_frequency

def simple_tokenize(sentence: str) -> List[str]:
    """
    Simple tokenizer that splits on whitespace and removes punctuation.
    """
    return re.findall(r'\b\w+\b', sentence.lower())

def compute_length_features(tokens: List[str], sentence: str) -> List[float]:
    """
    Computes Sentence Length (SL).
    Scalar: number of words in the sentence.
    Size 1.
    """
    return [float(len(tokens))]

def compute_position_features(idx: int, n: int) -> List[float]:
    """
    Computes Sentence Position (SP).
    One-hot vector for the sentence’s position within the passage (1st, 2nd, 3rd, 4th).
    Size 4.
    """
    # 1-based index
    # We assume max 4 positions as per description.
    # If idx > 4, we might want to clip or just handle it. 
    # Given the description "1st, 2nd, 3rd, 4th", we'll create a size 4 vector.
    sp = [0.0] * 4
    if 1 <= idx <= 4:
        sp[idx - 1] = 1.0
    return sp

def compute_word_position_features(tokens: List[str]) -> List[float]:
    """
    Computes Word Position (WP).
    A 9-dimensional feature per word: a ramping positional signal plus a smoothed one-hot over the 8 within-sentence positions.
    Word-based, then treated as a sentence-level feature (averaged).
    """
    n_words = len(tokens)
    if n_words == 0:
        return [0.0] * 9
        
    word_features = []
    for i in range(n_words):
        # 0-based index i
        # Ramping signal: linear position in [0, 1]
        # Using (i+1)/n_words or i/(n_words-1)? 
        # Usually i / (n-1) covers 0 to 1. If n=1, it's 0 (or 1).
        if n_words > 1:
            pos_norm = i / (n_words - 1)
        else:
            pos_norm = 1.0 # Single word is both start and end
            
        # Smoothed one-hot over 8 positions
        # We'll use 8 Gaussian kernels centered at 0, 1/7, 2/7, ..., 1.
        # Centers for 8 bins over [0, 1]
        centers = np.linspace(0, 1, 8)
        sigma = 1.0 / 8.0 # Width of the bins
        
        smoothed_one_hot = np.exp(-((pos_norm - centers) ** 2) / (2 * sigma ** 2))
        
        # Combine: [ramping, smoothed_one_hot...]
        row = [pos_norm] + smoothed_one_hot.tolist()
        word_features.append(row)
        
    # Aggregate to sentence level: Average
    avg_features = np.mean(word_features, axis=0).tolist()
    return avg_features

def compute_frequency_features(tokens: List[str]) -> List[float]:
    """
    Computes frequency and lexical richness features.
    (Keeping this as it wasn't explicitly complained about, but maybe user wants ONLY the 3 mentioned?)
    User said: "Are you using these (and other) features...". 
    So keeping frequency features is probably safe/good.
    """
    if not tokens:
        return [0.0, 0.0, 0.0]
        
    freqs = [zipf_frequency(t, "en") for t in tokens]
    mean_logfreq = np.mean(freqs)
    std_logfreq = np.std(freqs)
    
    n_unique = len(set(tokens))
    ttr = n_unique / len(tokens)
    
    return [
        float(mean_logfreq),
        float(std_logfreq),
        float(ttr)
    ]

def compute_objective_features(
    sentences: List[str],
    passage_ids: List[Hashable],
    sentence_indices: List[int],
    sentence_counts_per_passage: Dict[Hashable, int],
) -> np.ndarray:
    """
    Returns X_obj with shape (n_sentences, n_features_obj).
    The ordering of rows must match the benchmark's stimulus ordering.
    """
    features = []
    
    for i, sentence in enumerate(sentences):
        tokens = simple_tokenize(sentence)
        
        # Length features (1)
        len_feats = compute_length_features(tokens, sentence)
        
        # Position features (4)
        pid = passage_ids[i]
        idx = sentence_indices[i]
        n = sentence_counts_per_passage.get(pid, 1)
        pos_feats = compute_position_features(idx, n)
        
        # Word Position features (9)
        wp_feats = compute_word_position_features(tokens)
        
        # Frequency features (3) - Keeping these as "other" features
        freq_feats = compute_frequency_features(tokens)
        
        # Concatenate
        row = len_feats + pos_feats + wp_feats + freq_feats
        features.append(row)
        
    return np.array(features)
