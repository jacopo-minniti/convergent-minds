import re
import numpy as np
from typing import List, Hashable, Dict
from wordfreq import zipf_frequency

def simple_tokenize(sentence: str) -> List[str]:
    """
    Simple tokenizer that splits on whitespace and removes punctuation.
    """
    # Lowercase and split by non-alphanumeric characters (keeping them as separate tokens if needed, 
    # but for wordfreq we usually want words). 
    # The requirement says "lowercased word-like tokens".
    # Let's use a regex to find words.
    return re.findall(r'\b\w+\b', sentence.lower())

def compute_length_features(tokens: List[str], sentence: str) -> List[float]:
    """
    Computes length and load features.
    """
    num_tokens = len(tokens)
    num_chars = len(sentence) # Decided to count spaces as per "Just be consistent"
    avg_chars_per_token = num_chars / max(num_tokens, 1)
    
    # Content words (not in stoplist)
    # Using a minimal stoplist for reproducibility without NLTK dependency if possible, 
    # or just a standard small set. 
    # "standard English stopword list (e.g. NLTK stopwords or a custom list)"
    # Let's define a small custom list for simplicity and speed.
    stopwords = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
    }
    num_content_words = sum(1 for t in tokens if t not in stopwords)
    
    # Binary indicators
    has_digit = 1.0 if any(c.isdigit() for c in sentence) else 0.0
    has_quotes = 1.0 if any(c in "\"'`" for c in sentence) else 0.0
    has_colon_semicolon = 1.0 if any(c in ":;" for c in sentence) else 0.0
    # Special punct: anything not in .,?!:; and not alphanumeric/space
    # "Contains punctuation other than .,?!:; (for example ()[]{} or - etc.)"
    special_chars = set("()[]{}-_@#$%^&*+=|\\/<>\u2014\u2013") # added em-dash/en-dash
    has_special_punct = 1.0 if any(c in special_chars for c in sentence) else 0.0
    
    return [
        float(num_tokens), 
        float(num_chars), 
        float(avg_chars_per_token), 
        float(num_content_words),
        has_digit,
        has_quotes,
        has_colon_semicolon,
        has_special_punct
    ]

def compute_position_features(idx: int, n: int) -> List[float]:
    """
    Computes position and passage context features.
    idx is 1-based index in passage.
    n is number of sentences in passage.
    """
    pos_norm = idx / n if n > 0 else 0
    is_first = 1.0 if idx == 1 else 0.0
    is_last = 1.0 if idx == n else 0.0
    dist_from_center = abs(idx - (n + 1) / 2)
    
    return [
        float(idx),
        float(pos_norm),
        is_first,
        is_last,
        float(dist_from_center)
    ]

def compute_frequency_features(tokens: List[str]) -> List[float]:
    """
    Computes frequency and lexical richness features.
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
        
        # Length features (8)
        len_feats = compute_length_features(tokens, sentence)
        
        # Position features (5)
        pid = passage_ids[i]
        idx = sentence_indices[i]
        n = sentence_counts_per_passage.get(pid, 1) # Default to 1 if missing, though shouldn't happen
        pos_feats = compute_position_features(idx, n)
        
        # Frequency features (3)
        freq_feats = compute_frequency_features(tokens)
        
        # Concatenate
        row = len_feats + pos_feats + freq_feats
        features.append(row)
        
    return np.array(features)
