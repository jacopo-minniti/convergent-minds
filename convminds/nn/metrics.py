from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer
    from jiwer import wer
    _HAS_NLP_LIBS = True
except ImportError:
    _HAS_NLP_LIBS = False

def calculate_nlp_metrics(pred_text: str, target_text: str) -> dict[str, float]:
    """
    Standardized NLP evaluation for Brain-to-LLM decoding.
    Calculates BLEU-1, ROUGE-1, ROUGE-L, and WER (Word Error Rate).
    
    If libraries are missing, returns empty metrics.
    """
    if not _HAS_NLP_LIBS:
        return {"bleu": 0.0, "rouge1": 0.0, "rougeL": 0.0, "wer": 0.0, "meteor": 0.0}
    
    # Tokenization
    pred_tokens = pred_text.lower().split()
    target_tokens = target_text.lower().split()
    
    if not target_tokens:
        return {"bleu": 0.0, "rouge1": 0.0, "rougeL": 0.0, "wer": 0.0, "meteor": 0.0}
    
    # BLEU-1
    # We use smoothing 1 to avoid zero scores for short segments
    bleu = sentence_bleu([target_tokens], pred_tokens, weights=(1, 0, 0, 0), 
                         smoothing_function=SmoothingFunction().method1)
    
    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge_val = scorer.score(target_text, pred_text)
    
    # WER
    try:
        wer_val = wer(target_text, pred_text)
    except:
        wer_val = 1.0
        
    # METEOR (requires nltk installation of wordnet/omw-1.4 usually)
    try:
        met = meteor_score([target_tokens], pred_tokens)
    except:
        met = 0.0
        
    return {
        "bleu": bleu,
        "rouge1": rouge_val['rouge1'].fmeasure,
        "rougeL": rouge_val['rougeL'].fmeasure,
        "wer": min(wer_val, 1.0),
        "meteor": met
    }

def identification_accuracy(predicted_vecs: torch.Tensor, target_vecs: torch.Tensor, top_k: int = 10) -> float:
    """
    Common metric in brain decoding (Tang et al., 2023).
    Calculates how often the correct target is among the top-k matches for a predicted vector.
    
    Args:
        predicted_vecs: Model's brain latents (B, D)
        target_vecs: Reference LLM embeddings (B, D)
        top_k: How many neighbors to consider.
    """
    if predicted_vecs.shape[0] < 1:
        return 0.0
        
    # Compute Cosine Similarity Matrix (B, B)
    cos_sim = F.cosine_similarity(predicted_vecs.unsqueeze(1), target_vecs.unsqueeze(0), dim=-1)
    
    correct = 0
    batch_size = predicted_vecs.shape[0]
    for i in range(batch_size):
        # Indices of the top k candidates for this sample
        indices = torch.topk(cos_sim[i], k=min(top_k, batch_size)).indices
        if i in indices:
            correct += 1
            
    return correct / batch_size
