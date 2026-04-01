from __future__ import annotations

from typing import Iterable, List, Tuple


def BLEU(predictions_or_model, references_or_dataloader, ngram: int = 4) -> float:
    if isinstance(predictions_or_model, (list, tuple)):
        predictions = list(predictions_or_model)
        references = list(references_or_dataloader)
    else:
        predictions, references = _collect_pairs(references_or_dataloader)
    return _bleu_score(predictions, references, ngram)

def _collect_pairs(dataloader) -> Tuple[List[str], List[str]]:
    predictions: List[str] = []
    references: List[str] = []
    for batch in dataloader:
        if isinstance(batch, dict) and "predictions" in batch and "references" in batch:
            predictions.extend(batch["predictions"])
            references.extend(batch["references"])
            continue
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            preds, refs = batch
            predictions.extend(preds)
            references.extend(refs)
            continue
        raise ValueError("BLEU expects dataloader batches with predictions and references.")
    return predictions, references

def _bleu_score(predictions: List[str], references: List[str], ngram: int) -> float:
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must be the same length.")
    scores = []
    for pred, ref in zip(predictions, references):
        scores.append(_sentence_bleu(pred, ref, ngram))
    return sum(scores) / max(len(scores), 1)

def _sentence_bleu(prediction: str, reference: str, ngram: int) -> float:
    pred_tokens = prediction.split()
    ref_tokens = reference.split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    precisions = []
    for n in range(1, ngram + 1):
        pred_ngrams = _ngrams(pred_tokens, n)
        ref_ngrams = _ngrams(ref_tokens, n)
        match = 0
        for ng in pred_ngrams:
            if ng in ref_ngrams:
                match += 1
        precisions.append(match / max(len(pred_ngrams), 1))
    geo_mean = 1.0
    for precision in precisions:
        geo_mean *= max(precision, 1e-9)
    geo_mean **= 1 / ngram
    brevity = min(1.0, len(pred_tokens) / max(len(ref_tokens), 1))
    return geo_mean * brevity

def _ngrams(tokens: List[str], n: int) -> List[tuple]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]