from __future__ import annotations

from typing import Iterable, List, Tuple


class BLEU:
    def __init__(self, ngram: int = 4):
        self.ngram = ngram

    def compute(self, predictions_or_model, references_or_dataloader) -> float:
        if isinstance(predictions_or_model, (list, tuple)):
            predictions = list(predictions_or_model)
            references = list(references_or_dataloader)
        else:
            predictions, references = self._collect_pairs(references_or_dataloader)
        return self._bleu_score(predictions, references)

    def _collect_pairs(self, dataloader) -> Tuple[List[str], List[str]]:
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

    def _bleu_score(self, predictions: List[str], references: List[str]) -> float:
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must be the same length.")
        scores = []
        for pred, ref in zip(predictions, references):
            scores.append(self._sentence_bleu(pred, ref))
        return sum(scores) / max(len(scores), 1)

    def _sentence_bleu(self, prediction: str, reference: str) -> float:
        pred_tokens = prediction.split()
        ref_tokens = reference.split()
        if not pred_tokens or not ref_tokens:
            return 0.0
        precisions = []
        for n in range(1, self.ngram + 1):
            pred_ngrams = self._ngrams(pred_tokens, n)
            ref_ngrams = self._ngrams(ref_tokens, n)
            match = 0
            for ngram in pred_ngrams:
                if ngram in ref_ngrams:
                    match += 1
            precisions.append(match / max(len(pred_ngrams), 1))
        geo_mean = 1.0
        for precision in precisions:
            geo_mean *= max(precision, 1e-9)
        geo_mean **= 1 / self.ngram
        brevity = min(1.0, len(pred_tokens) / max(len(ref_tokens), 1))
        return geo_mean * brevity

    def _ngrams(self, tokens: List[str], n: int) -> List[tuple]:
        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


class BERTScore:
    def compute(self, predictions, references) -> float:
        raise NotImplementedError("BERTScore requires an external dependency; provide your own evaluator.")
