from __future__ import annotations

import torch


class NextTokenCrossEntropy:
    def __init__(self, ignore_index: int = -100):
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def __call__(self, outputs, labels: torch.Tensor) -> torch.Tensor:
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        vocab_size = shift_logits.size(-1)
        return self.loss_fn(shift_logits.view(-1, vocab_size), shift_labels.view(-1))


class PenalizedCrossEntropy:
    """
    Phase 3: Computes Cross Entropy and adds a lambda-weighted penalty.
    L_total = L_CE + lambda * penalty
    """
    def __init__(self, ignore_index: int = -100, lambda_weight: float = 1.0):
        self.ce_loss = NextTokenCrossEntropy(ignore_index=ignore_index)
        self.lambda_weight = lambda_weight

    def __call__(self, outputs, labels: torch.Tensor, penalty: torch.Tensor | None = None) -> torch.Tensor:
        loss = self.ce_loss(outputs, labels)
        if penalty is not None:
            # penalty is likely shape (batch, seq_len), so we average it along with CE loss
            loss = loss + self.lambda_weight * penalty.mean()
        return loss
