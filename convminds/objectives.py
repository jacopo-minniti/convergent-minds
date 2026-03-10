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
