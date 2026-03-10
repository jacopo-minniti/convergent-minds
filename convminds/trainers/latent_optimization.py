from __future__ import annotations

import torch


class LatentOptimizationTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn,
        lr: float = 1e-2,
        steps: int = 100,
        device: torch.device | None = None,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.steps = steps
        self.device = device or self._infer_device()

        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def fit(self, latents: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        latents = latents.detach().clone().to(self.device).requires_grad_(True)
        targets = targets.to(self.device)
        optimizer = torch.optim.Adam([latents], lr=self.lr)

        for _ in range(self.steps):
            optimizer.zero_grad(set_to_none=True)
            outputs = self.model(latents)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

        return latents.detach()

    def _infer_device(self) -> torch.device:
        for param in self.model.parameters():
            return param.device
        return torch.device("cpu")
