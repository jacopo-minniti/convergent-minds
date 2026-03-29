import torch
import inspect
from tqdm import tqdm


class GradientTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn,
        lr: float = 1e-4,
        optimizer_cls=torch.optim.AdamW,
        device: torch.device | None = None,
        max_grad_norm: float | None = None,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.device = device or self._infer_device()
        self.max_grad_norm = max_grad_norm
        self.optimizer = optimizer_cls(self.model.parameters(), lr=lr)

    def fit(self, dataloader: Iterable, epochs: int = 1, *, target_key: str | None = None) -> None:
        self.model.train()
        for epoch in range(epochs):
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
            for batch in pbar:
                self.optimizer.zero_grad(set_to_none=True)
                loss = self._step(batch, target_key=target_key)
                if loss is None:
                    continue
                loss.backward()
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

    def _step(self, batch, *, target_key: str | None = None):
        batch = self._move_to_device(batch)
        if isinstance(batch, dict):
            targets = None
            inputs = dict(batch)
            if target_key is not None and target_key in inputs:
                targets = inputs.pop(target_key)
            inputs = self._filter_inputs(inputs)
            outputs = self.model(**inputs) if inputs else self.model()
            if targets is not None:
                return self.loss_fn(outputs, targets)
            if hasattr(outputs, "loss") and outputs.loss is not None:
                return outputs.loss
            if "labels" in batch:
                return self.loss_fn(outputs, batch["labels"])
            return self.loss_fn(outputs)
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            inputs, targets = batch
            outputs = self.model(*inputs) if isinstance(inputs, (list, tuple)) else self.model(inputs)
            return self.loss_fn(outputs, targets)
        outputs = self.model(batch)
        return self.loss_fn(outputs)

    def _move_to_device(self, batch):
        if torch.is_tensor(batch):
            return batch.to(self.device)
        if hasattr(batch, "to") and callable(getattr(batch, "to")):
            try:
                return batch.to(self.device)
            except Exception:
                pass
        if isinstance(batch, dict):
            return {k: self._move_to_device(v) for k, v in batch.items()}
        if isinstance(batch, (list, tuple)):
            return type(batch)(self._move_to_device(v) for v in batch)
        return batch

    def _infer_device(self) -> torch.device:
        for param in self.model.parameters():
            return param.device
        return torch.device("cpu")

    def _filter_inputs(self, inputs: dict) -> dict:
        signature = inspect.signature(self.model.forward)
        params = signature.parameters
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
            return inputs
        return {key: value for key, value in inputs.items() if key in params}
