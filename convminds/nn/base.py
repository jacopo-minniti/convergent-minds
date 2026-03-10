from __future__ import annotations

import torch.nn as nn


class Module(nn.Module):
    def freeze_module(self, module: nn.Module) -> None:
        for param in module.parameters():
            param.requires_grad = False
        module.eval()

    def freeze_base_model(self) -> None:
        for name in ("llm", "base_model", "model"):
            module = getattr(self, name, None)
            if module is not None:
                self.freeze_module(module)
                return
