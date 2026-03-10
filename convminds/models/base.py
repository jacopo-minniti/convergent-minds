from __future__ import annotations

from convminds.nn import Module


class BrainLanguageModel(Module):
    def generate(self, *args, **kwargs):
        base_model = (
            getattr(self, "llm", None)
            or getattr(self, "base_model", None)
            or getattr(self, "model", None)
        )
        if base_model is None or not hasattr(base_model, "generate"):
            raise AttributeError("No base model with generate() found on this model.")
        return base_model.generate(*args, **kwargs)
