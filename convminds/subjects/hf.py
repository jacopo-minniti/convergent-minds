from __future__ import annotations

import logging
from importlib import import_module
from typing import Any, Iterable

import numpy as np

logger = logging.getLogger(__name__)

from convminds.cache import load_cache, save_cache as save_cache_payload
from convminds.subjects.base import ArtificialSubject


def _resolve_backend():
    try:
        torch = import_module("torch")
    except ModuleNotFoundError as error:
        raise RuntimeError("HFArtificialSubject.record requires PyTorch to be installed.") from error
    return torch


def _resolve_transformers():
    try:
        return import_module("transformers")
    except ModuleNotFoundError as error:
        raise RuntimeError("HFArtificialSubject requires transformers to be installed.") from error


def _layer_indices_from_config(config, layers: Iterable[int | str] | None) -> list[int]:
    if layers is None:
        for attr in ("num_hidden_layers", "n_layer", "num_layers", "n_layers"):
            if hasattr(config, attr):
                return [int(getattr(config, attr)) - 1]
        raise ValueError("Could not infer the final hidden layer from the model config.")

    resolved = []
    for layer in layers:
        if isinstance(layer, int):
            resolved.append(layer)
            continue
        digits = "".join(character for character in str(layer) if character.isdigit())
        if not digits:
            raise ValueError(f"Layer {layer!r} could not be resolved to an integer layer index.")
        resolved.append(int(digits))
    return resolved


class HFArtificialSubject(ArtificialSubject):
    def __init__(
        self,
        model_id: str,
        *,
        trained: bool = True,
        layers: Iterable[int | str] | None = None,
        device: str | None = None,
        representation_token_index: int = -1,
    ) -> None:
        super().__init__()
        self.model_id = model_id
        self.trained = trained
        self.layers = list(layers) if layers is not None else None
        self.device = device
        self.representation_token_index = representation_token_index
        self._model = None
        self._tokenizer = None
        self._layer_indices = None

    def identifier(self) -> str:
        return self.model_id

    def subject_config(self) -> dict[str, Any]:
        return {
            "kind": "hf-artificial",
            "model_id": self.model_id,
            "trained": self.trained,
            "layers": list(self.layers) if self.layers is not None else None,
            "representation_token_index": self.representation_token_index,
        }

    def _load_model(self):
        if self._model is not None and self._tokenizer is not None and self._layer_indices is not None:
            return self._model, self._tokenizer, self._layer_indices

        torch = _resolve_backend()
        transformers = _resolve_transformers()
        AutoConfig = getattr(transformers, "AutoConfig")
        AutoModelForCausalLM = getattr(transformers, "AutoModelForCausalLM")
        AutoTokenizer = getattr(transformers, "AutoTokenizer")

        logger.info(f"Loading {self.model_id} from Hugging Face...")
        config = AutoConfig.from_pretrained(self.model_id)
        if self.trained:
            model = AutoModelForCausalLM.from_pretrained(self.model_id)
        else:
            model = AutoModelForCausalLM.from_config(config)
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        self._model = model
        self._tokenizer = tokenizer
        self.device = device
        self._layer_indices = _layer_indices_from_config(config, self.layers)
        return self._model, self._tokenizer, self._layer_indices

    def _extract_group_activations(self, texts: list[str]) -> np.ndarray:
        torch = _resolve_backend()
        model, tokenizer, layer_indices = self._load_model()

        activations = []
        context_parts: list[str] = []
        for text in texts:
            context_parts.append(text)
            context = " ".join(part.strip() for part in context_parts if part.strip())
            encoded = tokenizer(context, return_tensors="pt", truncation=True)
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                output = model(**encoded, output_hidden_states=True)
            vectors = []
            for layer_index in layer_indices:
                hidden_state = output.hidden_states[layer_index + 1]
                vectors.append(hidden_state[:, self.representation_token_index, :].detach().cpu().numpy().reshape(-1))
            activations.append(np.concatenate(vectors, axis=0))
        return np.asarray(activations, dtype=float)

    def record(self, benchmark, *, save_cache: bool = False, force: bool = False):
        cache_config = {
            "kind": "artificial-activations",
            "subject": self.subject_config(),
            "benchmark": benchmark.benchmark_config(),
        }
        if not force:
            cached = load_cache("activations", config=cache_config)
            if cached is not None:
                logger.info(f"Loaded cached activations for {self.model_id} on {benchmark.identifier}")
                self._load_cache_payload(cached)
                return self
        
        logger.info(f"Computing new activations for {self.model_id} on {benchmark.identifier}")

        from tqdm import tqdm
        all_values = []
        all_stimulus_ids = []
        for indices in tqdm(benchmark.iter_context_groups(), desc=f"Recording activations ({self.model_id})", leave=False):
            texts = [benchmark.stimuli[index].text for index in indices]
            values = self._extract_group_activations(texts)
            for local_index, global_index in enumerate(indices):
                all_values.append(values[local_index])
                all_stimulus_ids.append(benchmark.stimuli[global_index].stimulus_id)

        if self._layer_indices is None:
            _, _, layer_indices = self._load_model()
        else:
            layer_indices = self._layer_indices
        width = len(all_values[0]) if all_values else 0
        units_per_layer = width // len(layer_indices) if layer_indices else 0
        feature_ids = []
        for layer_index in layer_indices:
            for unit in range(units_per_layer):
                feature_ids.append(f"layer-{layer_index}--unit-{unit}")

        metadata = {
            "subject_identifier": self.identifier(),
            "benchmark_identifier": benchmark.identifier,
            "layers": list(layer_indices),
            "trained": self.trained,
        }
        self._store_recordings(
            benchmark,
            np.asarray(all_values, dtype=float),
            stimulus_ids=all_stimulus_ids,
            feature_ids=feature_ids,
            metadata=metadata,
        )
        if save_cache:
            save_cache_payload("activations", config=cache_config, payload=self._cache_payload())
        return self
