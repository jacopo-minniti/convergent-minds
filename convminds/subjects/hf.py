from typing import Iterable, Optional

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from convminds.brainscore.artificial_subject import ArtificialSubject
from convminds.brainscore.model_helpers.huggingface import HuggingfaceSubject, get_layer_names


class HFLLMSubject(HuggingfaceSubject):
    """
    Convenience subject: instantiate a HuggingFace causal LM by model id,
    optionally untrained, with sensible default language-system layer mapping.
    """

    def __init__(
        self,
        model_id: str,
        layers: Optional[Iterable[str]] = None,
        untrained: bool = False,
        use_localizer: bool = False,
        localizer_kwargs: Optional[dict] = None,
        device: Optional[str] = None,
        representation_token_index: int = -1,
    ):
        if layers is None:
            all_layers = get_layer_names(model_id)
            layers = [all_layers[-1]]

        tokenizer = AutoTokenizer.from_pretrained(model_id, truncation_side="left")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if untrained:
            config = AutoConfig.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_config(config)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id)

        super().__init__(
            model_id=model_id,
            region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: list(layers)},
            model=model,
            tokenizer=tokenizer,
            use_localizer=use_localizer,
            localizer_kwargs=localizer_kwargs,
            device=device,
            representation_token_index=representation_token_index,
        )
