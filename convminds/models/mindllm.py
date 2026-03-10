from __future__ import annotations

import convminds.nn as cnn
from convminds.models.prompt_conditioned import PromptConditionedLM


class SpatialPrefixLM(PromptConditionedLM):
    """
    Spatial attention encoder with prefix fusion for LLM conditioning.
    """

    def __init__(self, llm_id: str = "lmsys/vicuna-7b-v1.5", num_queries: int = 128, llm_dim: int = 4096):
        encoder = cnn.encoders.SpatialAttention(
            num_queries=num_queries,
            query_dim=llm_dim,
            use_coords=True,
        )
        fusion = cnn.fusion.PrefixFusion()
        super().__init__(llm_id=llm_id, encoder=encoder, fusion=fusion)
