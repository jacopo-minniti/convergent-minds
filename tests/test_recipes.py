from __future__ import annotations

import types
import torch
import convminds as cm
from convminds.recipes.residual_steer import ResidualSteerRecipe


def test_residual_steer_recipe_init(monkeypatch):
    # Patch transformers to avoid downloads
    import transformers
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: types.SimpleNamespace(
            config=types.SimpleNamespace(hidden_size=8),
            model=types.SimpleNamespace(layers=torch.nn.ModuleList([torch.nn.Identity()])),
            parameters=lambda: iter([]),
            to=lambda x: None,
        ),
    )
    
    # Mock benchmark
    benchmark = types.SimpleNamespace(
        identifier="test",
        split_indices={"train": [0], "test": [0]},
        stimuli=[{"stimulus_id": "1", "text": "test"}],
        human_values=[torch.randn(10, 4)] # 10 TRs, 4 voxels
    )
    
    recipe = ResidualSteerRecipe(
        benchmark=benchmark,
        llm_id="test",
        batch_size=1
    )
    
    assert recipe.model is not None
    assert isinstance(recipe.model, cm.models.ResidualSteerLM)
    assert recipe.datamodule is not None
