from __future__ import annotations

import torch.nn as nn


class ResidualInjector(nn.Module):
    """
    Wraps a Transformer layer to inject an intervention module output
    into the residual stream without breaking HuggingFace cache flow.
    """

    def __init__(self, base_layer: nn.Module, intervention_module: nn.Module, kwarg_name: str = "brain_context"):
        super().__init__()
        self.base_layer = base_layer
        self.intervention_module = intervention_module
        self.kwarg_name = kwarg_name

    def forward(self, hidden_states, *args, **kwargs):
        brain_context = kwargs.pop(self.kwarg_name, None)
        layer_outputs = self.base_layer(hidden_states, *args, **kwargs)

        if isinstance(layer_outputs, tuple):
            new_hidden_states = layer_outputs[0]
        else:
            new_hidden_states = layer_outputs

        if brain_context is not None:
            steer_update = self.intervention_module(new_hidden_states, brain_context)
            new_hidden_states = new_hidden_states + steer_update

        if isinstance(layer_outputs, tuple):
            return (new_hidden_states,) + layer_outputs[1:]
        return new_hidden_states
