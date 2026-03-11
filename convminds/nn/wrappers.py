from __future__ import annotations

import torch
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


class SteerInjector(ResidualInjector):
    """
    Phase 3: Residual Injection & Norm Penalty.
    Calculates the step size (L2 norm squared) of the injection so it can be 
    added as a penalty to the total loss.
    """
    def __init__(self, base_layer: nn.Module, intervention_module: nn.Module, kwarg_name: str = "brain_context"):
        super().__init__(base_layer, intervention_module, kwarg_name)
        self.last_penalty = None

    def forward(self, hidden_states, *args, **kwargs):
        brain_context = kwargs.pop(self.kwarg_name, None)
        layer_outputs = self.base_layer(hidden_states, *args, **kwargs)

        if isinstance(layer_outputs, tuple):
            new_hidden_states = layer_outputs[0]
        else:
            new_hidden_states = layer_outputs

        if brain_context is not None:
            # h_steer * W_out
            steer_update = self.intervention_module(new_hidden_states, brain_context)
            
            # Record the L2 norm squared penalty
            # mean over batch/sequence to keep magnitude stable, or sum depending on desired scaling
            # The prompt asked for \| h_{steer} W_{out} \|_2^2
            self.last_penalty = torch.norm(steer_update, p=2, dim=-1).pow(2)
            
            new_hidden_states = new_hidden_states + steer_update
        else:
            self.last_penalty = None

        if isinstance(layer_outputs, tuple):
            return (new_hidden_states,) + layer_outputs[1:]
        return new_hidden_states
