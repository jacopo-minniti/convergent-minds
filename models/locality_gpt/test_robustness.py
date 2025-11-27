import pytest
import torch
import numpy as np
from transformers import AutoConfig
from .model import LocalityGPT2

@pytest.fixture(scope="module")
def gpt2_config():
    """Provides a minimal GPT-2 config for testing."""
    config = AutoConfig.from_pretrained("gpt2", n_layer=2, n_head=2, n_embd=64)
    config.attn_implementation = "eager"
    return config

def _get_last_layer_attention(model, inputs):
    """
    Robustly retrieves the attention weights of the last layer.
    Attempts standard output_attentions first, then falls back to a forward hook
    if the model (e.g. via SDPA or specific configs) returns None.
    """
    # Ensure config requests attentions
    model.config.output_attentions = True
    
    # 1. Try standard forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        if outputs.attentions is not None and len(outputs.attentions) > 0:
            last_attn = outputs.attentions[-1]
            if last_attn is not None:
                return last_attn

    # 2. Fallback: Hook capture
    # Locate last transformer layer. For GPT2, it's model.transformer.h[-1]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        last_layer = model.transformer.h[-1]
    elif hasattr(model, "h"):
        last_layer = model.h[-1]
    else:
        raise ValueError("Could not locate transformer layers for hook capture.")

    # Target the attention module
    target_module = getattr(last_layer, "attn", getattr(last_layer, "self_attn", None))
    if target_module is None:
         raise ValueError("Could not locate attention module in last layer.")

    captured = {}
    def hook(_module, _inputs, output):
        # GPT2Attention forward returns: (attn_output, present, attn_weights)
        # output[2] is attn_weights if present
        if isinstance(output, tuple):
            if len(output) >= 3:
                captured["attn"] = output[2]
            elif len(output) == 2:
                # If only 2 returned, it might be (output, present).
                # This happens if output_attentions=False was effectively passed.
                pass
        elif torch.is_tensor(output):
            pass

    handle = target_module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        handle.remove()

    return captured.get("attn")

def test_deterministic_behavior(gpt2_config):
    """Test that the model produces deterministic outputs for the same input."""
    model_id = "gpt2"
    dummy_mapping = {'dummy_region': 0}
    
    subject = LocalityGPT2(
        model_id,
        region_layer_mapping=dummy_mapping,
        untrained=True, # Use untrained for speed/independence
        decay_rate=1.0
    )
    subject.model.eval()
    
    input_text = "The quick brown fox"
    tokenizer = subject.tokenizer
    inputs = tokenizer(input_text, return_tensors="pt")
    
    with torch.no_grad():
        output1 = subject.model(**inputs).logits
        output2 = subject.model(**inputs).logits
        
    torch.testing.assert_close(output1, output2, msg="Model output should be deterministic")

def test_decay_rate_sensitivity(gpt2_config):
    """Test that changing the decay rate affects the model output and attention weights."""
    model_id = "gpt2"
    dummy_mapping = {'dummy_region': 0}
    input_text = "The quick brown fox jumps over the lazy dog"
    
    # Model with decay = 0.0 (no locality)
    subject_0 = LocalityGPT2(
        model_id,
        region_layer_mapping=dummy_mapping,
        untrained=True,
        decay_rate=0.0
    )
    subject_0.model.eval()
    tokenizer = subject_0.tokenizer
    inputs = tokenizer(input_text, return_tensors="pt")
    
    # Model with decay = 2.0 (strong locality)
    subject_high = LocalityGPT2(
        model_id,
        region_layer_mapping=dummy_mapping,
        untrained=True,
        decay_rate=2.0
    )
    # Force weights to match subject_0 to isolate decay rate effect
    subject_high.model.load_state_dict(subject_0.model.state_dict())
    subject_high.model.eval()
    
    with torch.no_grad():
        logits_0 = subject_0.model(**inputs).logits
        logits_high = subject_high.model(**inputs).logits
        
    # They should be different
    assert not torch.allclose(logits_0, logits_high), "Decay rate should affect model output"
    
    # Check that attention patterns are different using robust capture
    attn_0 = _get_last_layer_attention(subject_0.model, inputs)
    attn_high = _get_last_layer_attention(subject_high.model, inputs)
    
    assert attn_0 is not None, "Failed to capture attention for decay=0.0"
    assert attn_high is not None, "Failed to capture attention for decay=2.0"
    
    assert not torch.allclose(attn_0, attn_high), "Decay rate should affect attention weights"

def test_untrained_vs_trained_structure():
    """Test that untrained flag actually does something (at least doesn't crash and returns valid model)."""
    model_id = "gpt2"
    dummy_mapping = {'dummy_region': 0}
    
    subject_untrained = LocalityGPT2(
        model_id,
        region_layer_mapping=dummy_mapping,
        untrained=True
    )
    
    subject_trained = LocalityGPT2(
        model_id,
        region_layer_mapping=dummy_mapping,
        untrained=False
    )
    
    # Just check they are both valid models
    assert subject_untrained.model is not None
    assert subject_trained.model is not None
    
    # Their weights should likely be different
    w_untrained = subject_untrained.model.transformer.h[0].attn.c_attn.weight
    w_trained = subject_trained.model.transformer.h[0].attn.c_attn.weight
    
    assert not torch.allclose(w_untrained, w_trained), "Untrained model should have different weights from trained model"

def test_score_object_integrity():
    """Test that the model subject exposes expected methods for BrainScore."""
    model_id = "gpt2"
    dummy_mapping = {'language_system': 'transformer.h.0'}
    
    subject = LocalityGPT2(
        model_id,
        region_layer_mapping=dummy_mapping,
        untrained=True
    )
    
    assert hasattr(subject, 'start_neural_recording')
    assert hasattr(subject, 'digest_text')

if __name__ == "__main__":
    pytest.main([__file__])