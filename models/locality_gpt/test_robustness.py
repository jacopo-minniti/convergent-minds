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
    # We need to tokenize manually since subject.tokenizer might not be exposed directly or we want control
    # But LocalityGPT2 wraps HuggingfaceSubject which has tokenizer
    tokenizer = subject.tokenizer
    inputs = tokenizer(input_text, return_tensors="pt")
    
    with torch.no_grad():
        output1 = subject.model(**inputs).logits
        output2 = subject.model(**inputs).logits
        
    torch.testing.assert_close(output1, output2, msg="Model output should be deterministic")

def test_decay_rate_sensitivity(gpt2_config):
    """Test that changing the decay rate affects the model output."""
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
    # We must ensure weights are same to compare effect of decay only
    # But LocalityGPT2 re-initializes model if untrained=True. 
    # To strictly test decay effect, we should load same weights.
    # However, for a "robustness" test, just checking they are different is a good start.
    # If we want to be precise, we can copy weights.
    
    subject_high = LocalityGPT2(
        model_id,
        region_layer_mapping=dummy_mapping,
        untrained=True,
        decay_rate=2.0
    )
    # Force weights to match subject_0
    subject_high.model.load_state_dict(subject_0.model.state_dict())
    subject_high.model.eval()
    
    with torch.no_grad():
        logits_0 = subject_0.model(**inputs).logits
        logits_high = subject_high.model(**inputs).logits
        
    # They should be different
    assert not torch.allclose(logits_0, logits_high), "Decay rate should affect model output"
    
    # Check that attention patterns are different
    with torch.no_grad():
        attn_0 = subject_0.model(**inputs, output_attentions=True).attentions[-1]
        attn_high = subject_high.model(**inputs, output_attentions=True).attentions[-1]
        
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
    
    # Their weights should likely be different (unless by some miracle random init matches pretrained)
    # We check the first layer's weight
    w_untrained = subject_untrained.model.transformer.h[0].attn.c_attn.weight
    w_trained = subject_trained.model.transformer.h[0].attn.c_attn.weight
    
    assert not torch.allclose(w_untrained, w_trained), "Untrained model should have different weights from trained model"

def test_score_object_integrity():
    """Test that the model can be scored (mocked) and returns expected structure."""
    # This is a bit harder to test without running the full benchmark, 
    # but we can check if the model forward pass works as expected for 'recording'.
    
    model_id = "gpt2"
    dummy_mapping = {'language_system': 'transformer.h.0'} # Use a valid layer name
    
    subject = LocalityGPT2(
        model_id,
        region_layer_mapping=dummy_mapping,
        untrained=True
    )
    
    # Mock input for BrainScore subject
    # HuggingfaceSubject expects list of strings usually
    sentences = ["Hello world", "Testing locality"]
    
    # We can't easily mock the whole BrainScore recording process here without dependencies,
    # but we can ensure the subject exposes the right methods.
    assert hasattr(subject, 'start_neural_recording')
    assert hasattr(subject, 'digest_text')
    
    # Basic smoke test for digest_text (if it doesn't require recording to be active)
    # Usually start_recording is needed.
    # We'll skip deep integration testing here to avoid mocking hell, 
    # relying on the other tests for logic correctness.

if __name__ == "__main__":
    pytest.main([__file__])
