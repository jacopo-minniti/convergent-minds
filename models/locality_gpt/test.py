import pytest
import torch
from transformers import AutoConfig, GPT2LMHeadModel
from brainscore.model_helpers.huggingface import HuggingfaceSubject
from .model import LocalityGPT2, LocalityGPT2Attention


@pytest.fixture(scope="module")
def gpt2_config():
    """Provides a minimal, non-scaled GPT-2 config for testing."""
    config = AutoConfig.from_pretrained(
        "gpt2",
        n_head=2,
        n_layer=1,
        n_embd=64
    )
    config.scale_attn_weights = False
    config.attn_implementation = "eager"
    return config


def test_model_instantiation(gpt2_config):
    """Tests if the LocalityGPT2 wrapper correctly passes the decay_rate."""
    model_id = "gpt2"
    dummy_mapping = {'dummy_region': 0}
    test_decay_rate = 1.5

    model_wrapper = LocalityGPT2(
        model_id,
        region_layer_mapping=dummy_mapping,
        untrained=False,
        decay_rate=test_decay_rate
    )

    assert isinstance(model_wrapper, HuggingfaceSubject)
    for layer in model_wrapper.model.transformer.h:
        assert isinstance(layer.attn, LocalityGPT2Attention)
        assert layer.attn.decay_rate == test_decay_rate


def test_attention_with_default_decay(gpt2_config):
    """Tests attention decay with the default decay_rate of 1.0."""
    gpt2_config.n_head = 1
    attn_layer = LocalityGPT2Attention(gpt2_config, layer_idx=0)
    attn_layer.eval()

    seq_len = 4
    head_dim = gpt2_config.n_embd // gpt2_config.n_head
    query = torch.ones(1, 1, seq_len, head_dim)
    key = torch.ones(1, 1, seq_len, head_dim)
    value = torch.ones(1, 1, seq_len, head_dim)

    # Manual calculation for decay_rate = 1.0
    raw_weights = torch.full((1, 1, seq_len, seq_len), float(head_dim))
    i = torch.arange(seq_len).unsqueeze(1)
    j = torch.arange(seq_len).unsqueeze(0)
    distance = torch.abs(i - j).float()
    distance = torch.abs(i - j).float()
    decay_penalty = 1.0 * distance
    decayed_weights = raw_weights - decay_penalty.unsqueeze(0).unsqueeze(0)
    
    causal_mask = attn_layer.bias[:, :, :seq_len, :seq_len].bool()
    masked_bias = attn_layer.masked_bias
    final_weights = torch.where(causal_mask, decayed_weights, masked_bias)
    expected_attn = torch.nn.Softmax(dim=-1)(final_weights)
    
    _, actual_attn = attn_layer._attn(query, key, value)
    
    torch.testing.assert_close(actual_attn, expected_attn)
    assert actual_attn[0, 0, 2, 0] < actual_attn[0, 0, 2, 1]
    assert actual_attn[0, 0, 2, 1] < actual_attn[0, 0, 2, 2]


@pytest.mark.parametrize("decay_rate", [0.5, 2.0])
def test_attention_positive_decay_rate(gpt2_config, decay_rate):
    """Tests that positive decay rates correctly enforce locality."""
    gpt2_config.n_head = 1
    attn_layer = LocalityGPT2Attention(gpt2_config, layer_idx=0, decay_rate=decay_rate)
    attn_layer.eval()
    
    seq_len = 4
    head_dim = gpt2_config.n_embd // gpt2_config.n_head
    query = torch.ones(1, 1, seq_len, head_dim)
    key = torch.ones(1, 1, seq_len, head_dim)
    value = torch.ones(1, 1, seq_len, head_dim)
    
    _, attn_weights = attn_layer._attn(query, key, value)
    
    # Attention to nearer tokens should be higher than to farther tokens.
    assert attn_weights[0, 0, 3, 1] < attn_weights[0, 0, 3, 2]
    assert attn_weights[0, 0, 3, 0] < attn_weights[0, 0, 3, 1]


def test_attention_negative_decay_rate(gpt2_config):
    """Tests that a negative decay rate correctly enforces anti-locality."""
    gpt2_config.n_head = 1
    attn_layer = LocalityGPT2Attention(gpt2_config, layer_idx=0, decay_rate=-0.1)
    attn_layer.eval()
    
    seq_len = 4
    head_dim = gpt2_config.n_embd // gpt2_config.n_head
    query = torch.ones(1, 1, seq_len, head_dim)
    key = torch.ones(1, 1, seq_len, head_dim)
    value = torch.ones(1, 1, seq_len, head_dim)

    _, attn_weights = attn_layer._attn(query, key, value)
    
    # Attention to nearer tokens should be lower than to farther tokens.
    assert attn_weights[0, 0, 3, 2] < attn_weights[0, 0, 3, 1]
    assert attn_weights[0, 0, 3, 1] < attn_weights[0, 0, 3, 0]


def test_attention_zero_decay_rate(gpt2_config):
    """Tests that a zero decay rate results in uniform attention (no bias)."""
    gpt2_config.n_head = 1
    attn_layer = LocalityGPT2Attention(gpt2_config, layer_idx=0, decay_rate=0.0)
    attn_layer.eval()
    
    seq_len = 3
    head_dim = gpt2_config.n_embd // gpt2_config.n_head
    query = torch.ones(1, 1, seq_len, head_dim)
    key = torch.ones(1, 1, seq_len, head_dim)
    value = torch.ones(1, 1, seq_len, head_dim)

    _, attn_weights = attn_layer._attn(query, key, value)
    
    # All allowed positions should have equal attention scores.
    torch.testing.assert_close(attn_weights[0, 0, 2, 0], attn_weights[0, 0, 2, 1])
    torch.testing.assert_close(attn_weights[0, 0, 2, 1], attn_weights[0, 0, 2, 2])

