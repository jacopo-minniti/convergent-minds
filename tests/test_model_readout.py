import torch
import convminds as cm
import convminds.nn as cnn

class LinearBaseline(torch.nn.Module):
    """Refactored version of the model from base_pipeline."""
    def __init__(self, n_voxels=333, dim=768):
        super().__init__()
        self.proj = torch.nn.Linear(n_voxels, dim)
        
    def forward(self, brain_tensor):
        # We now check that it's stimulus-level or we handle it
        x = brain_tensor.signal.mean(dim=1)
        return self.proj(x)

def test_linear_readout_batch_flow():
    # Simulate a batch from BrainDataModule
    B, T, N = 8, 1, 333
    brain = cm.data.primitives.BrainTensor(
        signal=torch.randn(B, T, N),
        coords=torch.randn(B, N, 3),
        category=cm.data.types.DataCategory.STIMULUS_LEVEL
    )
    
    model = LinearBaseline(n_voxels=N, dim=768)
    out = model(brain)
    
    assert out.shape == (B, 768)
    assert not torch.isnan(out).any()

def test_spatial_attention_flow():
    # Simulate spatial grounding flow
    B, T, N = 4, 1, 333
    brain = cm.data.primitives.BrainTensor(
        signal=torch.randn(B, T, N),
        coords=torch.randn(B, N, 3),
        category=cm.data.types.DataCategory.STIMULUS_LEVEL
    )
    
    encoder = cnn.encoders.SpatialAttentionEncoder(num_queries=16, query_dim=768)
    latents = encoder(brain) # B, 16, 768
    
    assert latents.shape == (B, 16, 768)
