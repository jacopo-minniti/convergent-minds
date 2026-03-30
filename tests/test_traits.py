import pytest
import torch
from convminds.data.primitives import BrainTensor
from convminds.data.types import DataCategory, check_trait
from convminds.errors import DataTraitMismatchError
from convminds.transforms import HRFWindow

def test_data_trait_mismatch_error():
    # Create stimulus-level data (like Pereira)
    brain = BrainTensor(
        signal=torch.randn(1, 10), 
        coords=torch.randn(10, 3),
        category=DataCategory.STIMULUS_LEVEL
    )
    
    # Try to check for Token Level (should fail)
    with pytest.raises(DataTraitMismatchError) as excinfo:
        check_trait(brain, DataCategory.TOKEN_LEVEL, operation_name="TestOp")
    
    assert "Operation 'TestOp' requires trait TOKEN_LEVEL" in str(excinfo.value)
    assert "categorized as STIMULUS_LEVEL" in str(excinfo.value)

def test_hrf_window_trait_enforcement():
    # Create stimulus-level data
    brain = BrainTensor(
        signal=torch.randn(1, 10), 
        coords=torch.randn(10, 3),
        category=DataCategory.STIMULUS_LEVEL
    )
    
    transform = HRFWindow(t=1)
    
    # Applying HRF window to stimulus-level data should raise the custom error
    with pytest.raises(DataTraitMismatchError):
        transform(brain)

def test_hrf_window_success_on_token_level():
    # Create token-level data
    brain = BrainTensor(
        signal=torch.randn(5, 10), 
        coords=torch.randn(10, 3),
        category=DataCategory.TOKEN_LEVEL
    )
    
    transform = HRFWindow(t=2)
    out = transform(brain)
    
    assert out.signal.shape[0] == 2
    assert out.category == DataCategory.TOKEN_LEVEL
