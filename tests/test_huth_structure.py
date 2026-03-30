import torch
import numpy as np
import pytest
from convminds.data.alignment import lanczos_interp2d, lanczos_kernel
from convminds.transforms.fir import FIRDelay
from convminds.data.primitives import BrainTensor

def test_lanczos_kernel():
    # Test t=0 case
    val = lanczos_kernel(1.0, np.array([0.0]), window=3)
    assert val[0] == 1.0
    
    # Test cutoff and window
    val = lanczos_kernel(1.0, np.array([4.0]), window=3)
    assert val[0] == 0.0

def test_lanczos_interp2d():
    # Synthetic data: (10 samples, 2 features)
    data = np.ones((10, 2))
    old_times = np.arange(10)
    new_times = np.arange(0, 10, 2) # Downsample by 2
    
    newdata = lanczos_interp2d(data, old_times, new_times, window=3)
    assert newdata.shape == (5, 2)
    # With all ones, interp should be close to 1
    assert np.allclose(newdata, 1.0, atol=0.1)

def test_fir_delay():
    # (T=10, F=2)
    signal = torch.ones((10, 2))
    brain = BrainTensor(signal=signal, coords=torch.zeros((2, 3)))
    
    # Delay by 1 and 2
    fir = FIRDelay(delays=[0, 1, 2])
    out = fir(brain)
    
    assert out.signal.shape == (10, 6) # (T, F * 3)
    # Check that t=0 has zeros for delays 1 and 2
    # delay 0: [1, 1], delay 1: [0, 0], delay 2: [0, 0]
    assert torch.all(out.signal[0, 2:] == 0)
    # Check that t=2 has ones for everyone
    assert torch.all(out.signal[2, :] == 1)

def test_fir_delay_batched():
    # (B=2, T=10, F=2)
    signal = torch.ones((2, 10, 2))
    brain = BrainTensor(signal=signal, coords=torch.zeros((2, 3)))
    
    fir = FIRDelay(delays=[0, 1])
    out = fir(brain)
    
    assert out.signal.shape == (2, 10, 4)
    assert torch.all(out.signal[:, 0, 2:] == 0)

if __name__ == "__main__":
    test_lanczos_kernel()
    test_lanczos_interp2d()
    test_fir_delay()
    test_fir_delay_batched()
    print("All tests passed!")
