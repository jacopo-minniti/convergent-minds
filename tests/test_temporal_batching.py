import torch
import numpy as np
import pytest
from convminds.data.primitives import BrainTensor
from convminds.data.collate import collate_brains
from convminds.transforms.timeseries import TrimTRs, RandomWindow, SlidingWindow
from convminds.transforms.fir import FIRDelay
from convminds.transforms.roi import ROIFilter

def test_temporal_collation():
    # Two stories of different lengths: (T=10, V=2) and (T=5, V=2)
    b1 = BrainTensor(
        signal=torch.ones((10, 2)),
        coords=torch.zeros((2, 3)),
        rois={"earlyV": torch.BoolTensor([True, False])}
    )
    b2 = BrainTensor(
        signal=torch.ones((5, 2)) * 2,
        coords=torch.zeros((2, 3)),
        rois={"earlyV": torch.BoolTensor([True, False])}
    )
    
    batch = collate_brains([b1, b2])
    
    # Expected shape: (B=2, T=10, V=2)
    assert batch.signal.shape == (2, 10, 2)
    # Check values
    assert torch.all(batch.signal[0, :, :] == 1)
    assert torch.all(batch.signal[1, :5, :] == 2)
    assert torch.all(batch.signal[1, 5:, :] == 0) # Padded with zero
    
    # Check padding mask
    assert batch.padding_mask.shape == (2, 10, 2)
    assert torch.all(batch.padding_mask[1, 5:, :] == True)
    assert torch.all(batch.padding_mask[1, :5, :] == False)

def test_roi_filter():
    signal = torch.arange(6).reshape(1, 3, 2).float() # (B=1, T=3, V=2)
    # V0: [0, 2, 4], V1: [1, 3, 5]
    b = BrainTensor(
        signal=signal,
        coords=torch.zeros((2, 3)),
        rois={"earlyV": torch.BoolTensor([True, False])} # Only pick V0
    )
    
    roi_filter = ROIFilter(roi="earlyV")
    out = roi_filter(b)
    
    assert out.signal.shape == (1, 3, 1)
    assert torch.all(out.signal.squeeze() == torch.tensor([0., 2., 4.]))

def test_sequence_transforms_on_padded_batch():
    # (B=1, T=10, V=2) with padding from T=5 onwards
    signal = torch.ones((1, 10, 2))
    mask = torch.zeros((1, 10, 2), dtype=torch.bool)
    mask[:, 5:, :] = True
    
    b = BrainTensor(signal=signal, coords=torch.zeros((2, 3)), padding_mask=mask)
    
    # 1. TrimTRs
    trim = TrimTRs(start=2, end=2)
    out_trim = trim(b)
    # 10 - 2 - 2 = 6
    assert out_trim.signal.shape == (1, 6, 2)
    # Original padding was at T=5..9. In the trimmed 2..7, padding starts at T'=3 (original T=5)
    assert out_trim.padding_mask[0, 3:, 0].all()
    assert not out_trim.padding_mask[0, :3, 0].any()

    # 2. FIRDelay
    fir = FIRDelay(delays=[0, 1])
    out_fir = fir(b)
    assert out_fir.signal.shape == (1, 10, 4)
    # t=0 has zeros for delay=1
    assert torch.all(out_fir.signal[0, 0, 2:] == 0)

def test_window_transforms():
    signal = torch.arange(20).reshape(1, 10, 2).float()
    b = BrainTensor(signal=signal, coords=torch.zeros((2, 3)))
    
    # RandomWindow
    rw = RandomWindow(window_size=4)
    out_rw = rw(b)
    assert out_rw.signal.shape == (1, 4, 2)
    
    # SlidingWindow
    sw = SlidingWindow(window_size=4, offset=2)
    out_sw = sw(b)
    assert out_sw.signal.shape == (1, 4, 2)
    # Offset 2, so should start at index 2 (values 4, 5, ...)
    assert out_sw.signal[0, 0, 0] == 4.0

if __name__ == "__main__":
    test_temporal_collation()
    test_roi_filter()
    test_sequence_transforms_on_padded_batch()
    test_window_transforms()
    print("All temporal batching tests passed!")
