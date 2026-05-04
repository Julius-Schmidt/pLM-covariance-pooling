import pytest
import torch


@pytest.fixture
def small_batch():
    """Three proteins, different lengths, padded to L=10, d=8.

    Padded positions are explicitly zeroed so that tests can verify masking
    without also needing to check initialisation.
    """
    torch.manual_seed(42)
    B, L, d = 3, 10, 8
    lengths = torch.tensor([10, 7, 4])
    X = torch.randn(B, L, d)
    mask = torch.arange(L).unsqueeze(0) < lengths.unsqueeze(1)   # [B, L] bool
    X = X * mask.unsqueeze(-1).float()                            # zero padding
    return X, mask, lengths
