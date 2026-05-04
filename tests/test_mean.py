import torch
import pytest
from sop.pooling.mean import MeanPooler


def test_full_mask_equals_torch_mean():
    """With all positions valid, pool() must equal X.mean(dim=1)."""
    torch.manual_seed(0)
    X = torch.randn(4, 8, 16)
    mask = torch.ones(4, 8, dtype=torch.bool)
    pooler = MeanPooler(16)
    assert torch.allclose(pooler.pool(X, mask), X.mean(dim=1), atol=1e-6)


def test_single_valid_residue():
    """When only the first residue is valid, the output must equal that residue."""
    torch.manual_seed(1)
    d = 12
    X = torch.randn(1, 6, d)
    mask = torch.zeros(1, 6, dtype=torch.bool)
    mask[0, 0] = True
    pooler = MeanPooler(d)
    assert torch.allclose(pooler.pool(X, mask)[0], X[0, 0], atol=1e-6)


def test_embedding_dim_property():
    assert MeanPooler(32).embedding_dim == 32


def test_output_shape(small_batch):
    X, mask, _ = small_batch
    B, _, d = X.shape
    pooler = MeanPooler(d)
    out = pooler.pool(X, mask)
    assert out.shape == (B, d)
