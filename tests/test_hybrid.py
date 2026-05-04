import torch

from sop.pooling.covariance import CovariancePooler
from sop.pooling.hybrid import HybridPooler
from sop.pooling.mean import MeanPooler


def test_output_dim_is_d_plus_dc_squared():
    d, dc = 32, 8
    pooler = HybridPooler(d, CovariancePooler(d, dc))
    assert pooler.embedding_dim == d + dc ** 2


def test_concat_order_is_mean_then_cov():
    """First d entries must equal MeanPooler(d) output, remainder = CovariancePooler output."""
    torch.manual_seed(0)
    d, dc, L = 16, 4, 8
    cov = CovariancePooler(d, dc)
    hybrid = HybridPooler(d, cov)
    mean = MeanPooler(d)

    X = torch.randn(2, L, d)
    mask = torch.ones(2, L, dtype=torch.bool)

    out = hybrid.pool(X, mask)         # [2, d + dc²]
    assert out.shape == (2, d + dc ** 2)
    assert torch.allclose(out[:, :d], mean.pool(X, mask), atol=1e-6)
    assert torch.allclose(out[:, d:], cov.pool(X, mask), atol=1e-6)


def test_dim_mismatch_raises():
    import pytest
    cov = CovariancePooler(d=16, dc=4)
    with pytest.raises(ValueError, match="match"):
        HybridPooler(d=32, cov=cov)
