"""Masking invariance tests.

Core property: the pooled result must be identical whether or not extra
zero-padded rows are appended, as long as the mask correctly marks them.
Any pooler that fails this test is subtly broken.
"""
import torch
import pytest

from sop.pooling.mean import MeanPooler
from sop.pooling.covariance import CovariancePooler


def pad(X: torch.Tensor, extra: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Append `extra` zero rows; return (X_padded [L+extra, d], mask [L+extra])."""
    L, d = X.shape
    X_pad = torch.cat([X, torch.zeros(extra, d)], dim=0)
    mask = torch.zeros(L + extra, dtype=torch.bool)
    mask[:L] = True
    return X_pad, mask


def _fit_covariance_pooler(d: int, dc: int, center: bool = True) -> CovariancePooler:
    torch.manual_seed(99)
    proteins = [(torch.randn(15, d), torch.ones(15, dtype=torch.bool)) for _ in range(30)]

    def get_iter():
        return iter(proteins)

    pooler = CovariancePooler(d, dc, center=center)
    pooler.fit(get_iter)
    return pooler


class TestMeanPoolingMaskInvariance:
    @pytest.mark.parametrize("extra", [1, 5, 20])
    def test_padding_does_not_change_result(self, extra):
        torch.manual_seed(0)
        d, L = 16, 12
        X = torch.randn(L, d)
        mask_full = torch.ones(L, dtype=torch.bool)
        pooler = MeanPooler(d)

        ref = pooler.pool(X.unsqueeze(0), mask_full.unsqueeze(0))

        X_pad, mask_pad = pad(X, extra)
        out = pooler.pool(X_pad.unsqueeze(0), mask_pad.unsqueeze(0))

        assert torch.allclose(ref, out, atol=1e-6), (
            f"MeanPooler output changed when {extra} padding rows were added."
        )


class TestCovariancePoolingMaskInvariance:
    @pytest.mark.parametrize("extra", [1, 5, 20])
    @pytest.mark.parametrize("center", [True, False])
    def test_padding_does_not_change_result(self, extra, center):
        torch.manual_seed(1)
        d, dc, L = 16, 4, 12
        pooler = _fit_covariance_pooler(d, dc, center)
        X = torch.randn(L, d)
        mask_full = torch.ones(L, dtype=torch.bool)

        ref = pooler.pool(X.unsqueeze(0), mask_full.unsqueeze(0))

        X_pad, mask_pad = pad(X, extra)
        out = pooler.pool(X_pad.unsqueeze(0), mask_pad.unsqueeze(0))

        assert torch.allclose(ref, out, atol=1e-5), (
            f"CovariancePooler (center={center}) output changed with {extra} padding rows."
        )
