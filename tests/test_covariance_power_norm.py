"""Matrix-power-normalised covariance poolers: invariance + persistence."""
import pytest
import torch

from sop.pooling.covariance import CovariancePooler
from sop.pooling.covariance_pca import CovariancePCAPooler


def pad(X: torch.Tensor, extra: int) -> tuple[torch.Tensor, torch.Tensor]:
    L, d = X.shape
    X_pad = torch.cat([X, torch.zeros(extra, d)], dim=0)
    mask = torch.zeros(L + extra, dtype=torch.bool)
    mask[:L] = True
    return X_pad, mask


def test_power_norm_changes_cov_output():
    torch.manual_seed(0)
    d, dc, L = 16, 4, 10
    X = torch.randn(1, L, d)
    mask = torch.ones(1, L, dtype=torch.bool)

    torch.manual_seed(1)
    plain = CovariancePooler(d, dc, power_norm=False)
    torch.manual_seed(1)
    sqrt = CovariancePooler(d, dc, power_norm=True)
    assert not torch.allclose(plain.pool(X, mask), sqrt.pool(X, mask), atol=1e-4)


@pytest.mark.parametrize("extra", [1, 5, 20])
def test_cov_power_norm_mask_invariant(extra):
    torch.manual_seed(2)
    d, dc, L = 16, 4, 12
    pooler = CovariancePooler(d, dc, power_norm=True)
    X = torch.randn(L, d)
    ref = pooler.pool(X.unsqueeze(0), torch.ones(L, dtype=torch.bool).unsqueeze(0))
    X_pad, mask_pad = pad(X, extra)
    out = pooler.pool(X_pad.unsqueeze(0), mask_pad.unsqueeze(0))
    assert torch.allclose(ref, out, atol=1e-5)


def test_cov_power_norm_survives_roundtrip(tmp_path):
    torch.manual_seed(3)
    d, dc = 16, 4
    pooler = CovariancePooler(d, dc, power_norm=True)
    path = tmp_path / "pooler.pt"
    pooler.save_state(path)
    loaded = CovariancePooler.from_pretrained(path)
    assert loaded.power_norm  # flag restored
    X = torch.randn(2, 10, d)
    mask = torch.ones(2, 10, dtype=torch.bool)
    assert torch.allclose(pooler.pool(X, mask), loaded.pool(X, mask), atol=1e-5)


def test_cov_power_norm_gradients_flow():
    torch.manual_seed(4)
    d, dc = 16, 4
    pooler = CovariancePooler(d, dc, power_norm=True)
    X = torch.randn(2, 8, d)
    mask = torch.ones(2, 8, dtype=torch.bool)
    pooler.pool(X, mask).sum().backward()
    assert pooler.proj_l.weight.grad.abs().sum() > 0
    assert pooler.proj_r.weight.grad.abs().sum() > 0


def test_pca_power_norm_setter_and_psd_sqrt():
    """PCA pool is PSD, so C^{1/2} is exact: its eigenvalues are sqrt of C's."""
    torch.manual_seed(5)
    d, dc, L = 12, 4, 30
    pooler = CovariancePCAPooler(d, dc, center=True)
    # Fit on a few synthetic proteins.
    data = [(torch.randn(L, d), torch.ones(L, dtype=torch.bool)) for _ in range(8)]
    pooler.fit(lambda: iter(data))

    X = torch.randn(1, L, d)
    mask = torch.ones(1, L, dtype=torch.bool)
    plain = pooler.pool(X, mask).reshape(dc, dc)

    pooler.set_power_norm(True)
    sqrt = pooler.pool(X, mask).reshape(dc, dc)

    eig_plain = torch.linalg.eigvalsh(plain).clamp_min(0)
    eig_sqrt = torch.linalg.eigvalsh(sqrt).clamp_min(0)
    assert torch.allclose(eig_sqrt, eig_plain.sqrt(), atol=1e-2, rtol=1e-2)
