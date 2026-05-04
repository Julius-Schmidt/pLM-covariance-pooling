import pytest
import torch

from sop.pooling.covariance_pca import CovariancePCAPooler


def _fitted(d: int, dc: int, n: int = 25, L: int = 15, center: bool = True) -> CovariancePCAPooler:
    torch.manual_seed(42)
    proteins = [(torch.randn(L, d), torch.ones(L, dtype=torch.bool)) for _ in range(n)]
    pooler = CovariancePCAPooler(d, dc, center=center)
    pooler.fit(lambda: iter(proteins))
    return pooler


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

def test_output_shape_batched():
    pooler = _fitted(32, 8)
    X = torch.randn(5, 10, 32)
    mask = torch.ones(5, 10, dtype=torch.bool)
    assert pooler.pool(X, mask).shape == (5, 64)


def test_output_shape_unbatched():
    pooler = _fitted(16, 4)
    X = torch.randn(10, 16)
    mask = torch.ones(10, dtype=torch.bool)
    assert pooler.pool(X, mask).shape == (16,)


def test_embedding_dim_property():
    assert CovariancePCAPooler(16, 4).embedding_dim == 16


def test_unfitted_raises():
    pooler = CovariancePCAPooler(16, 4)
    X = torch.randn(1, 5, 16)
    mask = torch.ones(1, 5, dtype=torch.bool)
    with pytest.raises(RuntimeError, match="fit"):
        pooler.pool(X, mask)


# ---------------------------------------------------------------------------
# Mathematical properties
# ---------------------------------------------------------------------------

def test_projection_is_orthonormal():
    """Top-dc eigenvectors of a symmetric matrix must be orthonormal."""
    pooler = _fitted(32, 8)
    UtU = pooler.proj.T @ pooler.proj
    assert torch.allclose(UtU, torch.eye(8), atol=1e-5)


def test_output_is_symmetric():
    """C = (1/L)(XU)ᵀ(XU) must be symmetric (PSD) — single shared projection."""
    pooler = _fitted(32, 8)
    X = torch.randn(1, 10, 32)
    mask = torch.ones(1, 10, dtype=torch.bool)
    flat = pooler.pool(X, mask).reshape(8, 8)
    assert torch.allclose(flat, flat.T, atol=1e-5)


# ---------------------------------------------------------------------------
# Masking invariance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("extra", [1, 5, 20])
@pytest.mark.parametrize("center", [True, False])
def test_padding_does_not_change_result(extra, center):
    pooler = _fitted(16, 4, center=center)
    X = torch.randn(12, 16)
    mask = torch.ones(12, dtype=torch.bool)

    ref = pooler.pool(X.unsqueeze(0), mask.unsqueeze(0))

    X_pad = torch.cat([X, torch.zeros(extra, 16)], dim=0)
    mask_pad = torch.zeros(12 + extra, dtype=torch.bool)
    mask_pad[:12] = True

    out = pooler.pool(X_pad.unsqueeze(0), mask_pad.unsqueeze(0))
    assert torch.allclose(ref, out, atol=1e-5), (
        f"PCA pool changed with {extra} pad rows (center={center})."
    )


# ---------------------------------------------------------------------------
# Fit refuses empty input
# ---------------------------------------------------------------------------

def test_fit_raises_on_empty():
    pooler = CovariancePCAPooler(8, 2)
    with pytest.raises(ValueError, match="No valid residues"):
        pooler.fit(lambda: iter([]))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_save_and_from_pretrained_roundtrip(tmp_path):
    pooler = _fitted(16, 4)
    path = tmp_path / "pca.pt"
    pooler.save_state(path)

    loaded = CovariancePCAPooler.from_pretrained(path)
    X = torch.randn(2, 10, 16)
    mask = torch.ones(2, 10, dtype=torch.bool)
    assert torch.allclose(pooler.pool(X, mask), loaded.pool(X, mask))
    assert loaded.is_fitted


def test_freeze_is_noop():
    """No parameters means freeze() is purely cosmetic — must not error."""
    pooler = _fitted(16, 4)
    pooler.freeze()  # smoke test
    # Buffers stay buffers.
    assert "proj" in dict(pooler.named_buffers())
