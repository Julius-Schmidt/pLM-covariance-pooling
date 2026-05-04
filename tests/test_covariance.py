import torch
import pytest
from sop.pooling.covariance import CovariancePooler


def make_fitted_pooler(d: int, dc: int, center: bool = True, n: int = 25) -> CovariancePooler:
    torch.manual_seed(42)
    data = [(torch.randn(15, d), torch.ones(15, dtype=torch.bool)) for _ in range(n)]

    def get_iter():
        return iter(data)

    return CovariancePooler(d, dc, center).fit(get_iter)


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

def test_output_shape_batched():
    pooler = make_fitted_pooler(32, 8)
    X = torch.randn(5, 10, 32)
    mask = torch.ones(5, 10, dtype=torch.bool)
    assert pooler.pool(X, mask).shape == (5, 64)


def test_output_shape_unbatched():
    """pool() must handle a single [L, d] tensor without a batch dimension."""
    pooler = make_fitted_pooler(16, 4)
    X = torch.randn(10, 16)
    mask = torch.ones(10, dtype=torch.bool)
    assert pooler.pool(X, mask).shape == (16,)


def test_embedding_dim_property():
    pooler = make_fitted_pooler(16, 4)
    assert pooler.embedding_dim == 16   # dc² = 4² = 16


def test_not_fitted_raises():
    pooler = CovariancePooler(16, 4)
    with pytest.raises(RuntimeError, match="fit"):
        pooler.pool(torch.randn(1, 5, 16), torch.ones(1, 5, dtype=torch.bool))


# ---------------------------------------------------------------------------
# Mathematical properties
# ---------------------------------------------------------------------------

def test_covariance_matrix_is_symmetric():
    """C = (1/L)(XU)ᵀ(XU) is symmetric (PSD) for any X when L=R=U."""
    torch.manual_seed(7)
    d, dc = 16, 4
    pooler = make_fitted_pooler(d, dc)
    X = torch.randn(1, 10, d)
    mask = torch.ones(1, 10, dtype=torch.bool)
    out = pooler.pool(X, mask)           # [1, dc²]
    C = out.reshape(dc, dc)
    assert torch.allclose(C, C.T, atol=1e-5), "Pooled covariance matrix is not symmetric."


def test_pca_projection_is_orthonormal():
    """Top-dc eigenvectors of a symmetric matrix must be orthonormal."""
    pooler = make_fitted_pooler(32, 8)
    U = pooler._proj   # [d, dc]
    product = U.T @ U  # should be I_dc
    eye = torch.eye(8)
    assert torch.allclose(product, eye, atol=1e-5), "PCA projection is not orthonormal."


# ---------------------------------------------------------------------------
# Serialisation round-trip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip(tmp_path):
    d, dc = 16, 4
    pooler = make_fitted_pooler(d, dc)
    path = tmp_path / "proj.pt"
    pooler.save(path)

    loaded = CovariancePooler.load(path)
    X = torch.randn(2, 10, d)
    mask = torch.ones(2, 10, dtype=torch.bool)
    assert torch.allclose(pooler.pool(X, mask), loaded.pool(X, mask))
