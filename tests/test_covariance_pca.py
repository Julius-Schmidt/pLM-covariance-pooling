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
# Input validation
# ---------------------------------------------------------------------------

def test_dc_exceeds_d_raises():
    with pytest.raises(ValueError, match="dc"):
        CovariancePCAPooler(d=8, dc=16)


def test_dc_zero_raises():
    with pytest.raises(ValueError, match="dc"):
        CovariancePCAPooler(d=16, dc=0)


def test_dc_equals_d_is_valid():
    """dc == d is a legal edge case (no compression)."""
    pooler = _fitted(d=8, dc=8)
    assert pooler.embedding_dim == 64


def test_empty_protein_skipped():
    """Proteins with all-False masks must not affect the fitted projection."""
    torch.manual_seed(0)
    d, dc = 16, 4
    valid = [(torch.randn(10, d), torch.ones(10, dtype=torch.bool)) for _ in range(20)]
    empty = (torch.randn(10, d), torch.zeros(10, dtype=torch.bool))

    p_valid = CovariancePCAPooler(d, dc).fit(lambda: iter(valid))
    p_with_empty = CovariancePCAPooler(d, dc).fit(lambda: iter(valid + [empty]))

    # Sign of eigenvectors is arbitrary — compare absolute values.
    assert torch.allclose(p_valid.proj.abs(), p_with_empty.proj.abs(), atol=1e-5)


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


def test_protein_weighted_differs_from_residue_weighted():
    """When protein lengths vary by >10×, protein-weighting and residue-
    weighting yield meaningfully different principal directions. This locks
    in the protein-weighting choice — flipping the implementation back to
    residue-weighted would make the subspaces match and fail this test.
    """
    torch.manual_seed(99)
    d, dc = 32, 4

    # Mix of short (L=5) and long (L=80) proteins. Residue-weighting lets long
    # proteins dominate; protein-weighting does not.
    short = [(torch.randn(5,  d), torch.ones(5,  dtype=torch.bool)) for _ in range(10)]
    long_ = [(torch.randn(80, d), torch.ones(80, dtype=torch.bool)) for _ in range(10)]
    mixed = short + long_

    pooler_pw = CovariancePCAPooler(d, dc).fit(lambda: iter(mixed))

    # Manual residue-weighted reference for comparison.
    cov_rw = torch.zeros(d, d, dtype=torch.float64)
    sum_x = torch.zeros(d, dtype=torch.float64)
    n_res = 0
    for X, mask in mixed:
        Xv = X[mask.bool()].double()
        sum_x += Xv.sum(0)
        n_res += Xv.shape[0]
    mean = sum_x / n_res
    for X, mask in mixed:
        Xv = X[mask.bool()].double() - mean
        cov_rw += Xv.T @ Xv
    cov_rw /= n_res - 1
    _, eigvecs_rw = torch.linalg.eigh(cov_rw)
    proj_rw = eigvecs_rw[:, -dc:].float()

    # Compare subspaces via singular values of the overlap. Identical subspaces
    # would give all singular values = 1; we require some difference.
    overlap = (pooler_pw.proj.T @ proj_rw).abs()
    svd_vals = torch.linalg.svdvals(overlap)
    assert not torch.allclose(svd_vals, torch.ones(dc), atol=1e-4), (
        "Protein-weighted and residue-weighted projections are unexpectedly "
        "identical for a dataset with highly variable protein lengths."
    )


def test_precision_float64_accumulation():
    """Catastrophic cancellation guard — when residues cluster tightly around
    a large mean, a float32 ``X − μ`` loses most of its significant digits.
    The projection must still come out orthonormal."""
    torch.manual_seed(5)
    d, dc = 16, 4

    base = torch.full((d,), 1e4)
    noisy = [(base + torch.randn(20, d) * 1e-3, torch.ones(20, dtype=torch.bool))
             for _ in range(30)]

    pooler = CovariancePCAPooler(d, dc).fit(lambda: iter(noisy))
    UtU = pooler.proj.T @ pooler.proj
    assert torch.allclose(UtU, torch.eye(dc), atol=1e-4), (
        "Projection lost orthonormality — likely a float64 precision regression "
        "in fit() (subtraction in float32, or eigh on float32 cov)."
    )


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
