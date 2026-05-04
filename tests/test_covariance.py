import torch

from sop.pooling.covariance import CovariancePooler


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

def test_output_shape_batched():
    pooler = CovariancePooler(32, 8)
    X = torch.randn(5, 10, 32)
    mask = torch.ones(5, 10, dtype=torch.bool)
    assert pooler.pool(X, mask).shape == (5, 64)


def test_output_shape_unbatched():
    """pool() must handle a single [L, d] tensor without a batch dimension."""
    pooler = CovariancePooler(16, 4)
    X = torch.randn(10, 16)
    mask = torch.ones(10, dtype=torch.bool)
    assert pooler.pool(X, mask).shape == (16,)


def test_embedding_dim_property():
    assert CovariancePooler(16, 4).embedding_dim == 16   # dc² = 4² = 16


def test_forward_alias():
    """forward() must call pool() (nn.Module entry point)."""
    pooler = CovariancePooler(16, 4)
    X = torch.randn(2, 5, 16)
    mask = torch.ones(2, 5, dtype=torch.bool)
    assert torch.equal(pooler(X, mask), pooler.pool(X, mask))


# ---------------------------------------------------------------------------
# Mathematical properties
# ---------------------------------------------------------------------------

def test_matches_manual_bilinear():
    """Pooler output equals (XL)ᵀ(XR)/L_valid when L=R=identity-shaped weights."""
    torch.manual_seed(0)
    d, dc, L = 16, 4, 8
    pooler = CovariancePooler(d, dc)
    X = torch.randn(1, L, d)
    mask = torch.ones(1, L, dtype=torch.bool)

    out = pooler.pool(X, mask).reshape(dc, dc)

    Lw = pooler.proj_l.weight.T   # [d, dc]
    Rw = pooler.proj_r.weight.T   # [d, dc]
    expected = (X[0] @ Lw).T @ (X[0] @ Rw) / L
    assert torch.allclose(out, expected, atol=1e-5)


def test_asymmetric_when_l_neq_r():
    """With independent L, R (random init), C is generally not symmetric."""
    torch.manual_seed(1)
    d, dc = 16, 4
    pooler = CovariancePooler(d, dc)
    X = torch.randn(1, 12, d)
    mask = torch.ones(1, 12, dtype=torch.bool)
    C = pooler.pool(X, mask).reshape(dc, dc)
    assert not torch.allclose(C, C.T, atol=1e-3), (
        "Random L, R should produce an asymmetric C; got symmetric."
    )


def test_symmetric_when_l_eq_r():
    """If we tie L = R, C must be symmetric (PCA special case)."""
    torch.manual_seed(2)
    d, dc = 16, 4
    pooler = CovariancePooler(d, dc)
    with torch.no_grad():
        pooler.proj_r.weight.copy_(pooler.proj_l.weight)
    X = torch.randn(1, 12, d)
    mask = torch.ones(1, 12, dtype=torch.bool)
    C = pooler.pool(X, mask).reshape(dc, dc)
    assert torch.allclose(C, C.T, atol=1e-5)


# ---------------------------------------------------------------------------
# Gradient flow (supervised regime)
# ---------------------------------------------------------------------------

def test_gradients_reach_projections():
    """Backprop from a scalar loss must produce non-zero grads on L and R."""
    torch.manual_seed(3)
    d, dc = 16, 4
    pooler = CovariancePooler(d, dc)
    X = torch.randn(2, 8, d)
    mask = torch.ones(2, 8, dtype=torch.bool)
    out = pooler.pool(X, mask)
    out.sum().backward()
    assert pooler.proj_l.weight.grad is not None
    assert pooler.proj_r.weight.grad is not None
    assert pooler.proj_l.weight.grad.abs().sum() > 0
    assert pooler.proj_r.weight.grad.abs().sum() > 0


def test_freeze_disables_grads():
    pooler = CovariancePooler(16, 4).freeze()
    assert not pooler.proj_l.weight.requires_grad
    assert not pooler.proj_r.weight.requires_grad


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_save_and_from_pretrained_roundtrip(tmp_path):
    torch.manual_seed(4)
    d, dc = 16, 4
    pooler = CovariancePooler(d, dc)
    path = tmp_path / "pooler.pt"
    pooler.save_state(path)

    loaded = CovariancePooler.from_pretrained(path)
    X = torch.randn(2, 10, d)
    mask = torch.ones(2, 10, dtype=torch.bool)
    assert torch.allclose(pooler.pool(X, mask), loaded.pool(X, mask))
    # from_pretrained must freeze.
    assert not loaded.proj_l.weight.requires_grad
