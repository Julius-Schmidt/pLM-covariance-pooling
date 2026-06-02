"""Tests for the differentiable matrix square root (iSQRT-COV)."""
import torch

from sop.pooling.matrix_power import isqrt_cov


def _random_spd(d: int, batch: int = 1, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    A = torch.randn(batch, d, d)
    # M M^T is symmetric PSD; + eps I keeps it well away from singular.
    return A @ A.transpose(-2, -1) + 1e-2 * torch.eye(d)


def test_sqrt_squares_back_to_input():
    """C^{1/2} @ C^{1/2} should reconstruct the (symmetric PSD) input C."""
    C = _random_spd(8, batch=4, seed=1)
    S = isqrt_cov(C, n_iter=15)
    recon = S @ S
    assert torch.allclose(recon, C, atol=1e-3, rtol=1e-3)


def test_output_is_symmetric():
    C = _random_spd(6, batch=2, seed=2)
    S = isqrt_cov(C)
    assert torch.allclose(S, S.transpose(-2, -1), atol=1e-5)


def test_sqrt_compresses_eigenvalue_spectrum():
    """Eigenvalues of C^{1/2} are the sqrt of those of C → smaller condition #."""
    C = _random_spd(8, batch=1, seed=3)
    S = isqrt_cov(C, n_iter=20)
    cond_C = torch.linalg.eigvalsh(C)[0]
    eig_C = torch.linalg.eigvalsh(C[0])
    eig_S = torch.linalg.eigvalsh(S[0])
    cond_before = eig_C.max() / eig_C.min()
    cond_after = eig_S.max() / eig_S.min()
    # sqrt halves the condition number on a log scale.
    assert cond_after < cond_before
    assert torch.allclose(eig_S.clamp_min(0), eig_C.clamp_min(0).sqrt(), atol=1e-2, rtol=1e-2)


def test_identity_maps_to_identity():
    I = torch.eye(5).unsqueeze(0)
    assert torch.allclose(isqrt_cov(I, n_iter=10), I, atol=1e-4)


def test_differentiable():
    """Gradient must flow back through the Newton–Schulz iteration."""
    C = _random_spd(6, batch=1, seed=4).requires_grad_(True)
    out = isqrt_cov(C).pow(2).sum()
    out.backward()
    assert C.grad is not None
    assert torch.isfinite(C.grad).all()
    assert C.grad.abs().sum() > 0


def test_handles_asymmetric_via_symmetric_part():
    """Asymmetric input is projected onto its symmetric part (no NaNs)."""
    torch.manual_seed(5)
    C = _random_spd(6, batch=1, seed=5) + 0.05 * torch.randn(1, 6, 6)
    S = isqrt_cov(C)
    assert torch.isfinite(S).all()
    assert torch.allclose(S, S.transpose(-2, -1), atol=1e-5)
