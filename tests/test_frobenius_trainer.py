"""Tests for the unsupervised covariance autoencoder.

The key correctness check is that ``frobenius_recon_loss`` computed via the
trick (XXᵀ + small dc×dc terms) equals the direct ‖XᵀX − L C̃ Rᵀ‖²_F
computation that explicitly materialises the d×d matrix.
"""
import torch

from sop.pooling.covariance import CovariancePooler
from sop.unsupervised.frobenius_trainer import (
    frobenius_recon_loss,
    train_unsupervised_pooler,
)


def _direct_recon_loss(X: torch.Tensor, mask: torch.Tensor, pooler: CovariancePooler) -> torch.Tensor:
    """Reference implementation that materialises XᵀX and L C̃ Rᵀ explicitly."""
    Xv = X[mask.bool()]
    L_valid = Xv.shape[0]
    Lw = pooler.proj_l.weight.T
    Rw = pooler.proj_r.weight.T

    S = Xv.T @ Xv                                  # [d, d]
    C_tilde = (Xv @ Lw).T @ (Xv @ Rw)              # [dc, dc]
    S_hat = Lw @ C_tilde @ Rw.T                    # [d, d]

    return ((S - S_hat) ** 2).sum() / (L_valid ** 2)


def test_frobenius_loss_matches_direct_computation():
    """The Frobenius-trick loss must equal the direct ‖XᵀX − L C̃ Rᵀ‖²_F / L²."""
    torch.manual_seed(0)
    d, dc, L = 32, 6, 25
    pooler = CovariancePooler(d, dc)
    X = torch.randn(L, d)
    mask = torch.ones(L, dtype=torch.bool)

    fast = frobenius_recon_loss(X, mask, pooler)
    slow = _direct_recon_loss(X, mask, pooler)

    assert torch.allclose(fast, slow, atol=1e-4, rtol=1e-4), (
        f"Frobenius trick disagrees with direct computation: {fast.item()} vs {slow.item()}"
    )


def test_frobenius_loss_ignores_padded_positions():
    """Adding zero-padded rows must not change the loss (with mask updated)."""
    torch.manual_seed(1)
    d, dc, L = 16, 4, 12
    pooler = CovariancePooler(d, dc)
    X = torch.randn(L, d)
    mask = torch.ones(L, dtype=torch.bool)

    ref = frobenius_recon_loss(X, mask, pooler)

    X_pad = torch.cat([X, torch.zeros(7, d)], dim=0)
    mask_pad = torch.cat([mask, torch.zeros(7, dtype=torch.bool)])

    out = frobenius_recon_loss(X_pad, mask_pad, pooler)
    assert torch.allclose(ref, out, atol=1e-5)


def test_loss_is_non_negative():
    torch.manual_seed(2)
    d, dc, L = 16, 4, 12
    pooler = CovariancePooler(d, dc)
    X = torch.randn(L, d)
    mask = torch.ones(L, dtype=torch.bool)
    loss = frobenius_recon_loss(X, mask, pooler)
    assert loss.item() >= -1e-6


def test_training_decreases_loss():
    """Smoke test: a few epochs of SGD must reduce mean reconstruction loss."""
    torch.manual_seed(3)
    d, dc = 16, 4
    proteins = [(torch.randn(20, d), torch.ones(20, dtype=torch.bool)) for _ in range(40)]

    def get_iter():
        return iter(proteins)

    pooler = CovariancePooler(d, dc)

    def avg_loss():
        with torch.no_grad():
            return sum(
                float(frobenius_recon_loss(X, mask, pooler)) for X, mask in proteins
            ) / len(proteins)

    before = avg_loss()
    train_unsupervised_pooler(
        pooler, get_iter, epochs=10, batch_size=8, lr=1e-2, log=lambda _: None,
    )
    after = avg_loss()

    assert after < before, f"Training did not reduce loss: {before:.4e} → {after:.4e}"
