"""Tests for light-attention-weighted covariance pooling (docs §5.1)."""
import pytest
import torch

from sop.pooling.attention_covariance import AttentionCovariancePooler


def pad(X: torch.Tensor, extra: int) -> tuple[torch.Tensor, torch.Tensor]:
    L, d = X.shape
    X_pad = torch.cat([X, torch.zeros(extra, d)], dim=0)
    mask = torch.zeros(L + extra, dtype=torch.bool)
    mask[:L] = True
    return X_pad, mask


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

def test_output_shape_batched():
    pooler = AttentionCovariancePooler(32, 8).eval()
    X = torch.randn(5, 10, 32)
    mask = torch.ones(5, 10, dtype=torch.bool)
    assert pooler.pool(X, mask).shape == (5, 64)   # dc²


def test_output_shape_unbatched():
    pooler = AttentionCovariancePooler(16, 4).eval()
    X = torch.randn(10, 16)
    mask = torch.ones(10, dtype=torch.bool)
    assert pooler.pool(X, mask).shape == (16,)


def test_embedding_dim_property():
    assert AttentionCovariancePooler(16, 4).embedding_dim == 16


# ---------------------------------------------------------------------------
# Mathematical properties
# ---------------------------------------------------------------------------

def test_covariance_is_symmetric_psd():
    """Tied projection + non-negative attention weights → symmetric PSD C."""
    torch.manual_seed(0)
    d, dc, L = 16, 4, 12
    pooler = AttentionCovariancePooler(d, dc).eval()
    X = torch.randn(1, L, d)
    mask = torch.ones(1, L, dtype=torch.bool)
    C = pooler.pool(X, mask).reshape(dc, dc)
    assert torch.allclose(C, C.T, atol=1e-5)
    eigs = torch.linalg.eigvalsh(C)
    assert (eigs >= -1e-5).all(), f"C is not PSD; min eig = {eigs.min()}"


def test_matches_manual_attention_weighted_bilinear():
    """Output equals Σ_i α_i (U x_i)(U x_i)ᵀ with α the masked soft-max."""
    torch.manual_seed(1)
    d, dc, L = 12, 4, 8
    pooler = AttentionCovariancePooler(d, dc).eval()
    X = torch.randn(1, L, d)
    mask = torch.ones(1, L, dtype=torch.bool)

    out = pooler.pool(X, mask).reshape(dc, dc)

    scores = pooler.attention_conv(X.transpose(1, 2)).squeeze(1)   # [1, L]
    alpha = torch.softmax(scores, dim=-1).squeeze(0)               # [L]
    U = pooler.proj.weight.T                                       # [d, dc]
    XU = X[0] @ U                                                  # [L, dc]
    expected = (XU * alpha.unsqueeze(-1)).T @ XU
    assert torch.allclose(out, expected, atol=1e-5)


def test_power_norm_changes_output():
    torch.manual_seed(2)
    d, dc, L = 16, 4, 10
    X = torch.randn(1, L, d)
    mask = torch.ones(1, L, dtype=torch.bool)

    torch.manual_seed(3)
    plain = AttentionCovariancePooler(d, dc, power_norm=False).eval()
    torch.manual_seed(3)
    sqrt = AttentionCovariancePooler(d, dc, power_norm=True).eval()

    assert sqrt.power_norm and not plain.power_norm
    assert not torch.allclose(plain.pool(X, mask), sqrt.pool(X, mask), atol=1e-4)


# ---------------------------------------------------------------------------
# Masking invariance
# ---------------------------------------------------------------------------

class TestAttentionCovMaskInvariance:
    @pytest.mark.parametrize("power_norm", [False, True])
    @pytest.mark.parametrize("extra", [1, 5, 20])
    def test_padding_does_not_change_result(self, extra, power_norm):
        torch.manual_seed(4)
        d, dc, L = 16, 4, 12
        pooler = AttentionCovariancePooler(d, dc, power_norm=power_norm).eval()
        X = torch.randn(L, d)
        mask_full = torch.ones(L, dtype=torch.bool)

        ref = pooler.pool(X.unsqueeze(0), mask_full.unsqueeze(0))
        X_pad, mask_pad = pad(X, extra)
        out = pooler.pool(X_pad.unsqueeze(0), mask_pad.unsqueeze(0))

        assert torch.allclose(ref, out, atol=1e-5), (
            f"AttentionCovariancePooler changed with {extra} pads "
            f"(power_norm={power_norm})."
        )


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("power_norm", [False, True])
def test_gradients_reach_projection_and_attention(power_norm):
    torch.manual_seed(5)
    d, dc = 16, 4
    pooler = AttentionCovariancePooler(d, dc, power_norm=power_norm)
    X = torch.randn(2, 8, d)
    mask = torch.ones(2, 8, dtype=torch.bool)
    pooler.pool(X, mask).sum().backward()
    assert pooler.proj.weight.grad.abs().sum() > 0
    assert pooler.attention_conv.weight.grad.abs().sum() > 0
