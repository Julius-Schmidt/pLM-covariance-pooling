"""Tests for Light Attention pooling (Stärk et al. 2021)."""
import pytest
import torch

from sop.pooling.light_attention import LightAttentionPooler


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
    pooler = LightAttentionPooler(32).eval()
    X = torch.randn(5, 10, 32)
    mask = torch.ones(5, 10, dtype=torch.bool)
    assert pooler.pool(X, mask).shape == (5, 64)   # 2·d


def test_output_shape_unbatched():
    pooler = LightAttentionPooler(16).eval()
    X = torch.randn(10, 16)
    mask = torch.ones(10, dtype=torch.bool)
    assert pooler.pool(X, mask).shape == (32,)


def test_embedding_dim_property():
    assert LightAttentionPooler(64).embedding_dim == 128


def test_forward_alias():
    pooler = LightAttentionPooler(16).eval()
    X = torch.randn(2, 5, 16)
    mask = torch.ones(2, 5, dtype=torch.bool)
    assert torch.equal(pooler(X, mask), pooler.pool(X, mask))


# ---------------------------------------------------------------------------
# Masking invariance — eval() so dropout is disabled
# ---------------------------------------------------------------------------

class TestLightAttentionMaskInvariance:
    @pytest.mark.parametrize("extra", [1, 5, 20])
    def test_padding_does_not_change_result(self, extra):
        torch.manual_seed(0)
        d, L = 16, 12
        pooler = LightAttentionPooler(d).eval()
        X = torch.randn(L, d)
        mask_full = torch.ones(L, dtype=torch.bool)

        ref = pooler.pool(X.unsqueeze(0), mask_full.unsqueeze(0))

        X_pad, mask_pad = pad(X, extra)
        out = pooler.pool(X_pad.unsqueeze(0), mask_pad.unsqueeze(0))

        assert torch.allclose(ref, out, atol=1e-5), (
            f"LightAttentionPooler output changed with {extra} padding rows."
        )


def test_attention_weights_sum_to_one_over_valid_positions():
    """The softmax distribution per channel must concentrate on valid residues:
    a sequence with a non-zero region followed by padding ignores the padding."""
    torch.manual_seed(1)
    d, L = 8, 6
    pooler = LightAttentionPooler(d).eval()
    X = torch.randn(L, d)
    mask = torch.ones(L, dtype=torch.bool)
    # Padding-extended version must match the unpadded pool exactly.
    X_pad, mask_pad = pad(X, 7)
    assert torch.allclose(
        pooler.pool(X.unsqueeze(0), mask.unsqueeze(0)),
        pooler.pool(X_pad.unsqueeze(0), mask_pad.unsqueeze(0)),
        atol=1e-5,
    )


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

def test_gradients_reach_both_convolutions():
    torch.manual_seed(2)
    d = 16
    pooler = LightAttentionPooler(d)
    X = torch.randn(2, 8, d)
    mask = torch.ones(2, 8, dtype=torch.bool)
    pooler.pool(X, mask).sum().backward()
    assert pooler.feature_conv.weight.grad.abs().sum() > 0
    assert pooler.attention_conv.weight.grad.abs().sum() > 0
