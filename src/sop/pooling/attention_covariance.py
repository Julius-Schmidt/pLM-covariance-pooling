"""Light-attention-weighted covariance pooling (docs §5.1).

Combines Stärk-style light attention with second-order pooling: a per-residue
scalar attention weight ``α_i`` (Σ_i α_i = 1) replaces the uniform ``1/L`` in
the covariance, so the pool focuses on the residues that matter:

    C = Σ_i α_i (U x_i)(U x_i)ᵀ   ∈   R^{dc × dc}.

A single shared projection ``U`` (tied L = R) is used, so ``C`` is symmetric
PSD — which is exactly what matrix-power normalisation wants. Setting
``power_norm=True`` applies ``C -> C^{1/2}`` (iSQRT-COV) on top, giving the full
LA + covariance + matrix-power stack.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import Pooler
from .matrix_power import isqrt_cov


class AttentionCovariancePooler(Pooler):
    """Attention-weighted second-order pool with a tied projection.

    Trainable end-to-end (projection + attention conv) like the supervised
    covariance pooler; output dim is ``dc²``.
    """

    def __init__(
        self,
        d: int,
        dc: int,
        power_norm: bool = False,
        kernel_size: int = 9,
        conv_dropout: float = 0.0,
    ) -> None:
        """
        Args:
            d: Input embedding dimension.
            dc: Bottleneck dimension; output is ``dc²`` floats.
            power_norm: If True, apply matrix square-root normalisation to C.
            kernel_size: Conv width of the attention scorer (Stärk uses 9).
            conv_dropout: Optional dropout on the attention logits.
        """
        super().__init__()
        self._d = d
        self._dc = dc
        self._power_norm = power_norm
        self.proj = nn.Linear(d, dc, bias=False)
        padding = kernel_size // 2
        # One scalar attention logit per residue (vs LA's per-channel scores):
        # the covariance needs a single distribution over residues to weight the
        # outer products.
        self.attention_conv = nn.Conv1d(d, 1, kernel_size, padding=padding)
        self.dropout = nn.Dropout(conv_dropout) if conv_dropout > 0 else nn.Identity()

    def pool(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        single = X.dim() == 2
        if single:
            X = X.unsqueeze(0)
            mask = mask.unsqueeze(0)

        mask_b = mask.bool()
        # Zero padded rows: keeps both the attention conv and the bilinear sum
        # invariant to the amount of padding.
        Xm = X * mask_b.unsqueeze(-1).to(X.dtype)

        scores = self.attention_conv(Xm.transpose(1, 2)).squeeze(1)   # [B, L]
        scores = self.dropout(scores)
        # Soft-max over residues (float32 for fp16-autocast safety); pads -> 0.
        alpha = torch.softmax(scores.float().masked_fill(~mask_b, float("-inf")), dim=-1)
        alpha = alpha.to(X.dtype)                                     # [B, L], Σ=1

        XU = self.proj(Xm)                                           # [B, L, dc]
        XUa = XU * alpha.unsqueeze(-1)                               # weight rows
        C = torch.bmm(XUa.transpose(1, 2), XU)                      # [B, dc, dc]

        if self._power_norm:
            C = isqrt_cov(C)

        out = C.flatten(1)                                          # [B, dc²]
        return out.squeeze(0) if single else out

    @property
    def embedding_dim(self) -> int:
        return self._dc ** 2

    @property
    def d(self) -> int:
        return self._d

    @property
    def dc(self) -> int:
        return self._dc

    @property
    def power_norm(self) -> bool:
        return self._power_norm
