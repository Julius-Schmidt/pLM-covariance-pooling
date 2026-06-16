"""Light Attention pooling (Stärk et al. 2021).

Reimplementation of the pooling block from *"Light attention predicts protein
location from the language of life"* (Stärk, Dallago, Heinzinger, Rost; 2021),
adapted to the project's ``Pooler`` interface so it feeds the same shared
``ProbeFNN`` head as every other method.

Two parallel 1-D convolutions over the residue axis produce per-channel
**values** and per-channel **attention scores**. The scores are soft-maxed over
the sequence length (per channel, with padding masked to ``-inf``) and used to
take an attention-weighted sum of the values; a max-pool over the values is
concatenated alongside. Output dimension is ``2 * d``.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import Pooler


class LightAttentionPooler(Pooler):
    """Light-attention pooling: ``[ Σ_i softmax(e)_i · v_i ; max_i v_i ]``.

    ``v`` (feature convolution) and ``e`` (attention convolution) are both
    ``[B, d, L]``; the soft-max runs over ``L`` independently per channel, so
    every feature gets its own attention distribution over residues. The
    concatenation of the attention-weighted sum and the channel-wise max gives a
    ``2·d`` protein vector.

    Note: this is a pure first-order (attention-weighted) pool — its output is a
    vector, not an SPD matrix, so matrix-power normalisation does not apply to
    it (use :class:`AttentionCovariancePooler` for LA + covariance).
    """

    def __init__(
        self,
        d: int,
        d_out: int | None = None,
        kernel_size: int = 9,
        conv_dropout: float = 0.25,
    ) -> None:
        """
        Args:
            d: Input embedding dimension.
            d_out: Number of output channels per conv branch. Output dim is
                ``2 * d_out``. Defaults to ``d`` (original Stärk behaviour).
                Set to a smaller value (e.g. via ``--dc`` sweep) to trade
                capacity for a compact embedding comparable to other methods.
            kernel_size: Conv width over the residue axis (Stärk uses 9).
            conv_dropout: Dropout on the value-convolution output.
        """
        super().__init__()
        self._d = d
        self._d_out = d_out if d_out is not None else d
        padding = kernel_size // 2
        self.feature_conv = nn.Conv1d(d, self._d_out, kernel_size, padding=padding)
        self.attention_conv = nn.Conv1d(d, self._d_out, kernel_size, padding=padding)
        self.dropout = nn.Dropout(conv_dropout)

    def pool(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        single = X.dim() == 2
        if single:
            X = X.unsqueeze(0)
            mask = mask.unsqueeze(0)

        mask_b = mask.bool()
        # Zero padded rows so the convolutions never mix in pad content at the
        # sequence boundary — keeps the pool invariant to the amount of padding.
        Xm = X * mask_b.unsqueeze(-1).to(X.dtype)
        x = Xm.transpose(1, 2)                                   # [B, d, L]

        v = self.dropout(self.feature_conv(x))                  # [B, d, L]
        scores = self.attention_conv(x)                         # [B, d, L]

        neg = ~mask_b.unsqueeze(1)                              # [B, 1, L]
        # Soft-max over L per channel; do it in float32 for fp16-autocast safety.
        alpha = torch.softmax(scores.float().masked_fill(neg, float("-inf")), dim=-1)
        attended = (v * alpha.to(v.dtype)).sum(dim=-1)         # [B, d]

        maxed = v.masked_fill(neg, float("-inf")).max(dim=-1).values  # [B, d]

        out = torch.cat([attended, maxed], dim=-1)             # [B, 2d]
        return out.squeeze(0) if single else out

    @property
    def embedding_dim(self) -> int:
        return 2 * self._d_out

    @property
    def d(self) -> int:
        return self._d
