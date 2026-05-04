"""Train the CovariancePooler projections L, R unsupervised.

The reconstruction objective is

    ‖XᵀX − L (XL)ᵀ(XR) Rᵀ‖²_F

per protein. Materialising the d×d matrix XᵀX is expensive (d ≈ 1024 →
1M entries). Expanding the squared Frobenius norm and using the identities

    ‖XᵀX‖²_F = ‖XXᵀ‖²_F                                  (Frobenius equivalence)
    ⟨XᵀX, L (XL)ᵀ(XR) Rᵀ⟩_F = ‖(XL)ᵀ(XR)‖²_F            (cyclic trace)

reduces every per-protein quantity to either an L×L Gram matrix or to small
dc×dc matrices. d×d matrices never get materialised.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import torch
import torch.nn as nn
from torch.optim import Adam

from ..pooling.covariance import CovariancePooler

# A factory returning a fresh iterator of (X [L, d], mask [L] bool) pairs.
EmbeddingFactory = Callable[[], Iterable[tuple[torch.Tensor, torch.Tensor]]]


def frobenius_recon_loss(
    X: torch.Tensor,
    mask: torch.Tensor,
    pooler: CovariancePooler,
) -> torch.Tensor:
    """Per-protein squared Frobenius reconstruction loss, normalised by L².

    Computed via the trick described in the module docstring — never builds
    a d×d matrix.

    Args:
        X:      [L, d] per-residue embeddings (single protein).
        mask:   [L] bool, True for valid (non-padded) positions.
        pooler: CovariancePooler whose proj_l, proj_r are being trained.

    Returns:
        scalar loss.
    """
    # Drop padded rows so they cannot influence any term.
    Xv = X[mask.bool()]                                       # [L_valid, d]
    L_valid = Xv.shape[0]
    if L_valid == 0:
        return X.sum() * 0.0  # zero with grad to keep batches uniform

    L_proj = pooler.proj_l.weight.T                            # [d, dc]
    R_proj = pooler.proj_r.weight.T                            # [d, dc]

    # ‖XᵀX‖²_F via the L×L Gram matrix XXᵀ (Frobenius equivalence).
    XXt = Xv @ Xv.T                                            # [L, L]
    sq_target = (XXt ** 2).sum()

    # ‖C̃‖²_F where C̃ = (XL)ᵀ(XR), used twice below.
    XL = Xv @ L_proj                                           # [L, dc]
    XR = Xv @ R_proj                                           # [L, dc]
    C_tilde = XL.T @ XR                                        # [dc, dc]
    sq_cross = (C_tilde ** 2).sum()                            # = ⟨XᵀX, L C̃ Rᵀ⟩_F

    # ‖L C̃ Rᵀ‖²_F = trace(LᵀL · C̃ · RᵀR · C̃ᵀ) — all small.
    LtL = L_proj.T @ L_proj                                    # [dc, dc]
    RtR = R_proj.T @ R_proj                                    # [dc, dc]
    sq_recon = torch.einsum("ab,bc,cd,da->", LtL, C_tilde, RtR, C_tilde.T)

    loss = sq_target - 2.0 * sq_cross + sq_recon
    # Normalise so long proteins don't dominate (XᵀX scales with L²).
    return loss / (L_valid ** 2)


def train_unsupervised_pooler(
    pooler: CovariancePooler,
    get_iter: EmbeddingFactory,
    *,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
    log_every: int = 50,
    log: Callable[[str], None] = print,
) -> CovariancePooler:
    """Train ``pooler`` in-place by Frobenius reconstruction of XᵀX.

    Args:
        pooler:     A CovariancePooler with random or warm-started weights.
        get_iter:   Factory returning a fresh iterator of (X, mask) pairs.
                    Called once per epoch.
        epochs:     Number of full passes over the data.
        batch_size: Number of proteins to accumulate gradients over before
                    each optimiser step (no padding, just summed losses).
        lr:         Adam learning rate.
        device:     'cpu' or 'cuda'.
        log_every:  Emit a log line every N optimiser steps.
        log:        Sink for log lines (defaults to print).

    Returns:
        The same pooler, with trained weights and gradients still enabled.
        Call ``pooler.freeze()`` afterwards if you intend to plug it into a
        downstream task as a frozen module.
    """
    pooler.to(device).train()
    opt = Adam(pooler.parameters(), lr=lr)

    step = 0
    for epoch in range(epochs):
        running = 0.0
        seen = 0
        opt.zero_grad()
        accum = 0

        for X, mask in get_iter():
            X = X.to(device)
            mask = mask.to(device)
            loss = frobenius_recon_loss(X, mask, pooler)
            loss.backward()
            running += float(loss.detach())
            seen += 1
            accum += 1

            if accum >= batch_size:
                opt.step()
                opt.zero_grad()
                accum = 0
                step += 1
                if step % log_every == 0:
                    log(f"  epoch {epoch} step {step:5d}  avg_loss={running / seen:.4e}")

        # Flush any leftover gradient (incomplete final batch).
        if accum > 0:
            opt.step()
            opt.zero_grad()

        log(f"epoch {epoch} done — avg_loss={running / max(seen, 1):.4e}  proteins={seen}")

    return pooler


def save_trained_pooler(pooler: CovariancePooler, path: Path | str) -> None:
    """Convenience wrapper around ``pooler.save_state``."""
    pooler.save_state(path)
