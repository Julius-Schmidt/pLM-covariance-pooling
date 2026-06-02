"""Closed-form (PCA) covariance pooling.

A deterministic, training-free alternative to the learnable autoencoder in
``covariance.py``. The projection ``U ∈ R^{d × dc}`` is fit once via
streaming top-dc PCA over the residue matrix, then reused unchanged across
tasks.

This is the **symmetric, tied-weights** special case of the autoencoder
(``L = R = U``), so the resulting C is symmetric and PSD. It is included as
a baseline to answer: does the autoencoder's extra freedom (asymmetric L, R
trained by SGD) buy anything over the closed-form symmetric solution?
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator

import torch

from .base import Pooler
from .matrix_power import isqrt_cov

# A factory returning a fresh iterator of (X [L, d], mask [L] bool) pairs.
# Must be callable multiple times — the streaming fit makes two passes.
EmbeddingFactory = Callable[[], Iterator[tuple[torch.Tensor, torch.Tensor]]]


class CovariancePCAPooler(Pooler):
    """Compressed covariance pooling via dataset-wide top-dc PCA.

        C = (1 / L_valid) (X U)ᵀ (X U)   ∈   R^{dc × dc}

    where ``U`` holds the top-dc eigenvectors of the protein-weighted dataset
    covariance:

        Σ = (1 / N_proteins) Σ_t  (1 / Lₜ) Xₜ_centeredᵀ Xₜ_centered

    Weighting by 1/Lₜ makes each protein contribute equally regardless of
    sequence length, so long proteins do not dominate the principal directions.

    Streaming two-pass fit so the d × N residue matrix never has to live in
    memory all at once. All accumulation runs in float64 — pLM embeddings
    cluster tightly around a non-zero mean and a float32 (``X − μ``) loses
    most of its significant digits to cancellation.
    """

    def __init__(self, d: int, dc: int, center: bool = True,
                 power_norm: bool = False) -> None:
        super().__init__()
        if dc <= 0:
            raise ValueError(f"dc must be positive, got {dc}")
        if dc > d:
            raise ValueError(f"dc ({dc}) cannot exceed d ({d})")

        self._d = d
        self._dc = dc
        self._center = center
        # C is symmetric PSD here (tied projection), so the matrix square root
        # is exact in the eigenbasis — the ideal case for power normalisation.
        self._power_norm = power_norm
        # Buffers (not Parameters) — these are not learned by SGD and should
        # move with ``.to(device)`` and serialise via ``state_dict``.
        self.register_buffer("proj", torch.zeros(d, dc))
        self.register_buffer("mean", torch.zeros(d))
        self._fitted = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, get_iter: EmbeddingFactory) -> "CovariancePCAPooler":
        """Streaming two-pass fit of the top-dc PCA projection.

        Args:
            get_iter: Callable that returns a fresh iterator of (X, mask)
                      pairs each time it is invoked. Called twice.
        """
        # ---- Pass 1 — residue-weighted mean (float64 throughout) ----
        n_residues = 0
        n_proteins = 0
        sum_x = torch.zeros(self._d, dtype=torch.float64)
        for X, mask in get_iter():
            X_valid = X[mask.bool()].double()        # cast BEFORE any arithmetic
            L = X_valid.shape[0]
            if L == 0:
                continue
            sum_x += X_valid.sum(0)
            n_residues += L
            n_proteins += 1
        if n_residues == 0:
            raise ValueError("No valid residues found in the provided embeddings.")

        # Keep mean64 in float64 so pass 2's subtraction does not lose precision.
        mean64 = (sum_x / n_residues) if self._center \
            else torch.zeros(self._d, dtype=torch.float64)

        # ---- Pass 2 — protein-weighted scatter matrix (float64 throughout) ----
        cov = torch.zeros(self._d, self._d, dtype=torch.float64)
        for X, mask in get_iter():
            X_valid = X[mask.bool()].double()
            L = X_valid.shape[0]
            if L == 0:
                continue
            Xc = X_valid - mean64                   # subtraction in float64
            cov += (Xc.T @ Xc) / L                  # per-protein contribution
        cov /= max(n_proteins - 1, 1)               # unbiased over proteins

        # Keep float64 through the eigendecomposition; cast result down once.
        _, eigvecs = torch.linalg.eigh(cov)
        self.proj.copy_(eigvecs[:, -self._dc:].float().contiguous())
        self.mean.copy_(mean64.float())
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Pooling
    # ------------------------------------------------------------------

    def pool(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if not self._fitted:
            raise RuntimeError(
                "CovariancePCAPooler has no projection. "
                "Call fit() or load via from_pretrained() before pool()."
            )

        single = X.dim() == 2
        if single:
            X = X.unsqueeze(0)
            mask = mask.unsqueeze(0)

        # Centre using dataset-wide mean, THEN re-zero padded positions.
        # Order matters: masking must come AFTER centring so that originally
        # zero padding rows (which become -μ after centring) are zeroed
        # before the bilinear sum.
        if self._center:
            X = X - self.mean.to(X.device)

        mask_f = mask.to(X.dtype).unsqueeze(-1)
        lengths = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
        X = X * mask_f

        Xp = X @ self.proj.to(X.device)                          # [B, L, dc]
        C = torch.bmm(Xp.transpose(1, 2), Xp)                    # [B, dc, dc]
        C = C / lengths.squeeze(-1).unsqueeze(-1).to(Xp.dtype)

        if self._power_norm:
            C = isqrt_cov(C)                                      # [B, dc, dc]

        out = C.flatten(1)                                       # [B, dc²]
        return out.squeeze(0) if single else out

    def freeze(self) -> "CovariancePCAPooler":
        """No-op — proj/mean are buffers and never carry gradients."""
        return self

    def set_power_norm(self, power_norm: bool) -> "CovariancePCAPooler":
        """Toggle matrix-power normalisation (e.g. after ``from_pretrained``)."""
        self._power_norm = power_norm
        return self

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
    def is_fitted(self) -> bool:
        return self._fitted

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "d": self._d,
                "dc": self._dc,
                "center": self._center,
                "power_norm": self._power_norm,
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def from_pretrained(cls, path: Path | str) -> "CovariancePCAPooler":
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        obj = cls(ckpt["d"], ckpt["dc"], center=ckpt["center"],
                  power_norm=ckpt.get("power_norm", False))
        obj.load_state_dict(ckpt["state_dict"])
        obj._fitted = True
        return obj
