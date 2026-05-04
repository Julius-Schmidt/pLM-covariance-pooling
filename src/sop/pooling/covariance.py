from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator

import torch

from .base import Pooler

# A factory that returns a fresh iterator of (X [L, d], mask [L] bool) pairs.
# Must be callable multiple times (i.e. NOT a bare generator expression).
EmbeddingFactory = Callable[[], Iterator[tuple[torch.Tensor, torch.Tensor]]]


class CovariancePooler(Pooler):
    """Compressed second-order (covariance) pooling.

    Given per-residue embeddings X ∈ R^{L×d}, projects to a bottleneck
    dimension dc via a dataset-wide PCA projection U ∈ R^{d×dc}, then forms:

        C = (1 / L_valid) (XU)ᵀ (XU)  ∈  R^{dc×dc}

    C is flattened to a dc²-dimensional protein embedding.

    U is the matrix of the top-dc eigenvectors of the dataset-wide residue
    covariance Σ = E[(x - μ)(x - μ)ᵀ], fitted once on training data and
    reused across all downstream tasks.

    Two-pass fitting streams data from disk so that the full residue matrix
    never needs to be materialised in memory.
    """

    def __init__(self, d: int, dc: int, center: bool = True) -> None:
        """
        Args:
            d:      Input embedding dimension (e.g. 1024 for ProtX).
            dc:     Bottleneck dimension.  Output is dc² floats.
            center: Subtract dataset-wide residue mean before projection.
        """
        self._d = d
        self._dc = dc
        self._center = center
        self._proj: torch.Tensor | None = None   # U: [d, dc]
        self._mean: torch.Tensor | None = None   # μ: [d]

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, get_iter: EmbeddingFactory) -> "CovariancePooler":
        """Fit the PCA projection from training embeddings.

        Args:
            get_iter: Callable with no arguments that returns a fresh iterator
                      of (X [L, d], mask [L] bool) pairs.  Called twice.

        Returns:
            self, for chaining.
        """
        # Pass 1 — accumulate mean (float64 for numerical stability)
        n = 0
        sum_x = torch.zeros(self._d, dtype=torch.float64)
        for X, mask in get_iter():
            X_valid = X[mask.bool()].double()
            sum_x += X_valid.sum(0)
            n += X_valid.shape[0]

        if n == 0:
            raise ValueError("No valid residues found in the provided embeddings.")

        mean = (sum_x / n).float() if self._center else torch.zeros(self._d)

        # Pass 2 — accumulate (centred) scatter matrix
        cov = torch.zeros(self._d, self._d, dtype=torch.float64)
        for X, mask in get_iter():
            Xc = (X[mask.bool()] - mean).double()
            cov += Xc.T @ Xc
        cov /= max(n - 1, 1)   # unbiased sample covariance

        # PCA: eigh returns eigenvectors sorted by ascending eigenvalue
        _, eigvecs = torch.linalg.eigh(cov.float())
        # Take the dc eigenvectors with largest eigenvalues
        self._proj = eigvecs[:, -self._dc:].contiguous()   # [d, dc]
        self._mean = mean

        return self

    # ------------------------------------------------------------------
    # Pooling
    # ------------------------------------------------------------------

    def pool(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute compressed covariance embeddings.

        Args:
            X:    [B, L, d] or [L, d] (single protein, no batch dim).
            mask: [B, L] or [L] bool, True for valid positions.

        Returns:
            [B, dc²] or [dc²] (matches input batch rank).
        """
        if self._proj is None:
            raise RuntimeError("Call fit() (or load()) before pool().")

        single = X.dim() == 2
        if single:
            X = X.unsqueeze(0)
            mask = mask.unsqueeze(0)

        U = self._proj.to(X.device)       # [d, dc]

        # Centre using dataset-wide mean, then re-zero padded positions.
        # Order matters: masking must come AFTER centering so that the
        # previously-zero padding rows (which became -μ after centering)
        # are zeroed out again before the outer product.
        if self._center and self._mean is not None:
            X = X - self._mean.to(X.device)

        mask_f = mask.to(X.dtype)
        lengths = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)   # [B, 1]
        X = X * mask_f.unsqueeze(-1)                                # zero padding

        Xp = X @ U                                                  # [B, L, dc]
        C = torch.bmm(Xp.transpose(1, 2), Xp)                      # [B, dc, dc]
        C = C / lengths.unsqueeze(-1)                               # [B, dc, dc]

        out = C.flatten(1)                                          # [B, dc²]
        return out.squeeze(0) if single else out

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def embedding_dim(self) -> int:
        return self._dc ** 2

    @property
    def is_fitted(self) -> bool:
        return self._proj is not None

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Save fitted projection and mean to a .pt file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "d": self._d,
                "dc": self._dc,
                "center": self._center,
                "proj": self._proj,
                "mean": self._mean,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "CovariancePooler":
        """Restore a fitted CovariancePooler from disk."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        obj = cls(state["d"], state["dc"], state["center"])
        obj._proj = state["proj"]
        obj._mean = state["mean"]
        return obj
