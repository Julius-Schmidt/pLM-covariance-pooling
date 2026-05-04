from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .base import Pooler


class CovariancePooler(Pooler):
    """Compressed second-order (covariance) pooling with two learnable projections.

    Given per-residue embeddings X ∈ R^{L×d}, applies two distinct linear
    projections L, R ∈ R^{d×dc} and forms the bilinear "covariance":

        C = (1 / L_valid) (X L)ᵀ (X R)   ∈   R^{dc × dc}

    C is flattened to a dc²-dimensional protein embedding.

    Two training regimes — same module, different optimisation paths:

    * **Supervised**: instantiate fresh, plug into a model that ends in a probe
      head, train end-to-end. Gradients flow into L, R.
    * **Unsupervised**: train via Frobenius reconstruction of XᵀX
      (see ``sop.unsupervised.frobenius_trainer``), call
      ``freeze()`` and load into downstream tasks.

    Note: ``L`` and ``R`` are independent (not tied), so C is asymmetric in
    general. This is what distinguishes the autoencoder formulation from the
    PCA special case (where L = R and C is symmetric / PSD).
    """

    def __init__(self, d: int, dc: int) -> None:
        """
        Args:
            d:  Input embedding dimension (e.g. 1024 for ProtX).
            dc: Bottleneck dimension. Output is dc² floats.
        """
        super().__init__()
        self._d = d
        self._dc = dc
        self.proj_l = nn.Linear(d, dc, bias=False)
        self.proj_r = nn.Linear(d, dc, bias=False)

    def pool(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute compressed covariance embeddings.

        Args:
            X:    [B, L, d] or [L, d] (single protein, no batch dim).
            mask: [B, L] or [L] bool, True for valid positions.

        Returns:
            [B, dc²] or [dc²] (matches input batch rank).
        """
        single = X.dim() == 2
        if single:
            X = X.unsqueeze(0)
            mask = mask.unsqueeze(0)

        mask_f = mask.to(X.dtype).unsqueeze(-1)                  # [B, L, 1]
        lengths = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0) # [B, 1, 1]

        # Zero padded rows so they cannot contribute to the bilinear sum,
        # even if the caller passed non-zero values at masked positions.
        Xm = X * mask_f
        XL = self.proj_l(Xm)                                      # [B, L, dc]
        XR = self.proj_r(Xm)                                      # [B, L, dc]

        C = torch.bmm(XL.transpose(1, 2), XR) / lengths.squeeze(-1).unsqueeze(-1)
        out = C.flatten(1)                                        # [B, dc²]
        return out.squeeze(0) if single else out

    def freeze(self) -> "CovariancePooler":
        """Disable gradients on L, R for use as a frozen unsupervised pooler."""
        for p in self.parameters():
            p.requires_grad_(False)
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

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self, path: Path | str) -> None:
        """Save the pooler's state_dict + shape metadata to a .pt file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"d": self._d, "dc": self._dc, "state_dict": self.state_dict()},
            path,
        )

    @classmethod
    def from_pretrained(cls, path: Path | str) -> "CovariancePooler":
        """Restore a CovariancePooler with frozen projections from disk."""
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        obj = cls(ckpt["d"], ckpt["dc"])
        obj.load_state_dict(ckpt["state_dict"])
        return obj.freeze()
