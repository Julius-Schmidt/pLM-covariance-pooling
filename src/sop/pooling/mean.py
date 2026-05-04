import torch

from .base import Pooler


class MeanPooler(Pooler):
    """Masked mean over residue positions — zero-parameter baseline.

    μ = (1 / L_valid) Σ_{valid i} x_i
    """

    def __init__(self, d: int) -> None:
        self._d = d

    def pool(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.to(X.dtype)
        lengths = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
        return (X * mask_f.unsqueeze(-1)).sum(dim=1) / lengths

    @property
    def embedding_dim(self) -> int:
        return self._d
