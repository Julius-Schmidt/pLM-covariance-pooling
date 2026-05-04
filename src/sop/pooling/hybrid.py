import torch

from .base import Pooler
from .covariance import CovariancePooler
from .mean import MeanPooler


class HybridPooler(Pooler):
    """Concatenate mean pooling and covariance pooling: [μ ; flat(C)].

    Output dim is ``d + dc²``. Useful for testing whether second-order info
    *adds* to first-order info, rather than replacing it.

    The covariance branch shares the same CovariancePooler interface, so it can
    be either supervised (random init, gradients enabled) or unsupervised
    (loaded from a frozen checkpoint).
    """

    def __init__(self, d: int, cov: CovariancePooler) -> None:
        super().__init__()
        if cov.d != d:
            raise ValueError(
                f"CovariancePooler input dim ({cov.d}) does not match d ({d})."
            )
        self._d = d
        self.mean = MeanPooler(d)
        self.cov = cov

    def pool(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mu = self.mean.pool(X, mask)        # [B, d] or [d]
        c = self.cov.pool(X, mask)          # [B, dc²] or [dc²]
        return torch.cat([mu, c], dim=-1)

    @property
    def embedding_dim(self) -> int:
        return self._d + self.cov.embedding_dim
