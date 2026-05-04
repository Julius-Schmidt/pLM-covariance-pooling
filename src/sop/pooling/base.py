from abc import abstractmethod

import torch
import torch.nn as nn


class Pooler(nn.Module):
    """Base class for pooling strategies.

    All poolers are nn.Modules so that supervised covariance pooling can flow
    gradients into the projection matrices alongside the probe head, while
    parameter-free or frozen poolers use the same interface.
    """

    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.pool(X, mask)

    @abstractmethod
    def pool(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Reduce per-residue embeddings to a single protein-level vector.

        Args:
            X:    [B, L, d] per-residue embeddings (may be padded).
            mask: [B, L] bool, True for valid (non-padded) positions.

        Returns:
            [B, embedding_dim] protein-level representations.
        """
        ...

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Output dimensionality of pool()."""
        ...
