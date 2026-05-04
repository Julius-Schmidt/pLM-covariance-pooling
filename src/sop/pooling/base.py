from abc import ABC, abstractmethod

import torch


class Pooler(ABC):
    """Interface shared by all pooling strategies."""

    @abstractmethod
    def pool(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Reduce per-residue embeddings to a single protein-level vector.

        Args:
            X:    [B, L, d]  per-residue embeddings (may be padded).
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
