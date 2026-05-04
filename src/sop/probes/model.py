import torch
import torch.nn as nn

from ..pooling.base import Pooler
from .fnn import ProbeFNN


class PoolingProbeModel(nn.Module):
    """Composition of a pooling layer and an FNN probe head.

    The pooler reduces per-residue embeddings to a single protein-level vector,
    which the probe then maps to logits or a scalar. Both submodules are
    nn.Modules, so gradients flow into the pooler when it has trainable
    parameters (e.g. supervised covariance) and stop at the pooler boundary
    when it doesn't (mean) or when it has been frozen (unsupervised covariance).
    """

    def __init__(self, pooler: Pooler, probe: ProbeFNN) -> None:
        super().__init__()
        self.pooler = pooler
        self.probe = probe

    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pooled = self.pooler(X, mask)
        return self.probe(pooled)
