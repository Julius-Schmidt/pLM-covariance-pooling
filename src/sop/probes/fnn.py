import torch
import torch.nn as nn


class ProbeFNN(nn.Module):
    """Small feed-forward probe head shared by every pooling method.

    One hidden layer with ReLU + dropout by default. Output is logits for
    classification (apply CrossEntropyLoss) or a raw scalar for regression
    (apply MSELoss).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
