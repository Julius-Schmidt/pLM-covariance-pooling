import torch


def make_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """Boolean mask of shape [B, max_len]; True = valid (non-padded) position.

    Args:
        lengths: [B] int tensor, actual (unpadded) sequence lengths.
        max_len: padded sequence length.
    """
    idx = torch.arange(max_len, device=lengths.device)
    return idx.unsqueeze(0) < lengths.unsqueeze(1)


def apply_mask(X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Zero out padded positions in place (non-destructive via clone).

    Args:
        X:    [B, L, d] per-residue embeddings.
        mask: [B, L] bool, True for valid positions.

    Returns:
        [B, L, d] with padded positions zeroed.
    """
    return X * mask.unsqueeze(-1).to(X.dtype)
