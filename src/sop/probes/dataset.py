from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset

from ..data.store import EmbeddingStore


class ProteinEmbeddingDataset(Dataset):
    """Random-access dataset over a single HDF5 embedding store.

    Yields ``(X [L, d] float32, length int, y)`` per protein. Labels are
    fetched from a caller-provided ``id → label`` mapping; proteins with no
    label entry are skipped at construction time.

    The HDF5 file is opened lazily on first __getitem__ and kept open for the
    lifetime of the dataset. Safe for the default single-worker DataLoader;
    use ``persistent_workers=False`` if increasing num_workers.
    """

    def __init__(self, store_path: Path | str, labels: dict[str, object]) -> None:
        self.path = Path(store_path)
        self._store: EmbeddingStore | None = None
        with EmbeddingStore(self.path) as s:
            self.ids: list[str] = [k for k in s.keys() if k in labels]
        self.labels = {k: labels[k] for k in self.ids}

    def _open(self) -> EmbeddingStore:
        if self._store is None:
            self._store = EmbeddingStore(self.path).__enter__()
        return self._store

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, object]:
        seq_id = self.ids[idx]
        X, length = self._open().read(seq_id)
        return X, length, self.labels[seq_id]


def collate_pad(
    batch: Sequence[tuple[torch.Tensor, int, object]],
    label_to_index: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable-length proteins to the batch max length.

    Args:
        batch:          List of (X [L_i, d], length_i, label_i) tuples.
        label_to_index: For classification, mapping from label string → int
                        class index. If None, labels are kept as-is and stacked
                        as floats (regression).

    Returns:
        X_padded: [B, L_max, d] float32
        mask:     [B, L_max] bool, True for valid positions
        y:        [B] long (classification) or float (regression)
    """
    Xs, lengths, ys = zip(*batch)
    B = len(Xs)
    d = Xs[0].shape[1]
    L_max = max(lengths)

    X_padded = torch.zeros(B, L_max, d, dtype=torch.float32)
    mask = torch.zeros(B, L_max, dtype=torch.bool)
    for i, (X, L) in enumerate(zip(Xs, lengths)):
        X_padded[i, :L] = X
        mask[i, :L] = True

    if label_to_index is not None:
        y = torch.tensor([label_to_index[v] for v in ys], dtype=torch.long)
    else:
        y = torch.tensor([float(v) for v in ys], dtype=torch.float32)

    return X_padded, mask, y
