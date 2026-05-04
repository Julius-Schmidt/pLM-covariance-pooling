from __future__ import annotations

from pathlib import Path
from typing import Iterator

import h5py
import numpy as np
import torch


class EmbeddingStore:
    """HDF5-backed store for per-residue pLM embeddings.

    Layout
    ------
    Each protein lives in its own group named by sequence ID::

        /{seq_id}/embeddings   — float32 [L, d]
        /{seq_id}.attrs['length'] — int, number of valid (non-padded) residues

    Usage (write)::

        with EmbeddingStore("train.h5", mode="w") as store:
            store.write("P12345", embeddings_np)  # [L, d] ndarray

    Usage (read)::

        with EmbeddingStore("train.h5") as store:
            for seq_id, X, mask in store.iter_embeddings():
                pooled = pooler.pool(X.unsqueeze(0), mask.unsqueeze(0))
    """

    def __init__(self, path: Path | str, mode: str = "r") -> None:
        self.path = Path(path)
        self._mode = mode
        self._file: h5py.File | None = None

    def __enter__(self) -> "EmbeddingStore":
        self._file = h5py.File(self.path, self._mode)
        return self

    def __exit__(self, *_) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    # ------------------------------------------------------------------

    def write(self, seq_id: str, embeddings: np.ndarray) -> None:
        """Store per-residue embeddings for one protein.

        Args:
            seq_id:     Unique identifier (FASTA header without '>').
            embeddings: [L, d] float32 array.  L is the *actual* (unpadded)
                        sequence length.
        """
        assert self._file is not None, "Use inside a 'with' block."
        grp = self._file.require_group(seq_id)
        if "embeddings" in grp:
            del grp["embeddings"]
        grp.create_dataset(
            "embeddings",
            data=embeddings.astype(np.float32),
            compression="lzf",
            chunks=True,
        )
        grp.attrs["length"] = embeddings.shape[0]

    def read(self, seq_id: str) -> tuple[torch.Tensor, int]:
        """Load embeddings for one protein.

        Returns:
            (X [L, d] float32 tensor, L: valid length)
        """
        assert self._file is not None, "Use inside a 'with' block."
        grp = self._file[seq_id]
        X = torch.from_numpy(grp["embeddings"][:])
        length = int(grp.attrs["length"])
        return X, length

    def keys(self) -> list[str]:
        assert self._file is not None, "Use inside a 'with' block."
        return list(self._file.keys())

    def iter_embeddings(self) -> Iterator[tuple[str, torch.Tensor, torch.Tensor]]:
        """Yield (seq_id, X [L, d], mask [L] bool) for every protein."""
        for seq_id in self.keys():
            X, length = self.read(seq_id)
            mask = torch.zeros(X.shape[0], dtype=torch.bool)
            mask[:length] = True
            yield seq_id, X, mask

    def __len__(self) -> int:
        assert self._file is not None, "Use inside a 'with' block."
        return len(self._file)
