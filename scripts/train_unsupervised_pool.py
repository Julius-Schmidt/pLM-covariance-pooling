#!/usr/bin/env python
"""Train and freeze the unsupervised covariance autoencoder.

Streams cached per-residue embeddings from one or more HDF5 stores, fits a
CovariancePooler by minimising the Frobenius reconstruction loss
‖XᵀX − L (XL)ᵀ(XR) Rᵀ‖²_F (computed via the trick — see
``sop.unsupervised.frobenius_trainer``), and saves the trained pooler to a
.pt file that downstream experiments can load via ``CovariancePooler.from_pretrained``.

Usage
-----
    python scripts/train_unsupervised_pool.py \\
        --embeddings data/embeddings/deeploc_train.h5 \\
        --d 1024 --dc 32 \\
        --epochs 5 --batch-size 32 --lr 1e-3 \\
        --output models/unsup_pooler_dc32.pt
"""
import argparse
import logging
from pathlib import Path

import torch

from sop.data.store import EmbeddingStore
from sop.pooling.covariance import CovariancePooler
from sop.unsupervised.frobenius_trainer import train_unsupervised_pooler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, nargs="+", required=True,
                        help="One or more HDF5 embedding files to train on.")
    parser.add_argument("--d", type=int, required=True,
                        help="ProtX hidden dimension (e.g. 1024).")
    parser.add_argument("--dc", type=int, required=True,
                        help="Bottleneck dimension.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Number of proteins per gradient step (no padding).")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, required=True,
                        help="Destination .pt file.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    pooler = CovariancePooler(args.d, args.dc)

    def get_iter():
        """Yield (X, mask) for every protein in every input store."""
        for path in args.embeddings:
            with EmbeddingStore(path) as store:
                for _, X, mask in store.iter_embeddings():
                    yield X, mask

    log.info(
        "Training d=%d dc=%d  epochs=%d  batch_size=%d  lr=%.0e  device=%s",
        args.d, args.dc, args.epochs, args.batch_size, args.lr, args.device,
    )
    train_unsupervised_pooler(
        pooler, get_iter,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        log=lambda s: log.info(s),
    )

    pooler.cpu().freeze().save_state(args.output)
    log.info("Saved frozen pooler → %s", args.output)


if __name__ == "__main__":
    main()
