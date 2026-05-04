#!/usr/bin/env python
"""Fit and freeze the algorithmic (PCA) covariance pooler.

Closed-form alternative to ``train_unsupervised_pool.py``: no SGD, no
hyperparameters beyond ``dc``. Streams the cached embeddings twice
(pass 1: residue mean, pass 2: centred scatter matrix) and saves the top-dc
eigenvectors as a frozen checkpoint that downstream experiments load via
``CovariancePCAPooler.from_pretrained``.

Usage
-----
    python scripts/fit_pca_pool.py \\
        --embeddings data/embeddings/deeploc_train.h5 \\
        --d 1024 --dc 32 \\
        --output models/pca_pooler_dc32.pt
"""
import argparse
import logging
from pathlib import Path

from sop.data.store import EmbeddingStore
from sop.pooling.covariance_pca import CovariancePCAPooler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, nargs="+", required=True,
                        help="One or more HDF5 embedding files to fit on.")
    parser.add_argument("--d", type=int, required=True,
                        help="ProtX hidden dimension (e.g. 1024).")
    parser.add_argument("--dc", type=int, required=True,
                        help="Bottleneck dimension.")
    parser.add_argument("--no-center", action="store_true",
                        help="Skip mean centring before computing the scatter.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Destination .pt file.")
    args = parser.parse_args()

    pooler = CovariancePCAPooler(args.d, args.dc, center=not args.no_center)

    def get_iter():
        for path in args.embeddings:
            with EmbeddingStore(path) as store:
                for _, X, mask in store.iter_embeddings():
                    yield X, mask

    log.info(
        "Fitting PCA d=%d dc=%d  center=%s  files=%d",
        args.d, args.dc, not args.no_center, len(args.embeddings),
    )
    pooler.fit(get_iter)
    pooler.save_state(args.output)
    log.info("Saved fitted PCA pooler → %s", args.output)


if __name__ == "__main__":
    main()
