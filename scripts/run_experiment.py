#!/usr/bin/env python
"""Train a pooler+probe model on cached pLM embeddings and report metrics.

Reads per-residue embeddings from HDF5, instantiates the requested pooler
(mean / covariance-supervised / covariance-unsupervised / hybrid) plus an FNN
probe head, trains end-to-end with the generic torch loop, and writes a JSON
summary to ``results/runs/``.

Usage
-----
    # Mean pooling baseline
    python scripts/run_experiment.py --config configs/scl/mean.yaml

    # Covariance pooling, supervised, dc=32
    python scripts/run_experiment.py --config configs/scl/cov_supervised.yaml

    # Sweep over dc values for one config
    python scripts/run_experiment.py --config configs/scl/cov_supervised.yaml \\
        --dc 8 16 24 32 48

Config keys
-----------
See configs/scl/mean.yaml for a documented template.
"""
import argparse
import csv
import json
import logging
import re
from functools import partial
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from sop.data.store import EmbeddingStore
from sop.pooling.base import Pooler
from sop.pooling.covariance import CovariancePooler
from sop.pooling.covariance_pca import CovariancePCAPooler
from sop.pooling.hybrid import HybridPooler
from sop.pooling.mean import MeanPooler
from sop.probes.dataset import ProteinEmbeddingDataset, collate_pad
from sop.probes.fnn import ProbeFNN
from sop.probes.model import PoolingProbeModel
from sop.probes.train_loop import train_probe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_labels(path: Path) -> dict[str, str]:
    """Load a two-column CSV/TSV (id, label) into a dict."""
    labels: dict[str, str] = {}
    delimiter = "\t" if path.suffix in {".tsv", ".tab"} else ","
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        for row in reader:
            labels[row["id"]] = row["label"]
    return labels


def build_pooler(pooling_cfg: dict) -> Pooler:
    """Construct (and optionally load) the pooler described by the config."""
    method = pooling_cfg["method"]
    d = pooling_cfg["d"]

    if method == "mean":
        return MeanPooler(d)

    if method == "cov_supervised":
        dc = pooling_cfg["dc"]
        return CovariancePooler(d, dc)

    if method == "cov_unsupervised":
        ckpt = Path(pooling_cfg["pretrained_path"])
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Frozen covariance checkpoint not found: {ckpt}. "
                "Run scripts/train_unsupervised_pool.py first."
            )
        return CovariancePooler.from_pretrained(ckpt)

    if method == "cov_pca":
        ckpt = Path(pooling_cfg["pretrained_path"])
        if not ckpt.exists():
            raise FileNotFoundError(
                f"PCA covariance checkpoint not found: {ckpt}. "
                "Run scripts/fit_pca_pool.py first."
            )
        return CovariancePCAPooler.from_pretrained(ckpt)

    if method == "hybrid":
        dc = pooling_cfg["dc"]
        cov = CovariancePooler(d, dc)
        if "pretrained_path" in pooling_cfg:
            cov = CovariancePooler.from_pretrained(pooling_cfg["pretrained_path"])
        return HybridPooler(d, cov)

    raise ValueError(
        f"Unknown pooling method '{method}'. "
        "Choose mean | cov_supervised | cov_unsupervised | cov_pca | hybrid."
    )


def make_loaders(
    cfg: dict,
    task: str,
    device: str = "cpu",
) -> tuple[DataLoader, DataLoader, dict | None, int]:
    """Build train + test DataLoaders. Returns (train, test, label_to_index, n_classes)."""
    train_labels = load_labels(Path(cfg["data"]["train_labels"]))
    test_labels = load_labels(Path(cfg["data"]["test_labels"]))

    label_to_index: dict | None = None
    if task == "classification":
        all_labels = sorted(set(train_labels.values()) | set(test_labels.values()))
        label_to_index = {lbl: i for i, lbl in enumerate(all_labels)}
        n_classes = len(label_to_index)
    else:
        n_classes = 1

    train_ds = ProteinEmbeddingDataset(cfg["data"]["train_embeddings"], train_labels)
    test_ds = ProteinEmbeddingDataset(cfg["data"]["test_embeddings"], test_labels)

    pin = device.startswith("cuda")
    batch_size = cfg["probe"].get("batch_size", 16)
    collate = partial(collate_pad, label_to_index=label_to_index)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate, num_workers=0, pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate, num_workers=0, pin_memory=pin,
    )
    return train_loader, test_loader, label_to_index, n_classes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_one(cfg: dict, dc_override: int | None, output_dir: Path, config_stem: str) -> dict:
    task: str = cfg["task"]
    seeds: list[int] = cfg.get("seeds", [0, 1, 2])

    pooling_cfg = dict(cfg["pooling"])
    if dc_override is not None:
        pooling_cfg["dc"] = dc_override
        if "pretrained_path" in pooling_cfg:
            base = Path(pooling_cfg["pretrained_path"])
            stem_clean = re.sub(r"_dc\d+$", "", base.stem)
            pooling_cfg["pretrained_path"] = str(
                base.parent / f"{stem_clean}_dc{dc_override}{base.suffix}"
            )

    probe_cfg = cfg.get("probe", {})
    device = probe_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, label_to_index, n_classes = make_loaders(cfg, task, device)

    per_seed: list[dict] = []
    embedding_dim: int | None = None

    for seed in seeds:
        torch.manual_seed(seed)
        pooler = build_pooler(pooling_cfg)
        probe = ProbeFNN(
            in_dim=pooler.embedding_dim,
            out_dim=n_classes if task == "classification" else 1,
            hidden_dim=probe_cfg.get("hidden_dim", 256),
            dropout=probe_cfg.get("dropout", 0.1),
        )
        model = PoolingProbeModel(pooler, probe)

        if seed == seeds[0]:
            embedding_dim = pooler.embedding_dim
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            log.info(
                "method=%s dc=%s embed_dim=%d  trainable_params=%d  device=%s",
                pooling_cfg["method"], pooling_cfg.get("dc"), embedding_dim,
                n_trainable, device,
            )

        result = train_probe(
            model, train_loader, val_loader=test_loader,
            task=task,
            epochs=probe_cfg.get("epochs", 30),
            lr=probe_cfg.get("lr", 1e-3),
            weight_decay=probe_cfg.get("weight_decay", 0.0),
            device=device,
            log=lambda s: log.info("    " + s),
        )
        per_seed.append({
            "seed": seed,
            result["metric_key"]: result["best_metric"],
            "best_epoch": result["best_epoch"],
        })
        log.info("  seed=%d  best_%s=%.4f", seed, result["metric_key"], result["best_metric"])

    metric_key = "accuracy" if task == "classification" else "spearman_r"
    values = [r[metric_key] for r in per_seed]
    mean = sum(values) / len(values)
    std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5

    summary = {
        "config": config_stem,
        "method": pooling_cfg["method"],
        "dc": pooling_cfg.get("dc"),
        "embedding_dim": embedding_dim,
        "task": task,
        f"{metric_key}_mean": mean,
        f"{metric_key}_std": std,
        "per_seed": per_seed,
    }

    tag = pooling_cfg["method"] + (f"_dc{pooling_cfg.get('dc')}" if pooling_cfg.get("dc") else "")
    out_path = output_dir / f"{config_stem}_{tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    log.info(
        "→ %s=%.4f ± %.4f  saved to %s",
        metric_key, mean, std, out_path,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dc", type=int, nargs="+", default=None,
                        help="Override dc from config (enables quick sweep).")
    parser.add_argument("--output-dir", type=Path, default=Path("results/runs"))
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)

    dc_values: list[int | None] = args.dc if args.dc else [None]

    summaries = [
        run_one(cfg, dc, args.output_dir, args.config.stem) for dc in dc_values
    ]

    if len(summaries) > 1:
        sweep_path = args.output_dir / f"{args.config.stem}_sweep.json"
        with open(sweep_path, "w") as fh:
            json.dump(summaries, fh, indent=2)
        log.info("Sweep results → %s", sweep_path)


if __name__ == "__main__":
    main()
