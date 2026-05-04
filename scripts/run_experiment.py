#!/usr/bin/env python
"""Train and evaluate a linear probe on mean or covariance-pooled embeddings.

Reads raw per-residue embeddings from HDF5, applies the selected pooling
strategy on-the-fly, then trains a sklearn probe and reports the metric.

Usage
-----
    # Mean pooling baseline
    python scripts/run_experiment.py --config configs/deeploc_mean.yaml

    # Covariance pooling, dc=32
    python scripts/run_experiment.py --config configs/deeploc_cov_dc32.yaml

    # Sweep over dc values
    python scripts/run_experiment.py --config configs/deeploc_cov_dc32.yaml \\
        --dc 8 16 24 32 48

Config keys
-----------
See configs/deeploc_mean.yaml for a documented template.
"""
import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sop.data.store import EmbeddingStore
from sop.pooling.covariance import CovariancePooler
from sop.pooling.mean import MeanPooler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading helpers
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


def pool_split(
    store_path: Path,
    pooler,
    labels: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Apply pooler to every protein in the store and align with labels.

    Proteins not present in `labels` are silently skipped.

    Returns:
        X_pooled: [N, embed_dim] float32
        y:        [N] label array (dtype depends on task)
        ids:      list of sequence IDs in the same order
    """
    X_list: list[np.ndarray] = []
    y_list: list[str] = []
    ids: list[str] = []

    with EmbeddingStore(store_path) as store:
        for seq_id, emb, mask in store.iter_embeddings():
            if seq_id not in labels:
                continue
            pooled = pooler.pool(emb.unsqueeze(0), mask.unsqueeze(0))   # [1, dim]
            X_list.append(pooled.squeeze(0).numpy())
            y_list.append(labels[seq_id])
            ids.append(seq_id)

    return np.array(X_list, dtype=np.float32), np.array(y_list), ids


# ---------------------------------------------------------------------------
# Pooler construction
# ---------------------------------------------------------------------------

def build_pooler(pooling_cfg: dict, train_store_path: Path):
    """Construct and (if needed) fit the pooler described by the config."""
    method = pooling_cfg["method"]
    d = pooling_cfg["d"]

    if method == "mean":
        return MeanPooler(d)

    if method == "covariance":
        dc = pooling_cfg["dc"]
        center = pooling_cfg.get("center", True)
        proj_cache = Path(pooling_cfg["proj_cache"]) if "proj_cache" in pooling_cfg else None

        if proj_cache and proj_cache.exists():
            log.info("Loading cached projection from %s", proj_cache)
            return CovariancePooler.load(proj_cache)

        log.info("Fitting covariance projection (dc=%d) on %s …", dc, train_store_path)
        pooler = CovariancePooler(d, dc, center)

        def get_iter():
            with EmbeddingStore(train_store_path) as store:
                for _, X, mask in store.iter_embeddings():
                    yield X, mask

        pooler.fit(get_iter)

        if proj_cache:
            pooler.save(proj_cache)
            log.info("Saved projection → %s", proj_cache)

        return pooler

    raise ValueError(f"Unknown pooling method '{method}'. Choose 'mean' or 'covariance'.")


# ---------------------------------------------------------------------------
# Probe training
# ---------------------------------------------------------------------------

def run_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task: str,
    probe_cfg: dict,
    seed: int,
) -> dict[str, float]:
    # Scale features — important for covariance embeddings whose magnitude can
    # vary substantially across dc² entries.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if task == "classification":
        clf = LogisticRegression(
            C=probe_cfg.get("C", 1.0),
            max_iter=probe_cfg.get("max_iter", 1000),
            random_state=seed,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return {"accuracy": float(accuracy_score(y_test, y_pred))}

    if task == "regression":
        reg = Ridge(alpha=probe_cfg.get("alpha", 1.0), random_state=seed)
        reg.fit(X_train, y_train.astype(float))
        y_pred = reg.predict(X_test)
        rho, _ = spearmanr(y_test.astype(float), y_pred)
        return {"spearman_r": float(rho)}

    raise ValueError(f"Unknown task '{task}'. Choose 'classification' or 'regression'.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dc", type=int, nargs="+", default=None,
                        help="Override dc from config (enables quick sweep).")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)

    train_store = Path(cfg["data"]["train_embeddings"])
    test_store = Path(cfg["data"]["test_embeddings"])
    train_labels = load_labels(Path(cfg["data"]["train_labels"]))
    test_labels = load_labels(Path(cfg["data"]["test_labels"]))
    task: str = cfg["task"]
    seeds: list[int] = cfg.get("seeds", [0, 1, 2])

    dc_values: list[int | None] = args.dc if args.dc else [cfg["pooling"].get("dc")]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_summaries: list[dict] = []

    for dc in dc_values:
        pooling_cfg = dict(cfg["pooling"])
        if dc is not None:
            pooling_cfg["dc"] = dc
            # Update proj_cache path to avoid cross-dc contamination
            if "proj_cache" in pooling_cfg:
                base = Path(pooling_cfg["proj_cache"])
                pooling_cfg["proj_cache"] = str(
                    base.parent / f"{base.stem}_dc{dc}{base.suffix}"
                )

        pooler = build_pooler(pooling_cfg, train_store)
        embed_dim = pooler.embedding_dim

        log.info("Pooling training split …")
        X_train, y_train, _ = pool_split(train_store, pooler, train_labels)
        log.info("Pooling test split …")
        X_test, y_test, _ = pool_split(test_store, pooler, test_labels)

        log.info(
            "method=%s dc=%s embed_dim=%d  train=%d  test=%d",
            pooling_cfg["method"], dc, embed_dim, len(X_train), len(X_test),
        )

        per_seed: list[dict] = []
        for seed in seeds:
            result = run_probe(X_train, y_train, X_test, y_test,
                               task, cfg.get("probe", {}), seed)
            result["seed"] = seed
            per_seed.append(result)
            log.info("  seed=%d  %s", seed, result)

        metric_key = "accuracy" if task == "classification" else "spearman_r"
        values = [r[metric_key] for r in per_seed]

        summary = {
            "method": pooling_cfg["method"],
            "dc": dc,
            "embedding_dim": embed_dim,
            "task": task,
            f"{metric_key}_mean": float(np.mean(values)),
            f"{metric_key}_std": float(np.std(values)),
            "per_seed": per_seed,
        }
        all_summaries.append(summary)

        tag = f"{pooling_cfg['method']}" + (f"_dc{dc}" if dc else "")
        out_path = args.output_dir / f"{args.config.stem}_{tag}.json"
        with open(out_path, "w") as fh:
            json.dump(summary, fh, indent=2)
        log.info(
            "→ %s=%.4f ± %.4f  saved to %s",
            metric_key, summary[f"{metric_key}_mean"], summary[f"{metric_key}_std"],
            out_path,
        )

    if len(all_summaries) > 1:
        sweep_path = args.output_dir / f"{args.config.stem}_sweep.json"
        with open(sweep_path, "w") as fh:
            json.dump(all_summaries, fh, indent=2)
        log.info("Sweep results → %s", sweep_path)


if __name__ == "__main__":
    main()
