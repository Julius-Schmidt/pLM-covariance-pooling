#!/usr/bin/env python
"""lxhalle CPU runner — self-contained twin of scripts/run_experiment_aws.py.

Why this exists
---------------
Same independent path as scripts/run_experiment_aws.py (inlines its own
training loop so it does not depend on src/sop/probes/train_loop.py), but
tuned for a CPU-only server (lxhalle) that you attach to via terminal:
    * no AMP / GradScaler / autocast (CPU has no fp16 benefit here)
    * pin_memory disabled, num_workers configurable via probe.num_workers
      (default 4 — lxhalle has many cores available for data loading)

JSON output format matches scripts/run_experiment.py / run_experiment_aws.py
(including per-seed y_true / y_pred / y_proba and best-epoch pooler weight
snapshots), so the DeepLoc per-class plots in scripts/make_plots.py work
unchanged.

Usage
-----
    python scripts/run_experiments_lxhalle.py --config configs/scl/mean.yaml
    python scripts/run_experiments_lxhalle.py --config configs/scl/cov_supervised.yaml --dc 8 16 24 32 48
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import subprocess
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import Adam
from torch.utils.data import DataLoader

from sop.pooling.attention_covariance import AttentionCovariancePooler
from sop.pooling.base import Pooler
from sop.pooling.covariance import CovariancePooler
from sop.pooling.covariance_pca import CovariancePCAPooler
from sop.pooling.hybrid import HybridPooler
from sop.pooling.light_attention import LightAttentionPooler
from sop.pooling.mean import MeanPooler
from sop.probes.dataset import ProteinEmbeddingDataset, collate_pad
from sop.probes.fnn import ProbeFNN
from sop.probes.metrics import accuracy, spearman_r
from sop.probes.model import PoolingProbeModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Run-level metadata (captured once per process)
# ---------------------------------------------------------------------------

def _run_meta() -> dict:
    """git SHA + torch/python versions so each result can be traced back."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        sha = None
    try:
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip())
    except Exception:
        dirty = None
    return {
        "git_sha": sha,
        "git_dirty": dirty,
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "device_name": "cpu",
    }


# ---------------------------------------------------------------------------
# Config / model construction
# ---------------------------------------------------------------------------

def load_labels(path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    delim = "\t" if path.suffix in {".tsv", ".tab"} else ","
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh, delimiter=delim):
            labels[row["id"]] = row["label"]
    return labels


def build_pooler(pooling_cfg: dict) -> Pooler:
    method = pooling_cfg["method"]
    d = pooling_cfg["d"]

    # Matrix-power (iSQRT-COV) normalisation toggle, shared by the cov methods.
    power_norm = pooling_cfg.get("power_norm", False)

    if method == "mean":
        return MeanPooler(d)
    if method == "cov_supervised":
        return CovariancePooler(d, pooling_cfg["dc"], power_norm=power_norm)
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
        pooler = CovariancePCAPooler.from_pretrained(ckpt)
        if power_norm:
            pooler.set_power_norm(True)
        return pooler
    if method == "hybrid":
        dc = pooling_cfg["dc"]
        cov = CovariancePooler(d, dc)
        if "pretrained_path" in pooling_cfg:
            cov = CovariancePooler.from_pretrained(pooling_cfg["pretrained_path"])
        return HybridPooler(d, cov)
    if method == "light_attention":
        return LightAttentionPooler(
            d,
            d_out=pooling_cfg.get("dc"),
            kernel_size=pooling_cfg.get("kernel_size", 9),
            conv_dropout=pooling_cfg.get("conv_dropout", 0.25),
        )
    if method == "attention_cov":
        return AttentionCovariancePooler(
            d,
            pooling_cfg["dc"],
            power_norm=power_norm,
            kernel_size=pooling_cfg.get("kernel_size", 9),
            conv_dropout=pooling_cfg.get("conv_dropout", 0.0),
        )

    raise ValueError(
        f"Unknown pooling method '{method}'. Choose mean | cov_supervised | "
        "cov_unsupervised | cov_pca | hybrid | light_attention | attention_cov."
    )


def make_loaders(
    cfg: dict, task: str,
) -> tuple[DataLoader, DataLoader, dict | None, int]:
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

    probe_cfg = cfg.get("probe", {})
    num_workers = probe_cfg.get("num_workers", 4)
    loader_kwargs: dict[str, Any] = dict(
        collate_fn=partial(collate_pad, label_to_index=label_to_index),
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    batch_size = probe_cfg.get("batch_size", 16)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
    return train_loader, test_loader, label_to_index, n_classes


# ---------------------------------------------------------------------------
# Training loop (inlined — does not depend on src/sop/probes/train_loop.py)
# ---------------------------------------------------------------------------

def _loss_and_metric(task: str) -> tuple[nn.Module, Callable[[np.ndarray, np.ndarray], float], str]:
    if task == "classification":
        return nn.CrossEntropyLoss(), accuracy, "accuracy"
    if task == "regression":
        return nn.MSELoss(), spearman_r, "spearman_r"
    raise ValueError(f"Unknown task '{task}'. Choose 'classification' or 'regression'.")


@torch.no_grad()
def evaluate(
    model: PoolingProbeModel, loader: DataLoader, task: str,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray | None]:
    """Return (metric, y_true, y_pred, y_proba).

    * y_pred is class index for classification, scalar for regression.
    * y_proba is the per-class softmax for classification, None for regression.
    """
    _, metric_fn, _ = _loss_and_metric(task)
    model.eval()
    preds: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    probas: list[np.ndarray] = []
    for X, mask, y in loader:
        out = model(X, mask)
        if task == "classification":
            proba = torch.softmax(out, dim=-1).numpy()
            probas.append(proba)
            p = proba.argmax(axis=-1)
        else:
            p = out.squeeze(-1).numpy()
        preds.append(p)
        truths.append(y.numpy())
    y_true = np.concatenate(truths)
    y_pred = np.concatenate(preds)
    y_proba = np.concatenate(probas) if probas else None
    return metric_fn(y_true, y_pred), y_true, y_pred, y_proba


def train_probe(
    model: PoolingProbeModel,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    *,
    task: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    log_fn: Callable[[str], None],
    patience: int | None = None,
) -> dict:
    """Train until convergence.

    * If `patience` is set, training stops early when the val metric hasn't
      improved for `patience` consecutive epochs. `epochs` then acts as a
      max-epoch ceiling.
    * If `patience` is None, training runs the full `epochs` count.
    """
    loss_fn, _, metric_key = _loss_and_metric(task)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = Adam(trainable, lr=lr, weight_decay=weight_decay)

    history: list[dict] = []
    best_metric = -float("inf")
    best_epoch = -1
    best_y_true: np.ndarray | None = None
    best_y_pred: np.ndarray | None = None
    best_y_proba: np.ndarray | None = None
    best_pooler_state: dict | None = None
    epochs_since_improvement = 0
    stopped_early = False

    for epoch in range(epochs):
        t0 = time.perf_counter()
        model.train()
        running = 0.0
        n_batches = 0
        for X, mask, y in train_loader:
            opt.zero_grad(set_to_none=True)
            out = model(X, mask)
            if task == "regression":
                out = out.squeeze(-1)
                target = y.float()
            else:
                target = y
            loss = loss_fn(out, target)
            loss.backward()
            opt.step()

            running += float(loss.detach())
            n_batches += 1
        t_train = time.perf_counter() - t0

        train_loss = running / max(n_batches, 1)
        val_metric = float("nan")
        t_val = 0.0
        if val_loader is not None:
            t1 = time.perf_counter()
            val_metric, y_true, y_pred, y_proba = evaluate(model, val_loader, task)
            t_val = time.perf_counter() - t1
            if val_metric > best_metric:
                best_metric = val_metric
                best_epoch = epoch
                best_y_true = y_true
                best_y_pred = y_pred
                best_y_proba = y_proba
                # Snapshot pooler weights at the best epoch (CPU copy; ignored
                # at save time for MeanPooler which has no parameters).
                best_pooler_state = {
                    k: v.detach().clone()
                    for k, v in model.pooler.state_dict().items()
                }
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            metric_key: val_metric,
            "epoch_time_s": round(t_train + t_val, 3),
            "train_time_s": round(t_train, 3),
            "val_time_s": round(t_val, 3),
        })
        log_fn(
            f"epoch {epoch:3d}  train_loss={train_loss:.4f}  "
            f"val_{metric_key}={val_metric:.4f}  "
            f"time={t_train + t_val:.1f}s (train={t_train:.1f}, val={t_val:.1f})  "
            f"no_improve={epochs_since_improvement}"
        )

        if patience is not None and epochs_since_improvement >= patience:
            log_fn(
                f"early stopping at epoch {epoch} — no improvement for "
                f"{patience} epochs (best was epoch {best_epoch}, "
                f"val_{metric_key}={best_metric:.4f})"
            )
            stopped_early = True
            break

    return {
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "metric_key": metric_key,
        "history": history,
        "best_y_true": best_y_true,
        "best_y_pred": best_y_pred,
        "best_y_proba": best_y_proba,
        "best_pooler_state": best_pooler_state,
        "stopped_early": stopped_early,
        "epochs_run": len(history),
    }


# ---------------------------------------------------------------------------
# Per-config driver (mirrors run_experiment.run_one for JSON-compat output)
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
    train_loader, test_loader, label_to_index, n_classes = make_loaders(cfg, task)

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
                "method=%s dc=%s embed_dim=%d  trainable_params=%d  device=cpu",
                pooling_cfg["method"], pooling_cfg.get("dc"), embedding_dim,
                n_trainable,
            )

        result = train_probe(
            model, train_loader, val_loader=test_loader,
            task=task,
            epochs=probe_cfg.get("epochs", 30),
            lr=probe_cfg.get("lr", 1e-3),
            weight_decay=probe_cfg.get("weight_decay", 0.0),
            log_fn=lambda s: log.info("    " + s),
            patience=probe_cfg.get("patience"),  # None = no early stopping
        )
        # Save pooler weights snapshot to a sibling .pt file (skip empty
        # state_dicts, e.g. MeanPooler).
        pooler_weights_path: str | None = None
        if result["best_pooler_state"]:
            weights_dir = output_dir / "pooler_weights"
            weights_dir.mkdir(parents=True, exist_ok=True)
            tag_dc = f"_dc{pooling_cfg.get('dc')}" if pooling_cfg.get("dc") else ""
            weights_file = weights_dir / f"{config_stem}_{pooling_cfg['method']}{tag_dc}_seed{seed}.pt"
            torch.save(result["best_pooler_state"], weights_file)
            pooler_weights_path = str(weights_file)

        per_seed.append({
            "seed": seed,
            result["metric_key"]: result["best_metric"],
            "best_epoch": result["best_epoch"],
            "epochs_run": result["epochs_run"],
            "stopped_early": result["stopped_early"],
            "history": result["history"],
            "y_true": result["best_y_true"].tolist() if result["best_y_true"] is not None else None,
            "y_pred": result["best_y_pred"].tolist() if result["best_y_pred"] is not None else None,
            "y_proba": result["best_y_proba"].tolist() if result["best_y_proba"] is not None else None,
            "pooler_weights_path": pooler_weights_path,
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
        "power_norm": pooling_cfg.get("power_norm", False),
        "embedding_dim": embedding_dim,
        "task": task,
        f"{metric_key}_mean": mean,
        f"{metric_key}_std": std,
        "label_to_index": label_to_index,  # None for regression; class-name → int for classification
        "meta": _run_meta(),
        "per_seed": per_seed,
    }

    tag = pooling_cfg["method"] + (f"_dc{pooling_cfg.get('dc')}" if pooling_cfg.get("dc") else "")
    out_path = output_dir / f"{config_stem}_{tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    log.info("→ %s=%.4f ± %.4f  saved to %s", metric_key, mean, std, out_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dc", type=int, nargs="+", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results/runs"))
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)

    dc_values: list[int | None] = args.dc if args.dc else [None]
    # Prefix with the parent directory (e.g. "scl" / "meltome") so SCL and
    # meltome runs of the same method don't overwrite each other's JSON.
    stem = f"{args.config.parent.name}_{args.config.stem}"
    summaries = [run_one(cfg, dc, args.output_dir, stem) for dc in dc_values]

    if len(summaries) > 1:
        sweep_path = args.output_dir / f"{stem}_sweep.json"
        with open(sweep_path, "w") as fh:
            json.dump(summaries, fh, indent=2)
        log.info("Sweep results → %s", sweep_path)


if __name__ == "__main__":
    main()
