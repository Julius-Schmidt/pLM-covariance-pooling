#!/usr/bin/env python
"""lxhalle CPU runner for DeepLoc 2 — multi-label subcellular localisation.

DeepLoc 2 proteins can be annotated with multiple compartments simultaneously,
so this script uses BCEWithLogitsLoss + micro-F1 instead of the single-label
CrossEntropyLoss + accuracy used by run_experiments_lxhalle.py.

Label CSV format
----------------
Two formats are accepted (auto-detected):

  Format 1 — single labels column (pipe-separated):
    id,labels
    Q9Y5V3,Nucleus|Cytoplasm
    P12345,Extracellular

  Format 2 — binary columns (one per class):
    id,Nucleus,Cytoplasm,Extracellular,...
    Q9Y5V3,1,1,0,...
    P12345,0,0,1,...

JSON output
-----------
Same schema as run_experiments_lxhalle.py, except:
  * task            = "multilabel_classification"
  * metric key      = "f1_micro"  (threshold 0.5 on sigmoid output)
  * per_seed y_true / y_pred / y_proba are 2-D lists [n_samples, n_classes]

This means make_plots.py per-class analysis works — just extend it to handle
the multilabel f1 metric and 2-D prediction arrays.

Usage
-----
    python scripts/run_experiments_lxhalle_deeploc2.py --config configs/deeploc2/mean.yaml
    python scripts/run_experiments_lxhalle_deeploc2.py --config configs/deeploc2/cov_supervised.yaml --dc 8 16 24 32 48
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
from sklearn.metrics import f1_score
from torch.optim import Adam
from torch.utils.data import DataLoader

from sop.pooling.attention_covariance import AttentionCovariancePooler
from sop.pooling.base import Pooler
from sop.pooling.covariance import CovariancePooler
from sop.pooling.covariance_pca import CovariancePCAPooler
from sop.pooling.hybrid import HybridPooler
from sop.pooling.light_attention import LightAttentionPooler
from sop.pooling.mean import MeanPooler
from sop.probes.dataset import ProteinEmbeddingDataset
from sop.probes.fnn import ProbeFNN
from sop.probes.model import PoolingProbeModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Run-level metadata
# ---------------------------------------------------------------------------

def _run_meta() -> dict:
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
# Multi-label CSV loading
# ---------------------------------------------------------------------------

_DEEPLOC2_META_COLS = {"Unnamed: 0", "Kingdom", "Partition", "Sequence"}

def load_multilabels(
    path: Path,
    sep: str = "|",
    id_col: str | None = None,
    meta_cols: set[str] | None = None,
) -> tuple[dict[str, list[str]], list[str]]:
    """Load a multi-label CSV.  Returns (id→[class_names], sorted_class_list).

    id_col    — which column holds the protein accession (auto-detected if None).
    meta_cols — columns to skip when finding class columns (non-binary metadata).

    Auto-detects format:
      - If a column named "labels" or "label" exists → pipe-separated values.
      - Otherwise → binary columns (1/0) after removing id_col and meta_cols.

    DeepLoc 2 format example:
      Unnamed: 0,ACC,Kingdom,Partition,Membrane,Cytoplasm,...,Sequence
    """
    if meta_cols is None:
        meta_cols = _DEEPLOC2_META_COLS
    delim = "\t" if path.suffix in {".tsv", ".tab"} else ","
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter=delim))
    if not rows:
        return {}, []

    columns = list(rows[0].keys())

    # Auto-detect id column: explicit arg > "id" > "ACC" > first column
    if id_col is None:
        if "id" in columns:
            id_col = "id"
        elif "ACC" in columns:
            id_col = "ACC"
        else:
            id_col = columns[0]

    id_to_labels: dict[str, list[str]] = {}
    all_classes: set[str] = set()

    if "labels" in columns or "label" in columns:
        lbl_col = "labels" if "labels" in columns else "label"
        for row in rows:
            classes = [c.strip() for c in row[lbl_col].split(sep) if c.strip()]
            id_to_labels[row[id_col]] = classes
            all_classes.update(classes)
    else:
        skip = meta_cols | {id_col}
        class_cols = [c for c in columns if c not in skip]
        all_classes = set(class_cols)
        for row in rows:
            classes = [
                c for c in class_cols
                if row[c].strip() in ("1", "1.0", "True", "true", "yes")
            ]
            id_to_labels[row[id_col]] = classes

    return id_to_labels, sorted(all_classes)


def collate_pad_multilabel(
    batch: list[tuple[torch.Tensor, int, list[str]]],
    label_to_index: dict[str, int],
    n_classes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable-length proteins; encode multi-label targets as binary vectors."""
    Xs, lengths, ys = zip(*batch)
    B = len(Xs)
    d = Xs[0].shape[1]
    L_max = max(lengths)

    X_padded = torch.zeros(B, L_max, d, dtype=torch.float32)
    mask = torch.zeros(B, L_max, dtype=torch.bool)
    for i, (X, L) in enumerate(zip(Xs, lengths)):
        X_padded[i, :L] = X
        mask[i, :L] = True

    y = torch.zeros(B, n_classes, dtype=torch.float32)
    for i, class_list in enumerate(ys):
        for cls in class_list:
            if cls in label_to_index:
                y[i, label_to_index[cls]] = 1.0

    return X_padded, mask, y


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_pooler(pooling_cfg: dict) -> Pooler:
    method = pooling_cfg["method"]
    d = pooling_cfg["d"]
    power_norm = pooling_cfg.get("power_norm", False)

    if method == "mean":
        return MeanPooler(d)
    if method == "cov_supervised":
        return CovariancePooler(d, pooling_cfg["dc"], power_norm=power_norm)
    if method == "cov_unsupervised":
        ckpt = Path(pooling_cfg["pretrained_path"])
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        return CovariancePooler.from_pretrained(ckpt)
    if method == "cov_pca":
        ckpt = Path(pooling_cfg["pretrained_path"])
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        pooler = CovariancePCAPooler.from_pretrained(ckpt)
        if power_norm:
            pooler.set_power_norm(True)
        return pooler
    if method == "hybrid":
        cov = CovariancePooler(d, pooling_cfg["dc"])
        if "pretrained_path" in pooling_cfg:
            cov = CovariancePooler.from_pretrained(pooling_cfg["pretrained_path"])
        return HybridPooler(d, cov)
    if method == "light_attention":
        return LightAttentionPooler(
            d,
            kernel_size=pooling_cfg.get("kernel_size", 9),
            conv_dropout=pooling_cfg.get("conv_dropout", 0.25),
        )
    if method == "attention_cov":
        return AttentionCovariancePooler(
            d, pooling_cfg["dc"],
            power_norm=power_norm,
            kernel_size=pooling_cfg.get("kernel_size", 9),
            conv_dropout=pooling_cfg.get("conv_dropout", 0.0),
        )
    raise ValueError(f"Unknown pooling method '{method}'.")


def make_loaders(
    cfg: dict,
    label_to_index: dict[str, int],
    n_classes: int,
) -> tuple[DataLoader, DataLoader]:
    probe_cfg = cfg.get("probe", {})
    num_workers = probe_cfg.get("num_workers", 4)
    batch_size = probe_cfg.get("batch_size", 64)
    collate = partial(collate_pad_multilabel,
                      label_to_index=label_to_index, n_classes=n_classes)
    loader_kwargs: dict[str, Any] = dict(
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    id_col = cfg["data"].get("id_col", None)
    meta_cols = set(cfg["data"].get("meta_cols", [])) or None
    train_labels_raw, _ = load_multilabels(Path(cfg["data"]["train_labels"]), id_col=id_col, meta_cols=meta_cols)
    test_labels_raw, _  = load_multilabels(Path(cfg["data"]["test_labels"]),  id_col=id_col, meta_cols=meta_cols)
    train_ds = ProteinEmbeddingDataset(cfg["data"]["train_embeddings"], train_labels_raw)
    test_ds  = ProteinEmbeddingDataset(cfg["data"]["test_embeddings"],  test_labels_raw)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **loader_kwargs)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **loader_kwargs)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: PoolingProbeModel,
    loader: DataLoader,
    threshold: float = 0.5,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Return (micro_f1, y_true [N,C], y_pred_binary [N,C], y_proba [N,C])."""
    model.eval()
    all_proba, all_true = [], []
    for X, mask, y in loader:
        logits = model(X, mask)
        all_proba.append(torch.sigmoid(logits).numpy())
        all_true.append(y.numpy())
    y_proba = np.concatenate(all_proba)
    y_true  = np.concatenate(all_true)
    y_pred  = (y_proba >= threshold).astype(np.int8)
    score   = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    return score, y_true, y_pred, y_proba


def train_probe(
    model: PoolingProbeModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    log_fn: Callable[[str], None],
    patience: int | None = None,
    threshold: float = 0.5,
) -> dict:
    loss_fn = nn.BCEWithLogitsLoss()
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = Adam(trainable, lr=lr, weight_decay=weight_decay)

    history: list[dict] = []
    best_metric = -float("inf")
    best_epoch  = -1
    best_y_true = best_y_pred = best_y_proba = None
    best_pooler_state = None
    epochs_since_improvement = 0
    stopped_early = False

    for epoch in range(epochs):
        t0 = time.perf_counter()
        model.train()
        running, n_batches = 0.0, 0
        for X, mask, y in train_loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(X, mask), y)
            loss.backward()
            opt.step()
            running += float(loss.detach())
            n_batches += 1
        t_train = time.perf_counter() - t0

        t1 = time.perf_counter()
        val_metric, y_true, y_pred, y_proba = evaluate(model, val_loader, threshold)
        t_val = time.perf_counter() - t1

        if val_metric > best_metric:
            best_metric = val_metric
            best_epoch  = epoch
            best_y_true, best_y_pred, best_y_proba = y_true, y_pred, y_proba
            best_pooler_state = {
                k: v.detach().clone()
                for k, v in model.pooler.state_dict().items()
            }
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        train_loss = running / max(n_batches, 1)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "f1_micro": val_metric,
            "epoch_time_s": round(t_train + t_val, 3),
        })
        log_fn(
            f"epoch {epoch:3d}  train_loss={train_loss:.4f}  "
            f"val_f1_micro={val_metric:.4f}  "
            f"time={t_train + t_val:.1f}s  no_improve={epochs_since_improvement}"
        )

        if patience is not None and epochs_since_improvement >= patience:
            log_fn(
                f"early stopping at epoch {epoch} — no improvement for "
                f"{patience} epochs (best epoch {best_epoch}, f1={best_metric:.4f})"
            )
            stopped_early = True
            break

    return {
        "best_metric": best_metric,
        "best_epoch":  best_epoch,
        "history":     history,
        "best_y_true":  best_y_true,
        "best_y_pred":  best_y_pred,
        "best_y_proba": best_y_proba,
        "best_pooler_state": best_pooler_state,
        "stopped_early": stopped_early,
        "epochs_run": len(history),
    }


# ---------------------------------------------------------------------------
# Per-config driver
# ---------------------------------------------------------------------------

def run_one(cfg: dict, dc_override: int | None, output_dir: Path, config_stem: str) -> dict:
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

    # Build label vocabulary from both splits so indices are consistent.
    id_col = cfg["data"].get("id_col", None)
    meta_cols = set(cfg["data"].get("meta_cols", [])) or None
    train_labels, train_classes = load_multilabels(Path(cfg["data"]["train_labels"]), id_col=id_col, meta_cols=meta_cols)
    test_labels,  test_classes  = load_multilabels(Path(cfg["data"]["test_labels"]),  id_col=id_col, meta_cols=meta_cols)
    all_classes   = sorted(set(train_classes) | set(test_classes))
    label_to_index = {cls: i for i, cls in enumerate(all_classes)}
    n_classes = len(all_classes)
    log.info("Multi-label classes (%d): %s", n_classes, all_classes)

    train_loader, test_loader = make_loaders(cfg, label_to_index, n_classes)

    per_seed: list[dict] = []
    embedding_dim: int | None = None

    for seed in seeds:
        torch.manual_seed(seed)
        pooler = build_pooler(pooling_cfg)
        probe  = ProbeFNN(
            in_dim=pooler.embedding_dim,
            out_dim=n_classes,
            hidden_dim=probe_cfg.get("hidden_dim", 256),
            dropout=probe_cfg.get("dropout", 0.1),
        )
        model = PoolingProbeModel(pooler, probe)

        if seed == seeds[0]:
            embedding_dim = pooler.embedding_dim
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            log.info(
                "method=%s dc=%s embed_dim=%d  trainable_params=%d  n_classes=%d  device=cpu",
                pooling_cfg["method"], pooling_cfg.get("dc"),
                embedding_dim, n_trainable, n_classes,
            )

        result = train_probe(
            model, train_loader, test_loader,
            epochs=probe_cfg.get("epochs", 50),
            lr=probe_cfg.get("lr", 1e-3),
            weight_decay=probe_cfg.get("weight_decay", 0.0),
            log_fn=lambda s: log.info("    " + s),
            patience=probe_cfg.get("patience"),
            threshold=probe_cfg.get("threshold", 0.5),
        )

        pooler_weights_path: str | None = None
        if result["best_pooler_state"]:
            weights_dir = output_dir / "pooler_weights"
            weights_dir.mkdir(parents=True, exist_ok=True)
            tag_dc = f"_dc{pooling_cfg.get('dc')}" if pooling_cfg.get("dc") else ""
            weights_file = weights_dir / f"{config_stem}_{pooling_cfg['method']}{tag_dc}_seed{seed}.pt"
            torch.save(result["best_pooler_state"], weights_file)
            pooler_weights_path = str(weights_file)

        per_seed.append({
            "seed":          seed,
            "f1_micro":      result["best_metric"],
            "best_epoch":    result["best_epoch"],
            "epochs_run":    result["epochs_run"],
            "stopped_early": result["stopped_early"],
            "history":       result["history"],
            "y_true":  result["best_y_true"].tolist()  if result["best_y_true"]  is not None else None,
            "y_pred":  result["best_y_pred"].tolist()  if result["best_y_pred"]  is not None else None,
            "y_proba": result["best_y_proba"].tolist() if result["best_y_proba"] is not None else None,
            "pooler_weights_path": pooler_weights_path,
        })
        log.info("  seed=%d  best_f1_micro=%.4f", seed, result["best_metric"])

    values = [r["f1_micro"] for r in per_seed]
    mean   = sum(values) / len(values)
    std    = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5

    summary = {
        "config":        config_stem,
        "method":        pooling_cfg["method"],
        "dc":            pooling_cfg.get("dc"),
        "power_norm":    pooling_cfg.get("power_norm", False),
        "embedding_dim": embedding_dim,
        "task":          "multilabel_classification",
        "f1_micro_mean": mean,
        "f1_micro_std":  std,
        "label_to_index": label_to_index,
        "meta":          _run_meta(),
        "per_seed":      per_seed,
    }

    tag = pooling_cfg["method"] + (f"_dc{pooling_cfg.get('dc')}" if pooling_cfg.get("dc") else "")
    out_path = output_dir / f"{config_stem}_{tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    log.info("→ f1_micro=%.4f ± %.4f  saved to %s", mean, std, out_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dc", type=int, nargs="+", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results/runs"))
    parser.add_argument("--label-sep", default="|",
                        help="Separator for multi-label column (default: |)")
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)

    dc_values: list[int | None] = args.dc if args.dc else [None]
    stem = f"{args.config.parent.name}_{args.config.stem}"
    summaries = [run_one(cfg, dc, args.output_dir, stem) for dc in dc_values]

    if len(summaries) > 1:
        sweep_path = args.output_dir / f"{stem}_sweep.json"
        with open(sweep_path, "w") as fh:
            json.dump(summaries, fh, indent=2)
        log.info("Sweep results → %s", sweep_path)


if __name__ == "__main__":
    main()
