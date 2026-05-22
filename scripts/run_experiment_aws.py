#!/usr/bin/env python
"""AWS / GPU runner — self-contained twin of scripts/run_experiment.py.

Why this exists
---------------
scripts/run_experiment.py and src/sop/probes/train_loop.py currently have
unresolved merge conflicts. This file is an independent path: it inlines its
own training loop and CUDA-tuned DataLoader config, so the AWS box can produce
results without depending on the conflicted modules.

Tuned for a single NVIDIA T4 (Turing — fp16 OK, bf16 NOT supported natively):
    * AMP enabled with fp16 + GradScaler
    * pin_memory, num_workers=2, persistent_workers, prefetch_factor=2

JSON output format matches scripts/run_experiment.py so downstream analysis
notebooks keep working unchanged.

Usage
-----
    python scripts/run_experiment_aws.py --config configs/scl/mean.yaml
    python scripts/run_experiment_aws.py --config configs/scl/cov_supervised.yaml --dc 8 16 24 32 48
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from functools import partial
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import Adam
from torch.utils.data import DataLoader

from sop.pooling.base import Pooler
from sop.pooling.covariance import CovariancePooler
from sop.pooling.covariance_pca import CovariancePCAPooler
from sop.pooling.hybrid import HybridPooler
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

    if method == "mean":
        return MeanPooler(d)
    if method == "cov_supervised":
        return CovariancePooler(d, pooling_cfg["dc"])
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
    cfg: dict, task: str, device: str,
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

    use_cuda = device.startswith("cuda")
    num_workers = 2 if use_cuda else 0
    loader_kwargs: dict[str, Any] = dict(
        collate_fn=partial(collate_pad, label_to_index=label_to_index),
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    batch_size = cfg["probe"].get("batch_size", 16)
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
def evaluate(model: PoolingProbeModel, loader: DataLoader, task: str, device: str) -> float:
    _, metric_fn, _ = _loss_and_metric(task)
    use_amp = device.startswith("cuda")
    model.eval()
    preds: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    for X, mask, y in loader:
        X = X.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            out = model(X, mask)
        if task == "classification":
            p = out.argmax(dim=-1).detach().cpu().numpy()
        else:
            p = out.squeeze(-1).float().detach().cpu().numpy()
        preds.append(p)
        truths.append(y.numpy())
    return metric_fn(np.concatenate(truths), np.concatenate(preds))


def train_probe(
    model: PoolingProbeModel,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    *,
    task: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: str,
    log_fn: Callable[[str], None],
) -> dict:
    loss_fn, _, metric_key = _loss_and_metric(task)
    model.to(device)

    use_amp = device.startswith("cuda")
    # T4 is Turing → use fp16 (bf16 not supported natively).
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = Adam(trainable, lr=lr, weight_decay=weight_decay)

    history: list[dict] = []
    best_metric = -float("inf")
    best_epoch = -1

    for epoch in range(epochs):
        model.train()
        running = 0.0
        n_batches = 0
        for X, mask, y in train_loader:
            X = X.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                out = model(X, mask)
                if task == "regression":
                    out = out.squeeze(-1)
                    target = y.float()
                else:
                    target = y
                loss = loss_fn(out, target)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += float(loss.detach())
            n_batches += 1

        train_loss = running / max(n_batches, 1)
        val_metric = float("nan")
        if val_loader is not None:
            val_metric = evaluate(model, val_loader, task, device)
            if val_metric > best_metric:
                best_metric = val_metric
                best_epoch = epoch

        history.append({"epoch": epoch, "train_loss": train_loss, metric_key: val_metric})
        log_fn(f"epoch {epoch:3d}  train_loss={train_loss:.4f}  val_{metric_key}={val_metric:.4f}")

    return {
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "metric_key": metric_key,
        "history": history,
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
    device = probe_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, _, n_classes = make_loaders(cfg, task, device)

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
            log_fn=lambda s: log.info("    " + s),
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
    summaries = [run_one(cfg, dc, args.output_dir, args.config.stem) for dc in dc_values]

    if len(summaries) > 1:
        sweep_path = args.output_dir / f"{args.config.stem}_sweep.json"
        with open(sweep_path, "w") as fh:
            json.dump(summaries, fh, indent=2)
        log.info("Sweep results → %s", sweep_path)


if __name__ == "__main__":
    main()
