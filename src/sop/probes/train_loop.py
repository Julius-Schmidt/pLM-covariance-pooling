"""Generic training + evaluation loop for the pooler+probe model.

Single entry point used by every pooling method (mean, supervised covariance,
unsupervised/frozen covariance, hybrid). Branches only on task type
(classification vs regression).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from .metrics import accuracy, spearman_r
from .model import PoolingProbeModel


def _loss_and_metric(task: str) -> tuple[nn.Module, Callable[[np.ndarray, np.ndarray], float], str]:
    if task == "classification":
        return nn.CrossEntropyLoss(), accuracy, "accuracy"
    if task == "regression":
        return nn.MSELoss(), spearman_r, "spearman_r"
    raise ValueError(f"Unknown task '{task}'. Choose 'classification' or 'regression'.")


@torch.no_grad()
def evaluate(
    model: PoolingProbeModel,
    loader: DataLoader,
    task: str,
    device: str,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run the model over loader and return (metric, y_true, y_pred)."""
    _, metric_fn, _ = _loss_and_metric(task)
    model.eval()
    device_type = device.split(":")[0]
    use_amp = device_type == "cuda"
    preds: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    for X, mask, y in loader:
        X = X.to(device)
        mask = mask.to(device)
        with torch.autocast(device_type=device_type, enabled=use_amp):
            out = model(X, mask)
        if task == "classification":
            p = out.argmax(dim=-1).cpu().numpy()
        else:
            p = out.squeeze(-1).cpu().numpy()
        preds.append(p)
        truths.append(y.numpy())
    y_true = np.concatenate(truths)
    y_pred = np.concatenate(preds)
    return metric_fn(y_true, y_pred), y_true, y_pred


def train_probe(
    model: PoolingProbeModel,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    *,
    task: str,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: str = "cpu",
    log: Callable[[str], None] = print,
) -> dict:
    """Train the pooler+probe model and report best validation metric.

    Only parameters with ``requires_grad=True`` are optimised — so frozen
    pooler weights stay frozen and supervised pooler weights get trained
    alongside the probe.

    Returns:
        dict with keys: best_metric, best_epoch, history (list of per-epoch
        train_loss + val_metric).
    """
    loss_fn, _, metric_key = _loss_and_metric(task)
    model.to(device)

    device_type = device.split(":")[0]
    use_amp = device_type == "cuda"
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
            X = X.to(device)
            mask = mask.to(device)
            y = y.to(device)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device_type, enabled=use_amp):
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
            val_metric, _, _ = evaluate(model, val_loader, task, device)
            if val_metric > best_metric:
                best_metric = val_metric
                best_epoch = epoch

        history.append({"epoch": epoch, "train_loss": train_loss, metric_key: val_metric})
        log(f"epoch {epoch:3d}  train_loss={train_loss:.4f}  val_{metric_key}={val_metric:.4f}")

    return {
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "metric_key": metric_key,
        "history": history,
    }
