#!/usr/bin/env python
"""Read result JSONs from results/runs/ and produce diagnostic plots.

For classification configs (task=classification):
    * Confusion matrix per seed + aggregated across seeds
    * Per-class precision / recall / F1

For regression configs (task=regression):
    * Scatter of y_true vs y_pred per seed
    * Residual histogram

Also writes a per-config learning-curve plot (train_loss + val_metric over
epochs) from the saved `history`.

Usage
-----
    python scripts/analyze_results.py                          # all JSONs in results/runs/
    python scripts/analyze_results.py --runs-dir results/runs  # explicit dir
    python scripts/analyze_results.py --file results/runs/scl_mean_mean.json  # one file
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)


def _load(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _class_names(label_to_index: dict | None) -> list[str] | None:
    if not label_to_index:
        return None
    return [name for name, _ in sorted(label_to_index.items(), key=lambda kv: kv[1])]


def plot_confusion(summary: dict, out_dir: Path) -> None:
    """Aggregated + per-seed confusion matrices for one classification run."""
    name = f"{summary['config']}_{summary['method']}"
    if summary.get("dc"):
        name += f"_dc{summary['dc']}"

    classes = _class_names(summary.get("label_to_index"))
    per_seed = summary["per_seed"]

    # Aggregated across seeds
    y_true_all = np.concatenate([np.array(s["y_true"]) for s in per_seed if s["y_true"]])
    y_pred_all = np.concatenate([np.array(s["y_pred"]) for s in per_seed if s["y_pred"]])
    cm = confusion_matrix(y_true_all, y_pred_all)
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(
        ax=ax, xticks_rotation=45, colorbar=True, values_format="d"
    )
    ax.set_title(f"{name}  —  aggregated over {len(per_seed)} seeds")
    plt.tight_layout()
    fig.savefig(out_dir / f"{name}_confusion.png", dpi=140)
    plt.close(fig)

    # Per-seed grid
    n = len(per_seed)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.5), squeeze=False)
    for ax, seed_data in zip(axes[0], per_seed):
        if not seed_data.get("y_true"):
            continue
        cm = confusion_matrix(seed_data["y_true"], seed_data["y_pred"])
        ConfusionMatrixDisplay(cm, display_labels=classes).plot(
            ax=ax, xticks_rotation=45, colorbar=False, values_format="d"
        )
        acc = seed_data.get("accuracy", float("nan"))
        ax.set_title(f"seed={seed_data['seed']}  acc={acc:.4f}")
    plt.tight_layout()
    fig.savefig(out_dir / f"{name}_confusion_per_seed.png", dpi=120)
    plt.close(fig)

    # Per-class report (printed + saved to .txt)
    report = classification_report(
        y_true_all, y_pred_all, target_names=classes, digits=3, zero_division=0
    )
    print(f"\n=== {name} per-class metrics (aggregated) ===\n{report}")
    (out_dir / f"{name}_classification_report.txt").write_text(report)


def plot_regression(summary: dict, out_dir: Path) -> None:
    name = f"{summary['config']}_{summary['method']}"
    if summary.get("dc"):
        name += f"_dc{summary['dc']}"

    per_seed = summary["per_seed"]
    n = len(per_seed)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 9), squeeze=False)
    for col, seed_data in enumerate(per_seed):
        if not seed_data.get("y_true"):
            continue
        y_true = np.array(seed_data["y_true"])
        y_pred = np.array(seed_data["y_pred"])
        # Scatter
        ax = axes[0, col]
        ax.scatter(y_true, y_pred, s=4, alpha=0.3)
        lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel("y_true")
        ax.set_ylabel("y_pred")
        rho = seed_data.get("spearman_r", float("nan"))
        ax.set_title(f"seed={seed_data['seed']}  ρ={rho:.4f}")
        # Residuals
        ax = axes[1, col]
        ax.hist(y_pred - y_true, bins=50)
        ax.set_xlabel("residual (y_pred − y_true)")
    fig.suptitle(name)
    plt.tight_layout()
    fig.savefig(out_dir / f"{name}_regression.png", dpi=130)
    plt.close(fig)


def plot_history(summary: dict, out_dir: Path) -> None:
    name = f"{summary['config']}_{summary['method']}"
    if summary.get("dc"):
        name += f"_dc{summary['dc']}"
    metric_key = "accuracy" if summary["task"] == "classification" else "spearman_r"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for s in summary["per_seed"]:
        hist = s.get("history") or []
        if not hist:
            continue
        epochs = [h["epoch"] for h in hist]
        axes[0].plot(epochs, [h["train_loss"] for h in hist], label=f"seed={s['seed']}")
        axes[1].plot(epochs, [h[metric_key] for h in hist], label=f"seed={s['seed']}")
    axes[0].set(xlabel="epoch", ylabel="train_loss", title="train loss")
    axes[1].set(xlabel="epoch", ylabel=f"val_{metric_key}", title=f"val {metric_key}")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(name)
    plt.tight_layout()
    fig.savefig(out_dir / f"{name}_history.png", dpi=130)
    plt.close(fig)


def analyze_file(path: Path, out_dir: Path) -> None:
    summary = _load(path)
    if "per_seed" not in summary:
        print(f"skip: {path.name} has no per_seed")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    have_predictions = any(s.get("y_true") for s in summary["per_seed"])

    if summary["task"] == "classification" and have_predictions:
        plot_confusion(summary, out_dir)
    elif summary["task"] == "regression" and have_predictions:
        plot_regression(summary, out_dir)
    else:
        print(f"  no y_true/y_pred in {path.name} — confusion matrix skipped "
              "(re-run with the updated script to capture predictions)")

    have_history = any(s.get("history") for s in summary["per_seed"])
    if have_history:
        plot_history(summary, out_dir)
    else:
        print(f"  no history in {path.name} — learning curves skipped")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-dir", type=Path, default=Path("results/runs"))
    p.add_argument("--file", type=Path, help="Analyze a single JSON instead of the whole dir.")
    p.add_argument("--out", type=Path, default=Path("results/figures"))
    args = p.parse_args()

    files = [args.file] if args.file else sorted(args.runs_dir.glob("*.json"))
    if not files:
        print(f"No JSON files found in {args.runs_dir}")
        return

    for f in files:
        print(f"→ {f}")
        analyze_file(f, args.out)
    print(f"\nFigures saved under {args.out}/")


if __name__ == "__main__":
    main()
