"""
plot_results.py  –  SCL pooling benchmark figures
---------------------------------------------------
Generates three publication-ready PNG figures from the DeepLoc
subcellular-localization benchmark results (dc=32, ProtX backbone).

Usage:
    python plot_results.py [--data DIR] [--out DIR]

    --data  folder containing the five JSON result files (default: same folder as script)
    --out   folder where PNGs are saved               (default: same folder as script)

Figures produced:
    fig1_main_results.png     – bar chart, all 5 methods, error bars (±1 std, 3 seeds)
    fig2_training_curves.png  – validation accuracy over epochs, mean ± std shaded
    fig3_per_class.png        – per-class accuracy, mean pooling vs cov supervised
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.metrics import confusion_matrix

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FILE_MAP = {
    "mean":             "scl_mean_mean.json",
    "cov_supervised":   "scl_cov_supervised_cov_supervised_dc32.json",
    "cov_unsupervised": "scl_cov_unsupervised_cov_unsupervised_dc32.json",
    "cov_pca":          "scl_cov_pca_cov_pca_dc32.json",
    "hybrid":           "scl_hybrid_hybrid_dc32.json",
}

LABELS = {
    "mean":             "Mean\n(baseline)",
    "cov_supervised":   "Cov\nsupervised",
    "cov_unsupervised": "Cov\nunsupervised",
    "cov_pca":          "Cov\nPCA",
    "hybrid":           "Hybrid\n[µ; C]",
}

COLORS = {
    "mean":             "#888780",
    "cov_supervised":   "#378ADD",
    "cov_unsupervised": "#85B7EB",
    "cov_pca":          "#D3D1C7",
    "hybrid":           "#1D9E75",
}

CLASS_NAMES = [
    "Cell membrane",
    "Cytoplasm",
    "ER",
    "Extracellular",
    "Golgi",
    "Lysosome/Vac.",
    "Mitochondrion",
    "Nucleus",
    "Peroxisome",
    "Plastid",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load(data_dir: Path, key: str) -> dict:
    path = data_dir / FILE_MAP[key]
    with open(path) as f:
        return json.load(f)


def seed_accuracies(data: dict) -> np.ndarray:
    return np.array([s["accuracy"] for s in data["per_seed"]])


def avg_history(data: dict, field: str = "accuracy") -> tuple[np.ndarray, np.ndarray]:
    """Return (mean_curve, std_curve) averaged over seeds, truncated to shortest."""
    curves = [np.array([ep[field] for ep in s["history"]]) for s in data["per_seed"]]
    min_len = min(len(c) for c in curves)
    mat = np.stack([c[:min_len] for c in curves])          # (n_seeds, epochs)
    return mat.mean(axis=0) * 100, mat.std(axis=0) * 100


def pooled_predictions(data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate y_true / y_pred across all seeds."""
    y_true = np.concatenate([s["y_true"] for s in data["per_seed"]])
    y_pred = np.concatenate([s["y_pred"] for s in data["per_seed"]])
    return y_true, y_pred


def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 10) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    with np.errstate(divide="ignore", invalid="ignore"):
        acc = np.where(cm.sum(axis=1) > 0,
                       cm.diagonal() / cm.sum(axis=1),
                       np.nan)
    return acc * 100


# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.size":          11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.8,
    "xtick.major.size":   0,
    "ytick.major.size":   3,
    "figure.dpi":         150,
})


# ---------------------------------------------------------------------------
# Figure 1 – Main results bar chart
# ---------------------------------------------------------------------------

def fig1_main_results(datasets: dict, out: Path):
    methods = list(FILE_MAP.keys())
    means = np.array([seed_accuracies(datasets[m]).mean() for m in methods]) * 100
    stds  = np.array([seed_accuracies(datasets[m]).std()  for m in methods]) * 100
    bar_colors = [COLORS[m] for m in methods]
    x = np.arange(len(methods))

    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    bars = ax.bar(x, means, width=0.55,
                  color=bar_colors,
                  edgecolor=[c for c in ["#5F5E5A","#185FA5","#378ADD","#888780","#0F6E56"]],
                  linewidth=0.8,
                  zorder=3)

    ax.errorbar(x, means, yerr=stds,
                fmt="none", color="black", capsize=5, capthick=1.2,
                elinewidth=1.2, zorder=4)

    # Baseline reference line
    baseline = means[0]
    ax.axhline(baseline, color=COLORS["mean"], linewidth=1.0,
               linestyle="--", alpha=0.6, zorder=2)

    # Value labels on bars
    for xi, (m, s) in enumerate(zip(means, stds)):
        ax.text(xi, m + s + 0.25, f"{m:.1f}%",
                ha="center", va="bottom", fontsize=9.5, color="#2C2C2A")

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in methods], fontsize=10)
    ax.set_ylabel("Test accuracy (%)", fontsize=11)
    ax.set_ylim(77, 88)
    ax.yaxis.grid(True, color="#E8E8E8", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("Subcellular localisation accuracy  ·  DeepLoc  ·  dc = 32",
                 fontsize=12, pad=10)

    # Embedding-size annotation under hybrid bar
    ax.text(4, 77.2, "2048-dim", ha="center", fontsize=8.5,
            color="#5F5E5A", style="italic")
    ax.text(0, 77.2, "1024-dim", ha="center", fontsize=8.5,
            color="#5F5E5A", style="italic")

    fig.tight_layout()
    path = out / "fig1_main_results.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Figure 2 – Training curves
# ---------------------------------------------------------------------------

def fig2_training_curves(datasets: dict, out: Path):
    show = ["mean", "cov_supervised", "cov_unsupervised", "hybrid"]
    nice = {
        "mean":             "Mean (baseline)",
        "cov_supervised":   "Cov supervised",
        "cov_unsupervised": "Cov unsupervised",
        "hybrid":           "Hybrid [µ; C]",
    }
    dashes = {
        "mean":             (None, None),
        "cov_supervised":   (None, None),
        "cov_unsupervised": (4, 2),
        "hybrid":           (None, None),
    }

    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    # Truncate all methods to the same number of epochs (shortest method)
    curves = {m: avg_history(datasets[m]) for m in show}
    min_epochs = min(len(mu) for mu, _ in curves.values())

    for m in show:
        mu, sd = curves[m]
        mu, sd = mu[:min_epochs], sd[:min_epochs]
        ep = np.arange(1, min_epochs + 1)
        ls = "--" if dashes[m][0] else "-"
        ax.plot(ep, mu, color=COLORS[m], linewidth=2.0,
                linestyle=ls, label=nice[m], zorder=3)
        ax.fill_between(ep, mu - sd, mu + sd,
                        color=COLORS[m], alpha=0.15, zorder=2)

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Validation accuracy (%)", fontsize=11)
    ax.set_title("Training dynamics  ·  mean ± 1 std across 3 seeds",
                 fontsize=12, pad=10)
    ax.yaxis.grid(True, color="#E8E8E8", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9.5, frameon=False, loc="lower right")

    fig.tight_layout()
    path = out / "fig2_training_curves.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Figure 3 – Per-class accuracy
# ---------------------------------------------------------------------------

def fig3_per_class(datasets: dict, out: Path):
    compare = ["mean", "cov_supervised"]
    n = len(CLASS_NAMES)
    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(9.5, 4.5))

    for i, m in enumerate(compare):
        yt, yp = pooled_predictions(datasets[m])
        acc = per_class_accuracy(yt, yp)
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, acc, width,
                      color=COLORS[m], alpha=0.9,
                      label=LABELS[m].replace("\n", " "),
                      edgecolor="white", linewidth=0.5, zorder=3)

    # Improvement arrows / difference line
    yt_m, yp_m = pooled_predictions(datasets["mean"])
    yt_s, yp_s = pooled_predictions(datasets["cov_supervised"])
    acc_mean = per_class_accuracy(yt_m, yp_m)
    acc_sup  = per_class_accuracy(yt_s, yp_s)
    delta = acc_sup - acc_mean

    ax2 = ax.twinx()
    ax2.plot(x, delta, "o-", color="#D85A30", linewidth=1.4,
             markersize=5, zorder=5, label="Δ (supervised − mean)")
    ax2.axhline(0, color="#D85A30", linewidth=0.7, linestyle="--", alpha=0.5)
    ax2.set_ylabel("Δ accuracy (pp)", fontsize=10, color="#D85A30")
    ax2.tick_params(axis="y", colors="#D85A30", labelsize=9)
    ax2.spines["right"].set_color("#D85A30")
    ax2.spines["right"].set_linewidth(0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=35, ha="right", fontsize=9.5)
    ax.set_ylabel("Per-class accuracy (%)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, color="#E8E8E8", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("Per-class accuracy  ·  mean vs cov supervised  ·  pooled across 3 seeds",
                 fontsize=12, pad=10)

    # Combined legend – placed below the plot
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2,
              fontsize=9.5, frameon=False, ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.28))

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)
    path = out / "fig3_per_class.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(Path(__file__).parent),
                        help="Folder containing the JSON result files")
    parser.add_argument("--out",  default=str(Path(__file__).parent),
                        help="Output folder for PNG figures")
    args = parser.parse_args()

    data_dir = Path(args.data)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    datasets = {k: load(data_dir, k) for k in FILE_MAP}

    print("Generating figures...")
    fig1_main_results(datasets, out_dir)
    fig2_training_curves(datasets, out_dir)
    fig3_per_class(datasets, out_dir)

    print("\nDone. Three figures saved to:", out_dir)


if __name__ == "__main__":
    main()
