"""
plot_results.py  –  DeepLoc + Meltome pooling benchmark figures
----------------------------------------------------------------
Generates five publication-ready PNG figures from the DeepLoc
subcellular-localisation and Meltome thermostability results.

Usage:
    python plot_results.py [--data DIR] [--out DIR]

    --data  folder containing the JSON result files (default: same folder as script)
    --out   folder where PNGs are saved               (default: same folder as script)

Figures produced:
    DeepLoc
        fig_deeploc_bar.png       – bar chart, accuracy ± std (3 seeds)
        fig_deeploc_training.png  – validation accuracy over epochs, mean ± std
        fig_deeploc_per_class.png – per-class accuracy, mean vs cov supervised

    Meltome
        fig_meltome_bar.png       – bar chart, Spearman R ± std (3 seeds)
        fig_meltome_training.png  – validation Spearman R over epochs, mean ± std
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# ---------------------------------------------------------------------------
# File map  –  edit here when new result files arrive
# ---------------------------------------------------------------------------

DEEPLOC_FILES = {
    "mean":             "scl_mean_mean.json",
    "cov_supervised":   "scl_cov_supervised_cov_supervised_dc32.json",
    "cov_unsupervised": "scl_cov_unsupervised_cov_unsupervised_dc32.json",
    "cov_pca":          "scl_cov_pca_cov_pca_dc32.json",
    "hybrid":           "scl_hybrid_hybrid_dc32.json",
}

MELTOME_FILES = {
    "mean":             "meltome_mean_mean.json",
    "cov_supervised":   "meltome_cov_supervised_cov_supervised_dc32.json",
    "cov_unsupervised": "meltome_cov_unsupervised_cov_unsupervised_dc32.json",
    "cov_pca":          "meltome_cov_pca_cov_pca_dc32.json",
}

# ---------------------------------------------------------------------------
# Display config
# ---------------------------------------------------------------------------

LABELS = {
    "mean":             "Mean",
    "cov_supervised":   "Cov\nsupervised",
    "cov_unsupervised": "Cov\nunsupervised",
    "cov_pca":          "Cov\nPCA",
    "hybrid":           "Hybrid\n[µ; C]",
}

LABELS_FLAT = {k: v.replace("\n", " ") for k, v in LABELS.items()}

COLORS = {
    "mean":             "#888780",
    "cov_supervised":   "#378ADD",
    "cov_unsupervised": "#85B7EB",
    "cov_pca":          "#D3D1C7",
    "hybrid":           "#1D9E75",
}

EDGE_COLORS = {
    "mean":             "#5F5E5A",
    "cov_supervised":   "#185FA5",
    "cov_unsupervised": "#378ADD",
    "cov_pca":          "#888780",
    "hybrid":           "#0F6E56",
}

DASHES = {
    "mean":             [],
    "cov_supervised":   [],
    "cov_unsupervised": [4, 2],
    "cov_pca":          [2, 2],
    "hybrid":           [],
}

CLASS_NAMES = [
    "Cell membrane", "Cytoplasm", "ER", "Extracellular",
    "Golgi", "Lysosome/Vac.", "Mitochondrion", "Nucleus",
    "Peroxisome", "Plastid",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load(data_dir: Path, file_map: dict) -> dict:
    return {k: json.load(open(data_dir / v)) for k, v in file_map.items()}


def metric_key(data: dict) -> str:
    return "spearman_r" if data["task"] == "regression" else "accuracy"


def seed_values(data: dict) -> np.ndarray:
    k = metric_key(data)
    return np.array([s[k] for s in data["per_seed"]])


def avg_history(data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Mean ± std validation curves, truncated to shortest seed. Returns raw values."""
    k = metric_key(data)
    curves = [np.array([ep[k] for ep in s["history"]]) for s in data["per_seed"]]
    min_len = min(len(c) for c in curves)
    mat = np.stack([c[:min_len] for c in curves])
    return mat.mean(axis=0), mat.std(axis=0)


def pooled_predictions(data: dict) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.concatenate([s["y_true"] for s in data["per_seed"]])
    y_pred = np.concatenate([s["y_pred"] for s in data["per_seed"]])
    return y_true, y_pred


def per_class_accuracy(y_true, y_pred, n_classes=10) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    with np.errstate(divide="ignore", invalid="ignore"):
        acc = np.where(cm.sum(axis=1) > 0,
                       cm.diagonal() / cm.sum(axis=1), np.nan)
    return acc * 100


# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "xtick.major.size":  0,
    "ytick.major.size":  3,
    "figure.dpi":        150,
})


def _bar_fig(datasets, ylabel, ylim, metric_fmt, title, scale=1.0, ytick_fmt=None):
    methods = list(datasets.keys())
    vals = np.array([seed_values(datasets[m]).mean() for m in methods]) * scale
    stds = np.array([seed_values(datasets[m]).std()  for m in methods]) * scale
    x = np.arange(len(methods))

    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    ax.bar(x, vals, width=0.55,
           color=[COLORS[m] for m in methods],
           edgecolor=[EDGE_COLORS[m] for m in methods],
           linewidth=0.8, zorder=3)

    ax.errorbar(x, vals, yerr=stds,
                fmt="none", color="black", capsize=5,
                capthick=1.2, elinewidth=1.2, zorder=4)

    for xi, (v, s) in enumerate(zip(vals, stds)):
        ax.text(xi, v + s + (ylim[1] - ylim[0]) * 0.012,
                metric_fmt.format(v),
                ha="center", va="bottom", fontsize=9.5)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in methods], fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(*ylim)
    ax.yaxis.grid(True, color="#E8E8E8", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=12, pad=10)
    if ytick_fmt:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(ytick_fmt))

    fig.tight_layout()
    return fig


def _curve_fig(datasets, ylabel, title, scale=1.0):
    curves = {m: avg_history(datasets[m]) for m in datasets}
    min_epochs = min(len(mu) for mu, _ in curves.values())

    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    for m, (mu, sd) in curves.items():
        mu, sd = mu[:min_epochs] * scale, sd[:min_epochs] * scale
        ep = np.arange(1, min_epochs + 1)
        ls = "--" if DASHES[m] else "-"
        ax.plot(ep, mu, color=COLORS[m], linewidth=2.0,
                linestyle=ls, label=LABELS_FLAT[m], zorder=3)
        ax.fill_between(ep, mu - sd, mu + sd,
                        color=COLORS[m], alpha=0.15, zorder=2)

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)
    ax.yaxis.grid(True, color="#E8E8E8", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9.5, frameon=False, loc="lower right")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# DeepLoc figures
# ---------------------------------------------------------------------------

def fig_deeploc_bar(deeploc: dict, out: Path):
    fig = _bar_fig(deeploc,
                   ylabel="Accuracy (%)",
                   ylim=(77, 88),
                   metric_fmt="{:.1f}%",
                   title="Subcellular localisation  ·  DeepLoc  ·  dc = 32",
                   scale=100)
    path = out / "fig_deeploc_bar.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def fig_deeploc_training(deeploc: dict, out: Path):
    fig = _curve_fig(deeploc,
                     ylabel="Validation accuracy (%)",
                     title="DeepLoc training dynamics  ·  mean ± 1 std, 3 seeds",
                     scale=100)
    path = out / "fig_deeploc_training.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def fig_deeploc_per_class(deeploc: dict, out: Path):
    compare = ["mean", "cov_supervised"]
    n = len(CLASS_NAMES)
    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(9.5, 4.5))

    for i, m in enumerate(compare):
        yt, yp = pooled_predictions(deeploc[m])
        acc = per_class_accuracy(yt, yp)
        offset = (i - 0.5) * width
        ax.bar(x + offset, acc, width,
               color=COLORS[m], alpha=0.9,
               label=LABELS_FLAT[m],
               edgecolor="white", linewidth=0.5, zorder=3)

    yt_m, yp_m = pooled_predictions(deeploc["mean"])
    yt_s, yp_s = pooled_predictions(deeploc["cov_supervised"])
    delta = per_class_accuracy(yt_s, yp_s) - per_class_accuracy(yt_m, yp_m)

    ax2 = ax.twinx()
    ax2.plot(x, delta, "o-", color="#D85A30", linewidth=1.4,
             markersize=5, zorder=5, label="Δ (cov supervised − mean)")
    ax2.axhline(0, color="#D85A30", linewidth=0.7, linestyle="--", alpha=0.5)
    ax2.set_ylabel("Δ accuracy (pp)", fontsize=10, color="#D85A30")
    ax2.tick_params(axis="y", colors="#D85A30", labelsize=9)
    ax2.spines["right"].set_color("#D85A30")
    ax2.spines["right"].set_linewidth(0.7)
    ax2.spines["top"].set_visible(False)

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=35, ha="right", fontsize=9.5)
    ax.set_ylabel("Per-class accuracy (%)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, color="#E8E8E8", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("Per-class accuracy  ·  DeepLoc  ·  pooled across 3 seeds",
                 fontsize=12, pad=10)

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2,
              fontsize=9.5, frameon=False, ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.28))

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)
    path = out / "fig_deeploc_per_class.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Meltome figures
# ---------------------------------------------------------------------------

def fig_meltome_bar(meltome: dict, out: Path):
    fig = _bar_fig(meltome,
                   ylabel="Spearman R",
                   ylim=(0.60, 0.73),
                   metric_fmt="{:.3f}",
                   title="Thermostability  ·  Meltome  ·  dc = 32",
                   ytick_fmt=lambda v, _: f"{v:.2f}")
    path = out / "fig_meltome_bar.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def fig_meltome_training(meltome: dict, out: Path):
    fig = _curve_fig(meltome,
                     ylabel="Validation Spearman R",
                     title="Meltome training dynamics  ·  mean ± 1 std, 3 seeds")
    path = out / "fig_meltome_training.png"
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
    deeploc = load(data_dir, DEEPLOC_FILES)
    meltome = load(data_dir, MELTOME_FILES)

    print("Generating figures...")

    # --- DeepLoc ---
    fig_deeploc_bar(deeploc, out_dir)
    fig_deeploc_training(deeploc, out_dir)
    fig_deeploc_per_class(deeploc, out_dir)

    # --- Meltome ---
    fig_meltome_bar(meltome, out_dir)
    fig_meltome_training(meltome, out_dir)

    print("\nDone. Five figures saved to:", out_dir)


if __name__ == "__main__":
    main()
