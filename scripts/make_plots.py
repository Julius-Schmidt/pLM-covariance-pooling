"""Generate the full visualisation suite for a results directory.

Walks a directory of per-run JSONs (and optional ``*_sweep.json``) and writes
PNGs under ``results/visualization/<task>/``. Designed for the
"detailed run 24 05 2026" layout but generic over any aggregated runs that
follow the same schema.

Run:
    python scripts/make_plots.py \
        --results "results/detailed run 24 05 2026" \
        --out results/visualization
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


METHOD_ORDER = ["mean", "cov_supervised", "cov_unsupervised", "cov_pca", "hybrid",
                "light_attention"]
METHOD_LABEL = {
    "mean":             "Mean",
    "cov_supervised":   "Cov supervised",
    "cov_unsupervised": "Cov unsupervised",
    "cov_pca":          "Cov PCA",
    "hybrid":           "Hybrid [μ; C]",
    "light_attention":  "Light Attention",
}
METHOD_COLOR = {
    "mean":             "#6f6f6f",
    "cov_supervised":   "#3a78c2",
    "cov_unsupervised": "#92b8e6",
    "cov_pca":          "#c8c2b0",
    "hybrid":           "#2ea27a",
    "light_attention":  "#7ECBA1",
}
METHOD_LS = {
    "mean":             "-",
    "cov_supervised":   "-",
    "cov_unsupervised": "--",
    "cov_pca":          "--",
    "hybrid":           "-",
    "light_attention":  "-",
}

DEEPLOC_SHORT = {
    "Cell membrane": "Cell membrane",
    "Cytoplasm": "Cytoplasm",
    "Endoplasmic reticulum": "ER",
    "Extracellular": "Extracellular",
    "Golgi apparatus": "Golgi",
    "Lysosome/Vacuole": "Lysosome/Vac.",
    "Mitochondrion": "Mitochondrion",
    "Nucleus": "Nucleus",
    "Peroxisome": "Peroxisome",
    "Plastid": "Plastid",
}

# Frequency-sorted order matching the DeepLoc training-set class distribution
# (n = 11,566). Used for per-class plots so the eye reads "head → tail".
DEEPLOC_CLASS_ORDER = [
    "Nucleus",
    "Cytoplasm",
    "Extracellular",
    "Mitochondrion",
    "Cell membrane",
    "Endoplasmic reticulum",
    "Plastid",
    "Golgi apparatus",
    "Lysosome/Vacuole",
    "Peroxisome",
]

plt.rcParams.update({
    "figure.dpi":        120,
    "savefig.dpi":       150,
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "axes.axisbelow":    True,
    "axes.grid":         True,
    "grid.color":        "#CCCCCC",
    "grid.linewidth":    0.7,
    "grid.linestyle":    "-",
    "xtick.major.size":  0,
    "ytick.major.size":  3,
})


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_runs(results_dir: Path) -> list[dict]:
    """Flatten every JSON in ``results_dir`` (including sweep files) into per-run dicts."""
    runs: list[dict] = []
    for path in sorted(results_dir.glob("*.json")):
        with open(path) as fh:
            payload = json.load(fh)
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            runs.append(item)
    # Deduplicate (sweep file may overlap with individual dc files): keep latest by file mtime.
    seen: dict[tuple, dict] = {}
    for r in runs:
        key = (r.get("config"), r["method"], r.get("dc"), r["task"])
        seen[key] = r
    return list(seen.values())


def split_by_task(runs: list[dict]) -> tuple[list[dict], list[dict]]:
    cls = [r for r in runs if r["task"] == "classification"]
    reg = [r for r in runs if r["task"] == "regression"]
    return cls, reg


def metric_key(task: str) -> str:
    return "accuracy" if task == "classification" else "spearman_r"


# ---------------------------------------------------------------------------
# Plot 1: main results bar chart per task @ dc=32
# ---------------------------------------------------------------------------

def plot_main_bar(runs: list[dict], task: str, out_path: Path, dc: int = 32) -> None:
    mkey = metric_key(task)
    rows: list[tuple[str, float, float]] = []
    for m in METHOD_ORDER:
        if m == "mean":
            cand = [r for r in runs if r["method"] == "mean"]
        elif m == "light_attention":
            cand = [r for r in runs if r["method"] == "light_attention"]
        else:
            cand = [r for r in runs if r["method"] == m and r.get("dc") == dc]
        if not cand:
            continue
        r = cand[0]
        rows.append((m, r[f"{mkey}_mean"], r[f"{mkey}_std"]))

    if not rows:
        return

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    xs = np.arange(len(rows))
    means = [r[1] for r in rows]
    stds = [r[2] for r in rows]
    colors = [METHOD_COLOR[r[0]] for r in rows]
    bars = ax.bar(xs, means, yerr=stds, capsize=4, color=colors,
                  edgecolor="none", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([METHOD_LABEL[r[0]].replace(" ", "\n", 1) for r in rows])
    if task == "classification":
        ax.set_ylabel("Accuracy (%)")
        for x, m, s in zip(xs, means, stds):
            ax.text(x, m + s + 0.003, f"{m*100:.1f}%", ha="center", fontsize=10)
        # Add headroom
        lo = min(means) - 0.04
        hi = max(means) + 0.03
        ax.set_ylim(lo, hi)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}"))
        ax.set_title(f"Subcellular localisation  ·  DeepLoc  ·  dc = {dc}")
    else:
        ax.set_ylabel("Spearman R")
        for x, m, s in zip(xs, means, stds):
            ax.text(x, m + s + 0.003, f"{m:.3f}", ha="center", fontsize=10)
        lo = min(means) - 0.05
        hi = max(means) + 0.03
        ax.set_ylim(lo, hi)
        ax.set_title(f"Thermostability  ·  Meltome  ·  dc = {dc}")
    ax.yaxis.grid(True, color="#CCCCCC", linewidth=0.7, zorder=0)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: dc-sweep efficiency curve (cov_supervised) with mean baseline
# ---------------------------------------------------------------------------

def plot_dc_sweep(runs: list[dict], task: str, out_path: Path,
                  la_run: dict | None = None) -> None:
    mkey = metric_key(task)
    # Every covariance method that was swept gets a full line; a method present
    # at only one dc falls back to a single marker. Annotate embedding dims on
    # cov_supervised only (the others would clutter the panel).
    cov_methods = ["cov_supervised", "cov_unsupervised", "cov_pca", "hybrid", "light_attention"]
    series: dict[str, list[dict]] = {}
    for m in cov_methods:
        pts = sorted((r for r in runs if r["method"] == m and r.get("dc") is not None),
                     key=lambda r: r["dc"])
        if pts:
            series[m] = pts
    if not series:
        return

    fig, ax = plt.subplots(figsize=(8, 4.8))
    all_dcs: set[int] = set()
    for m, pts in series.items():
        dcs = [r["dc"] for r in pts]
        all_dcs.update(dcs)
        means = [r[f"{mkey}_mean"] for r in pts]
        stds = [r[f"{mkey}_std"] for r in pts]
        if len(pts) >= 2:
            ax.errorbar(dcs, means, yerr=stds, marker="o", capsize=3,
                        color=METHOD_COLOR[m], linestyle=METHOD_LS[m],
                        label=METHOD_LABEL[m])
        else:
            ax.errorbar(dcs, means, yerr=stds, fmt="D", capsize=3, markersize=8,
                        color=METHOD_COLOR[m],
                        label=f"{METHOD_LABEL[m]} @ dc={dcs[0]}")
        if m == "cov_supervised":
            for x, y, r in zip(dcs, means, pts):
                ax.annotate(f"d={r['embedding_dim']}", (x, y),
                            textcoords="offset points", xytext=(6, 8),
                            fontsize=8, color="#444")

    mean_r = next((r for r in runs if r["method"] == "mean"), None)
    if mean_r is not None:
        m, s = mean_r[f"{mkey}_mean"], mean_r[f"{mkey}_std"]
        ax.axhline(m, color=METHOD_COLOR["mean"], linestyle="--",
                   label=f"Mean baseline (d=1024) = "
                         + (f"{m*100:.1f}%" if task == "classification" else f"{m:.3f}"))
        ax.axhspan(m - s, m + s, color=METHOD_COLOR["mean"], alpha=0.12)

    if la_run is not None and task == "classification" and "light_attention" not in series:
        m, s = la_run[f"{mkey}_mean"], la_run[f"{mkey}_std"]
        ax.axhline(m, color=METHOD_COLOR["light_attention"], linestyle="--",
                   label=f"Light Attention = {m*100:.1f}%")
        ax.axhspan(m - s, m + s, color=METHOD_COLOR["light_attention"], alpha=0.12)

    ax.set_xlabel("Bottleneck dimension dc  (covariance is dc × dc)")
    if task == "classification":
        ax.set_ylabel("Accuracy (%)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}"))
        ax.set_title("DeepLoc · dc sweep · all pooling methods")
    else:
        ax.set_ylabel("Spearman R")
        ax.set_title("Meltome · dc sweep · all pooling methods")
    ax.set_xticks(sorted(all_dcs))
    y_lo, y_hi = ax.get_ylim()
    ax.set_ylim(y_lo - 0.005, y_hi + 0.005)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=9, frameon=True)
    ax.yaxis.grid(True, color="#CCCCCC", linewidth=0.7, zorder=0)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3: training dynamics, mean ± std across seeds
# ---------------------------------------------------------------------------

def plot_training_dynamics(runs: list[dict], task: str, out_path: Path, dc: int = 32) -> None:
    mkey = metric_key(task)
    fig, ax = plt.subplots(figsize=(8, 4.8))

    for m in METHOD_ORDER:
        if m == "mean":
            r = next((x for x in runs if x["method"] == "mean"), None)
        elif m == "light_attention":
            r = next((x for x in runs if x["method"] == "light_attention"), None)
        else:
            r = next((x for x in runs
                      if x["method"] == m and x.get("dc") == dc), None)
        if r is None:
            continue

        # Pad histories to equal length within this method
        hist_metric = [[h[mkey] for h in ps["history"]] for ps in r["per_seed"]]
        max_len = max(len(h) for h in hist_metric)
        arr = np.full((len(hist_metric), max_len), np.nan)
        for i, h in enumerate(hist_metric):
            arr[i, :len(h)] = h
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        xs = np.arange(1, max_len + 1)
        ax.plot(xs, mean, color=METHOD_COLOR[m], linestyle=METHOD_LS[m],
                linewidth=2.2, label=METHOD_LABEL[m])
        ax.fill_between(xs, mean - std, mean + std,
                        color=METHOD_COLOR[m], alpha=0.18, linewidth=0)

    ax.set_xlabel("Epoch")
    if task == "classification":
        ax.set_ylabel("Validation accuracy (%)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.1f}"))
        ax.set_title("DeepLoc training dynamics  ·  mean ± 1 std, 3 seeds")
    else:
        ax.set_ylabel("Validation Spearman R")
        ax.set_title("Meltome training dynamics  ·  mean ± 1 std, 3 seeds")
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.yaxis.grid(True, color="#CCCCCC", linewidth=0.7, zorder=0)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 4: per-class accuracy for DeepLoc, all methods vs mean
# ---------------------------------------------------------------------------

def _per_class_accuracy_per_seed(r: dict, n_classes: int) -> np.ndarray:
    """Return [n_seeds, n_classes] per-class accuracy for each seed."""
    out = np.full((len(r["per_seed"]), n_classes), np.nan)
    for s, ps in enumerate(r["per_seed"]):
        yt = np.asarray(ps["y_true"], dtype=int)
        yp = np.asarray(ps["y_pred"], dtype=int)
        for c in range(n_classes):
            mask = yt == c
            if mask.sum() > 0:
                out[s, c] = (yp[mask] == c).mean()
    return out


def plot_per_class_all_methods(runs: list[dict], out_path: Path, dc: int = 32) -> None:
    cls_runs = [r for r in runs if r["task"] == "classification"]
    mean_r = next((r for r in cls_runs if r["method"] == "mean"), None)
    if mean_r is None or "label_to_index" not in mean_r:
        return
    label_map = mean_r["label_to_index"]
    # Display order: frequency-sorted, head → tail (matches the project slide
    # showing the DeepLoc class distribution). Fall back to label-index order
    # for any class the map doesn't recognise.
    ordered_names = [c for c in DEEPLOC_CLASS_ORDER if c in label_map]
    ordered_names += [c for c in sorted(label_map, key=lambda k: label_map[k])
                      if c not in DEEPLOC_CLASS_ORDER]
    order_idx = np.array([label_map[c] for c in ordered_names])
    n = len(ordered_names)

    methods = [m for m in METHOD_ORDER if m != "mean"]
    series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    arr = _per_class_accuracy_per_seed(mean_r, len(label_map))[:, order_idx]
    series["mean"] = (np.nanmean(arr, axis=0), np.nanstd(arr, axis=0))
    for m in methods:
        if m == "light_attention":
            r = next((x for x in cls_runs if x["method"] == "light_attention"), None)
        else:
            r = next((x for x in cls_runs if x["method"] == m and x.get("dc") == dc), None)
        if r is not None:
            arr = _per_class_accuracy_per_seed(r, len(label_map))[:, order_idx]
            series[m] = (np.nanmean(arr, axis=0), np.nanstd(arr, axis=0))

    fig, ax = plt.subplots(figsize=(11.5, 5))
    width = 0.85 / len(series)
    xs = np.arange(n)
    for i, (m, (mean, std)) in enumerate(series.items()):
        offset = i * width - 0.425 + width / 2
        ax.bar(xs + offset, mean * 100, width=width,
               yerr=std * 100, capsize=2.5,
               error_kw=dict(elinewidth=0.8, ecolor="#222"),
               color=METHOD_COLOR[m], label=METHOD_LABEL[m],
               edgecolor="none", linewidth=0.5)

    ax.set_xticks(xs)
    ax.set_xticklabels([DEEPLOC_SHORT[c] for c in ordered_names],
                       rotation=20, ha="right")
    ax.set_ylabel("Per-class accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title(
        f"DeepLoc · per-class accuracy by pooling method  ·  dc = {dc}  ·  "
        "classes ordered by training-set frequency  ·  mean ± 1 std (3 seeds)"
    )
    ax.legend(ncols=len(series), loc="upper center", bbox_to_anchor=(0.5, -0.12),
              frameon=False, fontsize=10)
    ax.yaxis.grid(True, color="#CCCCCC", linewidth=0.7, zorder=0)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 5: confusion matrix for the best DeepLoc method (hybrid)
# ---------------------------------------------------------------------------

def plot_confusion_matrix(runs: list[dict], method: str, out_path: Path, dc: int = 32) -> None:
    cls_runs = [r for r in runs if r["task"] == "classification"]
    if method in ("mean", "light_attention"):
        r = next((x for x in cls_runs if x["method"] == method), None)
    else:
        r = next((x for x in cls_runs if x["method"] == method and x.get("dc") == dc), None)
    if r is None or "label_to_index" not in r:
        return
    label_map = r["label_to_index"]
    classes = sorted(label_map, key=lambda k: label_map[k])
    n = len(classes)

    cm = np.zeros((n, n), dtype=float)
    for ps in r["per_seed"]:
        yt = np.asarray(ps["y_true"], dtype=int)
        yp = np.asarray(ps["y_pred"], dtype=int)
        for t, p in zip(yt, yp):
            cm[t, p] += 1
    # Row-normalise
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sum > 0, cm / row_sum, 0)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm_norm * 100, cmap="Blues", vmin=0, vmax=100, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([DEEPLOC_SHORT[c] for c in classes], rotation=45, ha="right")
    ax.set_yticklabels([DEEPLOC_SHORT[c] for c in classes])
    for i in range(n):
        for j in range(n):
            v = cm_norm[i, j] * 100
            if v > 0.5:
                ax.text(j, i, f"{v:.0f}",
                        ha="center", va="center",
                        color="white" if v > 50 else "#222",
                        fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(
        f"DeepLoc confusion matrix · {METHOD_LABEL[method]} · row-normalised %\n"
        f"(pooled across 3 seeds)"
    )
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="%")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 6: Meltome predicted vs true grid
# ---------------------------------------------------------------------------

def plot_meltome_scatter(runs: list[dict], out_path: Path, dc: int = 32) -> None:
    reg_runs = [r for r in runs if r["task"] == "regression"]
    panels: list[tuple[str, dict]] = []
    for m in METHOD_ORDER:
        if m == "mean":
            r = next((x for x in reg_runs if x["method"] == "mean"), None)
        else:
            r = next((x for x in reg_runs if x["method"] == m and x.get("dc") == dc), None)
        if r is not None:
            panels.append((m, r))
    if not panels:
        return

    n = len(panels)
    cols = min(n, 3)
    rows_ = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_, cols, figsize=(4.0 * cols, 4.0 * rows_),
                             squeeze=False, sharex=True, sharey=True,
                             constrained_layout=True)

    for ax, (method, r) in zip(axes.flat, panels):
        ps = r["per_seed"][0]
        yt = np.asarray(ps["y_true"], dtype=float)
        yp = np.asarray(ps["y_pred"], dtype=float)
        ax.scatter(yt, yp, s=6, alpha=0.35, color=METHOD_COLOR[method])
        lo = float(min(yt.min(), yp.min()))
        hi = float(max(yt.max(), yp.max()))
        ax.plot([lo, hi], [lo, hi], color="#444", linestyle="--", linewidth=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", "box")
        sr = r["spearman_r_mean"]
        ax.set_title(f"{METHOD_LABEL[method]}\nSpearman R = {sr:.3f}", fontsize=10)
        ax.yaxis.grid(True, color="#CCCCCC", linewidth=0.7, zorder=0)

    # Hide unused axes
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)

    for ax in axes[-1]:
        ax.set_xlabel("True T_m (°C)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Predicted T_m (°C)")

    fig.suptitle("Meltome · predicted vs true melting temperature (seed 0)",
                 fontsize=12)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 7: parameter-efficiency — accuracy vs embedding dim (log), all methods
# ---------------------------------------------------------------------------

def plot_param_efficiency(runs: list[dict], task: str, out_path: Path) -> None:
    mkey = metric_key(task)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    # Cov-supervised sweep, if multiple dcs present
    sup_runs = sorted(
        [r for r in runs if r["method"] == "cov_supervised" and r.get("dc") is not None],
        key=lambda r: r["embedding_dim"],
    )
    if sup_runs:
        xs = [r["embedding_dim"] for r in sup_runs]
        ys = [r[f"{mkey}_mean"] for r in sup_runs]
        es = [r[f"{mkey}_std"] for r in sup_runs]
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3,
                    color=METHOD_COLOR["cov_supervised"],
                    label=METHOD_LABEL["cov_supervised"])

    # Reference points: other methods at their available dc
    for m in ("cov_unsupervised", "cov_pca", "hybrid"):
        cand = sorted(
            [r for r in runs if r["method"] == m and r.get("dc") is not None],
            key=lambda r: r["embedding_dim"],
        )
        if not cand:
            continue
        if len(cand) > 1:
            xs = [r["embedding_dim"] for r in cand]
            ys = [r[f"{mkey}_mean"] for r in cand]
            es = [r[f"{mkey}_std"] for r in cand]
            ax.errorbar(xs, ys, yerr=es, marker="s", capsize=3,
                        color=METHOD_COLOR[m], label=METHOD_LABEL[m])
        else:
            r = cand[0]
            ax.errorbar([r["embedding_dim"]], [r[f"{mkey}_mean"]],
                        yerr=[r[f"{mkey}_std"]], fmt="D", capsize=3,
                        color=METHOD_COLOR[m], label=METHOD_LABEL[m])

    mean_r = next((r for r in runs if r["method"] == "mean"), None)
    if mean_r is not None:
        m_v, s_v = mean_r[f"{mkey}_mean"], mean_r[f"{mkey}_std"]
        d = mean_r["embedding_dim"]
        ax.axhline(m_v, color=METHOD_COLOR["mean"], linestyle="--",
                   label=f"Mean (d={d})")
        ax.axhspan(m_v - s_v, m_v + s_v, color=METHOD_COLOR["mean"], alpha=0.12)
        ax.axvline(d, color=METHOD_COLOR["mean"], linestyle=":", alpha=0.5)

    la_r = next((r for r in runs if r["method"] == "light_attention"), None)
    if la_r is not None:
        m_v, s_v = la_r[f"{mkey}_mean"], la_r[f"{mkey}_std"]
        d = la_r["embedding_dim"]
        ax.errorbar([d], [m_v], yerr=[s_v], fmt="*", capsize=3, markersize=11,
                    color=METHOD_COLOR["light_attention"],
                    label=f"{METHOD_LABEL['light_attention']} (d={d})")

    ax.set_xscale("log")
    ax.set_xlabel("Pooled embedding dimension (log)")
    if task == "classification":
        ax.set_ylabel("Accuracy (%)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}"))
        ax.set_title("DeepLoc · parameter efficiency")
    else:
        ax.set_ylabel("Spearman R")
        ax.set_title("Meltome · parameter efficiency")
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    ax.yaxis.grid(True, color="#CCCCCC", linewidth=0.7, zorder=0)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary table TSV
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# External benchmark comparison: Stärk et al. 2021 (Light Attention paper)
# Figures read off the published bar chart (setDeepLoc / setHARD, Q10).
# ---------------------------------------------------------------------------

LIGHT_ATTN_BENCHMARK_DEEPLOC = {
    # (head, model): setDeepLoc accuracy in %
    ("EAT", "UniRep"): 61, ("EAT", "SeqVec"): 61, ("EAT", "ProtBert"): 65,
    ("EAT", "ESM-1b"): 70, ("EAT", "ProtT5"): 74,
    ("FNN", "UniRep"): 68, ("FNN", "SeqVec"): 71, ("FNN", "ProtBert"): 76,
    ("FNN", "ESM-1b"): 80, ("FNN", "ProtT5"): 82,
    ("LA",  "UniRep"): 71, ("LA",  "SeqVec"): 76, ("LA",  "ProtBert"): 80,
    ("LA",  "ESM-1b"): 83, ("LA",  "ProtT5"): 86,
}
BASELINES_DEEPLOC = {
    "Majority": 29, "CELLO": 55, "LocTree2": 61, "DeepLoc62": 74, "DeepLoc": 78,
}


def plot_light_attention_comparison(runs: list[dict], out_path: Path,
                                    dc: int = 32) -> None:
    """Overlay our DeepLoc accuracies on the Stärk et al. 2021 benchmark bars."""
    cls_runs = [r for r in runs if r["task"] == "classification"]
    if not cls_runs:
        return

    # Build the bar ordering: baselines | EAT | FNN | LA
    sections: list[tuple[str, list[tuple[str, float]]]] = [
        ("Baselines", list(BASELINES_DEEPLOC.items())),
    ]
    for head in ("EAT", "FNN", "LA"):
        sections.append((head, [
            (model, LIGHT_ATTN_BENCHMARK_DEEPLOC[(head, model)])
            for model in ("UniRep", "SeqVec", "ProtBert", "ESM-1b", "ProtT5")
        ]))

    # Collect our methods (all use FNN probe on ProtT5 = distilled ProtT5)
    SHORT_LABELS = {
        "cov_supervised":  "Cov sup",
        "cov_unsupervised":"Cov unsup",
        "cov_pca":         "Cov PCA",
        "hybrid":          "Hybrid",
        "light_attention": "Light Att.",
    }
    ours: list[tuple[str, float, float, str]] = []
    for m in METHOD_ORDER:
        if m == "mean":
            r = next((x for x in cls_runs if x["method"] == "mean"), None)
            label = "ProtT5\nMean"
        elif m == "light_attention":
            r = next((x for x in cls_runs if x["method"] == "light_attention"), None)
            label = f"ProtT5\n{SHORT_LABELS[m]}"
        else:
            r = next((x for x in cls_runs
                      if x["method"] == m and x.get("dc") == dc), None)
            label = f"ProtT5\n{SHORT_LABELS[m]}"
        if r is not None:
            ours.append((label, r["accuracy_mean"] * 100,
                         r["accuracy_std"] * 100, m))
    sections.append(("Ours (FNN, ProtT5)", [(lbl, val) for lbl, val, _, _ in ours]))

    # Build flat x-position list with a gap between sections
    fig, ax = plt.subplots(figsize=(13, 4.8))
    x = 0
    positions: list[float] = []
    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []
    section_spans: list[tuple[float, float, str]] = []

    base_color = "#bcbcbc"
    head_colors = {"EAT": "#bcbcbc", "FNN": "#bcbcbc", "LA": "#bcbcbc"}
    for sec_name, entries in sections:
        start = x
        for name, val in entries:
            positions.append(x)
            labels.append(name)
            values.append(val)
            if sec_name == "Ours (FNN, ProtT5)":
                # use our colour palette for the our-method bars
                idx = len(positions) - 1 - (len(positions) - 1 - x)  # simplifies
                # find matching method by label suffix
                method = next(m for lbl, _, _, m in ours if lbl == name)
                colors.append(METHOD_COLOR[method])
            else:
                colors.append(base_color)
            x += 1
        section_spans.append((start, x - 1, sec_name))
        x += 1  # gap between sections

    bars = ax.bar(positions, values, color=colors, edgecolor="none",
                  linewidth=0.7)

    # Add error bars + percentage labels for our bars only
    for lbl, val, std, method in ours:
        idx = labels.index(lbl)
        ax.errorbar(positions[idx], val, yerr=std, fmt="none",
                    ecolor="black", capsize=3, linewidth=1)

    # Top labels with values
    for px, val in zip(positions, values):
        ax.text(px, val + 1.0, f"{val:.0f}", ha="center", fontsize=8.5)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("10-state accuracy Q10 (setDeepLoc, %)")
    ax.set_ylim(0, 100)
    ax.set_title("DeepLoc Q10 benchmark  ·  Stärk et al. 2021 (grey) vs ours (coloured)")
    ax.yaxis.grid(True, color="#CCCCCC", linewidth=0.7, zorder=0)

    # Section dividers + headers
    for i, (lo, hi, name) in enumerate(section_spans):
        if i > 0:
            ax.axvline(lo - 0.5, color="#e07a3a", linewidth=1.5)
        ax.text((lo + hi) / 2, 96, name, ha="center", fontsize=11,
                color="#e07a3a", weight="bold")

    # ProtT5+FNN dashed reference (the directly comparable bar)
    pt5_fnn = LIGHT_ATTN_BENCHMARK_DEEPLOC[("FNN", "ProtT5")]
    ax.axhline(pt5_fnn, color="#3a78c2", linestyle="--", linewidth=1, alpha=0.6,
               label=f"ProtT5+FNN reference ({pt5_fnn}%)")
    pt5_la = LIGHT_ATTN_BENCHMARK_DEEPLOC[("LA", "ProtT5")]
    ax.axhline(pt5_la, color="#2ea27a", linestyle=":", linewidth=1, alpha=0.6,
               label=f"ProtT5+LA SOTA ({pt5_la}%)")
    ax.legend(loc="lower left", fontsize=9, frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def write_summary_table(runs: list[dict], out_path: Path) -> None:
    lines = ["task\tmethod\tdc\tembed_dim\tmetric\tmean\tstd\tn_seeds"]
    for r in sorted(runs, key=lambda x: (x["task"], x["method"], x.get("dc") or 0)):
        mkey = metric_key(r["task"])
        lines.append(
            f"{r['task']}\t{r['method']}\t{r.get('dc')}\t{r['embedding_dim']}"
            f"\t{mkey}\t{r[f'{mkey}_mean']:.4f}\t{r[f'{mkey}_std']:.4f}"
            f"\t{len(r['per_seed'])}"
        )
    out_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True,
                        help="Directory containing per-run JSON files")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output base directory for figures")
    parser.add_argument("--dc", type=int, default=32,
                        help="Bottleneck dimension to feature in per-task plots")
    args = parser.parse_args()

    runs = load_runs(args.results)
    if not runs:
        raise SystemExit(f"No JSON results found under {args.results}")

    deeploc_dir = args.out / "DeepLoc"
    meltome_dir = args.out / "Meltome"
    deeploc_dir.mkdir(parents=True, exist_ok=True)
    meltome_dir.mkdir(parents=True, exist_ok=True)

    cls_runs, reg_runs = split_by_task(runs)

    write_summary_table(runs, args.out / "summary_table.tsv")

    la_run = next((r for r in cls_runs if r.get("method") == "light_attention"), None)
    sweep_dcs = [8, 16, 24, 32, 48]

    if cls_runs:
        plot_training_dynamics(cls_runs, "classification",
                               deeploc_dir / "fig_deeploc_training.png", dc=args.dc)
        plot_per_class_all_methods(cls_runs, deeploc_dir / "fig_deeploc_per_class.png",
                                   dc=args.dc)
        # dc-sweep bar and per-class plots (one per dc, new files)
        for dc_val in sweep_dcs:
            plot_main_bar(cls_runs, "classification",
                          deeploc_dir / f"fig_deeploc_bar_dc{dc_val}.png", dc=dc_val)
            plot_per_class_all_methods(cls_runs,
                                       deeploc_dir / f"fig_deeploc_per_class_dc{dc_val}.png",
                                       dc=dc_val)
        # dc-sweep line chart with LA reference
        plot_dc_sweep(cls_runs, "classification",
                      deeploc_dir / "fig_deeploc_dc_sweep_la.png", la_run=la_run)
        plot_param_efficiency(cls_runs, "classification",
                              deeploc_dir / "fig_deeploc_param_efficiency.png")
        plot_confusion_matrix(cls_runs, "hybrid",
                              deeploc_dir / "fig_deeploc_confusion_hybrid.png", dc=args.dc)
        plot_confusion_matrix(cls_runs, "mean",
                              deeploc_dir / "fig_deeploc_confusion_mean.png")
        plot_confusion_matrix(cls_runs, "light_attention",
                              deeploc_dir / "fig_deeploc_confusion_light_attention.png")
        # LA comparison with light_attention bar (dc=48 = best performing)
        plot_light_attention_comparison(
            cls_runs, deeploc_dir / "fig_deeploc_vs_light_attention_la.png", dc=48)

    if reg_runs:
        plot_main_bar(reg_runs, "regression", meltome_dir / "fig_meltome_bar.png", dc=args.dc)
        plot_training_dynamics(reg_runs, "regression",
                               meltome_dir / "fig_meltome_training.png", dc=args.dc)
        plot_meltome_scatter(reg_runs, meltome_dir / "fig_meltome_scatter.png", dc=args.dc)
        # per-dc bar charts (one per dc)
        for dc_val in sweep_dcs:
            plot_main_bar(reg_runs, "regression",
                          meltome_dir / f"fig_meltome_bar_dc{dc_val}.png", dc=dc_val)
        plot_dc_sweep(reg_runs, "regression",
                      meltome_dir / "fig_meltome_dc_sweep_la.png")
        plot_param_efficiency(reg_runs, "regression",
                              meltome_dir / "fig_meltome_param_efficiency.png")

    print(f"Wrote figures to {args.out}")


if __name__ == "__main__":
    main()
