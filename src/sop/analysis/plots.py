"""Required (MVP) plots: bar chart, efficiency curve, layer-sweep heatmap.

All functions take a long-format DataFrame produced by ``aggregate_runs`` and
return a matplotlib Figure so the caller controls saving.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main_results_bar(df: pd.DataFrame, task: str) -> plt.Figure:
    """Per-task bar chart: pooling methods side by side, error bars from seeds."""
    sub = df[df["task"] == task]
    summary = (
        sub.groupby("method")["metric_value"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("method")
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(summary["method"], summary["mean"], yerr=summary["std"], capsize=4)
    metric_label = sub["metric"].iloc[0] if not sub.empty else "metric"
    ax.set_ylabel(metric_label)
    ax.set_title(f"{task} — pooling methods")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def efficiency_curve(df: pd.DataFrame, task: str, mean_method: str = "mean") -> plt.Figure:
    """Embedding dim (log-x) vs metric. Covariance variants as curves; mean as horizontal."""
    sub = df[df["task"] == task]
    fig, ax = plt.subplots(figsize=(6, 4))

    cov_methods = sorted(m for m in sub["method"].unique() if m != mean_method)
    for method in cov_methods:
        m_df = (
            sub[sub["method"] == method]
            .groupby("embedding_dim")["metric_value"]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("embedding_dim")
        )
        ax.errorbar(
            m_df["embedding_dim"], m_df["mean"], yerr=m_df["std"],
            marker="o", capsize=3, label=method,
        )

    mean_rows = sub[sub["method"] == mean_method]
    if not mean_rows.empty:
        m, s = mean_rows["metric_value"].mean(), mean_rows["metric_value"].std()
        d = mean_rows["embedding_dim"].iloc[0]
        ax.axhline(m, color="gray", linestyle="--", label=f"{mean_method} (d={d})")
        ax.axvline(d, color="gray", linestyle=":", alpha=0.5)
        ax.fill_between(ax.get_xlim(), m - s, m + s, color="gray", alpha=0.1)

    ax.set_xscale("log")
    ax.set_xlabel("embedding dim (log)")
    ax.set_ylabel(sub["metric"].iloc[0] if not sub.empty else "metric")
    ax.set_title(f"{task} — efficiency curve")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig


def layer_sweep_heatmap(
    df: pd.DataFrame,
    task: str,
    layer_col: str = "layer",
) -> plt.Figure:
    """Heatmap of pooling method × layer for one task.

    Expects ``df`` to have a ``layer`` column added by the caller (the raw
    aggregator doesn't know about layers — they're encoded in the embedding
    file path; the caller adds the column when building the layer-sweep dataset).
    """
    sub = df[df["task"] == task]
    pivot = (
        sub.groupby(["method", layer_col])["metric_value"]
        .mean()
        .reset_index()
        .pivot(index="method", columns=layer_col, values="metric_value")
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(7, 0.5 + 0.5 * len(pivot)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel(layer_col)
    ax.set_ylabel("method")
    ax.set_title(f"{task} — layer sweep")
    fig.colorbar(im, ax=ax, label=sub["metric"].iloc[0] if not sub.empty else "metric")
    fig.tight_layout()
    return fig
