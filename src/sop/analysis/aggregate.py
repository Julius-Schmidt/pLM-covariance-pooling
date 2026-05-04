"""Walk results/runs/ and assemble run metadata into a long-format DataFrame."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def aggregate_runs(runs_dir: Path | str) -> pd.DataFrame:
    """Load every per-run JSON under ``runs_dir`` into a tidy DataFrame.

    One row per (run, seed). Columns include: config, method, dc,
    embedding_dim, task, seed, metric, metric_value.
    """
    rows: list[dict] = []
    for path in Path(runs_dir).rglob("*.json"):
        # Skip sweep-aggregate files; we'll re-aggregate from the per-run JSONs.
        if path.stem.endswith("_sweep"):
            continue
        with open(path) as fh:
            summary = json.load(fh)

        # Identify metric key by convention.
        metric_key = "accuracy" if summary["task"] == "classification" else "spearman_r"

        for seed_entry in summary["per_seed"]:
            rows.append({
                "config": summary["config"],
                "method": summary["method"],
                "dc": summary.get("dc"),
                "embedding_dim": summary["embedding_dim"],
                "task": summary["task"],
                "seed": seed_entry["seed"],
                "metric": metric_key,
                "metric_value": seed_entry[metric_key],
                "best_epoch": seed_entry.get("best_epoch"),
                "source_file": str(path),
            })

    return pd.DataFrame(rows)
