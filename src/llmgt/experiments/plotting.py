"""Plotting utilities for experiment summaries.

All functions gracefully skip if ``matplotlib`` is not installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
import math


def _get_k(r: dict) -> int:
    """Extract the communication-budget value from a summary row."""
    if "k" in r:
        return int(r["k"])
    if "K" in r:
        return int(r["K"])
    raise KeyError("Row is missing 'k' (or legacy 'K')")


def plot_metric_by_k(
    rows: Sequence[dict],
    *,
    metric: str,
    title: str,
    ylabel: str,
    out_path: Path,
) -> bool:
    """Plot a single metric against *k* and save to *out_path*.

    Automatically picks up ``{metric}_std`` columns for error bars if present.
    Returns *True* on success, *False* if plotting was skipped.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot_metric_by_k] Plotting skipped: {e}")
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)

    ks_all = [_get_k(r) for r in rows]
    vals_all = [r.get(metric) for r in rows]
    vals_all = [float("nan") if v is None else float(v) for v in vals_all]

    if all(math.isnan(v) for v in vals_all):
        print(f"[plot_metric_by_k] Skipped '{metric}' (all values are None/NaN).")
        return False

    # Check for a matching *_std column for error bars
    std_key = f"{metric}_std"
    has_std = all(std_key in r for r in rows)

    # Filter out NaN rows
    filtered = [
        (k, v, r)
        for k, v, r in zip(ks_all, vals_all, rows)
        if not math.isnan(v)
    ]
    ks = [t[0] for t in filtered]
    vals = [t[1] for t in filtered]

    plt.figure(figsize=(7, 4.5))

    if has_std:
        stds = [
            float(t[2].get(std_key, 0) or 0)
            for t in filtered
        ]
        plt.errorbar(ks, vals, yerr=stds, marker="o", capsize=3, linewidth=1.5)
    else:
        plt.plot(ks, vals, marker="o", linewidth=1.5)

    plt.xlabel("k (max communication rounds)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, format="png", dpi=200)
    plt.close()
    return True
