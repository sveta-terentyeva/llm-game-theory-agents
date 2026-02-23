from __future__ import annotations

from pathlib import Path
from typing import Sequence
import math


def _get_k(r: dict) -> int:
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

    ks = [k for k, v in zip(ks_all, vals_all) if not math.isnan(v)]
    vals = [v for v in vals_all if not math.isnan(v)]

    plt.figure()
    plt.plot(ks, vals, marker="o")
    plt.xlabel("k (max communication rounds)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, format="png", dpi=200)
    plt.close()
    return True
