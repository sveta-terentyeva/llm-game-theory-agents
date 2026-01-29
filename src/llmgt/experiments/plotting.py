from __future__ import annotations

from pathlib import Path
from typing import Sequence


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
        matplotlib.use("Agg")  # headless backend
        import matplotlib.pyplot as plt
    except Exception as e:
        print(
            f"[plot_metric_by_k] Plotting skipped (matplotlib/numpy not available): {e}"
        )
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)

    ks = [int(r["K"]) for r in rows]
    vals = [r.get(metric) for r in rows]
    vals = [float("nan") if v is None else float(v) for v in vals]

    plt.figure()
    plt.plot(ks, vals, marker="o")
    plt.xlabel("K (max communication rounds)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, format="png", dpi=200)
    plt.close()
    return True

