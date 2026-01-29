from pathlib import Path
from typing import Sequence


def plot_metric_by_k(
    rows: Sequence[dict],
    *,
    metric: str,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:

    try:
        import matplotlib.pyplot as plt
    except Exception:
        out_path.touch()
        return

    ks = [r["K"] for r in rows]
    values = [r[metric] for r in rows]

    plt.figure()
    plt.plot(ks, values, marker="o")
    plt.xlabel("K (max communication rounds)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

