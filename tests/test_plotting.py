from pathlib import Path

from llmgt.experiments.plotting import plot_metric_by_k


def test_plot_metric_by_k(tmp_path: Path):
    rows = [
        {"K": 0, "agreement_rate": 0.1},
        {"K": 1, "agreement_rate": 0.4},
        {"K": 2, "agreement_rate": 0.8},
    ]

    out = tmp_path / "plot.png"

    plot_metric_by_k(
        rows,
        metric="agreement_rate",
        title="Test plot",
        ylabel="Agreement",
        out_path=out,
    )

    assert out.exists()
