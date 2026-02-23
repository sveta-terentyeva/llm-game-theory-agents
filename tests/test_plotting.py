from pathlib import Path

from llmgt.experiments.plotting import plot_metric_by_k


def test_plot_metric_by_k_accepts_legacy_K(tmp_path: Path):
    rows = [
        {"K": 0, "agreement_rate": 0.1},
        {"K": 1, "agreement_rate": 0.4},
        {"K": 2, "agreement_rate": 0.8},
    ]

    out = tmp_path / "plot_legacy.png"

    plot_metric_by_k(
        rows,
        metric="agreement_rate",
        title="Test plot (legacy K)",
        ylabel="Agreement",
        out_path=out,
    )

    assert out.exists()


def test_plot_metric_by_k_accepts_k(tmp_path: Path):
    rows = [
        {"k": 0, "agreement_rate": 0.1},
        {"k": 1, "agreement_rate": 0.4},
        {"k": 2, "agreement_rate": 0.8},
    ]

    out = tmp_path / "plot_k.png"

    plot_metric_by_k(
        rows,
        metric="agreement_rate",
        title="Test plot (k)",
        ylabel="Agreement",
        out_path=out,
    )

    assert out.exists()
