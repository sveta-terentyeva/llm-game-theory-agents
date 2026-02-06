from __future__ import annotations

from llmgt.agents.simple import FixedActionAgent
from llmgt.experiments.sweep import run_comm_sweep, summarize_by_k, write_csv
from llmgt.experiments.plotting import plot_metric_by_k
from llmgt.logging.jsonl_logger import JsonlLogger
from llmgt.logging.run_meta import write_run_meta
from llmgt.sim.run_dir import make_run_dir
from llmgt.games.prisoners_dilemma import PrisonersDilemma


def main() -> None:
    game = PrisonersDilemma()
    agent_a = FixedActionAgent(name="A", action="D")
    agent_b = FixedActionAgent(name="B", action="D")

    run = make_run_dir(tag="pd_fixed_sweep")
    write_run_meta(
        run.root / "run_meta.json",
        {
            "tag": "pd_fixed_sweep",
            "game": game.name,
            "mode": "no_workflow",
            "k_values": list(range(0, 6)),
            "n_runs": 50,
            "agent_a": "FixedActionAgent(D)",
            "agent_b": "FixedActionAgent(D)",
        },
    )

    logger = JsonlLogger(out_dir=run.logs_dir, filename="episodes.jsonl", overwrite=True)

    records = run_comm_sweep(
        game=game,
        agent_a=agent_a,
        agent_b=agent_b,
        k_values=range(0, 6),
        n_runs=50,
        mode="no_workflow",
        logger=logger,
    )

    rows = summarize_by_k(records)
    write_csv(rows, run.root / "summary_by_k.csv")

    plot_metric_by_k(
        rows,
        metric="agreement_rate",
        title="prisoners_dilemma — agreement vs K (fixed baseline)",
        ylabel="Agreement rate",
        out_path=run.figures_dir / "agreement_rate.png",
    )
    plot_metric_by_k(
        rows,
        metric="mean_rounds_to_agreement",
        title="prisoners_dilemma — rounds-to-agreement vs K (fixed baseline)",
        ylabel="Mean rounds-to-agreement",
        out_path=run.figures_dir / "mean_rounds_to_agreement.png",
    )

    print(f"Saved to: {run.root}")


if __name__ == "__main__":
    main()
