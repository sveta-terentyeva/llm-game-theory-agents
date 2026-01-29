from pathlib import Path

from llmgt.agents.simple import FixedActionAgent
from llmgt.experiments.sweep import run_comm_sweep, summarize_by_k
from llmgt.experiments.plotting import plot_metric_by_k
from llmgt.games.prisoners_dilemma import PrisonersDilemma


def main():
    game = PrisonersDilemma()

    agent_a = FixedActionAgent(name="A", action="D")
    agent_b = FixedActionAgent(name="B", action="D")

    records = run_comm_sweep(
        game=game,
        agent_a=agent_a,
        agent_b=agent_b,
        k_values=range(0, 6),
        n_runs=50,
    )

    rows = summarize_by_k(records)

    out_dir = Path("data/figures")

    plot_metric_by_k(
        rows,
        metric="agreement_rate",
        title="Agreement rate vs communication budget (PD)",
        ylabel="Agreement rate",
        out_path=out_dir / "pd_agreement_rate.png",
    )

    plot_metric_by_k(
        rows,
        metric="mean_rounds_to_agreement",
        title="Rounds to agreement vs communication budget (PD)",
        ylabel="Mean rounds to agreement",
        out_path=out_dir / "pd_rounds_to_agreement.png",
    )


if __name__ == "__main__":
    main()
