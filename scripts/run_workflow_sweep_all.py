from __future__ import annotations

from pathlib import Path

from llmgt.experiments import run_comm_sweep, summarize_by_k, write_csv, make_workflow_agents
from llmgt.experiments.plotting import plot_metric_by_k

from llmgt.games.prisoners_dilemma import PrisonersDilemma
from llmgt.games.stag_hunt import StagHunt
from llmgt.games.battle_of_sexes import BattleOfSexes
from llmgt.games.ultimatum import UltimatumGame


def run_for_game(game, tag: str) -> None:
    a, b = make_workflow_agents(game)

    records = run_comm_sweep(
        game=game,
        agent_a=a,
        agent_b=b,
        k_values=range(0, 7),
        n_runs=200,
        mode="workflow",
    )

    rows = summarize_by_k(records)

    out_dir = Path("data/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{tag}_workflow_sweep.csv"
    write_csv(rows, csv_path)

    plot_metric_by_k(
        rows,
        metric="agreement_rate",
        title=f"Agreement rate vs K ({tag}, workflow baseline)",
        ylabel="Agreement rate",
        out_path=out_dir / f"{tag}_workflow_agreement_rate.png",
    )

    plot_metric_by_k(
        rows,
        metric="mean_rounds_to_agreement",
        title=f"Rounds to agreement vs K ({tag}, workflow baseline)",
        ylabel="Mean rounds to agreement",
        out_path=out_dir / f"{tag}_workflow_rounds_to_agreement.png",
    )

    print(f"[{tag}] saved CSV + plots to data/figures/")


def main() -> None:
    run_for_game(PrisonersDilemma(), "pd")
    run_for_game(StagHunt(), "stag_hunt")
    run_for_game(BattleOfSexes(), "battle_of_sexes")
    run_for_game(UltimatumGame(), "ultimatum")


if __name__ == "__main__":
    main()
