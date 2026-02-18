from __future__ import annotations

from llmgt.experiments import run_comm_sweep, summarize_by_k, write_csv, make_rule_based_workflow_agents
from llmgt.experiments.plotting import plot_metric_by_k
from llmgt.logging.jsonl_logger import JsonlLogger
from llmgt.logging.run_meta import write_run_meta
from llmgt.sim.run_dir import make_run_dir

from llmgt.games.prisoners_dilemma import PrisonersDilemma
from llmgt.games.stag_hunt import StagHunt
from llmgt.games.battle_of_sexes import BattleOfSexes
from llmgt.games.ultimatum import UltimatumGame


def run_for_game(game, run_root, tag: str) -> None:
    game_dir = run_root / tag
    logs_dir = game_dir / "logs"
    figs_dir = game_dir / "figures"
    logs_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    a, b = make_rule_based_workflow_agents(game)
    logger = JsonlLogger(out_dir=logs_dir, filename="episodes.jsonl", overwrite=True)

    k_values = list(range(0, 7))
    n_runs = 200
    mode = "workflow"

    records = run_comm_sweep(
        game=game,
        agent_a=a,
        agent_b=b,
        k_values=k_values,
        n_runs=n_runs,
        mode=mode,
        logger=logger,
    )

    rows = summarize_by_k(records)
    write_csv(rows, game_dir / "summary_by_k.csv")

    plot_metric_by_k(
        rows,
        metric="agreement_rate",
        title=f"{game.name} — agreement vs K ({mode})",
        ylabel="Agreement rate",
        out_path=figs_dir / "agreement_rate.png",
    )
    plot_metric_by_k(
        rows,
        metric="mean_rounds_to_agreement",
        title=f"{game.name} — rounds-to-agreement vs K ({mode})",
        ylabel="Mean rounds-to-agreement",
        out_path=figs_dir / "mean_rounds_to_agreement.png",
    )
    plot_metric_by_k(
        rows,
        metric="welfare_mean",
        title=f"{game.name} — welfare vs K ({mode})",
        ylabel="Mean welfare (A+B)",
        out_path=figs_dir / "welfare_mean.png",
    )

    print(f"[{tag}] saved to: {game_dir}")


def main() -> None:
    run = make_run_dir(tag="workflow_sweep_all")
    run_root = run.root

    write_run_meta(
        run_root / "run_meta.json",
        {
            "tag": "workflow_sweep_all",
            "mode": "workflow",
            "k_values": list(range(0, 7)),
            "n_runs": 200,
            "agent": "workflow_baseline",
            "notes": "Per-game outputs under run_root/<game>/",
        },
    )

    run_for_game(PrisonersDilemma(), run_root, "prisoners_dilemma")
    run_for_game(StagHunt(), run_root, "stag_hunt")
    run_for_game(BattleOfSexes(), run_root, "battle_of_sexes")
    run_for_game(UltimatumGame(), run_root, "ultimatum")

    print(f"\nDONE. Run directory: {run_root}")


if __name__ == "__main__":
    main()
