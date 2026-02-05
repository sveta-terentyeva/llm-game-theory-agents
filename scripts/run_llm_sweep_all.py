from __future__ import annotations

from pathlib import Path

from llmgt.agents.llm import LLMAgent
from llmgt.llm import HeuristicLLMClient
from llmgt.experiments.sweep import run_comm_sweep, summarize_by_k, write_csv
from llmgt.experiments.plotting import plot_metric_by_k
from llmgt.logging.jsonl_logger import JsonlLogger

from llmgt.games.prisoners_dilemma import PrisonersDilemma
from llmgt.games.stag_hunt import StagHunt
from llmgt.games.battle_of_sexes import BattleOfSexes
from llmgt.games.ultimatum import UltimatumGame

from llmgt.sim.run_dir import make_run_dir
from llmgt.logging.run_meta import write_run_meta


def run_for_game(game, out_prefix: str, run_root: Path) -> None:
    game_dir = run_root / out_prefix
    logs_dir = game_dir / "logs"
    figs_dir = game_dir / "figures"

    logs_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    client_a = HeuristicLLMClient()
    client_b = HeuristicLLMClient()

    agent_a = LLMAgent(name=f"llm_A_{out_prefix}", client=client_a, role="agent_a")
    agent_b = LLMAgent(name=f"llm_B_{out_prefix}", client=client_b, role="agent_b")

    logger = JsonlLogger(out_dir=logs_dir, filename=f"{out_prefix}_episodes.jsonl", overwrite=True)

    k_values = list(range(0, 7))
    n_runs = 200
    mode = "workflow"

    records = run_comm_sweep(
        game=game,
        agent_a=agent_a,
        agent_b=agent_b,
        k_values=k_values,
        n_runs=n_runs,
        mode=mode,
        logger=logger,
    )

    rows = summarize_by_k(records)

    csv_path = game_dir / f"{out_prefix}_llm_{mode}_sweep.csv"
    write_csv(rows, csv_path)

    plot_metric_by_k(
        rows,
        metric="agreement_rate",
        title=f"Agreement rate vs K ({out_prefix}, LLM {mode})",
        ylabel="Agreement rate",
        out_path=figs_dir / f"{out_prefix}_llm_{mode}_agreement_rate.png",
    )

    plot_metric_by_k(
        rows,
        metric="mean_rounds_to_agreement",
        title=f"Rounds to agreement vs K ({out_prefix}, LLM {mode})",
        ylabel="Mean rounds to agreement",
        out_path=figs_dir / f"{out_prefix}_llm_{mode}_rounds_to_agreement.png",
    )

    print(f"[{out_prefix}] saved CSV + plots to: {game_dir}")


def main() -> None:
    run = make_run_dir(tag="llm_sweep")
    run_root = run.root

    write_run_meta(
        run_root / "run_meta.json",
        {
            "tag": "llm_sweep",
            "mode": "workflow",
            "k_values": list(range(0, 7)),
            "n_runs": 200,
            "agent": "LLMAgent",
            "client": "HeuristicLLMClient",
            "notes": "Per-run outputs are stored under run_root/<game>/",
        },
    )

    run_for_game(PrisonersDilemma(), "pd", run_root)
    run_for_game(StagHunt(), "stag_hunt", run_root)
    run_for_game(BattleOfSexes(), "battle_of_sexes", run_root)
    run_for_game(UltimatumGame(), "ultimatum", run_root)

    print(f"\nDONE. Run directory: {run_root}")


if __name__ == "__main__":
    main()
