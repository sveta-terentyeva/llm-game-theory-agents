from pathlib import Path

from llmgt.agents.llm import LLMAgent
from llmgt.llm import HeuristicLLMClient
from llmgt.experiments.sweep import run_comm_sweep, summarize_by_k, write_csv
from llmgt.experiments.plotting import plot_metric_by_k

from llmgt.games.prisoners_dilemma import PrisonersDilemma
from llmgt.games.stag_hunt import StagHunt
from llmgt.games.battle_of_sexes import BattleOfSexes
from llmgt.games.ultimatum import UltimatumGame


def run_for_game(game, out_prefix: str) -> None:
    client_a = HeuristicLLMClient()
    client_b = HeuristicLLMClient()

    agent_a = LLMAgent(name=f"llm_A_{out_prefix}", client=client_a, role="agent_a")
    agent_b = LLMAgent(name=f"llm_B_{out_prefix}", client=client_b, role="agent_b")

    records = run_comm_sweep(
        game=game,
        agent_a=agent_a,
        agent_b=agent_b,
        k_values=range(0, 7),
        n_runs=200,
        mode="workflow",
    )

    rows = summarize_by_k(records)

    out_dir = Path("data/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(rows, Path(f"data/figures/{out_prefix}_llm_workflow_sweep.csv"))

    plot_metric_by_k(
        rows,
        metric="agreement_rate",
        title=f"Agreement rate vs K ({out_prefix}, LLM workflow)",
        ylabel="Agreement rate",
        out_path=out_dir / f"{out_prefix}_llm_workflow_agreement_rate.png",
    )

    plot_metric_by_k(
        rows,
        metric="mean_rounds_to_agreement",
        title=f"Rounds to agreement vs K ({out_prefix}, LLM workflow)",
        ylabel="Mean rounds to agreement",
        out_path=out_dir / f"{out_prefix}_llm_workflow_rounds_to_agreement.png",
    )

    print(f"[{out_prefix}] saved CSV + plots to data/figures/")


def main() -> None:
    run_for_game(PrisonersDilemma(), "pd")
    run_for_game(StagHunt(), "stag_hunt")
    run_for_game(BattleOfSexes(), "battle_of_sexes")
    run_for_game(UltimatumGame(), "ultimatum")


if __name__ == "__main__":
    main()
