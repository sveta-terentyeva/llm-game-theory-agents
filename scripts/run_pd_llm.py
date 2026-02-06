"""
Run Prisoner's Dilemma with LLM agents (heuristic backend).
Outputs are written to data/runs/<run_id>/...
"""
from __future__ import annotations

from llmgt.games.prisoners_dilemma import PrisonersDilemma
from llmgt.agents.llm import LLMAgent
from llmgt.llm import HeuristicLLMClient
from llmgt.logging.jsonl_logger import JsonlLogger
from llmgt.logging.run_meta import write_run_meta
from llmgt.sim.run_dir import make_run_dir
from llmgt.sim.runner import run_experiment, summarize_theory_hits, summarize_experiment


def main() -> None:
    game = PrisonersDilemma()

    client_a = HeuristicLLMClient()
    client_b = HeuristicLLMClient()

    agent_a = LLMAgent(name="llm_A", client=client_a, role="agent_a")
    agent_b = LLMAgent(name="llm_B", client=client_b, role="agent_b")

    n_episodes = 100
    max_comm_rounds = 5
    mode = "workflow"

    run = make_run_dir(tag="pd_llm_heuristic_workflow")
    write_run_meta(
        run.root / "run_meta.json",
        {
            "tag": "pd_llm_heuristic_workflow",
            "game": game.name,
            "mode": mode,
            "n_episodes": n_episodes,
            "max_comm_rounds": max_comm_rounds,
            "agent_a": "LLMAgent(HeuristicLLMClient)",
            "agent_b": "LLMAgent(HeuristicLLMClient)",
        },
    )

    logger = JsonlLogger(out_dir=run.logs_dir, filename="episodes.jsonl", overwrite=True)

    records = run_experiment(
        game=game,
        agent_a=agent_a,
        agent_b=agent_b,
        n_episodes=n_episodes,
        mode=mode,
        max_comm_rounds=max_comm_rounds,
        logger=logger,
        episode_id_prefix="pd_llm",
    )

    stats = summarize_theory_hits(records)
    summary = summarize_experiment(game, records)

    write_run_meta(
        run.root / "summary.json",
        {
            "agreement_rate": float(stats["agreement_rate"]),
            "nash_rate": float(stats["nash_rate"]),
            "pareto_rate": float(stats["pareto_rate"]),
            "conclusion": summary.conclusion,
        },
    )

    print("=== PD LLM (heuristic) workflow ===")
    print(f"Run dir: {run.root}")
    print(f"Logged:  {run.logs_dir / 'episodes.jsonl'}")
    print(f"Agreement rate: {stats['agreement_rate']:.2%}")
    print(f"Nash rate:      {stats['nash_rate']:.2%}")
    print(f"Pareto rate:    {stats['pareto_rate']:.2%}")


if __name__ == "__main__":
    main()
