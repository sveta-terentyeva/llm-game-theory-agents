"""
Run Prisoner's Dilemma with LLM agents (heuristic backend).
"""
from pathlib import Path

from llmgt.games.prisoners_dilemma import PrisonersDilemma
from llmgt.agents.llm import LLMAgent
from llmgt.llm import HeuristicLLMClient
from llmgt.logging.jsonl_logger import JsonlLogger
from llmgt.sim.runner import run_experiment, summarize_theory_hits


def main() -> None:
    game = PrisonersDilemma()

    client_a = HeuristicLLMClient()
    client_b = HeuristicLLMClient()

    agent_a = LLMAgent(name="llm_A", client=client_a)
    agent_b = LLMAgent(name="llm_B", client=client_b)

    out_dir = Path("data/runs")
    logger = JsonlLogger(out_dir=out_dir, filename="pd_llm_workflow.jsonl")

    records = run_experiment(
        game=game,
        agent_a=agent_a,
        agent_b=agent_b,
        n_episodes=100,
        mode="workflow",
        max_comm_rounds=5,
        logger=logger,
        episode_id_prefix="pd_llm",
    )

    stats = summarize_theory_hits(records)

    print("=== PD LLM workflow experiment ===")
    print(f"Agreement rate: {stats['agreement_rate']:.2%}")
    print(f"Nash rate: {stats['nash_rate']:.2%}")
    print(f"Pareto rate: {stats['pareto_rate']:.2%}")
    print(f"Logged to: {out_dir / 'pd_llm_workflow.jsonl'}")


if __name__ == "__main__":
    main()
