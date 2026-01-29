"""Run a simple Prisoner's Dilemma experiment and log episodes to JSONL.

  python -m scripts.run_pd
"""

from __future__ import annotations

from pathlib import Path

from llmgt.agents.simple import FixedActionAgent, EchoAgent
from llmgt.games.prisoners_dilemma import PrisonersDilemma
from llmgt.logging.jsonl_logger import JsonlLogger
from llmgt.sim.runner import run_experiment, summarize_theory_hits, summarize_experiment


def main() -> None:
    game = PrisonersDilemma()

    agent_a = FixedActionAgent(name="fixed_D_A", action=game.D)
    agent_b = FixedActionAgent(name="fixed_D_B", action=game.D)

    #agent_a = EchoAgent(name="echo_A", action=game.C)
    #agent_b = EchoAgent(name="echo_B", action=game.D)

    out_dir = Path("data/runs")
    logger = JsonlLogger(out_dir=out_dir, filename="pd_episodes.jsonl")

    records = run_experiment(
        game=game,
        agent_a=agent_a,
        agent_b=agent_b,
        n_episodes=50,
        mode="no_workflow",
        max_comm_rounds=0,
        logger=logger,
        episode_id_prefix="pd",
    )

    #print("First 5 action pairs:", [(r.action_a, r.action_b) for r in records[:5]])
    #print("Agent A config:", agent_a)
    #print("Agent B config:", agent_b)

    stats = summarize_theory_hits(records)
    summary = summarize_experiment(game, records)

    print("=== PD experiment ===")
    print(f"Logged: {out_dir / 'pd_episodes.jsonl'}")
    print(f"Nash rate: {stats['nash_rate']:.2%}")
    print(f"Pareto rate: {stats['pareto_rate']:.2%}")
    print(f"Agreement rate: {stats['agreement_rate']:.2%}")
    print("\nConclusion:")
    print(summary.conclusion)


if __name__ == "__main__":
    main()
