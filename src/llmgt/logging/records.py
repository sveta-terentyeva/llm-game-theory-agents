"""Run a simple Prisoner's Dilemma experiment and log episodes to JSONL.
  python -m scripts.run_pd
"""
from __future__ import annotations

from pathlib import Path

from llmgt.agents.simple import FixedActionAgent  # EchoAgent optional
from llmgt.games.prisoners_dilemma import PrisonersDilemma

# Logger import can vary depending on your file name.
# Prefer llmgt.logging.JsonlLogger if you export it there.
try:
    from llmgt.logging import JsonlLogger
except Exception:  # fallback
    from llmgt.logging.jsonl_logger import JsonlLogger

from llmgt.sim.runner import run_experiment, summarize_theory_hits


def main() -> None:
    game = PrisonersDilemma()

    # Baseline: always defect vs always defect
    agent_a = FixedActionAgent(name="fixed_D_A", action=game.D)
    agent_b = FixedActionAgent(name="fixed_D_B", action=game.D)

    out_dir = Path("data/runs")
    out_dir.mkdir(parents=True, exist_ok=True)

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

    stats = summarize_theory_hits(records)

    # Simple conclusion without summarize_experiment dependency
    theory_nash = sorted(list(game.nash_equilibria()))
    theory_pareto = sorted(list(game.pareto_optima()))

    conclusion = (
        f"Over n={int(stats['n_episodes'])} episodes: "
        f"Nash-hit rate {stats['nash_rate']:.2%}, "
        f"Pareto-hit rate {stats['pareto_rate']:.2%}, "
        f"Agreement rate {stats['agreement_rate']:.2%}. "
        f"Theoretical Nash set: {theory_nash}. "
        f"Theoretical Pareto set: {theory_pareto}."
    )

    print("=== PD experiment ===")
    print(f"Logged: {out_dir / 'pd_episodes.jsonl'}")
    print(f"Nash rate: {stats['nash_rate']:.2%}")
    print(f"Pareto rate: {stats['pareto_rate']:.2%}")
    print(f"Agreement rate: {stats['agreement_rate']:.2%}")
    print("\nConclusion:")
    print(conclusion)


if __name__ == "__main__":
    main()
