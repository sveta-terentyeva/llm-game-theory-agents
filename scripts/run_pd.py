"""Run a simple Prisoner's Dilemma experiment and log episodes to JSONL.

  python -m scripts.run_pd
"""
from __future__ import annotations

from llmgt.agents.simple import FixedActionAgent  # EchoAgent можна залишити за бажанням
from llmgt.games.prisoners_dilemma import PrisonersDilemma
from llmgt.logging.jsonl_logger import JsonlLogger
from llmgt.logging.run_meta import write_run_meta
from llmgt.sim.run_dir import make_run_dir
from llmgt.sim.runner import run_experiment, summarize_theory_hits, summarize_experiment


def main() -> None:
    game = PrisonersDilemma()

    agent_a = FixedActionAgent(name="fixed_D_A", action=game.D)
    agent_b = FixedActionAgent(name="fixed_D_B", action=game.D)

    run = make_run_dir(tag="pd_fixed_baseline")
    write_run_meta(
        run.root / "run_meta.json",
        {
            "tag": "pd_fixed_baseline",
            "game": game.name,
            "mode": "no_workflow",
            "n_episodes": 50,
            "max_comm_rounds": 0,
            "agent_a": "FixedActionAgent(D)",
            "agent_b": "FixedActionAgent(D)",
        },
    )

    logger = JsonlLogger(out_dir=run.logs_dir, filename="episodes.jsonl", overwrite=True)

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

    print("=== PD fixed baseline ===")
    print(f"Run dir: {run.root}")
    print(f"Logged:  {run.logs_dir / 'episodes.jsonl'}")
    print(f"Nash rate:      {stats['nash_rate']:.2%}")
    print(f"Pareto rate:    {stats['pareto_rate']:.2%}")
    print(f"Agreement rate: {stats['agreement_rate']:.2%}")
    print("\nConclusion:")
    print(summary.conclusion)


if __name__ == "__main__":
    main()
