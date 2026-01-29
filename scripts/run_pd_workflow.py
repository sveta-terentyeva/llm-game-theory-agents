from pathlib import Path

from llmgt.agents.workflow import WorkflowProposerAgent, WorkflowResponderAgent
from llmgt.games.prisoners_dilemma import PrisonersDilemma
from llmgt.logging import JsonlLogger
from llmgt.sim.runner import run_experiment, summarize_theory_hits


def main() -> None:
    g = PrisonersDilemma()

    agent_a = WorkflowProposerAgent(name="wf_A", propose_pair=(g.C, g.C))
    agent_b = WorkflowResponderAgent(name="wf_B", fallback_action=g.D)

    out_dir = Path("data/runs")
    logger = JsonlLogger(out_dir=out_dir, filename="pd_workflow.jsonl")

    recs = run_experiment(
        game=g,
        agent_a=agent_a,
        agent_b=agent_b,
        n_episodes=30,
        mode="workflow",
        max_comm_rounds=2,
        logger=logger,
        episode_id_prefix="pd-wf",
    )

    stats = summarize_theory_hits(recs)
    print("=== PD workflow experiment ===")
    print(f"Nash rate: {stats['nash_rate']:.2%}")
    print(f"Pareto rate: {stats['pareto_rate']:.2%}")
    print(f"Agreement rate: {stats['agreement_rate']:.2%}")
    print(f"Logged: {out_dir / 'pd_workflow.jsonl'}")


if __name__ == "__main__":
    main()
