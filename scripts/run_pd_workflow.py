from __future__ import annotations

from llmgt.agents.workflow import WorkflowProposerAgent, StochasticWorkflowResponderAgent
from llmgt.experiments.sweep import run_comm_sweep, summarize_by_k, write_csv
from llmgt.experiments.plotting import plot_metric_by_k
from llmgt.logging.jsonl_logger import JsonlLogger
from llmgt.logging.run_meta import write_run_meta
from llmgt.sim.run_dir import make_run_dir
from llmgt.games.prisoners_dilemma import PrisonersDilemma


def main() -> None:
    g = PrisonersDilemma()

    agent_a = WorkflowProposerAgent(name="wf_A", propose_pair=(g.C, g.C))
    agent_b = StochasticWorkflowResponderAgent(
        name="wf_B_stochastic",
        fallback_action=g.D,
        base_p=0.10,
        step_p=0.20,
        seed=42,
    )

    run = make_run_dir(tag="pd_workflow_stochastic")
    write_run_meta(
        run.root / "run_meta.json",
        {
            "tag": "pd_workflow_stochastic",
            "game": g.name,
            "mode": "workflow",
            "k_values": list(range(0, 7)),
            "n_runs": 200,
            "agent_a": "WorkflowProposerAgent",
            "agent_b": "StochasticWorkflowResponderAgent",
        },
    )

    logger = JsonlLogger(out_dir=run.logs_dir, filename="episodes.jsonl", overwrite=True)

    records = run_comm_sweep(
        game=g,
        agent_a=agent_a,
        agent_b=agent_b,
        k_values=range(0, 7),
        n_runs=200,
        mode="workflow",
        logger=logger,
    )

    rows = summarize_by_k(records)
    write_csv(rows, run.root / "summary_by_k.csv")

    plot_metric_by_k(
        rows,
        metric="agreement_rate",
        title="prisoners_dilemma — agreement vs K (workflow stochastic)",
        ylabel="Agreement rate",
        out_path=run.figures_dir / "agreement_rate.png",
    )
    plot_metric_by_k(
        rows,
        metric="mean_rounds_to_agreement",
        title="prisoners_dilemma — rounds-to-agreement vs K (workflow stochastic)",
        ylabel="Mean rounds-to-agreement",
        out_path=run.figures_dir / "mean_rounds_to_agreement.png",
    )

    print(f"Saved to: {run.root}")


if __name__ == "__main__":
    main()
