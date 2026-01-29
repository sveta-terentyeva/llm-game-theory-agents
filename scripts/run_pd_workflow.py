from pathlib import Path

from llmgt.agents.workflow import WorkflowProposerAgent, StochasticWorkflowResponderAgent
from llmgt.experiments.sweep import run_comm_sweep, summarize_by_k, write_csv
from llmgt.experiments.plotting import plot_metric_by_k
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

    records = run_comm_sweep(
        game=g,
        agent_a=agent_a,
        agent_b=agent_b,
        k_values=range(0, 7),
        n_runs=200,
        mode="workflow",
    )

    rows = summarize_by_k(records)

    out_dir = Path("data/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_metric_by_k(
        rows,
        metric="agreement_rate",
        title="Agreement rate vs communication budget (PD, workflow stochastic)",
        ylabel="Agreement rate",
        out_path=out_dir / "pd_workflow_agreement_rate.png",
    )

    plot_metric_by_k(
        rows,
        metric="mean_rounds_to_agreement",
        title="Rounds to agreement vs communication budget (PD, workflow stochastic)",
        ylabel="Mean rounds to agreement",
        out_path=out_dir / "pd_workflow_rounds_to_agreement.png",
    )

    write_csv(rows, Path("data/figures/pd_workflow_sweep.csv"))
    print("Saved figures to data/figures and CSV to data/figures/pd_workflow_sweep.csv")


if __name__ == "__main__":
    main()

