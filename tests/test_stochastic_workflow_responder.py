from llmgt.agents.workflow import WorkflowProposerAgent, StochasticWorkflowResponderAgent
from llmgt.games import PrisonersDilemma
from llmgt.sim.runner import run_episode


def test_stochastic_responder_accepts_with_high_prob_in_late_rounds():
    g = PrisonersDilemma()

    a = WorkflowProposerAgent(name="wf_A", propose_pair=(g.C, g.C))
    b = StochasticWorkflowResponderAgent(
        name="wf_B_stochastic",
        fallback_action=g.D,
        base_p=0.0,
        step_p=1.0,
        seed=123,
    )

    rec = run_episode(
        episode_id="wf-stoch-1",
        game=g,
        agent_a=a,
        agent_b=b,
        mode="workflow",
        max_comm_rounds=2,
    )

    assert rec.agreement_hit is True
    assert rec.action_a == g.C
    assert rec.action_b == g.C
    assert rec.rounds_to_agreement == 2
