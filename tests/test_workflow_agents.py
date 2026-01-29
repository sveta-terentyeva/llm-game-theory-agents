from llmgt.agents.workflow import WorkflowProposerAgent, WorkflowResponderAgent
from llmgt.games import PrisonersDilemma
from llmgt.sim.runner import run_episode


def test_workflow_agents_can_reach_accept():
    g = PrisonersDilemma()

    a = WorkflowProposerAgent(name="wf_A", propose_pair=(g.C, g.C))
    b = WorkflowResponderAgent(name="wf_B", fallback_action=g.D)

    rec = run_episode(
        episode_id="wf-1",
        game=g,
        agent_a=a,
        agent_b=b,
        mode="workflow",
        max_comm_rounds=2,
    )

    assert rec.action_a == g.C
    assert rec.action_b == g.C
    assert rec.agreement_hit is True
    assert rec.rounds_to_agreement == 1
