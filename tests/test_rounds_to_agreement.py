from llmgt.games import PrisonersDilemma
from llmgt.logging.records import ChatMessage
from llmgt.sim.rounds import compute_rounds_to_agreement
from llmgt.sim.rounds import compute_rounds_to_theory_hit



def test_rounds_to_agreement_workflow_accept_round_1():
    g = PrisonersDilemma()

    messages = [
        ChatMessage(role="agent_a", content="PROPOSE: (C,C)"),
        ChatMessage(role="agent_b", content="ACCEPT: (C,C)"),
    ]

    r = compute_rounds_to_agreement(
        game=g,
        mode="workflow",
        messages=messages,
        final_action_a="C",
        final_action_b="C",
        max_comm_rounds=3,
    )

    assert r == 1


def test_rounds_to_agreement_workflow_propose_matches():
    """Unified logic: PROPOSE: (C,C) + no ACCEPT still counts as agreement
    at round 1 because the PROPOSE fallback matches the final actions."""
    g = PrisonersDilemma()

    messages = [
        ChatMessage(role="agent_a", content="PROPOSE: (C,C)"),
        ChatMessage(role="agent_b", content="No"),
    ]

    r = compute_rounds_to_agreement(
        game=g,
        mode="workflow",
        messages=messages,
        final_action_a="C",
        final_action_b="C",
        max_comm_rounds=2,
    )

    assert r == 1


def test_rounds_to_agreement_no_protocol_markers():
    """No agreement when no protocol markers appear in messages."""
    g = PrisonersDilemma()

    messages = [
        ChatMessage(role="agent_a", content="Let's cooperate"),
        ChatMessage(role="agent_b", content="No way"),
    ]

    r = compute_rounds_to_agreement(
        game=g,
        mode="workflow",
        messages=messages,
        final_action_a="C",
        final_action_b="C",
        max_comm_rounds=2,
    )

    assert r is None


def test_rounds_to_theory_hit_nash_without_chat_no_workflow():
    g = PrisonersDilemma()

    r = compute_rounds_to_theory_hit(
        game=g,
        mode="no_workflow",
        messages=[],
        final_action_a="D",
        final_action_b="D",
        max_comm_rounds=3,
    )

    assert r == 1



