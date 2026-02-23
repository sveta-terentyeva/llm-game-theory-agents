"""Tests for the fixed compute_rounds_to_theory_hit function."""

from llmgt.games import PrisonersDilemma
from llmgt.logging.records import ChatMessage
from llmgt.sim.rounds import compute_rounds_to_theory_hit


def test_theory_hit_without_comm_rounds():
    """If theory hit + no comm → return 1."""
    g = PrisonersDilemma()
    r = compute_rounds_to_theory_hit(
        game=g,
        mode="no_workflow",
        messages=[],
        final_action_a="D",
        final_action_b="D",
        max_comm_rounds=0,
    )
    assert r == 1


def test_theory_hit_no_theory_outcome():
    """If NOT a theory hit → return None."""
    g = PrisonersDilemma()
    r = compute_rounds_to_theory_hit(
        game=g,
        mode="no_workflow",
        messages=[],
        final_action_a="C",
        final_action_b="D",
        max_comm_rounds=3,
    )
    assert r is None


def test_theory_hit_with_accept_in_workflow():
    """If theory hit + ACCEPT found in round 1 → return 1."""
    g = PrisonersDilemma()
    # Nash for PD is (D,D) — theory target is (D,D)
    msgs = [
        ChatMessage(role="agent_a", content="PROPOSE: (D,D)"),
        ChatMessage(role="agent_b", content="ACCEPT: (D,D)"),
    ]
    r = compute_rounds_to_theory_hit(
        game=g,
        mode="workflow",
        messages=msgs,
        final_action_a="D",
        final_action_b="D",
        max_comm_rounds=3,
    )
    assert r == 1


def test_theory_hit_with_delayed_accept():
    """ACCEPT appears only in round 2."""
    g = PrisonersDilemma()
    msgs = [
        ChatMessage(role="agent_a", content="PROPOSE: (C,C)"),
        ChatMessage(role="agent_b", content="COUNTER: (D,D)"),
        ChatMessage(role="agent_a", content="PROPOSE: (D,D)"),
        ChatMessage(role="agent_b", content="ACCEPT: (D,D)"),
    ]
    r = compute_rounds_to_theory_hit(
        game=g,
        mode="workflow",
        messages=msgs,
        final_action_a="D",
        final_action_b="D",
        max_comm_rounds=3,
    )
    assert r == 2


def test_theory_hit_no_agreement_in_chat_fallback():
    """Theory hit, comm exists, but no ACCEPT/PROPOSE matching final actions.
    Should still return a value (fallback to max_comm_rounds)."""
    g = PrisonersDilemma()
    msgs = [
        ChatMessage(role="agent_a", content="Hello"),
        ChatMessage(role="agent_b", content="Hi"),
    ]
    r = compute_rounds_to_theory_hit(
        game=g,
        mode="no_workflow",
        messages=msgs,
        final_action_a="D",
        final_action_b="D",
        max_comm_rounds=3,
    )
    # The agents ended up at (D,D) which is Nash, but there's no agreement
    # marker in the chat. The function should return max_comm_rounds.
    assert r == 3

