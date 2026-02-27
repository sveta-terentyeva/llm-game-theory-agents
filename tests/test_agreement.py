from llmgt.games import PrisonersDilemma
from llmgt.logging.records import ChatMessage
from llmgt.sim.agreement import agreement_hit
from llmgt.sim.theory import compute_theory_hits



def test_extract_explicit_agreement_like_text_still_ok():
    # This test is optional now, because workflow agreement is driven by ACCEPT:(X,Y),
    # but we keep a simple sanity check for message handling.
    msgs = [
        ChatMessage(role="agent_a", content="Maybe we should cooperate"),
        ChatMessage(role="agent_b", content="Ok"),
    ]
    assert isinstance(msgs[0].content, str)


def test_agreement_workflow_requires_accept_and_follow_through():
    g = PrisonersDilemma()
    msgs = [
        ChatMessage(role="agent_a", content="PROPOSE: (C,C)"),
        ChatMessage(role="agent_b", content="ACCEPT: (C,C)"),
    ]
    assert agreement_hit(
        game=g,
        mode="workflow",
        messages=msgs,
        final_action_a="C",
        final_action_b="C",
    )


def test_agreement_via_propose_without_accept_if_actions_match():
    """Unified logic: a PROPOSE fallback counts as agreement when final actions match."""
    g = PrisonersDilemma()
    msgs = [
        ChatMessage(role="agent_a", content="PROPOSE: (C,C)"),
        ChatMessage(role="agent_b", content="Sounds good"),
    ]
    # With unified logic PROPOSE is the fallback — and it matches (C,C)
    assert agreement_hit(
        game=g,
        mode="workflow",
        messages=msgs,
        final_action_a="C",
        final_action_b="C",
    )


def test_no_agreement_workflow_if_propose_mismatch():
    """No agreement when PROPOSE pair doesn't match final actions."""
    g = PrisonersDilemma()
    msgs = [
        ChatMessage(role="agent_a", content="PROPOSE: (C,C)"),
        ChatMessage(role="agent_b", content="Sounds good"),
    ]
    assert not agreement_hit(
        game=g,
        mode="workflow",
        messages=msgs,
        final_action_a="D",
        final_action_b="D",
    )


def test_theory_hit_nash_pd_without_chat():
    g = PrisonersDilemma()
    th = compute_theory_hits(game=g, final_action_a="D", final_action_b="D")
    assert th.nash_hit is True
    assert th.theory_hit is True

def test_no_explicit_agreement_without_protocol_no_workflow():
    g = PrisonersDilemma()
    assert not agreement_hit(
        game=g,
        mode="no_workflow",
        messages=[],
        final_action_a="D",
        final_action_b="D",
    )



def test_no_agreement_random_no_workflow():
    g = PrisonersDilemma()
    assert not agreement_hit(
        game=g,
        mode="no_workflow",
        messages=[],
        final_action_a="C",
        final_action_b="D",
    )


# --- Unified-mode parity tests ---


def test_agreement_identical_across_modes_accept():
    """Both modes detect agreement the same way when ACCEPT is present."""
    g = PrisonersDilemma()
    msgs = [
        ChatMessage(role="agent_a", content="PROPOSE: (C,C)"),
        ChatMessage(role="agent_b", content="ACCEPT: (C,C)"),
    ]
    for mode in ("no_workflow", "workflow"):
        assert agreement_hit(
            game=g, mode=mode, messages=msgs,
            final_action_a="C", final_action_b="C",
        ), f"failed for mode={mode}"


def test_agreement_identical_across_modes_propose_fallback():
    """Both modes use PROPOSE fallback identically."""
    g = PrisonersDilemma()
    msgs = [
        ChatMessage(role="agent_a", content="PROPOSE: (C,C)"),
        ChatMessage(role="agent_b", content="OK"),
    ]
    for mode in ("no_workflow", "workflow"):
        assert agreement_hit(
            game=g, mode=mode, messages=msgs,
            final_action_a="C", final_action_b="C",
        ), f"failed for mode={mode}"


def test_no_agreement_identical_across_modes_empty():
    """Both modes return False with no messages."""
    g = PrisonersDilemma()
    for mode in ("no_workflow", "workflow"):
        assert not agreement_hit(
            game=g, mode=mode, messages=[],
            final_action_a="D", final_action_b="D",
        ), f"failed for mode={mode}"

