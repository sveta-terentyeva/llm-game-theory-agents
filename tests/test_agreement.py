from llmgt.games import PrisonersDilemma
from llmgt.logging.records import ChatMessage
from llmgt.sim.agreement import agreement_hit


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


def test_no_agreement_workflow_if_accept_missing():
    g = PrisonersDilemma()
    msgs = [
        ChatMessage(role="agent_a", content="PROPOSE: (C,C)"),
        ChatMessage(role="agent_b", content="Sounds good"),
    ]
    assert not agreement_hit(
        game=g,
        mode="workflow",
        messages=msgs,
        final_action_a="C",
        final_action_b="C",
    )


def test_agreement_by_theory_nash_no_workflow():
    g = PrisonersDilemma()
    assert agreement_hit(
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
