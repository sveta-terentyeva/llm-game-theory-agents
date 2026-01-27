from llmgt.games import PrisonersDilemma
from llmgt.logging.records import ChatMessage
from llmgt.sim.agreement import extract_agreed_action_pair, agreement_hit


def test_extract_explicit_agreement():
    msgs = [
        ChatMessage(role="agent_a", content="Maybe we should cooperate"),
        ChatMessage(role="agent_b", content="Agreed on (C,C)"),
    ]
    assert extract_agreed_action_pair(msgs) == ("C", "C")


def test_agreement_by_explicit_plan():
    g = PrisonersDilemma()
    msgs = [
        ChatMessage(role="agent_a", content="Let's do (C,C)"),
        ChatMessage(role="agent_b", content="Yes"),
    ]
    assert agreement_hit(
        game=g,
        messages=msgs,
        final_action_a="C",
        final_action_b="C",
    )


def test_agreement_by_theory_nash():
    g = PrisonersDilemma()
    assert agreement_hit(
        game=g,
        messages=[],
        final_action_a="D",
        final_action_b="D",
    )


def test_no_agreement_random():
    g = PrisonersDilemma()
    assert not agreement_hit(
        game=g,
        messages=[],
        final_action_a="C",
        final_action_b="D",
    )
