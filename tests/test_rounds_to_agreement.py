from llmgt.games import PrisonersDilemma
from llmgt.logging.records import ChatMessage
from llmgt.sim.rounds import compute_rounds_to_agreement


def test_rounds_to_agreement_explicit_round_1():
    g = PrisonersDilemma()

    messages = [
        ChatMessage(role="agent_a", content="Let's do (C,C)"),
        ChatMessage(role="agent_b", content="Agreed"),
    ]

    r = compute_rounds_to_agreement(
        game=g,
        messages=messages,
        final_action_a="C",
        final_action_b="C",
        max_comm_rounds=3,
    )

    assert r == 1


def test_rounds_to_agreement_never():
    g = PrisonersDilemma()

    messages = [
        ChatMessage(role="agent_a", content="I will defect"),
        ChatMessage(role="agent_b", content="I will cooperate"),
    ]

    r = compute_rounds_to_agreement(
        game=g,
        messages=messages,
        final_action_a="C",
        final_action_b="D",
        max_comm_rounds=2,
    )

    assert r is None


def test_rounds_to_agreement_nash_without_chat():
    g = PrisonersDilemma()

    r = compute_rounds_to_agreement(
        game=g,
        messages=[],
        final_action_a="D",
        final_action_b="D",
        max_comm_rounds=3,
    )

    # Nash is known immediately
    assert r == 1


