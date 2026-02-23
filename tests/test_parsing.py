"""Tests for the centralised parsing module."""

from llmgt.agents.parsing import (
    extract_accepted_pair,
    extract_last_pair,
    format_history,
    parse_action,
    parse_protocol_reply,
    payoff_table,
    sanitize_pair,
)
from llmgt.games import PrisonersDilemma, BattleOfSexes, UltimatumGame
from llmgt.logging.records import ChatMessage


def test_format_history_limits():
    msgs = [ChatMessage(role="agent_a", content=f"msg {i}") for i in range(20)]
    out = format_history(msgs, limit=5)
    lines = out.strip().splitlines()
    assert len(lines) == 5


def test_payoff_table_contains_actions():
    g = PrisonersDilemma()
    table = payoff_table(g)
    assert "C" in table
    assert "D" in table
    assert "3.0" in table


def test_extract_last_pair_multiple():
    msgs = [
        ChatMessage(role="agent_a", content="PROPOSE: (C,C)"),
        ChatMessage(role="agent_b", content="COUNTER: (D,D)"),
    ]
    pair = extract_last_pair(msgs)
    assert pair == ("D", "D")


def test_extract_accepted_pair():
    msgs = [
        ChatMessage(role="agent_a", content="PROPOSE: (C,C)"),
        ChatMessage(role="agent_b", content="ACCEPT: (C,C)"),
    ]
    pair = extract_accepted_pair(msgs)
    assert pair == ("C", "C")


def test_extract_accepted_pair_none():
    msgs = [
        ChatMessage(role="agent_a", content="PROPOSE: (C,C)"),
    ]
    assert extract_accepted_pair(msgs) is None


def test_sanitize_pair_valid():
    g = PrisonersDilemma()
    assert sanitize_pair(g, "C", "D") == ("C", "D")


def test_sanitize_pair_invalid():
    g = PrisonersDilemma()
    assert sanitize_pair(g, "X", "Y") is None


def test_sanitize_pair_ultimatum():
    g = UltimatumGame()
    assert sanitize_pair(g, "F", "A") == ("F", "A")
    assert sanitize_pair(g, "A", "F") is None  # swapped roles


def test_parse_protocol_reply_accept():
    g = PrisonersDilemma()
    result = parse_protocol_reply(g, "ACCEPT: (C,C)")
    assert result == "ACCEPT: (C,C)"


def test_parse_protocol_reply_propose():
    g = PrisonersDilemma()
    result = parse_protocol_reply(g, "PROPOSE: (D,D)")
    assert result == "PROPOSE: (D,D)"


def test_parse_protocol_reply_counter():
    g = PrisonersDilemma()
    result = parse_protocol_reply(g, "COUNTER: (C,D)")
    assert result == "COUNTER: (C,D)"


def test_parse_protocol_reply_invalid():
    g = PrisonersDilemma()
    result = parse_protocol_reply(g, "I think we should cooperate.")
    assert result is None


def test_parse_action_exact():
    assert parse_action("C", ("C", "D")) == "C"


def test_parse_action_prefix():
    assert parse_action("ACTION: D", ("C", "D")) == "D"


def test_parse_action_embedded():
    assert parse_action("I choose C for cooperation", ("C", "D")) == "C"


def test_parse_action_invalid():
    assert parse_action("nonsense", ("C", "D")) is None

