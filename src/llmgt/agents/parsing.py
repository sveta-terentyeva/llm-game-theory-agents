"""Shared parsing utilities for agent message/action extraction.

This module centralises the regex patterns and helpers that were previously
duplicated across ``llm.py``, ``strategic.py``, and ``workflow_reasoner.py``.
"""

from __future__ import annotations

import re
from typing import Optional

from llmgt.games.base import Game
from llmgt.logging.records import ChatMessage

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_PAIR_RE = re.compile(r"\(\s*([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_]+)\s*\)")
_PROPOSE_RE = re.compile(r"\bPROPOSE\s*:\s*" + _PAIR_RE.pattern)
_COUNTER_RE = re.compile(r"\bCOUNTER\s*:\s*" + _PAIR_RE.pattern)
_ACCEPT_RE = re.compile(r"\bACCEPT\s*:\s*" + _PAIR_RE.pattern)

# ---------------------------------------------------------------------------
# History formatting
# ---------------------------------------------------------------------------


def format_history(messages: list[ChatMessage], limit: int = 12) -> str:
    """Return the last *limit* messages formatted as ``role: content``."""
    tail = messages[-limit:] if len(messages) > limit else messages
    return "\n".join(f"{m.role}: {m.content}" for m in tail)


# ---------------------------------------------------------------------------
# Payoff-table rendering (for prompts)
# ---------------------------------------------------------------------------


def payoff_table(game: Game) -> str:
    """Render a compact ASCII payoff matrix for inclusion in LLM prompts."""
    a_actions = game.actions_a()
    b_actions = game.actions_b()

    header = "A\\B | " + " | ".join(b_actions)
    sep = "-" * len(header)
    rows = [header, sep]
    for a in a_actions:
        cells = []
        for b in b_actions:
            pa, pb = game.payoff(a, b)
            cells.append(f"{pa:.1f},{pb:.1f}")
        rows.append(f"{a:<3} | " + " | ".join(cells))
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Protocol line extraction
# ---------------------------------------------------------------------------


def extract_last_pair(messages: list[ChatMessage]) -> Optional[tuple[str, str]]:
    """Return the last ``(X, Y)`` pair from any PROPOSE/COUNTER/ACCEPT line."""
    last: Optional[tuple[str, str]] = None
    for m in messages:
        for rx in (_PROPOSE_RE, _COUNTER_RE, _ACCEPT_RE):
            mm = rx.search(m.content)
            if mm:
                last = (mm.group(1), mm.group(2))
    return last


def extract_accepted_pair(messages: list[ChatMessage]) -> Optional[tuple[str, str]]:
    """Return the last ``ACCEPT: (X, Y)`` pair, or *None*."""
    for m in messages:
        mm = _ACCEPT_RE.search(m.content)
        if mm:
            return (mm.group(1), mm.group(2))
    return None


def sanitize_pair(
    game: Game, x: str, y: str
) -> Optional[tuple[str, str]]:
    """Return ``(x, y)`` only if they are valid actions; otherwise *None*."""
    if x in game.actions_a() and y in game.actions_b():
        return (x, y)
    return None


def parse_protocol_reply(game: Game, text: str) -> Optional[str]:
    """Parse an LLM reply and return a valid protocol line, or *None*.

    Checks in order: ACCEPT, COUNTER, PROPOSE.
    """
    t = text.strip()

    for label, rx in [("ACCEPT", _ACCEPT_RE), ("COUNTER", _COUNTER_RE), ("PROPOSE", _PROPOSE_RE)]:
        m = rx.search(t)
        if m:
            pair = sanitize_pair(game, m.group(1), m.group(2))
            if pair:
                return f"{label}: ({pair[0]},{pair[1]})"

    return None


# ---------------------------------------------------------------------------
# Action token extraction
# ---------------------------------------------------------------------------


def parse_action(text: str, allowed: tuple[str, ...]) -> Optional[str]:
    """Extract a valid action token from *text*, or *None*.

    Tries several heuristics: exact match, prefix stripping, token search.
    """
    t = text.strip()
    candidates = [t]

    if "\n" in t:
        candidates.append(t.splitlines()[0].strip())

    for prefix in ("ACTION:", "Action:", "action:", "Final:", "final:"):
        if t.startswith(prefix):
            candidates.append(t[len(prefix):].strip())

    candidates.append(t.strip("()[]{} ").strip())

    # Exact match
    for c in candidates:
        for a in allowed:
            if c == a:
                return a

    # Substring match
    for a in allowed:
        if f" {a}" in f" {t} ":
            return a

    return None

