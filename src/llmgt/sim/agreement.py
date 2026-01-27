from __future__ import annotations
import re
from typing import Iterable

from llmgt.logging.records import ChatMessage
from llmgt.games.base import Game


_ACTION_PAIR_RE = re.compile(r"\(([A-Za-z]+)\s*,\s*([A-Za-z]+)\)")


def extract_agreed_action_pair(
    messages: Iterable[ChatMessage],
) -> tuple[str, str] | None:
    """
    Look for explicit agreement in chat, e.g. "(C,C)".
    Returns the first agreed pair if found.
    """
    for m in messages:
        match = _ACTION_PAIR_RE.search(m.content)
        if match:
            return match.group(1), match.group(2)
    return None


def agreement_hit(
    *,
    game: Game,
    messages: list[ChatMessage],
    final_action_a: str,
    final_action_b: str,
) -> bool:
    """
    Agreement is reached if:
      1) Explicit agreement in chat AND actions follow it, OR
      2) Final actions are Nash OR Pareto optimal.
    """
    agreed = extract_agreed_action_pair(messages)
    if agreed is not None:
        return agreed == (final_action_a, final_action_b)

    if (final_action_a, final_action_b) in game.nash_equilibria():
        return True

    if (final_action_a, final_action_b) in game.pareto_optima():
        return True

    return False
