"""Utilities for computing round-based metrics.

Each "round" is one pair of messages (agent_a, agent_b).  The helpers here
scan the conversation prefix up to round *r* to decide whether agreement /
theory-hit had already been established at that point.
"""

from __future__ import annotations

from typing import List, Optional

from llmgt.logging.records import ChatMessage
from llmgt.games.base import Game
from llmgt.sim.agreement import agreement_hit
from llmgt.sim.theory import compute_theory_hits


def _is_action_line(m: ChatMessage) -> bool:
    """Return *True* if the message represents a final action declaration."""
    return isinstance(m.content, str) and m.content.strip().upper().startswith("ACTION:")


def _comm_messages(messages: List[ChatMessage]) -> List[ChatMessage]:
    """Filter out system messages and ACTION: lines."""
    return [m for m in messages if m.role != "system" and not _is_action_line(m)]


def compute_rounds_to_agreement(
    *,
    game: Game,
    mode: str,
    messages: List[ChatMessage],
    final_action_a: str,
    final_action_b: str,
    max_comm_rounds: int,
) -> Optional[int]:
    """Return the earliest round at which agreement is detected, or *None*."""
    comm = _comm_messages(messages)

    for r in range(1, max_comm_rounds + 1):
        prefix = comm[: 2 * r]
        if agreement_hit(
            game=game,
            mode=mode,
            messages=prefix,
            final_action_a=final_action_a,
            final_action_b=final_action_b,
        ):
            return r

    return None


def compute_rounds_to_theory_hit(
    *,
    game: Game,
    mode: str,
    messages: List[ChatMessage],
    final_action_a: str,
    final_action_b: str,
    max_comm_rounds: int,
) -> Optional[int]:
    """Return the earliest round at which the final action pair is a theory hit.

    The theory-hit depends only on the *final* actions (not on intermediate
    proposals), so the metric really answers: "given that the agents ended up
    at this outcome, at which round could we first observe agreement on it?"

    If there are no communication messages and the outcome is a theory hit,
    we return 1 (immediate hit).
    """
    th = compute_theory_hits(
        game=game,
        final_action_a=final_action_a,
        final_action_b=final_action_b,
    )
    if not th.theory_hit:
        return None

    comm = _comm_messages(messages)
    if not comm:
        return 1

    # Scan prefixes: return the earliest round where an agreement on the
    # theory-hit outcome appears in the conversation prefix.
    for r in range(1, max_comm_rounds + 1):
        prefix = comm[: 2 * r]
        if agreement_hit(
            game=game,
            mode=mode,
            messages=prefix,
            final_action_a=final_action_a,
            final_action_b=final_action_b,
        ):
            return r

    # Theory hit happened but no explicit agreement was found — treat the
    # decision round itself as the hit point.
    return max_comm_rounds if max_comm_rounds > 0 else 1

