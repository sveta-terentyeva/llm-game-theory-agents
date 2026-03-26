"""Agreement detection between agents.

An "agreement" means the final actions match a pair that was explicitly
proposed/accepted during the communication phase.
"""

from __future__ import annotations

from llmgt.games.base import Game
from llmgt.logging.records import ChatMessage
from llmgt.sim.workflow import (
    extract_accepted_pair,
    extract_last_counter,
    extract_last_proposal,
)



def agreement_hit(
    *,
    game: Game,
    messages: list[ChatMessage],
    final_action_a: str,
    final_action_b: str,
    mode: str = "no_workflow",
) -> bool:
    """Detect agreement — unified logic for both modes.

    An agreement is detected when a structured protocol marker
    (ACCEPT > COUNTER > PROPOSE, in priority order) matches the
    final actions.  Both ``workflow`` and ``no_workflow`` modes use
    the same PROPOSE/COUNTER/ACCEPT negotiation protocol, so the
    detection logic is identical.
    """
    target = extract_accepted_pair(messages)
    if target is None:
        target = extract_last_counter(messages)
    if target is None:
        target = extract_last_proposal(messages)

    if target is None:
        return False

    return target == (final_action_a, final_action_b)
