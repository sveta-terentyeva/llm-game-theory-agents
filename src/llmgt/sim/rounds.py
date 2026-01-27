from __future__ import annotations
from typing import List, Optional

from llmgt.logging.records import ChatMessage
from llmgt.games.base import Game
from llmgt.sim.agreement import agreement_hit


def compute_rounds_to_agreement(
    *,
    game: Game,
    messages: List[ChatMessage],
    final_action_a: str,
    final_action_b: str,
    max_comm_rounds: int,
) -> Optional[int]:
    """
    Returns the minimal round index (1-based) after which agreement
    is already determined and consistent with final actions.

    If agreement is never reached, returns None.
    """

    non_system = [m for m in messages if m.role != "system"]

    for r in range(1, max_comm_rounds + 1):
        upto = non_system[: 2 * r]

        if agreement_hit(
            game=game,
            messages=upto,
            final_action_a=final_action_a,
            final_action_b=final_action_b,
        ):
            return r

    return None
