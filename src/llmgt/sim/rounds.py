from __future__ import annotations
from typing import List, Optional

from llmgt.logging.records import ChatMessage
from llmgt.games.base import Game
from llmgt.sim.agreement import agreement_hit


def compute_rounds_to_agreement(
    *,
    game: Game,
    mode: str,
    messages: List[ChatMessage],
    final_action_a: str,
    final_action_b: str,
    max_comm_rounds: int,
) -> Optional[int]:

    non_system = [m for m in messages if m.role != "system"]

    for r in range(1, max_comm_rounds + 1):
        upto = non_system[: 2 * r]

        if agreement_hit(
            game=game,
            mode=mode,
            messages=upto,
            final_action_a=final_action_a,
            final_action_b=final_action_b,
        ):
            return r

    return None


