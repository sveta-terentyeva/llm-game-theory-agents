from __future__ import annotations

from typing import List, Optional

from llmgt.logging.records import ChatMessage
from llmgt.games.base import Game
from llmgt.sim.agreement import agreement_hit


def _is_action_line(m: ChatMessage) -> bool:
    return isinstance(m.content, str) and m.content.strip().upper().startswith("ACTION:")


def compute_rounds_to_agreement(
    *,
    game: Game,
    mode: str,
    messages: List[ChatMessage],
    final_action_a: str,
    final_action_b: str,
    max_comm_rounds: int,
) -> Optional[int]:

    comm = [m for m in messages if m.role != "system" and not _is_action_line(m)]

    for r in range(1, max_comm_rounds + 1):
        upto = comm[: 2 * r]

        if agreement_hit(
            game=game,
            mode=mode,
            messages=upto,
            final_action_a=final_action_a,
            final_action_b=final_action_b,
        ):
            return r

    return None


