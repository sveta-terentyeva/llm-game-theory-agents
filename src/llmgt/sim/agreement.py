from __future__ import annotations

from llmgt.games.base import Game
from llmgt.logging.records import ChatMessage
from llmgt.sim.workflow import extract_accepted_pair


def agreement_hit(
    *,
    game: Game,
    messages: list[ChatMessage],
    final_action_a: str,
    final_action_b: str,
    mode: str = "no_workflow",
) -> bool:

    if mode == "workflow":
        accepted = extract_accepted_pair(messages)
        return accepted == (final_action_a, final_action_b)

    # no_workflow baseline:
    if (final_action_a, final_action_b) in game.nash_equilibria():
        return True
    if (final_action_a, final_action_b) in game.pareto_optima():
        return True
    return False
