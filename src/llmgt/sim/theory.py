from __future__ import annotations

from dataclasses import dataclass
from llmgt.games.base import Game


@dataclass(frozen=True)
class TheoryHits:
    nash_hit: bool
    pareto_hit: bool
    pareto_nash_hit: bool
    theory_hit: bool


def theory_target_set(game: Game) -> set[tuple[str, str]]:
    """
    Paper-style success set:
    - If there exists a Pareto-optimal Nash equilibrium, target that set.
    - Otherwise target the Nash equilibrium set.
    """
    nash = set(game.nash_equilibria())
    pareto = set(game.pareto_optima())
    pareto_nash = nash & pareto
    return pareto_nash if pareto_nash else nash


def compute_theory_hits(*, game: Game, final_action_a: str, final_action_b: str) -> TheoryHits:
    prof = (final_action_a, final_action_b)
    nash = prof in set(game.nash_equilibria())
    pareto = prof in set(game.pareto_optima())
    pareto_nash = nash and pareto
    theory = prof in theory_target_set(game)
    return TheoryHits(
        nash_hit=nash,
        pareto_hit=pareto,
        pareto_nash_hit=pareto_nash,
        theory_hit=theory,
    )
