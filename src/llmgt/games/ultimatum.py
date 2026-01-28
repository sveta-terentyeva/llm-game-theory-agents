from __future__ import annotations
from llmgt.games.base import Game


class UltimatumGame(Game):
    """
    Ultimatum Game.

    Actions:
      Proposer:
        L = Low offer
        F = Fair offer
      Responder:
        A = Accept
        R = Reject

    Payoffs (A,B):
      (L,A) -> (3,1)
      (F,A) -> (2,2)
      (*,R) -> (0,0)

    Theory:
      Nash equilibrium: (L,A)
      Pareto optimum:   (F,A)
    """

    name = "ultimatum_game"

    L = "L"
    F = "F"
    A = "A"
    R = "R"

    def actions(self) -> tuple[str, str, str, str]:
        return (self.L, self.F, self.A, self.R)

    def payoff(self, a: str, b: str) -> tuple[float, float]:
        if a == self.L and b == self.A:
            return (3.0, 1.0)
        if a == self.F and b == self.A:
            return (2.0, 2.0)
        if b == self.R:
            return (0.0, 0.0)
        raise ValueError(f"Invalid actions: a={a!r}, b={b!r}")

    def nash_equilibria(self) -> set[tuple[str, str]]:
        return {(self.L, self.A)}

    def pareto_optima(self) -> set[tuple[str, str]]:
        return {(self.F, self.A)}
