from __future__ import annotations
from llmgt.games.base import Game


class PrisonersDilemma(Game):
    """
    Classic Prisoner's Dilemma (2x2).

    Actions:
      C = cooperate
      D = defect

    Payoffs (A,B):
      (C,C) -> (3,3)
      (C,D) -> (0,5)
      (D,C) -> (5,0)
      (D,D) -> (1,1)

    Theory:
      Nash equilibrium: (D,D)
      Pareto optimum:   (C,C)
    """

    name = "prisoners_dilemma"

    C = "C"
    D = "D"

    def actions(self) -> tuple[str, str]:
        return (self.C, self.D)

    def payoff(self, a: str, b: str) -> tuple[float, float]:
        if a == self.C and b == self.C:
            return (3, 3)
        if a == self.C and b == self.D:
            return (0, 5)
        if a == self.D and b == self.C:
            return (5, 0)
        if a == self.D and b == self.D:
            return (1, 1)
        raise ValueError(f"Invalid actions: a={a!r}, b={b!r}")

    def nash_equilibria(self) -> set[tuple[str, str]]:
        return {(self.D, self.D)}

    def pareto_optima(self) -> set[tuple[str, str]]:
        return {(self.C, self.C)}
