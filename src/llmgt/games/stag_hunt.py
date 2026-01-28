from __future__ import annotations
from llmgt.games.base import Game


class StagHunt(Game):
    """
    Stag Hunt game.

    Actions:
      S = Stag
      H = Hare

    Payoffs (A,B):
      (S,S) -> (4,4)
      (S,H) -> (0,3)
      (H,S) -> (3,0)
      (H,H) -> (3,3)

    Theory:
      Nash equilibria: (S,S), (H,H)
      Pareto optimum:  (S,S)
    """

    name = "stag_hunt"

    S = "S"
    H = "H"

    def actions(self) -> tuple[str, str]:
        return (self.S, self.H)

    def payoff(self, a: str, b: str) -> tuple[float, float]:
        if a == self.S and b == self.S:
            return (4.0, 4.0)
        if a == self.S and b == self.H:
            return (0.0, 3.0)
        if a == self.H and b == self.S:
            return (3.0, 0.0)
        if a == self.H and b == self.H:
            return (3.0, 3.0)
        raise ValueError(f"Invalid actions: a={a!r}, b={b!r}")

    def nash_equilibria(self) -> set[tuple[str, str]]:
        return {(self.S, self.S), (self.H, self.H)}

    def pareto_optima(self) -> set[tuple[str, str]]:
        return {(self.S, self.S)}

