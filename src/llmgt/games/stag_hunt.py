from __future__ import annotations
from llmgt.games.base import Game


class StagHunt(Game):
    """
    Stag Hunt game.

    Actions:
      S = Stag
      H = Hare

    Payoffs (A,B) — aligned with paper (2411.05990):
      (S,S) -> (3,3)
      (S,H) -> (0,1)
      (H,S) -> (1,0)
      (H,H) -> (1,1)

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
            return (3.0, 3.0)
        if a == self.S and b == self.H:
            return (0.0, 1.0)
        if a == self.H and b == self.S:
            return (1.0, 0.0)
        if a == self.H and b == self.H:
            return (1.0, 1.0)
        raise ValueError(f"Invalid actions: a={a!r}, b={b!r}")

    def nash_equilibria(self) -> set[tuple[str, str]]:
        return {(self.S, self.S), (self.H, self.H)}

    def pareto_optima(self) -> set[tuple[str, str]]:
        return {(self.S, self.S)}



