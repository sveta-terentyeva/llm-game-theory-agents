from __future__ import annotations
from llmgt.games.base import Game


class BattleOfSexes(Game):
    """
    Battle of the Sexes (2x2).

    Actions:
      O = Opera
      F = Football

    Payoffs (A,B):
      (O,O) -> (2,1)
      (O,F) -> (0,0)
      (F,O) -> (0,0)
      (F,F) -> (1,2)

    Theory:
      Nash equilibria: (O,O), (F,F)
      Pareto optima:   (O,O), (F,F)
    """

    name = "battle_of_sexes"

    O = "O"
    F = "F"

    def actions(self) -> tuple[str, str]:
        return (self.O, self.F)

    def payoff(self, a: str, b: str) -> tuple[float, float]:
        if a == self.O and b == self.O:
            return (2.0, 1.0)
        if a == self.O and b == self.F:
            return (0.0, 0.0)
        if a == self.F and b == self.O:
            return (0.0, 0.0)
        if a == self.F and b == self.F:
            return (1.0, 2.0)
        raise ValueError(f"Invalid actions: a={a!r}, b={b!r}")

    def nash_equilibria(self) -> set[tuple[str, str]]:
        return {(self.O, self.O), (self.F, self.F)}

    def pareto_optima(self) -> set[tuple[str, str]]:
        return {(self.O, self.O), (self.F, self.F)}
