from __future__ import annotations
from abc import ABC, abstractmethod


class Game(ABC):
    name: str

    @abstractmethod
    def actions(self) -> tuple[str, ...]:
        raise NotImplementedError

    @abstractmethod
    def payoff(self, action_a: str, action_b: str) -> tuple[float, float]:
        raise NotImplementedError

    @abstractmethod
    def nash_equilibria(self) -> set[tuple[str, str]]:
        raise NotImplementedError

    @abstractmethod
    def pareto_optima(self) -> set[tuple[str, str]]:
        raise NotImplementedError
