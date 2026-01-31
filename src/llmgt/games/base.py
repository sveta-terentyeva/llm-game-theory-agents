from __future__ import annotations

from abc import ABC, abstractmethod


class Game(ABC):
    name: str

    @abstractmethod
    def actions(self) -> tuple[str, ...]:
        raise NotImplementedError

    def actions_a(self) -> tuple[str, ...]:
        return self.actions()

    def actions_b(self) -> tuple[str, ...]:
        return self.actions()

    def actions_for(self, role: str) -> tuple[str, ...]:
        if role == "agent_a":
            return self.actions_a()
        if role == "agent_b":
            return self.actions_b()
        return self.actions()

    @abstractmethod
    def payoff(self, action_a: str, action_b: str) -> tuple[float, float]:
        raise NotImplementedError

    @abstractmethod
    def nash_equilibria(self) -> set[tuple[str, str]]:
        raise NotImplementedError

    @abstractmethod
    def pareto_optima(self) -> set[tuple[str, str]]:
        raise NotImplementedError

