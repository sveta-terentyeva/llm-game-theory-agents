from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import random

from llmgt.games.base import Game
from llmgt.logging.records import ChatMessage
from llmgt.sim.workflow import extract_last_proposal, extract_last_counter, extract_accepted_pair


@dataclass
class WorkflowProposerAgent:
    name: str
    propose_pair: tuple[str, str]  # (action_a, action_b)

    def send_message(self, game: Game, messages: list[ChatMessage]) -> str:
        accepted = extract_accepted_pair(messages)
        if accepted is not None:
            return "OK"

        counter = extract_last_counter(messages)
        if counter is not None:
            return f"PROPOSE: ({counter[0]},{counter[1]})"

        a, b = self.propose_pair
        return f"PROPOSE: ({a},{b})"

    def act(self, game: Game, messages: list[ChatMessage]) -> str:
        accepted = extract_accepted_pair(messages)
        if accepted is not None:
            return accepted[0]
        return self.propose_pair[0]


@dataclass
class WorkflowResponderAgent:
    name: str
    fallback_action: str
    preferred_pair: Optional[tuple[str, str]] = None
    min_payoff: Optional[float] = None

    def send_message(self, game: Game, messages: list[ChatMessage]) -> str:
        accepted = extract_accepted_pair(messages)
        if accepted is not None:
            return "OK"

        last = extract_last_proposal(messages)
        if last is None:
            return "WAIT"

        a, b = last

        # If min_payoff is set, reject low-payoff proposals by countering
        if self.min_payoff is not None:
            _, payoff_b = game.payoff(a, b)
            if float(payoff_b) < float(self.min_payoff):
                if self.preferred_pair is not None:
                    pa, pb = self.preferred_pair
                    return f"COUNTER: ({pa},{pb})"
                return "REJECT"

        return f"ACCEPT: ({a},{b})"

    def act(self, game: Game, messages: list[ChatMessage]) -> str:
        accepted = extract_accepted_pair(messages)
        if accepted is not None:
            return accepted[1]
        return self.fallback_action


@dataclass
class StochasticWorkflowResponderAgent:
    """
    Acceptance probability increases with the round number:
      p_accept(r) = min(1.0, base_p + (r-1) * step_p)
    """

    name: str
    fallback_action: str
    base_p: float = 0.15
    step_p: float = 0.20
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def send_message(self, game: Game, messages: list[ChatMessage]) -> str:
        accepted = extract_accepted_pair(messages)
        if accepted is not None:
            return "OK"

        last = extract_last_proposal(messages)
        if last is None:
            return "WAIT"

        non_system = [m for m in messages if m.role != "system"]
        r = (len(non_system) // 2) + 1

        p = min(1.0, self.base_p + (r - 1) * self.step_p)
        if self._rng.random() < p:
            a, b = last
            return f"ACCEPT: ({a},{b})"

        return "REJECT"

    def act(self, game: Game, messages: list[ChatMessage]) -> str:
        accepted = extract_accepted_pair(messages)
        if accepted is not None:
            return accepted[1]
        return self.fallback_action
