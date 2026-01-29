from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from llmgt.games.base import Game
from llmgt.logging.records import ChatMessage
from llmgt.sim.workflow import extract_last_proposal, extract_accepted_pair

import random


@dataclass
class WorkflowProposerAgent:
    name: str
    propose_pair: tuple[str, str]  # (action_a, action_b)

    def send_message(self, game: Game, messages: list[ChatMessage]) -> str:
        # If already accepted, confirm briefly (optional)
        accepted = extract_accepted_pair(messages)
        if accepted is not None:
            return f"CONFIRM: {accepted}"
        a, b = self.propose_pair
        return f"PROPOSE: ({a},{b})"

    def act(self, game: Game, messages: list[ChatMessage]) -> str:
        accepted = extract_accepted_pair(messages)
        if accepted is not None:
            return accepted[0]  # agent_a part
        return self.propose_pair[0]


@dataclass
class WorkflowResponderAgent:
    name: str
    fallback_action: str

    def send_message(self, game: Game, messages: list[ChatMessage]) -> str:
        accepted = extract_accepted_pair(messages)
        if accepted is not None:
            return f"CONFIRM: {accepted}"

        last = extract_last_proposal(messages)
        if last is None:
            return "WAIT"
        a, b = last
        return f"ACCEPT: ({a},{b})"

    def act(self, game: Game, messages: list[ChatMessage]) -> str:
        accepted = extract_accepted_pair(messages)
        if accepted is not None:
            return accepted[1]  # agent_b part
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
            return f"CONFIRM: {accepted}"

        last = extract_last_proposal(messages)
        if last is None:
            return "WAIT"

        non_system = [m for m in messages if m.role != "system"]
        # Each round adds 2 messages (A then B). B's message happens after A's message in the same round.
        # When B is about to speak, round index is (len(non_system)//2)+1
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
