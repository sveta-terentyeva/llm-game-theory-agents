from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from llmgt.games.base import Game
from llmgt.logging.records import ChatMessage
from llmgt.sim.workflow import extract_last_proposal, extract_accepted_pair


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
