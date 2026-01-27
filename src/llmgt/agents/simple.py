from __future__ import annotations
from dataclasses import dataclass

from llmgt.logging.records import ChatMessage
from llmgt.games.base import Game


@dataclass(frozen=True)
class FixedActionAgent:
    name: str
    action: str

    def act(self, game: Game, messages: list[ChatMessage]) -> str:
        return self.action

@dataclass
class EchoAgent:
    """
    Simple communicative agent:
    - echoes last received message
    - fixed final action
    """
    name: str
    action: str

    def send_message(self, game: Game, messages: list[ChatMessage]) -> str:
        if messages:
            return f"I saw: {messages[-1].content}"
        return "Hello"

    def act(self, game: Game, messages: list[ChatMessage]) -> str:
        return self.action