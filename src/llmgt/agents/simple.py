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
