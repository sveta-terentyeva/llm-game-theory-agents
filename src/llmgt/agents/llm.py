from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from llmgt.games.base import Game
from llmgt.logging.records import ChatMessage
from llmgt.llm.client import LLMClient, LLMMessage


def _format_history(messages: list[ChatMessage], limit: int = 12) -> str:
    tail = messages[-limit:] if len(messages) > limit else messages
    lines: list[str] = []
    for m in tail:
        lines.append(f"{m.role}: {m.content}")
    return "\n".join(lines)


def _parse_action(text: str, allowed: tuple[str, ...]) -> Optional[str]:
    t = text.strip()

    candidates = [t]

    if "\n" in t:
        candidates.append(t.splitlines()[0].strip())

    for prefix in ("ACTION:", "Action:", "action:", "Final:", "final:"):
        if t.startswith(prefix):
            candidates.append(t[len(prefix):].strip())

    candidates.append(t.strip("()[]{} ").strip())

    for c in candidates:
        for a in allowed:
            if c == a:
                return a

    for a in allowed:
        if f" {a}" in f" {t} ":
            return a

    return None


@dataclass
class LLMAgent:
    name: str
    client: LLMClient
    temperature: float = 0.7

    role_label: str = "A"

    def send_message(self, game: Game, messages: list[ChatMessage]) -> str:
        allowed = game.actions()
        system = (
            "You are a game-theory agent negotiating with another agent.\n"
            f"Game: {game.name}\n"
            f"Valid actions: {list(allowed)}\n"
            "Goal: propose a mutually beneficial plan.\n"
            "If you propose an agreement, use format: (X,Y) where X is your action and Y is the other agent's action.\n"
        )
        user = (
            f"Conversation so far:\n{_format_history(messages)}\n\n"
            "Send ONE short negotiation message. If proposing a plan, include exactly one pair like (C,C)."
        )

        reply = self.client.complete(
            [LLMMessage(role="system", content=system), LLMMessage(role="user", content=user)],
            temperature=self.temperature,
        )
        return reply.strip() or "OK"

    def act(self, game: Game, messages: list[ChatMessage]) -> str:
        allowed = game.actions()
        system = (
            "You are a game-theory agent. Choose a final action.\n"
            f"Game: {game.name}\n"
            f"Valid actions: {list(allowed)}\n"
            "You MUST output exactly one valid action token, and nothing else."
        )
        user = (
            f"Conversation so far:\n{_format_history(messages)}\n\n"
            "Output your final action token."
        )

        reply = self.client.complete(
            [LLMMessage(role="system", content=system), LLMMessage(role="user", content=user)],
            temperature=self.temperature,
        )

        parsed = _parse_action(reply, allowed)
        if parsed is not None:
            return parsed

        return allowed[0]
