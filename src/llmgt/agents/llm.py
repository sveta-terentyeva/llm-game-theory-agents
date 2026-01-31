from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal
import re

from llmgt.games.base import Game
from llmgt.logging.records import ChatMessage
from llmgt.llm.client import LLMClient, LLMMessage


_PAIR_RE = re.compile(r"\(([A-Za-z]+)\s*,\s*([A-Za-z]+)\)")
_ACCEPT_RE = re.compile(r"\bACCEPT\b\s*:\s*" + _PAIR_RE.pattern)


def _format_history(messages: list[ChatMessage], limit: int = 12) -> str:
    tail = messages[-limit:] if len(messages) > limit else messages
    return "\n".join(f"{m.role}: {m.content}" for m in tail)


def _extract_accepted_pair(messages: list[ChatMessage]) -> Optional[tuple[str, str]]:
    for m in messages:
        mm = _ACCEPT_RE.search(m.content)
        if mm:
            return (mm.group(1), mm.group(2))
    return None


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
    role: Literal["agent_a", "agent_b"]
    temperature: float = 0.7

    def send_message(self, game: Game, messages: list[ChatMessage]) -> str:
        allowed = game.actions()
        system = (
            "You are a game-theory agent negotiating with another agent.\n"
            f"Game: {game.name}\n"
            f"Valid actions: {list(allowed)}\n"
            "Goal: propose or accept a plan.\n"
            "When proposing, use: PROPOSE: (X,Y)\n"
            "When accepting, use: ACCEPT: (X,Y)\n"
            "Here X is agent_a's final action and Y is agent_b's final action.\n"
        )
        user = (
            f"Conversation so far:\n{_format_history(messages)}\n\n"
            "Send ONE short message. If you accept a plan, output exactly: ACCEPT: (X,Y). "
            "If you propose, output exactly: PROPOSE: (X,Y)."
        )

        reply = self.client.complete(
            [LLMMessage(role="system", content=system), LLMMessage(role="user", content=user)],
            temperature=self.temperature,
        )
        return reply.strip() or "OK"

    def act(self, game: Game, messages: list[ChatMessage]) -> str:
        allowed = game.actions_for(self.role)

        accepted = _extract_accepted_pair(messages)
        if accepted is not None:
            x, y = accepted
            chosen = x if self.role == "agent_a" else y
            if chosen in allowed:
                return chosen

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
