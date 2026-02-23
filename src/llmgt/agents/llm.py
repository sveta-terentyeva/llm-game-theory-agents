"""Basic (non-strategic) LLM agent.

Uses free-form prompts with minimal game-theory guidance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from llmgt.agents.parsing import extract_accepted_pair, format_history, parse_action
from llmgt.games.base import Game
from llmgt.logging.records import ChatMessage
from llmgt.llm.client import LLMClient, LLMMessage


@dataclass
class LLMAgent:
    """Simple LLM agent: free-form messaging + action extraction."""

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
            f"Conversation so far:\n{format_history(messages)}\n\n"
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

        accepted = extract_accepted_pair(messages)
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
            f"Conversation so far:\n{format_history(messages)}\n\n"
            "Output your final action token."
        )

        reply = self.client.complete(
            [LLMMessage(role="system", content=system), LLMMessage(role="user", content=user)],
            temperature=self.temperature,
        )

        parsed = parse_action(reply, allowed)
        if parsed is not None:
            return parsed

        return allowed[0]
