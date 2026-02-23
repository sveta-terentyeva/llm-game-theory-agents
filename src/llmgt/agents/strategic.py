"""Strategic LLM agent that uses game-theory–aware prompts.

This agent sends structured negotiation messages (PROPOSE / COUNTER / ACCEPT)
and picks its final action using the payoff table context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from llmgt.agents.parsing import (
    extract_accepted_pair,
    extract_last_pair,
    format_history,
    parse_action,
    parse_protocol_reply,
    payoff_table,
)
from llmgt.games.base import Game
from llmgt.logging.records import ChatMessage
from llmgt.llm.client import LLMClient, LLMMessage


@dataclass
class StrategicLLMAgent:
    """LLM agent with game-theory–aware negotiation prompts."""

    name: str
    client: LLMClient
    role: Literal["agent_a", "agent_b"]
    temperature: float = 0.7

    # -- negotiation ---------------------------------------------------------

    def send_message(self, game: Game, messages: list[ChatMessage]) -> str:
        allowed_a = list(game.actions_a())
        allowed_b = list(game.actions_b())
        table = payoff_table(game)

        system = (
            "You are a strategic game-theory agent negotiating with another agent.\n"
            f"Game: {game.name}\n"
            f"Valid actions for agent_a: {allowed_a}\n"
            f"Valid actions for agent_b: {allowed_b}\n\n"
            "Payoff table entries are (payoff_a, payoff_b):\n"
            f"{table}\n\n"
            "You MUST follow the negotiation protocol EXACTLY.\n"
            "- agent_a should output: PROPOSE: (X,Y)\n"
            "- agent_b should output: COUNTER: (X,Y) or ACCEPT: (X,Y)\n"
            "Where X is agent_a's final action and Y is agent_b's final action.\n"
            "Do not add any extra words.\n"
        )

        history = format_history(messages)
        last_pair = extract_last_pair(messages)

        if self.role == "agent_a":
            user = (
                f"Conversation so far:\n{history}\n\n"
                "Output exactly one line: PROPOSE: (X,Y)\n"
            )
            if last_pair is None:
                user += "Pick a reasonable proposal given payoffs.\n"
        else:
            user = (
                f"Conversation so far:\n{history}\n\n"
                "Output exactly one line: COUNTER: (X,Y) or ACCEPT: (X,Y)\n"
                "If the last proposal benefits you enough, ACCEPT it; otherwise COUNTER.\n"
            )

        reply = self.client.complete(
            [LLMMessage(role="system", content=system), LLMMessage(role="user", content=user)],
            temperature=self.temperature,
        )

        parsed = parse_protocol_reply(game, reply)
        if parsed is not None:
            return parsed

        # Fallback: never crash the simulation
        x0 = game.actions_a()[0]
        y0 = game.actions_b()[0]
        if self.role == "agent_a":
            return f"PROPOSE: ({x0},{y0})"
        return f"COUNTER: ({x0},{y0})"

    # -- final action --------------------------------------------------------

    def act(self, game: Game, messages: list[ChatMessage]) -> str:
        allowed = game.actions_for(self.role)

        # Honour accepted pair if present
        accepted = extract_accepted_pair(messages)
        if accepted is not None:
            x, y = accepted
            chosen = x if self.role == "agent_a" else y
            if chosen in allowed:
                return chosen

        system = (
            "You are a strategic game-theory agent. Choose a final action.\n"
            f"Game: {game.name}\n"
            f"Valid actions: {list(allowed)}\n"
            "You MUST output exactly one valid action token, and nothing else."
        )
        user = f"Conversation so far:\n{format_history(messages)}\n\nOutput your final action token."

        reply = self.client.complete(
            [LLMMessage(role="system", content=system), LLMMessage(role="user", content=user)],
            temperature=self.temperature,
        )

        parsed = parse_action(reply, allowed)
        if parsed is not None:
            return parsed

        return allowed[0]
