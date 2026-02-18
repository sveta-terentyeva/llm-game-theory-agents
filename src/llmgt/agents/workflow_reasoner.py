from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from llmgt.agents.strategic import (
    _extract_accepted_pair,
    _extract_last_pair,
    _format_history,
    _parse_action,
    _parse_protocol_reply,
    _payoff_table,
)
from llmgt.games.base import Game
from llmgt.logging.records import ChatMessage
from llmgt.llm.client import LLMClient, LLMMessage


@dataclass
class WorkflowStrategicLLMAgent:
    """
    LLM agent that follows a *paper-style* game-theoretic workflow:
    dominant strategies -> best responses -> Nash equilibria -> Pareto optima,
    then selects a protocol action (PROPOSE/COUNTER/ACCEPT) or final action.

    Key constraint: the agent MUST output exactly one protocol line in negotiation,
    and exactly one action token in act().
    """

    name: str
    client: LLMClient
    role: Literal["agent_a", "agent_b"]
    temperature: float = 0.7
    workflow_level: int = 2  # 1 = light, 2 = standard, 3 = strict

    def _workflow_instructions(self) -> str:
        base = (
            "Apply a structured game-theoretic workflow BEFORE deciding.\n"
            "Workflow steps:\n"
            "1) (Optional) Check for strictly dominant actions for each player.\n"
            "2) Compute best responses for each possible opponent action.\n"
            "3) Identify Nash equilibria (strategy profiles where both are best responses).\n"
            "4) Identify Pareto-optimal outcomes.\n"
            "5) Decision rule:\n"
            "   - If there exists a Nash equilibrium that is also Pareto-optimal, prefer it.\n"
            "   - Else prefer a Pareto-optimal outcome that improves your payoff.\n"
            "   - Else choose the outcome that maximizes your payoff while remaining plausible.\n"
        )

        if self.workflow_level >= 3:
            base += (
                "\nStrict mode:\n"
                "- Do not use vague language.\n"
                "- If multiple equilibria exist, break ties by (a) higher own payoff, then (b) higher joint payoff.\n"
            )
        elif self.workflow_level == 1:
            base = (
                "Use a light game-theoretic workflow: best responses -> Nash (if any) -> otherwise good Pareto-ish choice.\n"
            )
        return base

    def send_message(self, game: Game, messages: list[ChatMessage]) -> str:
        allowed_a = list(game.actions_a())
        allowed_b = list(game.actions_b())
        table = _payoff_table(game)

        system = (
            "You are a strategic game-theory agent negotiating with another agent.\n"
            f"Game: {game.name}\n"
            f"Valid actions for agent_a: {allowed_a}\n"
            f"Valid actions for agent_b: {allowed_b}\n\n"
            "Payoff table entries are (payoff_a, payoff_b):\n"
            f"{table}\n\n"
            "IMPORTANT:\n"
            "- Think through the workflow privately.\n"
            "- Do NOT reveal your reasoning.\n"
            "- Output MUST be exactly ONE line in the protocol format. No extra words.\n\n"
            + self._workflow_instructions()
            + "\n\n"
            "Negotiation protocol:\n"
            "- agent_a MUST output: PROPOSE: (X,Y)\n"
            "- agent_b MUST output: COUNTER: (X,Y) or ACCEPT: (X,Y)\n"
            "Where X is agent_a's final action and Y is agent_b's final action.\n"
        )

        history = _format_history(messages)
        last_pair = _extract_last_pair(messages)

        if self.role == "agent_a":
            user = (
                f"Conversation so far:\n{history}\n\n"
                "Output exactly one line: PROPOSE: (X,Y)\n"
            )
            if last_pair is None:
                user += "No proposal yet. Propose a good outcome given the payoff table.\n"
        else:
            user = (
                f"Conversation so far:\n{history}\n\n"
                "Output exactly one line: COUNTER: (X,Y) or ACCEPT: (X,Y)\n"
                "If the last proposal is good for you according to the payoff table and your workflow, ACCEPT it.\n"
                "Otherwise COUNTER with a better outcome for you.\n"
            )

        reply = self.client.complete(
            [LLMMessage(role="system", content=system), LLMMessage(role="user", content=user)],
            temperature=self.temperature,
        )

        parsed = _parse_protocol_reply(game, reply)
        if parsed is not None:
            return parsed

        # Fallback to a valid protocol line (never crash the simulation)
        x0 = game.actions_a()[0]
        y0 = game.actions_b()[0]
        if self.role == "agent_a":
            return f"PROPOSE: ({x0},{y0})"
        return f"COUNTER: ({x0},{y0})"

    def act(self, game: Game, messages: list[ChatMessage]) -> str:
        allowed = game.actions_for(self.role)

        accepted = _extract_accepted_pair(messages)
        if accepted is not None:
            x, y = accepted
            chosen = x if self.role == "agent_a" else y
            if chosen in allowed:
                return chosen

        system = (
            "You are a strategic game-theory agent. Choose a final action.\n"
            f"Game: {game.name}\n"
            f"Valid actions: {list(allowed)}\n\n"
            "IMPORTANT:\n"
            "- Think through the workflow privately.\n"
            "- Do NOT reveal your reasoning.\n"
            "- Output MUST be exactly one valid action token, and nothing else.\n\n"
            + self._workflow_instructions()
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

