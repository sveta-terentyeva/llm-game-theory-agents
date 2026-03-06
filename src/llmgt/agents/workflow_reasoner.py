"""Workflow-guided strategic LLM agent.

Follows a paper-style game-theoretic workflow:
dominant strategies → best responses → Nash equilibria → Pareto optima,
then selects a protocol action (PROPOSE / COUNTER / ACCEPT) or final action.
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
class WorkflowStrategicLLMAgent:
    """LLM agent that applies a structured game-theoretic reasoning workflow
    before each negotiation step and final action selection.

    ``workflow_level`` controls depth:
        1 = light (best responses + Nash),
        2 = standard (full pipeline),
        3 = strict (deterministic tie-breaking rules).
    """

    name: str
    client: LLMClient
    role: Literal["agent_a", "agent_b"]
    temperature: float = 0.7
    workflow_level: int = 2

    # -- internal helpers ----------------------------------------------------

    def _workflow_instructions(self) -> str:
        if self.workflow_level == 1:
            return (
                "Use a light game-theoretic workflow: "
                "best responses -> Nash (if any) -> otherwise good Pareto-ish choice.\n"
            )

        base = (
            "Apply a structured game-theoretic workflow BEFORE deciding.\n"
            "Workflow steps:\n"
            "1) (Optional) Check for strictly dominant actions for each player.\n"
            "2) Compute best responses for each possible opponent action.\n"
            "3) Identify Nash equilibria (strategy profiles where both are best responses).\n"
            "4) Identify Pareto-optimal outcomes.\n"
            "5) Decision rule (choose a TARGET outcome (X,Y) to negotiate):\n"
            "   - If there exists a Nash equilibrium that is also Pareto-optimal, prefer it.\n"
            "   - Else prefer a Pareto-optimal outcome.\n"
            "   - Else choose the outcome that maximizes your payoff.\n"
        )

        if self.workflow_level >= 3:
            base += (
                "\nStrict mode tie-breaks (to be deterministic):\n"
                "- If multiple candidate outcomes exist, break ties by: "
                "(a) higher joint payoff (payoff_a+payoff_b), then (b) higher own payoff.\n"
            )
        return base

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
            "IMPORTANT OUTPUT CONSTRAINTS (must follow exactly):\n"
            "- Think through the workflow privately.\n"
            "- Do NOT reveal your reasoning.\n"
            "- Output MUST be exactly ONE line. No extra text, no explanation, no markdown.\n"
            "- Use ONLY these uppercase labels: PROPOSE, COUNTER, ACCEPT.\n"
            "- Use EXACT format with parentheses and comma: LABEL: (X,Y)\n\n"
            + self._workflow_instructions()
            + "\n\n"
            "Negotiation protocol:\n"
            "- agent_a MUST output: PROPOSE: (X,Y)\n"
            "- agent_b MUST output: COUNTER: (X,Y) or ACCEPT: (X,Y)\n"
            "Where X is agent_a's final action and Y is agent_b's final action.\n"
        )

        history = format_history(messages)
        last_pair = extract_last_pair(messages)

        if self.role == "agent_a":
            user = (
                f"Conversation so far:\n{history}\n\n"
                "Output exactly one line: PROPOSE: (X,Y)\n"
                "Choose one concrete outcome pair (X,Y) from the payoff table.\n"
            )
            if last_pair is None:
                user += (
                    "No proposal yet. Start by proposing a TARGET outcome using the workflow decision rule.\n"
                )
        else:
            user = (
                f"Conversation so far:\n{history}\n\n"
                "Output exactly one line: COUNTER: (X,Y) or ACCEPT: (X,Y)\n\n"
                "Decision rule for ACCEPT vs COUNTER (be mechanical; do not explain):\n"
                "- Let the last proposed pair be (X_last,Y_last). If none, treat as missing.\n"
                "- Using the payoff table, compute:\n"
                "    * The set NE = all Nash equilibria (pairs where each action is a best response).\n"
                "    * The set PO = all Pareto-optimal pairs.\n\n"
                "- Define an ACCEPTABLE set (what you are willing to accept immediately):\n"
                "    1) If (NE ∩ PO) is non-empty, ACCEPTABLE = (NE ∩ PO).\n"
                "    2) Else if PO is non-empty, ACCEPTABLE = PO.\n"
                "    3) Else ACCEPTABLE = all pairs that maximize your payoff_b (argmax over payoff_b).\n\n"
                "- Define a TARGET pair to counter with (what you propose if you do not accept):\n"
                "    * Choose a pair from ACCEPTABLE (same set as above).\n"
                "    * Tie-break deterministically among candidates by:\n"
                "        (a) higher joint payoff (payoff_a+payoff_b), then (b) higher your payoff_b.\n\n"
                "- If there IS a last proposal:\n"
                "    * If (X_last,Y_last) is in ACCEPTABLE, output ACCEPT: (X_last,Y_last).\n"
                "    * Otherwise output COUNTER: (X_target,Y_target) with your TARGET pair.\n"
                "- If there is NO last proposal:\n"
                "    * Output COUNTER: (X_target,Y_target) with your TARGET pair.\n"
            )

        reply = self.client.complete(
            [LLMMessage(role="system", content=system), LLMMessage(role="user", content=user)],
            temperature=self.temperature,
        )

        parsed = parse_protocol_reply(game, reply)
        if parsed is not None:
            return parsed

        # Fallback to a valid protocol line (never crash the simulation)
        x0 = game.actions_a()[0]
        y0 = game.actions_b()[0]
        if self.role == "agent_a":
            return f"PROPOSE: ({x0},{y0})"
        return f"COUNTER: ({x0},{y0})"

    # -- final action --------------------------------------------------------

    def act(self, game: Game, messages: list[ChatMessage]) -> str:
        allowed = game.actions_for(self.role)

        accepted = extract_accepted_pair(messages)
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

