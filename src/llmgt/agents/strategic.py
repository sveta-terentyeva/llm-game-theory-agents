from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional
import re

from llmgt.games.base import Game
from llmgt.logging.records import ChatMessage
from llmgt.llm.client import LLMClient, LLMMessage


_PAIR_RE = re.compile(r"\(\s*([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_]+)\s*\)")
_PROPOSE_RE = re.compile(r"\bPROPOSE\s*:\s*" + _PAIR_RE.pattern)
_COUNTER_RE = re.compile(r"\bCOUNTER\s*:\s*" + _PAIR_RE.pattern)
_ACCEPT_RE = re.compile(r"\bACCEPT\s*:\s*" + _PAIR_RE.pattern)


def _format_history(messages: list[ChatMessage], limit: int = 12) -> str:
    tail = messages[-limit:] if len(messages) > limit else messages
    return "\n".join(f"{m.role}: {m.content}" for m in tail)


def _payoff_table(game: Game) -> str:
    a_actions = game.actions_a()
    b_actions = game.actions_b()

    #ascii-табличка
    header = "A\\B | " + " | ".join(b_actions)
    sep = "-" * len(header)

    rows = [header, sep]
    for a in a_actions:
        cells = []
        for b in b_actions:
            pa, pb = game.payoff(a, b)
            cells.append(f"{pa:.1f},{pb:.1f}")
        rows.append(f"{a:<3} | " + " | ".join(cells))
    return "\n".join(rows)


def _extract_last_pair(messages: list[ChatMessage]) -> Optional[tuple[str, str]]:
    # остання пара, яка з’являлась у PROPOSE/COUNTER/ACCEPT будь-ким
    last: Optional[tuple[str, str]] = None
    for m in messages:
        for rx in (_PROPOSE_RE, _COUNTER_RE, _ACCEPT_RE):
            mm = rx.search(m.content)
            if mm:
                last = (mm.group(1), mm.group(2))
    return last


def _extract_accepted_pair(messages: list[ChatMessage]) -> Optional[tuple[str, str]]:
    for m in messages:
        mm = _ACCEPT_RE.search(m.content)
        if mm:
            return (mm.group(1), mm.group(2))
    return None


def _sanitize_pair(
    game: Game,
    x: str,
    y: str,
) -> Optional[tuple[str, str]]:
    if x in game.actions_a() and y in game.actions_b():
        return (x, y)
    return None


def _parse_protocol_reply(game: Game, text: str) -> Optional[str]:
    t = text.strip()

    m = _ACCEPT_RE.search(t)
    if m:
        pair = _sanitize_pair(game, m.group(1), m.group(2))
        if pair:
            return f"ACCEPT: ({pair[0]},{pair[1]})"

    m = _COUNTER_RE.search(t)
    if m:
        pair = _sanitize_pair(game, m.group(1), m.group(2))
        if pair:
            return f"COUNTER: ({pair[0]},{pair[1]})"

    m = _PROPOSE_RE.search(t)
    if m:
        pair = _sanitize_pair(game, m.group(1), m.group(2))
        if pair:
            return f"PROPOSE: ({pair[0]},{pair[1]})"

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
class StrategicLLMAgent:
    name: str
    client: LLMClient
    role: Literal["agent_a", "agent_b"]
    temperature: float = 0.7

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
            "You MUST follow the negotiation protocol EXACTLY.\n"
            "- agent_a should output: PROPOSE: (X,Y)\n"
            "- agent_b should output: COUNTER: (X,Y) or ACCEPT: (X,Y)\n"
            "Where X is agent_a's final action and Y is agent_b's final action.\n"
            "Do not add any extra words.\n"
        )

        history = _format_history(messages)
        last_pair = _extract_last_pair(messages)

        if self.role == "agent_a":
            user = (
                f"Conversation so far:\n{history}\n\n"
                "Output exactly one line: PROPOSE: (X,Y)\n"
            )
            # якщо немає контексту — підштовхує до пропозиції Pareto/Nash
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

        parsed = _parse_protocol_reply(game, reply)
        if parsed is not None:
            return parsed

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
            f"Valid actions: {list(allowed)}\n"
            "You MUST output exactly one valid action token, and nothing else."
        )
        user = f"Conversation so far:\n{_format_history(messages)}\n\nOutput your final action token."

        reply = self.client.complete(
            [LLMMessage(role="system", content=system), LLMMessage(role="user", content=user)],
            temperature=self.temperature,
        )

        parsed = _parse_action(reply, allowed)
        if parsed is not None:
            return parsed

        return allowed[0]
