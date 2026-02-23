"""Deterministic heuristic LLM client (no external API calls).

Always proposes cooperative / Pareto-optimal outcomes.  Used for fast
smoke-tests and as a baseline that requires no GPU or network.
"""

from __future__ import annotations

from typing import Sequence

from llmgt.llm.client import LLMMessage


def _guess_game(full: str) -> str:
    f = full.lower()
    if "prisoners_dilemma" in f:
        return "pd"
    if "stag_hunt" in f:
        return "sh"
    if "battle_of_sexes" in f:
        return "bos"
    if "ultimatum_game" in f:
        return "ug"
    return "unknown"


class HeuristicLLMClient:
    def complete(self, messages: Sequence[LLMMessage], *, temperature: float = 0.7) -> str:
        full = "\n".join(m.content for m in messages)
        game = _guess_game(full)

        is_negotiation = "Send ONE short message" in full or "negotiating" in full
        is_action = "Output your final action token" in full or "MUST output exactly one valid action token" in full

        if is_negotiation:
            for line in reversed(full.splitlines()):
                line = line.strip()
                if line.startswith("agent_a:") or line.startswith("agent_b:") or line.startswith("assistant:"):
                    # search proposal in any chat line
                    if "PROPOSE:" in line and "(" in line and ")" in line:
                        start = line.find("(")
                        end = line.find(")", start)
                        if start != -1 and end != -1:
                            pair = line[start : end + 1]
                            return f"ACCEPT: {pair}"

            if game == "pd":
                return "PROPOSE: (C,C)"
            if game == "sh":
                return "PROPOSE: (S,S)"
            if game == "bos":
                return "PROPOSE: (O,O)"
            if game == "ug":
                # proposer/responder collapsed into (offer, response) in your model
                return "PROPOSE: (F,A)"

            return "PROPOSE: (C,C)"

        if is_action:
            if game == "pd":
                return "C"
            if game == "sh":
                return "S"
            if game == "bos":
                return "O"
            if game == "ug":
                return "A"
            return "C"

        return "OK"
