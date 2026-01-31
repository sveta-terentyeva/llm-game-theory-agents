from __future__ import annotations

from typing import Sequence

from llmgt.llm.client import LLMClient, LLMMessage


class HeuristicLLMClient:
    def complete(self, messages: Sequence[LLMMessage], *, temperature: float = 0.7) -> str:
        full = "\n".join(m.content for m in messages)

        is_negotiation = "Send ONE short negotiation message" in full
        is_action = "Output your final action token" in full or "You MUST output exactly one valid action token" in full

        if is_negotiation:
            for line in reversed(full.splitlines()):
                line = line.strip()
                if "PROPOSE:" in line and "(" in line and ")" in line:
                    # Extract the "(X,Y)" substring
                    start = line.find("(")
                    end = line.find(")", start)
                    if start != -1 and end != -1:
                        pair = line[start : end + 1]
                        return f"ACCEPT: {pair}"

            return "PROPOSE: (C,C)"

        if is_action:
            # - If we see "ACCEPT: (C,C)" -> output C
            # - Else if accept exists, default to C; else D.
            if "ACCEPT: (C,C)" in full:
                return "C"
            if "ACCEPT:" in full:
                return "C"
            if "(C,C)" in full:
                return "C"
            return "D"

        return "OK"
