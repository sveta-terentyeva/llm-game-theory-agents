from __future__ import annotations

from typing import Sequence
from llmgt.llm.client import LLMClient, LLMMessage


class HeuristicLLMClient:
    def complete(self, messages: Sequence[LLMMessage], *, temperature: float = 0.7) -> str:
        full = " ".join(m.content for m in messages)

        if "Send ONE short negotiation message" in full:
            return "I suggest we cooperate: (C,C)."

        if "(C,C)" in full or "cooperate" in full.lower():
            return "C"

        return "D"
