from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class LLMMessage:
    role: str
    content: str


class LLMClient(Protocol):
    def complete(self, messages: Sequence[LLMMessage], *, temperature: float = 0.7) -> str: ...


class ScriptedLLMClient:
    def __init__(self, outputs: list[str]) -> None:
        if not outputs:
            raise ValueError("ScriptedLLMClient requires at least one output.")
        self._outputs = outputs
        self._i = 0

    def complete(self, messages: Sequence[LLMMessage], *, temperature: float = 0.7) -> str:
        if self._i < len(self._outputs):
            out = self._outputs[self._i]
            self._i += 1
            return out
        return self._outputs[-1]
