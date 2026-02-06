from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Optional

from llmgt.llm.client import LLMMessage


@dataclass
class OpenAIResponsesClient:
    model: str = "gpt-4o-mini"
    temperature_default: float = 0.7
    max_output_tokens: int = 128
    api_key: Optional[str] = None
    base_url: Optional[str] = None  # for proxies / compatible endpoints

    def __post_init__(self) -> None:
        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError(
                "openai package is not installed. Run: pip install openai"
            ) from e

        kwargs = {}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url

        self._client = OpenAI(**kwargs)

    def complete(self, messages: Sequence[LLMMessage], *, temperature: float | None = None) -> str:
        temp = self.temperature_default if temperature is None else temperature

        input_msgs = [{"role": m.role, "content": m.content} for m in messages]

        resp = self._client.responses.create(
            model=self.model,
            input=input_msgs,
            temperature=temp,
            max_output_tokens=self.max_output_tokens,
        )

        out = (resp.output_text or "").strip()
        return out if out else "OK"

