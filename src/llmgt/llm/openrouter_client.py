"""OpenRouter LLM client — unified access to OpenAI, Claude, Gemini, etc.

Uses the standard OpenAI Chat Completions API via ``base_url`` override,
which is natively supported by the ``openai`` Python package.

Docs: https://openrouter.ai/docs/quickstart
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

from llmgt.llm.client import LLMMessage

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass
class OpenRouterClient:
    """LLM client that routes requests through OpenRouter.

    Parameters
    ----------
    model : str
        OpenRouter model identifier, e.g. ``"google/gemini-2.0-flash-001"``,
        ``"openai/gpt-4o-mini"``, ``"anthropic/claude-3.5-haiku"``.
    api_key : str | None
        OpenRouter API key. Falls back to the ``OPENROUTER_API_KEY`` env var.
    temperature_default : float
        Default sampling temperature.
    max_tokens : int
        Maximum number of tokens in the completion.
    max_retries : int
        How many times to retry on transient errors.
    retry_backoff_s : float
        Base back-off between retries (doubles each attempt).
    """

    model: str = "google/gemini-2.0-flash-001"
    api_key: Optional[str] = None
    temperature_default: float = 0.7
    max_tokens: int = 128
    max_retries: int = 3
    retry_backoff_s: float = 1.0

    # Private — initialised in __post_init__
    _client: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai package is required for the OpenRouter backend. "
                "Run:  pip install 'openai>=1.0.0'"
            ) from exc

        key = self.api_key or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError(
                "OpenRouter API key not provided. "
                "Pass api_key= or set the OPENROUTER_API_KEY env variable."
            )

        self._client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=key,
            default_headers={
                "HTTP-Referer": "https://github.com/llm-game-theory-agents",
                "X-Title": "llm-game-theory-agents",
            },
        )

    # ------------------------------------------------------------------
    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        temperature: float | None = None,
    ) -> str:
        temp = self.temperature_default if temperature is None else temperature
        input_msgs = [{"role": m.role, "content": m.content} for m in messages]

        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(  # type: ignore[union-attr]
                    model=self.model,
                    messages=input_msgs,
                    temperature=temp,
                    max_tokens=self.max_tokens,
                )
                content = (resp.choices[0].message.content or "").strip()
                return content if content else "OK"

            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt < self.max_retries:
                    wait = self.retry_backoff_s * (2 ** attempt)
                    time.sleep(wait)

        raise RuntimeError(
            f"OpenRouter request failed after {self.max_retries + 1} attempts"
        ) from last_err

