"""OpenRouter LLM client — unified access to OpenAI, Claude, Gemini, etc.

Uses the standard OpenAI Chat Completions API via ``base_url`` override,
which is natively supported by the ``openai`` Python package.

Docs: https://openrouter.ai/docs/quickstart
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from llmgt.llm.client import LLMMessage

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _is_anthropic_model(model_id: str) -> bool:
    return model_id.strip().lower().startswith("anthropic/")


def _cache_control(ttl: str | None) -> dict[str, str]:
    cc: dict[str, str] = {"type": "ephemeral"}
    if ttl:
        cc["ttl"] = ttl
    return cc


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

    Prompt caching (Anthropic/Claude)
    -------------------------------
    OpenRouter supports Anthropic prompt caching via per-block ``cache_control``.

    Enable with ``prompt_caching=True`` (or env ``LLMGT_OPENROUTER_PROMPT_CACHING=1``).
    By default we cache the *system* message only, since it tends to be large and stable.

    ``prompt_cache_ttl`` supports OpenRouter/Anthropic TTL values:
      - None or "5m": 5 minutes (default)
      - "1h":         1 hour
    """

    model: str = "google/gemini-2.0-flash-001"
    api_key: Optional[str] = None
    temperature_default: float = 0.7
    max_tokens: int = 128
    max_retries: int = 3
    retry_backoff_s: float = 1.0

    # Prompt caching toggles (Claude only)
    prompt_caching: bool = False
    prompt_cache_ttl: str | None = None  # supported: None/"5m"/"1h"
    cache_system_message: bool = True
    cache_first_user_message: bool = False

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

        # Env opt-in for prompt caching (handy for scripts)
        if os.getenv("LLMGT_OPENROUTER_PROMPT_CACHING") == "1":
            self.prompt_caching = True
        if os.getenv("LLMGT_OPENROUTER_PROMPT_CACHE_TTL"):
            self.prompt_cache_ttl = os.getenv("LLMGT_OPENROUTER_PROMPT_CACHE_TTL")
        if os.getenv("LLMGT_OPENROUTER_CACHE_FIRST_USER") == "1":
            self.cache_first_user_message = True
        if os.getenv("LLMGT_OPENROUTER_CACHE_SYSTEM") == "0":
            self.cache_system_message = False

        self._client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=key,
            default_headers={
                "HTTP-Referer": "https://github.com/llm-game-theory-agents",
                "X-Title": "llm-game-theory-agents",
            },
        )

    def _serialize_messages(self, messages: Sequence[LLMMessage]) -> list[dict[str, Any]]:
        """Convert internal messages to OpenRouter chat.completions payload."""

        use_claude_caching = self.prompt_caching and _is_anthropic_model(self.model)

        out: list[dict[str, Any]] = []
        user_seen = 0
        for m in messages:
            if m.content_blocks is not None:
                # Pass through blocks as-is.
                out.append({"role": m.role, "content": list(m.content_blocks)})
                if m.role == "user":
                    user_seen += 1
                continue

            if not use_claude_caching:
                out.append({"role": m.role, "content": m.content})
                if m.role == "user":
                    user_seen += 1
                continue

            should_cache = False
            if m.role == "system" and self.cache_system_message:
                should_cache = True
            if m.role == "user" and self.cache_first_user_message and user_seen == 0:
                should_cache = True

            if should_cache and m.content:
                ttl = None if (self.prompt_cache_ttl in (None, "", "5m")) else str(self.prompt_cache_ttl)
                out.append(
                    {
                        "role": m.role,
                        "content": [
                            {
                                "type": "text",
                                "text": m.content,
                                "cache_control": _cache_control(ttl),
                            }
                        ],
                    }
                )
            else:
                out.append({"role": m.role, "content": m.content})

            if m.role == "user":
                user_seen += 1

        return out

    # ------------------------------------------------------------------
    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        temperature: float | None = None,
    ) -> str:
        temp = self.temperature_default if temperature is None else temperature
        input_msgs = self._serialize_messages(messages)

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
                    wait = self.retry_backoff_s * (2**attempt)
                    time.sleep(wait)

        raise RuntimeError(f"OpenRouter request failed after {self.max_retries + 1} attempts") from last_err
