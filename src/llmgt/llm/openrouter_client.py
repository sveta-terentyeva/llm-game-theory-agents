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


# OpenRouter provider routing names are lowercase; this is the canonical form used in docs.
_ANTHROPIC_PROVIDER_SLUG = "anthropic"


def _cache_control(*, ttl: str | None) -> dict[str, str]:
    cc: dict[str, str] = {"type": "ephemeral"}
    if ttl:
        cc["ttl"] = ttl
    return cc


def _normalize_ttl(ttl: str | None) -> str | None:
    """Map our configuration values to OpenRouter/Anthropic TTL encoding.

    Returns:
    - None: use default 5-minute TTL (safest for Bedrock/Vertex compatibility)
    - "1h": explicit 1-hour TTL (requires Anthropic-only routing)
    """
    if ttl is None:
        return None
    t = str(ttl).strip().lower()
    if t in {"", "5m", "5min", "5mins", "300s", "default"}:
        return None
    if t == "1h":
        return "1h"
    # Unknown TTL -> pass through; provider will reject if unsupported.
    return t


def _provider_rejects_cache_params(exc: Exception) -> bool:
    """Detect 400s caused by incompatible cache_control fields."""

    status = getattr(exc, "status_code", None)
    if status != 400:
        return False

    raw = ""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        try:
            meta = body.get("metadata") or {}
            raw = str(meta.get("raw") or "")
        except Exception:  # noqa: BLE001
            raw = ""

    msg = (str(exc) + "\n" + raw).lower()
    return "cache_control" in msg and (
        "extra inputs are not permitted" in msg
        or "unknown field" in msg
        or "is not permitted" in msg
        or "invalid" in msg
    )


def _format_openai_like_error(exc: Exception) -> str:
    """Extract as much useful info as possible from openai/OpenRouter errors."""

    parts: list[str] = [f"{type(exc).__name__}: {exc}"]

    # openai>=1.0 exceptions often have status_code and response body
    status = getattr(exc, "status_code", None)
    if status is not None:
        parts.append(f"status_code={status}")

    # Some exceptions expose a .response with a JSON body
    resp = getattr(exc, "response", None)
    if resp is not None:
        try:
            # httpx.Response
            if hasattr(resp, "json"):
                parts.append(f"response_json={resp.json()}")
            elif hasattr(resp, "text"):
                parts.append(f"response_text={resp.text}")
        except Exception:  # noqa: BLE001
            pass

    body = getattr(exc, "body", None)
    if body is not None:
        parts.append(f"body={body}")

    return " | ".join(parts)


def _is_retryable_openrouter_error(exc: Exception) -> bool:
    """Return False for clearly permanent errors (won't be fixed by retry)."""

    status = getattr(exc, "status_code", None)
    # Most 4xx errors are permanent: bad key, no credits, bad model, bad request.
    if isinstance(status, int) and 400 <= status < 500 and status not in {408, 409, 429}:
        return False

    msg = str(exc).lower()
    permanent_markers = [
        "invalid api key",
        "incorrect api key",
        "no api key",
        "unauthorized",
        "forbidden",
        "insufficient credits",
        "payment required",
        "model not found",
        "not found",
        "bad request",
    ]
    if any(m in msg for m in permanent_markers):
        return False

    return True


@dataclass
class OpenRouterClient:
    """LLM client that routes requests through OpenRouter.

    Prompt caching (Anthropic/Claude)
    -------------------------------

    OpenRouter supports Anthropic prompt caching via explicit per-block cache_control.

    Caching modes:
    1. "explicit" (recommended): Per-block cache_control on system/user messages.
       - Default: Uses 5-minute TTL (compatible with Bedrock/Vertex)
       - With 1h TTL: Requires anthropic_only=True to avoid Bedrock rejection
       - Most cost-effective: write @ 1.25x (5m) or 2x (1h), read @ 0.1x

    2. "auto" (legacy): Top-level cache_control.
       - Only works with Anthropic provider (excludes Bedrock/Vertex)
       - Simpler API but less flexible

    Stability notes:
    - Bedrock DOES NOT support "ttl" in per-block cache_control.
      If ttl is attached and Bedrock is used, it returns 400 "Extra inputs not permitted".
    - Solution: Use explicit_cache_include_ttl=False to omit ttl (defaults to 5m).
      Or: Use anthropic_only=True when you need 1h TTL.
    - This client auto-retries without TTL if Bedrock rejects the request.
    """

    model: str = "google/gemini-2.0-flash-001"
    api_key: Optional[str] = None
    temperature_default: float = 0.7
    max_tokens: int = 128
    max_retries: int = 3
    retry_backoff_s: float = 1.0

    # Prompt caching toggles (Claude only)
    prompt_caching: bool = False
    prompt_cache_ttl: str | None = None  # None/"5m"/"1h"
    cache_system_message: bool = True
    cache_first_user_message: bool = False

    # New: caching mode.
    # - "explicit": add cache_control to selected content blocks (works on Bedrock/Vertex).
    # - "auto": set top-level cache_control (forces Anthropic routing; best cache hit rate).
    prompt_caching_mode: str = "explicit"

    # New: whether to attach ttl to explicit per-block caching. Some providers may reject.
    explicit_cache_include_ttl: bool = True

    # New: force routing to Anthropic provider (disables Bedrock/Vertex fallback).
    # This is recommended for stability when using ttl="1h".
    anthropic_only: bool = False

    # If True, allow raising on caching incompatibility. If False, auto-fallback.
    # Default: fallback for robustness.
    fail_on_cache_incompatibility: bool = False

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
        if os.getenv("LLMGT_OPENROUTER_PROMPT_CACHING_MODE"):
            self.prompt_caching_mode = os.getenv("LLMGT_OPENROUTER_PROMPT_CACHING_MODE", "explicit")
        if os.getenv("LLMGT_OPENROUTER_EXPLICIT_CACHE_INCLUDE_TTL") == "0":
            self.explicit_cache_include_ttl = False
        if os.getenv("LLMGT_OPENROUTER_ANTHROPIC_ONLY") == "1":
            self.anthropic_only = True

        self._client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=key,
            default_headers={
                "HTTP-Referer": "https://github.com/llm-game-theory-agents",
                "X-Title": "llm-game-theory-agents",
            },
        )

    def _serialize_messages(
        self,
        messages: Sequence[LLMMessage],
        *,
        disable_prompt_caching: bool = False,
        explicit_include_ttl: bool = True,
    ) -> list[dict[str, Any]]:
        """Convert internal messages to OpenRouter chat.completions payload.

        Important: Bedrock doesn't support ttl in block-level cache_control.
        We only include ttl when explicitly requested AND not ruled out by
        provider compatibility checks.
        """

        use_claude_caching = (
            (not disable_prompt_caching)
            and self.prompt_caching
            and _is_anthropic_model(self.model)
            and self.prompt_caching_mode == "explicit"
        )

        out: list[dict[str, Any]] = []
        user_seen = 0
        # Only include TTL if explicitly requested AND compatible with the configured routing.
        # For Anthropic-only routing with 1h TTL, we include it.
        # Otherwise, omit TTL for Bedrock/Vertex compatibility.
        ttl = _normalize_ttl(self.prompt_cache_ttl) if explicit_include_ttl else None
        for m in messages:
            if m.content_blocks is not None:
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
                block_cc = _cache_control(ttl=ttl if explicit_include_ttl else None)
                out.append(
                    {
                        "role": m.role,
                        "content": [
                            {
                                "type": "text",
                                "text": m.content,
                                "cache_control": block_cc,
                            }
                        ],
                    }
                )
            else:
                out.append({"role": m.role, "content": m.content})

            if m.role == "user":
                user_seen += 1

        return out

    def _serialize_plain_messages(self, messages: Sequence[LLMMessage]) -> list[dict[str, Any]]:
        """Serialize messages without injecting cache_control blocks."""
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.content_blocks is not None:
                out.append({"role": m.role, "content": list(m.content_blocks)})
            else:
                out.append({"role": m.role, "content": m.content or ""})
        return out

    def _top_level_cache_control(self) -> dict[str, str] | None:
        if not (self.prompt_caching and _is_anthropic_model(self.model)):
            return None
        if self.prompt_caching_mode != "auto":
            return None
        return _cache_control(ttl=_normalize_ttl(self.prompt_cache_ttl))

    def _provider_routing(self) -> dict[str, Any] | None:
        """Optional OpenRouter provider routing config.

        We force Anthropic-only routing when explicitly requested, or when using 1h TTL
        (since some alternate Claude providers reject ttl fields).

        This keeps prompt caching enabled and avoids hard failures.
        """

        if not _is_anthropic_model(self.model):
            return None

        ttl = _normalize_ttl(self.prompt_cache_ttl)
        force_anthropic = self.anthropic_only or (ttl == "1h")
        if not force_anthropic:
            return None

        # OpenRouter standard routing hint. Avoid specifying order elsewhere in this repo.
        return {"order": [_ANTHROPIC_PROVIDER_SLUG]}

    # ------------------------------------------------------------------
    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        temperature: float | None = None,
    ) -> str:
        temp = self.temperature_default if temperature is None else temperature

        last_err: Exception | None = None

        # Explicit mode fallback: if provider rejects ttl, retry with caching but without ttl.
        tried_no_ttl = False

        for attempt in range(self.max_retries + 1):
            try:
                cache_top = self._top_level_cache_control()

                if self.prompt_caching_mode == "explicit":
                    input_msgs = self._serialize_messages(
                        messages,
                        disable_prompt_caching=False,
                        explicit_include_ttl=(self.explicit_cache_include_ttl and not tried_no_ttl),
                    )
                else:
                    input_msgs = self._serialize_plain_messages(messages)

                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": input_msgs,
                    "temperature": temp,
                    "max_tokens": self.max_tokens,
                }

                # OpenRouter supports extra request fields like cache_control/provider.
                # The openai>=1.x SDK is strict about accepted kwargs, so we must pass
                # OpenRouter extensions via `extra_body`.
                extra_body: dict[str, Any] = {}

                if cache_top is not None:
                    extra_body["cache_control"] = cache_top

                provider = self._provider_routing()
                if provider is not None:
                    extra_body["provider"] = provider

                if extra_body:
                    kwargs["extra_body"] = extra_body

                resp = self._client.chat.completions.create(  # type: ignore[union-attr]
                    **kwargs
                )
                content = (resp.choices[0].message.content or "").strip()
                return content if content else "OK"

            except Exception as exc:  # noqa: BLE001
                last_err = exc

                if (
                    self.prompt_caching_mode == "explicit"
                    and not tried_no_ttl
                    and self.explicit_cache_include_ttl
                    and _provider_rejects_cache_params(exc)
                ):
                    # Keep prompt caching enabled; just drop ttl and try again.
                    if self.fail_on_cache_incompatibility:
                        break
                    tried_no_ttl = True
                    continue

                if not _is_retryable_openrouter_error(exc):
                    break

                if attempt < self.max_retries:
                    wait = self.retry_backoff_s * (2**attempt)
                    time.sleep(wait)

        detail = _format_openai_like_error(last_err) if last_err else "<unknown>"
        raise RuntimeError(
            f"OpenRouter request failed after {self.max_retries + 1} attempts | model={self.model} | {detail}"
        ) from last_err

