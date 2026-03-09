from __future__ import annotations

from llmgt.llm.client import LLMMessage
from llmgt.llm.openrouter_client import OpenRouterClient


def _mk_client(*, model: str, **kwargs):
    # Avoid needing OPENROUTER_API_KEY / openai import: bypass __post_init__.
    c = OpenRouterClient.__new__(OpenRouterClient)
    # Populate dataclass fields used by _serialize_messages
    c.model = model
    c.api_key = None
    c.temperature_default = 0.7
    c.max_tokens = 64
    c.max_retries = 0
    c.retry_backoff_s = 0.0
    c.prompt_caching = kwargs.get("prompt_caching", False)
    c.prompt_cache_ttl = kwargs.get("prompt_cache_ttl", None)
    c.cache_system_message = kwargs.get("cache_system_message", True)
    c.cache_first_user_message = kwargs.get("cache_first_user_message", False)
    c._client = object()
    return c


def test_claude_prompt_caching_adds_cache_control_to_system_message() -> None:
    client = _mk_client(model="anthropic/claude-3.5-haiku", prompt_caching=True, prompt_cache_ttl="1h")
    msgs = [
        LLMMessage(role="system", content="SYSTEM BIG PROMPT"),
        LLMMessage(role="user", content="Hi"),
    ]

    payload = client._serialize_messages(msgs)

    assert payload[0]["role"] == "system"
    assert isinstance(payload[0]["content"], list)
    block = payload[0]["content"][0]
    assert block["type"] == "text"
    assert block["text"] == "SYSTEM BIG PROMPT"
    assert block["cache_control"]["type"] == "ephemeral"
    assert block["cache_control"]["ttl"] == "1h"

    # user message should remain string content by default
    assert payload[1] == {"role": "user", "content": "Hi"}


def test_non_anthropic_models_ignore_prompt_caching_flag() -> None:
    client = _mk_client(model="openai/gpt-4o-mini", prompt_caching=True, prompt_cache_ttl="1h")
    msgs = [LLMMessage(role="system", content="S"), LLMMessage(role="user", content="U")]

    payload = client._serialize_messages(msgs)
    assert payload == [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]


def test_passthrough_content_blocks() -> None:
    client = _mk_client(model="anthropic/claude-3.5-haiku", prompt_caching=True)
    msgs = [
        LLMMessage(
            role="system",
            content_blocks=[
                {"type": "text", "text": "A"},
                {"type": "text", "text": "B", "cache_control": {"type": "ephemeral"}},
            ],
        )
    ]

    payload = client._serialize_messages(msgs)
    assert payload == [{"role": "system", "content": [{"type": "text", "text": "A"}, {"type": "text", "text": "B", "cache_control": {"type": "ephemeral"}}]}]


def test_claude_prompt_caching_can_cache_first_user_message() -> None:
    client = _mk_client(
        model="anthropic/claude-3.5-haiku",
        prompt_caching=True,
        prompt_cache_ttl="5m",
        cache_first_user_message=True,
    )
    msgs = [
        LLMMessage(role="system", content="SYS"),
        LLMMessage(role="user", content="U1"),
        LLMMessage(role="user", content="U2"),
    ]

    payload = client._serialize_messages(msgs)

    # system cached by default
    assert isinstance(payload[0]["content"], list)
    assert payload[0]["content"][0]["cache_control"]["type"] == "ephemeral"

    # first user cached, second is not
    assert isinstance(payload[1]["content"], list)
    assert payload[1]["content"][0]["text"] == "U1"
    assert payload[1]["content"][0]["cache_control"]["type"] == "ephemeral"
    assert payload[2] == {"role": "user", "content": "U2"}
