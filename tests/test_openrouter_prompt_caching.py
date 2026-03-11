from __future__ import annotations

from llmgt.llm.client import LLMMessage
from llmgt.llm.openrouter_client import OpenRouterClient


def _mk_client(*, model: str, **kwargs):
    # Avoid needing OPENROUTER_API_KEY / openai import: bypass __post_init__.
    c = OpenRouterClient.__new__(OpenRouterClient)
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

    c.prompt_caching_mode = kwargs.get("prompt_caching_mode", "explicit")
    c.explicit_cache_include_ttl = kwargs.get("explicit_cache_include_ttl", True)
    c.fail_on_cache_incompatibility = kwargs.get("fail_on_cache_incompatibility", False)

    c._client = object()
    return c


def test_claude_prompt_caching_adds_cache_control_to_system_message() -> None:
    client = _mk_client(
        model="anthropic/claude-3.5-haiku",
        prompt_caching=True,
        prompt_cache_ttl=None,
        prompt_caching_mode="explicit",
        explicit_cache_include_ttl=True,
    )
    msgs = [
        LLMMessage(role="system", content="SYSTEM BIG PROMPT"),
        LLMMessage(role="user", content="Hi"),
    ]

    payload = client._serialize_messages(msgs)

    block = payload[0]["content"][0]
    assert block["cache_control"]["type"] == "ephemeral"
    # default TTL (5m) => no ttl field
    assert "ttl" not in block["cache_control"]


def test_claude_prompt_caching_can_include_ttl_when_configured() -> None:
    client = _mk_client(
        model="anthropic/claude-3.5-haiku",
        prompt_caching=True,
        prompt_cache_ttl="1h",
        prompt_caching_mode="explicit",
        explicit_cache_include_ttl=True,
    )
    msgs = [
        LLMMessage(role="system", content="SYSTEM BIG PROMPT"),
        LLMMessage(role="user", content="Hi"),
    ]

    payload = client._serialize_messages(msgs)

    block = payload[0]["content"][0]
    assert block["cache_control"]["ttl"] == "1h"


def test_claude_prompt_caching_can_disable_ttl_for_explicit_blocks() -> None:
    client = _mk_client(
        model="anthropic/claude-3.5-haiku",
        prompt_caching=True,
        prompt_cache_ttl="1h",
        prompt_caching_mode="explicit",
        explicit_cache_include_ttl=False,
    )
    msgs = [
        LLMMessage(role="system", content="SYSTEM BIG PROMPT"),
        LLMMessage(role="user", content="Hi"),
    ]

    payload = client._serialize_messages(msgs, explicit_include_ttl=False)

    block = payload[0]["content"][0]
    assert block["cache_control"]["type"] == "ephemeral"
    assert "ttl" not in block["cache_control"]


def test_non_anthropic_models_ignore_prompt_caching_flag() -> None:
    client = _mk_client(model="openai/gpt-4o-mini", prompt_caching=True, prompt_cache_ttl="1h")
    msgs = [
        LLMMessage(role="system", content="S"),
        LLMMessage(role="user", content="U"),
    ]

    payload = client._serialize_messages(msgs)

    assert payload == [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
    ]


def test_passthrough_content_blocks() -> None:
    client = _mk_client(model="anthropic/claude-3.5-haiku", prompt_caching=True, prompt_cache_ttl="1h")
    msgs = [
        LLMMessage(
            role="system",
            content="",
            content_blocks=[
                {"type": "text", "text": "hello"},
                {"type": "text", "text": "world", "cache_control": {"type": "ephemeral"}},
            ],
        ),
        LLMMessage(role="user", content="Hi"),
    ]

    payload = client._serialize_messages(msgs)

    assert payload[0] == {
        "role": "system",
        "content": [
            {"type": "text", "text": "hello"},
            {"type": "text", "text": "world", "cache_control": {"type": "ephemeral"}},
        ],
    }


def test_claude_prompt_caching_can_cache_first_user_message() -> None:
    client = _mk_client(
        model="anthropic/claude-3.5-haiku",
        prompt_caching=True,
        prompt_cache_ttl=None,
        cache_system_message=False,
        cache_first_user_message=True,
    )
    msgs = [
        LLMMessage(role="system", content="S"),
        LLMMessage(role="user", content="FIRST USER LONG"),
        LLMMessage(role="assistant", content="OK"),
        LLMMessage(role="user", content="second user"),
    ]

    payload = client._serialize_messages(msgs)

    assert payload[0] == {"role": "system", "content": "S"}

    assert payload[1]["role"] == "user"
    assert isinstance(payload[1]["content"], list)
    assert payload[1]["content"][0]["cache_control"]["type"] == "ephemeral"
    # default TTL (5m)
    assert "ttl" not in payload[1]["content"][0]["cache_control"]

    assert payload[3] == {"role": "user", "content": "second user"}


def test_provider_routing_passed_via_extra_body_not_kwarg() -> None:
    class _DummyChatCompletions:
        def __init__(self):
            self.last_kwargs = None

        def create(self, **kwargs):
            self.last_kwargs = kwargs

            class _Msg:
                content = "ok"

            class _Choice:
                message = _Msg()

            class _Resp:
                choices = [_Choice()]

            return _Resp()

    class _DummyClient:
        def __init__(self, chat):
            self.chat = chat

    chat = type("_Chat", (), {})()
    chat.completions = _DummyChatCompletions()

    client = _mk_client(
        model="anthropic/claude-3.5-haiku",
        prompt_caching=True,
        prompt_cache_ttl="1h",
        prompt_caching_mode="explicit",
        explicit_cache_include_ttl=True,
    )
    client.anthropic_only = True
    client._client = _DummyClient(chat)

    out = client.complete([
        LLMMessage(role="system", content="SYSTEM BIG PROMPT"),
        LLMMessage(role="user", content="Hi"),
    ])
    assert out == "ok"

    sent = chat.completions.last_kwargs
    assert sent is not None

    # Critical: openai SDK rejects provider=... at top level.
    assert "provider" not in sent

    extra = sent.get("extra_body")
    assert isinstance(extra, dict)
    assert extra["provider"] == {"order": ["anthropic"]}


def test_cached_system_preamble_env_toggle(monkeypatch) -> None:
    from llmgt.llm.prompt_caching import maybe_prepend_cached_preamble

    def _count_tokens_approx(text: str) -> int:
        try:
            import tiktoken  # type: ignore

            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            return max(1, int(len(text) / 4))

    monkeypatch.delenv("LLMGT_OPENROUTER_CACHE_PREAMBLE", raising=False)
    base = "SYSTEM"
    assert maybe_prepend_cached_preamble(base) == base

    monkeypatch.setenv("LLMGT_OPENROUTER_CACHE_PREAMBLE", "1")
    out = maybe_prepend_cached_preamble(base)
    assert out.endswith(base)

    # Should exceed Claude 3.5 Haiku minimum (~2048 tokens) but not be enormous.
    approx = _count_tokens_approx(out)
    assert approx >= 2048
    assert approx <= 3200
