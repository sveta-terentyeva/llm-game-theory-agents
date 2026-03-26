"""Factory functions for creating LLM-backed game-theory agents.

Centralises LLM client construction and agent wiring for both
``no_workflow`` and ``workflow`` modes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal, Optional

from llmgt.agents.llm import LLMAgent
from llmgt.agents.strategic import StrategicLLMAgent
from llmgt.agents.workflow_reasoner import WorkflowStrategicLLMAgent
from llmgt.games.base import Game
from llmgt.llm.heuristic import HeuristicLLMClient


Backend = Literal["heuristic", "openai", "ollama", "hf", "openrouter"]
Mode = Literal["no_workflow", "workflow"]

# Type alias — any agent that exposes ``send_message`` + ``act``
AgentPair = tuple[Any, Any]


def _load_openrouter_config_from_env() -> dict[str, Any]:
    """Load OpenRouter prompt caching config from environment variables.

    This centralizes all the LLMGT_OPENROUTER_* variables into a dict
    that can be passed to LLMBackendConfig.

    For backward compatibility, also checks LLMGT_CLAUDE_* aliases.
    """
    config = {}

    # Prompt caching enabled?
    cache_enabled = os.getenv("LLMGT_OPENROUTER_PROMPT_CACHING")
    if cache_enabled is None:
        cache_enabled = os.getenv("LLMGT_CLAUDE_PROMPT_CACHING")
    config["openrouter_prompt_caching"] = cache_enabled == "1"

    # Cache TTL (5m default, 1h optional)
    cache_ttl = os.getenv("LLMGT_OPENROUTER_PROMPT_CACHE_TTL")
    if cache_ttl is None:
        cache_ttl = os.getenv("LLMGT_CLAUDE_PROMPT_CACHE_TTL")
    if cache_ttl:
        config["openrouter_prompt_cache_ttl"] = cache_ttl

    # Caching mode (explicit or auto)
    caching_mode = os.getenv("LLMGT_OPENROUTER_PROMPT_CACHING_MODE")
    if caching_mode is None:
        caching_mode = os.getenv("LLMGT_CLAUDE_PROMPT_CACHING_MODE")
    if caching_mode:
        config["openrouter_prompt_caching_mode"] = caching_mode  # type: ignore

    # Include TTL in explicit blocks?
    explicit_ttl = os.getenv("LLMGT_OPENROUTER_EXPLICIT_CACHE_INCLUDE_TTL")
    if explicit_ttl is None:
        explicit_ttl = os.getenv("LLMGT_CLAUDE_EXPLICIT_INCLUDE_TTL")
    if explicit_ttl is not None:
        config["openrouter_explicit_cache_include_ttl"] = explicit_ttl == "1"

    # Force Anthropic-only routing?
    anthropic_only = os.getenv("LLMGT_OPENROUTER_ANTHROPIC_ONLY")
    if anthropic_only is None:
        anthropic_only = os.getenv("LLMGT_CLAUDE_ANTHROPIC_ONLY")
    if anthropic_only is not None:
        config["openrouter_anthropic_only"] = anthropic_only == "1"

    # Cache system message?
    cache_system = os.getenv("LLMGT_OPENROUTER_CACHE_SYSTEM")
    if cache_system:
        config["openrouter_cache_system_message"] = cache_system != "0"

    # Cache first user message?
    cache_first_user = os.getenv("LLMGT_OPENROUTER_CACHE_FIRST_USER")
    if cache_first_user:
        config["openrouter_cache_first_user_message"] = cache_first_user == "1"

    return config


@dataclass(frozen=True)
class LLMBackendConfig:
    """Immutable configuration for LLM backend and agent behaviour."""

    backend: Backend = "heuristic"

    # Shared options
    temperature: float = 0.7
    max_output_tokens: int = 64

    # OpenAI
    openai_model: str = "gpt-4o-mini"
    base_url: Optional[str] = None

    # Ollama
    ollama_model: str = "llama3.1:8b"
    ollama_host: str = "http://localhost:11434"
    ollama_timeout_s: float = 120.0

    # HuggingFace (Transformers)
    hf_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    hf_max_new_tokens: int = 128

    # OpenRouter (access to OpenAI, Claude, Gemini, etc.)
    # Default to a widely-available free model so OpenRouter works out-of-the-box.
    openrouter_model: str = "meta-llama/llama-3.3-70b-instruct"
    openrouter_api_key: Optional[str] = None

    # OpenRouter prompt caching (Anthropic/Claude via cache_control blocks)
    openrouter_prompt_caching: bool = False
    openrouter_prompt_cache_ttl: Optional[str] = None  # None/"5m"/"1h"
    openrouter_cache_system_message: bool = True
    openrouter_cache_first_user_message: bool = False

    # New: OpenRouter prompt caching mode + routing controls
    openrouter_prompt_caching_mode: Literal["explicit", "auto"] = "explicit"
    openrouter_explicit_cache_include_ttl: bool = True
    openrouter_anthropic_only: bool = False

    # Agent behaviour
    agent_style: Literal["basic", "strategic"] = "strategic"
    workflow_level: int = 2  # only used in workflow mode


# ---------------------------------------------------------------------------
# Client construction
# ---------------------------------------------------------------------------


def _make_client(cfg: LLMBackendConfig) -> Any:
    """Create a *single* LLM client from *cfg*."""
    if cfg.backend == "heuristic":
        return HeuristicLLMClient()

    if cfg.backend == "openai":
        from llmgt.llm.openai_client import OpenAIResponsesClient

        return OpenAIResponsesClient(
            model=cfg.openai_model,
            temperature_default=cfg.temperature,
            max_output_tokens=cfg.max_output_tokens,
            base_url=cfg.base_url,
        )

    if cfg.backend == "ollama":
        from llmgt.llm.ollama_client import OllamaChatClient

        return OllamaChatClient(
            model=cfg.ollama_model,
            host=cfg.ollama_host,
            temperature_default=cfg.temperature,
            num_predict=cfg.max_output_tokens,
            timeout_s=cfg.ollama_timeout_s,
        )

    if cfg.backend == "hf":
        from llmgt.llm.hf_client import HuggingFaceChatClient

        return HuggingFaceChatClient(
            model_id=cfg.hf_model,
            max_new_tokens=cfg.hf_max_new_tokens,
            temperature_default=cfg.temperature,
        )

    if cfg.backend == "openrouter":
        from llmgt.llm.openrouter_client import OpenRouterClient

        client = OpenRouterClient(
            model=cfg.openrouter_model,
            api_key=cfg.openrouter_api_key,
            temperature_default=cfg.temperature,
            max_tokens=cfg.max_output_tokens,
            prompt_caching=cfg.openrouter_prompt_caching,
            prompt_cache_ttl=cfg.openrouter_prompt_cache_ttl,
            cache_system_message=cfg.openrouter_cache_system_message,
            cache_first_user_message=cfg.openrouter_cache_first_user_message,
            prompt_caching_mode=cfg.openrouter_prompt_caching_mode,
            explicit_cache_include_ttl=cfg.openrouter_explicit_cache_include_ttl,
            anthropic_only=cfg.openrouter_anthropic_only,
        )
        return client

    raise ValueError(f"Unknown backend: {cfg.backend}")


def _make_client_pair(cfg: LLMBackendConfig) -> tuple[Any, Any]:
    """Create two independent clients (for agent A and agent B)."""
    return _make_client(cfg), _make_client(cfg)


# ---------------------------------------------------------------------------
# Agent construction
# ---------------------------------------------------------------------------


def make_llm_agents(game: Game, cfg: LLMBackendConfig) -> AgentPair:
    """Create a pair of LLM agents (``no_workflow`` mode)."""
    client_a, client_b = _make_client_pair(cfg)

    if cfg.agent_style == "strategic":
        return (
            StrategicLLMAgent(
                name=f"llm_A_{cfg.backend}",
                client=client_a,
                role="agent_a",
                temperature=cfg.temperature,
            ),
            StrategicLLMAgent(
                name=f"llm_B_{cfg.backend}",
                client=client_b,
                role="agent_b",
                temperature=cfg.temperature,
            ),
        )

    return (
        LLMAgent(name=f"llm_A_{cfg.backend}", client=client_a, role="agent_a", temperature=cfg.temperature),
        LLMAgent(name=f"llm_B_{cfg.backend}", client=client_b, role="agent_b", temperature=cfg.temperature),
    )


def make_agents_for_mode(game: Game, cfg: LLMBackendConfig, mode: Mode) -> AgentPair:
    """Create agents appropriate for *mode*.

    - ``no_workflow``: LLM agents WITHOUT workflow prompts (baseline).
    - ``workflow``:    LLM agents WITH paper-style workflow prompts.
    """
    if mode == "workflow":
        client_a, client_b = _make_client_pair(cfg)
        return (
            WorkflowStrategicLLMAgent(
                name=f"wf_llm_A_{cfg.backend}",
                client=client_a,
                role="agent_a",
                temperature=cfg.temperature,
                workflow_level=cfg.workflow_level,
            ),
            WorkflowStrategicLLMAgent(
                name=f"wf_llm_B_{cfg.backend}",
                client=client_b,
                role="agent_b",
                temperature=cfg.temperature,
                workflow_level=cfg.workflow_level,
            ),
        )


    return make_llm_agents(game, cfg)
