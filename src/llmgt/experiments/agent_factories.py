from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from llmgt.agents.llm import LLMAgent
from llmgt.games.base import Game
from llmgt.llm.heuristic import HeuristicLLMClient


Backend = Literal["heuristic", "openai", "ollama"]


@dataclass(frozen=True)
class LLMBackendConfig:
    backend: Backend = "heuristic"

    # Shared-ish options
    temperature: float = 0.7
    max_output_tokens: int = 128

    # OpenAI options
    openai_model: str = "gpt-4o-mini"
    base_url: Optional[str] = None

    # Ollama options
    ollama_model: str = "llama3.1:8b"
    ollama_host: str = "http://localhost:11434"
    ollama_timeout_s: float = 120.0


def make_llm_agents(game: Game, cfg: LLMBackendConfig) -> tuple[LLMAgent, LLMAgent]:
    if cfg.backend == "heuristic":
        client_a = HeuristicLLMClient()
        client_b = HeuristicLLMClient()

    elif cfg.backend == "openai":
        from llmgt.llm.openai_client import OpenAIResponsesClient

        client_a = OpenAIResponsesClient(
            model=cfg.openai_model,
            temperature_default=cfg.temperature,
            max_output_tokens=cfg.max_output_tokens,
            base_url=cfg.base_url,
        )
        client_b = OpenAIResponsesClient(
            model=cfg.openai_model,
            temperature_default=cfg.temperature,
            max_output_tokens=cfg.max_output_tokens,
            base_url=cfg.base_url,
        )

    elif cfg.backend == "ollama":
        from llmgt.llm.ollama_client import OllamaChatClient

        client_a = OllamaChatClient(
            model=cfg.ollama_model,
            host=cfg.ollama_host,
            temperature_default=cfg.temperature,
            num_predict=cfg.max_output_tokens,
            timeout_s=cfg.ollama_timeout_s,
        )
        client_b = OllamaChatClient(
            model=cfg.ollama_model,
            host=cfg.ollama_host,
            temperature_default=cfg.temperature,
            num_predict=cfg.max_output_tokens,
            timeout_s=cfg.ollama_timeout_s,
        )

    else:
        raise ValueError(f"Unknown backend: {cfg.backend}")

    agent_a = LLMAgent(
        name=f"llm_A_{cfg.backend}",
        client=client_a,
        role="agent_a",
        temperature=cfg.temperature,
    )
    agent_b = LLMAgent(
        name=f"llm_B_{cfg.backend}",
        client=client_b,
        role="agent_b",
        temperature=cfg.temperature,
    )
    return agent_a, agent_b
