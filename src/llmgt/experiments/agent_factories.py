from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from llmgt.agents import LLMAgent
from llmgt.games.base import Game
from llmgt.llm.heuristic import HeuristicLLMClient
from llmgt.agents.strategic import StrategicLLMAgent


Backend = Literal["heuristic", "openai", "ollama", "hf"]


@dataclass(frozen=True)
class LLMBackendConfig:
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

    agent_style: Literal["basic", "strategic"] = "strategic"


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

    elif cfg.backend == "hf":
        from llmgt.llm.hf_client import HuggingFaceChatClient

        client_a = HuggingFaceChatClient(
            model_id=cfg.hf_model,
            max_new_tokens=cfg.max_output_tokens,
            temperature_default=cfg.temperature,
        )
        client_b = HuggingFaceChatClient(
            model_id=cfg.hf_model,
            max_new_tokens=cfg.max_output_tokens,
            temperature_default=cfg.temperature,
        )

    else:
        raise ValueError(f"Unknown backend: {cfg.backend}")

    if cfg.agent_style == "strategic":
        agent_a = StrategicLLMAgent(
            name=f"llm_A_{cfg.backend}",
            client=client_a,
            role="agent_a",
            temperature=cfg.temperature,
        )
        agent_b = StrategicLLMAgent(
            name=f"llm_B_{cfg.backend}",
            client=client_b,
            role="agent_b",
            temperature=cfg.temperature,
        )
    else:
        agent_a = LLMAgent(
            name=f"llm_A_{cfg.backend}",
            client=client_a,
            role="agent_a",
        )
        agent_b = LLMAgent(
            name=f"llm_B_{cfg.backend}",
            client=client_b,
            role="agent_b",
        )

    return agent_a, agent_b

