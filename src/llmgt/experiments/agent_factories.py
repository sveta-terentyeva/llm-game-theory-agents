from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from llmgt.agents.llm import LLMAgent
from llmgt.games.base import Game
from llmgt.llm.heuristic import HeuristicLLMClient


Backend = Literal["heuristic", "openai"]


@dataclass(frozen=True)
class LLMBackendConfig:
    backend: Backend = "heuristic"

    # OpenAI options:
    openai_model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_output_tokens: int = 128
    base_url: Optional[str] = None


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

    else:
        raise ValueError(f"Unknown backend: {cfg.backend}")

    agent_a = LLMAgent(name=f"llm_A_{cfg.backend}", client=client_a, role="agent_a", temperature=cfg.temperature)
    agent_b = LLMAgent(name=f"llm_B_{cfg.backend}", client=client_b, role="agent_b", temperature=cfg.temperature)
    return agent_a, agent_b
