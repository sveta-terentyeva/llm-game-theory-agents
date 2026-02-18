from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from llmgt.agents import LLMAgent
from llmgt.agents.strategic import StrategicLLMAgent
from llmgt.agents.workflow_reasoner import WorkflowStrategicLLMAgent
from llmgt.games.base import Game
from llmgt.llm.heuristic import HeuristicLLMClient


Backend = Literal["heuristic", "openai", "ollama", "hf"]
Mode = Literal["no_workflow", "workflow"]


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
    hf_max_new_tokens: int = 128

    # Agent behavior
    agent_style: Literal["basic", "strategic"] = "strategic"
    workflow_level: int = 2  # only used in workflow mode


def _make_clients(cfg: LLMBackendConfig):
    """Create two independent clients (A and B) with the same config."""
    if cfg.backend == "heuristic":
        return HeuristicLLMClient(), HeuristicLLMClient()

    if cfg.backend == "openai":
        from llmgt.llm.openai_client import OpenAIResponsesClient

        a = OpenAIResponsesClient(
            model=cfg.openai_model,
            temperature_default=cfg.temperature,
            max_output_tokens=cfg.max_output_tokens,
            base_url=cfg.base_url,
        )
        b = OpenAIResponsesClient(
            model=cfg.openai_model,
            temperature_default=cfg.temperature,
            max_output_tokens=cfg.max_output_tokens,
            base_url=cfg.base_url,
        )
        return a, b

    if cfg.backend == "ollama":
        from llmgt.llm.ollama_client import OllamaChatClient

        a = OllamaChatClient(
            model=cfg.ollama_model,
            host=cfg.ollama_host,
            temperature_default=cfg.temperature,
            num_predict=cfg.max_output_tokens,
            timeout_s=cfg.ollama_timeout_s,
        )
        b = OllamaChatClient(
            model=cfg.ollama_model,
            host=cfg.ollama_host,
            temperature_default=cfg.temperature,
            num_predict=cfg.max_output_tokens,
            timeout_s=cfg.ollama_timeout_s,
        )
        return a, b

    if cfg.backend == "hf":
        from llmgt.llm.hf_client import HuggingFaceChatClient

        a = HuggingFaceChatClient(
            model_id=cfg.hf_model,
            max_new_tokens=cfg.hf_max_new_tokens,
            temperature_default=cfg.temperature,
        )
        b = HuggingFaceChatClient(
            model_id=cfg.hf_model,
            max_new_tokens=cfg.hf_max_new_tokens,
            temperature_default=cfg.temperature,
        )
        return a, b

    raise ValueError(f"Unknown backend: {cfg.backend}")


def make_llm_agents(game: Game, cfg: LLMBackendConfig) -> tuple[LLMAgent, LLMAgent]:
    client_a, client_b = _make_clients(cfg)

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
        agent_a = LLMAgent(name=f"llm_A_{cfg.backend}", client=client_a, role="agent_a")
        agent_b = LLMAgent(name=f"llm_B_{cfg.backend}", client=client_b, role="agent_b")

    return agent_a, agent_b


def make_agents_for_mode(game: Game, cfg: LLMBackendConfig, mode: Mode) -> tuple[LLMAgent, LLMAgent]:
    """
    - no_workflow: LLM agents WITHOUT workflow prompts (baseline)
    - workflow:    LLM agents WITH paper-style workflow prompts
    """
    client_a, client_b = _make_clients(cfg)

    if mode == "workflow":
        agent_a = WorkflowStrategicLLMAgent(
            name=f"wf_llm_A_{cfg.backend}",
            client=client_a,
            role="agent_a",
            temperature=cfg.temperature,
            workflow_level=cfg.workflow_level,
        )
        agent_b = WorkflowStrategicLLMAgent(
            name=f"wf_llm_B_{cfg.backend}",
            client=client_b,
            role="agent_b",
            temperature=cfg.temperature,
            workflow_level=cfg.workflow_level,
        )
        return agent_a, agent_b

    # no_workflow
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
        agent_a = LLMAgent(name=f"llm_A_{cfg.backend}", client=client_a, role="agent_a")
        agent_b = LLMAgent(name=f"llm_B_{cfg.backend}", client=client_b, role="agent_b")

    return agent_a, agent_b

