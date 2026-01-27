from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ChatMessage(BaseModel):
    role: Literal["system", "agent_a", "agent_b"]
    content: str
    ts_utc: str = Field(default_factory=utc_now_iso)


class EpisodeRecord(BaseModel):
    episode_id: str
    game: str
    mode: Literal["no_workflow", "workflow"]
    max_comm_rounds: int
    used_comm_rounds: int

    model_a: str
    model_b: str

    messages: list[ChatMessage] = Field(default_factory=list)
    action_a: Optional[str] = None
    action_b: Optional[str] = None

    payoff_a: Optional[float] = None
    payoff_b: Optional[float] = None

    nash_hit: Optional[bool] = None
    pareto_hit: Optional[bool] = None
    agreement_hit: Optional[bool] = None
    rounds_to_agreement: Optional[int] = None

    extra: dict[str, Any] = Field(default_factory=dict)

    started_at_utc: str = Field(default_factory=utc_now_iso)
    finished_at_utc: Optional[str] = None

    winner: Optional[Literal["agent_a", "agent_b", "tie"]] = None


class ExperimentSummary(BaseModel):
    game: str
    mode: Literal["no_workflow", "workflow"]
    n_episodes: int
    max_comm_rounds: int

    model_a: str
    model_b: str

    action_pair_counts: dict[str, int] = Field(default_factory=dict)
    action_pair_freq: dict[str, float] = Field(default_factory=dict)

    nash_hit_rate: float
    pareto_hit_rate: float
    agreement_hit_rate: float

    avg_payoff_a: float
    avg_payoff_b: float

    theory_nash: list[tuple[str, str]] = Field(default_factory=list)
    theory_pareto: list[tuple[str, str]] = Field(default_factory=list)

    conclusion: str
    generated_at_utc: str = Field(default_factory=utc_now_iso)


