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

    extra: dict[str, Any] = Field(default_factory=dict)

    started_at_utc: str = Field(default_factory=utc_now_iso)
    finished_at_utc: Optional[str] = None
