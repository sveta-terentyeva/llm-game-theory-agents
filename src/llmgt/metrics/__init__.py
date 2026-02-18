from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from llmgt.games.base import Game
from llmgt.logging.records import EpisodeRecord
from llmgt.sim.workflow import extract_last_proposal, extract_last_counter, extract_accepted_pair


def _words(text: str) -> int:
    return len(text.strip().split()) if text else 0


def _is_action_line(text: str) -> bool:
    return bool(text) and text.strip().upper().startswith("ACTION:")


@dataclass(frozen=True)
class EpisodeCommStats:
    n_messages_total: int
    n_messages_system: int
    n_messages_agent_a: int
    n_messages_agent_b: int
    n_words_total: int
    n_words_agent_a: int
    n_words_agent_b: int

    has_propose: bool
    has_counter: bool
    has_accept: bool
    proposed_pair: Optional[tuple[str, str]]
    counter_pair: Optional[tuple[str, str]]
    accepted_pair: Optional[tuple[str, str]]

    actions_follow_accept: Optional[bool]


def compute_episode_comm_stats(rec: EpisodeRecord) -> EpisodeCommStats:
    msgs = rec.messages or []
    n_total = len(msgs)
    n_sys = sum(1 for m in msgs if m.role == "system")

    comm_msgs = [
        m for m in msgs
        if m.role != "system" and not _is_action_line(m.content)
    ]

    n_a = sum(1 for m in comm_msgs if m.role == "agent_a")
    n_b = sum(1 for m in comm_msgs if m.role == "agent_b")

    words_total = sum(_words(m.content) for m in comm_msgs)
    words_a = sum(_words(m.content) for m in comm_msgs if m.role == "agent_a")
    words_b = sum(_words(m.content) for m in comm_msgs if m.role == "agent_b")

    proposed = extract_last_proposal(comm_msgs)
    counter = extract_last_counter(comm_msgs)
    accepted = extract_accepted_pair(comm_msgs)

    follow: Optional[bool] = None
    if accepted is not None and rec.action_a is not None and rec.action_b is not None:
        follow = (rec.action_a, rec.action_b) == accepted

    return EpisodeCommStats(
        n_messages_total=n_total,
        n_messages_system=n_sys,
        n_messages_agent_a=n_a,
        n_messages_agent_b=n_b,
        n_words_total=words_total,
        n_words_agent_a=words_a,
        n_words_agent_b=words_b,
        has_propose=(proposed is not None),
        has_counter=(counter is not None),
        has_accept=(accepted is not None),
        proposed_pair=proposed,
        counter_pair=counter,
        accepted_pair=accepted,
        actions_follow_accept=follow,
    )


def regret_a(game: Game, action_a: str, action_b: str) -> float:
    actual, _ = game.payoff(action_a, action_b)
    best = max(game.payoff(a_alt, action_b)[0] for a_alt in game.actions_a())
    return float(best - actual)


def regret_b(game: Game, action_a: str, action_b: str) -> float:
    _, actual = game.payoff(action_a, action_b)
    best = max(game.payoff(action_a, b_alt)[1] for b_alt in game.actions_b())
    return float(best - actual)


def welfare_gap(game: Game, action_a: str, action_b: str) -> float:
    achieved = sum(game.payoff(action_a, action_b))
    best = max(sum(game.payoff(a, b)) for a in game.actions_a() for b in game.actions_b())
    return float(best - achieved)

