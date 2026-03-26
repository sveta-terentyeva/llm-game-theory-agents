"""Core simulation runner: episodes, experiments, and summaries."""

from __future__ import annotations

from typing import Protocol, Any, Optional, Iterable

from llmgt.games.base import Game
from llmgt.logging.records import EpisodeRecord, ChatMessage, utc_now_iso
from llmgt.logging.jsonl_logger import JsonlLogger

from llmgt.sim.agreement import agreement_hit
from llmgt.sim.rounds import compute_rounds_to_agreement
from llmgt.sim.workflow import workflow_has_agreement, extract_accepted_pair

from llmgt.sim.theory import compute_theory_hits
from llmgt.sim.rounds import compute_rounds_to_theory_hit



class Agent(Protocol):
    name: str

    def act(self, game: Game, messages: list[ChatMessage]) -> str: ...
    def send_message(self, game: Game, messages: list[ChatMessage]) -> str: ...


def run_episode(
    *,
    episode_id: str,
    game: Game,
    agent_a: Agent,
    agent_b: Agent,
    mode: str = "no_workflow",   # "no_workflow" | "workflow"
    max_comm_rounds: int = 0,
    logger: Optional[JsonlLogger] = None,
    extra: Optional[dict[str, Any]] = None,
) -> EpisodeRecord:

    rec = EpisodeRecord(
        episode_id=episode_id,
        game=game.name,
        mode=mode,
        max_comm_rounds=max_comm_rounds,
        used_comm_rounds=0,
        model_a=agent_a.name,
        model_b=agent_b.name,
        extra=extra or {},
        messages=[
            ChatMessage(role="system", content=f"Episode {episode_id} started for game={game.name}."),
        ],
        started_at_utc=utc_now_iso(),
    )

    used_rounds = 0
    accepted_pair: Optional[tuple[str, str]] = None

    for _ in range(max_comm_rounds):
        if not hasattr(agent_a, "send_message") or not hasattr(agent_b, "send_message"):
            break

        msg_a = agent_a.send_message(game, rec.messages)
        rec.messages.append(ChatMessage(role="agent_a", content=msg_a))

        msg_b = agent_b.send_message(game, rec.messages)
        rec.messages.append(ChatMessage(role="agent_b", content=msg_b))

        used_rounds += 1

        # Early exit: both modes use PROPOSE/COUNTER/ACCEPT protocol,
        # so stop communication as soon as an ACCEPT is detected.
        if workflow_has_agreement(rec.messages):
            accepted_pair = extract_accepted_pair(rec.messages)
            break

    rec.used_comm_rounds = used_rounds

    if accepted_pair is not None:
        rec.extra["accepted_pair"] = list(accepted_pair)

    allowed_a = set(game.actions_for("agent_a"))
    allowed_b = set(game.actions_for("agent_b"))

    a = agent_a.act(game, rec.messages)
    if a not in allowed_a:
        raise ValueError(f"agent_a returned invalid action {a!r}. Allowed: {sorted(allowed_a)}")
    rec.messages.append(ChatMessage(role="agent_a", content=f"ACTION: {a}"))

    b = agent_b.act(game, rec.messages)
    if b not in allowed_b:
        raise ValueError(f"agent_b returned invalid action {b!r}. Allowed: {sorted(allowed_b)}")
    rec.messages.append(ChatMessage(role="agent_b", content=f"ACTION: {b}"))

    rec.action_a = a
    rec.action_b = b

    payoff_a, payoff_b = game.payoff(a, b)
    rec.payoff_a = float(payoff_a)
    rec.payoff_b = float(payoff_b)

    if rec.payoff_a is not None and rec.payoff_b is not None:
        if rec.payoff_a > rec.payoff_b:
            rec.winner = "agent_a"
        elif rec.payoff_b > rec.payoff_a:
            rec.winner = "agent_b"
        else:
            rec.winner = "tie"

    th = compute_theory_hits(game=game, final_action_a=a, final_action_b=b)
    rec.nash_hit = th.nash_hit
    rec.pareto_hit = th.pareto_hit
    rec.pareto_nash_hit = th.pareto_nash_hit
    rec.theory_hit = th.theory_hit


    rec.agreement_hit = agreement_hit(
        game=game,
        mode=mode,
        messages=rec.messages,
        final_action_a=a,
        final_action_b=b,
    )

    rec.rounds_to_agreement = compute_rounds_to_agreement(
        game=game,
        mode=mode,
        messages=rec.messages,
        final_action_a=a,
        final_action_b=b,
        max_comm_rounds=max_comm_rounds,
    )

    rec.rounds_to_theory_hit = compute_rounds_to_theory_hit(
        game=game,
        mode=mode,
        messages=rec.messages,
        final_action_a=a,
        final_action_b=b,
        max_comm_rounds=max_comm_rounds,
    )

    rec.finished_at_utc = utc_now_iso()

    if logger is not None:
        logger.log_episode(rec)

    return rec


def run_experiment(
    *,
    game: Game,
    agent_a: Agent,
    agent_b: Agent,
    n_episodes: int,
    mode: str = "no_workflow",
    max_comm_rounds: int = 0,
    logger: Optional[JsonlLogger] = None,
    episode_id_prefix: str = "ep",
) -> list[EpisodeRecord]:
    out: list[EpisodeRecord] = []
    for i in range(1, n_episodes + 1):
        out.append(
            run_episode(
                episode_id=f"{episode_id_prefix}-{i}",
                game=game,
                agent_a=agent_a,
                agent_b=agent_b,
                mode=mode,
                max_comm_rounds=max_comm_rounds,
                logger=logger,
            )
        )
    return out


def summarize_theory_hits(records: Iterable[EpisodeRecord]) -> dict[str, float]:
    recs = list(records)
    n = len(recs) or 1

    nash_hits = sum(1 for r in recs if r.nash_hit)
    pareto_hits = sum(1 for r in recs if r.pareto_hit)
    agreement_hits = sum(1 for r in recs if r.agreement_hit)
    pareto_nash_hits = sum(1 for r in recs if r.pareto_nash_hit)
    theory_hits = sum(1 for r in recs if r.theory_hit)

    return {
        "n_episodes": float(len(recs)),
        "nash_hits": float(nash_hits),
        "pareto_hits": float(pareto_hits),
        "agreement_hits": float(agreement_hits),
        "nash_rate": nash_hits / n,
        "pareto_rate": pareto_hits / n,
        "agreement_rate": agreement_hits / n,
        "pareto_nash_hits": float(pareto_nash_hits),
        "theory_hits": float(theory_hits),
        "pareto_nash_rate": pareto_nash_hits / n,
        "theory_rate": theory_hits / n,
    }
