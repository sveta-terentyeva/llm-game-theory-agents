from __future__ import annotations

from typing import Protocol, Any, Optional, Iterable

from llmgt.games.base import Game
from llmgt.logging.records import EpisodeRecord, ChatMessage, utc_now_iso
from llmgt.logging.jsonl_logger import JsonlLogger

from collections import Counter
from llmgt.logging.records import EpisodeRecord, ExperimentSummary

from llmgt.sim.agreement import agreement_hit
from llmgt.sim.rounds import compute_rounds_to_agreement




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
    for t in range(max_comm_rounds):
        msg_a = agent_a.send_message(game, rec.messages)
        rec.messages.append(
            ChatMessage(role="agent_a", content=msg_a)
        )

        msg_b = agent_b.send_message(game, rec.messages)
        rec.messages.append(
            ChatMessage(role="agent_b", content=msg_b)
        )

        used_rounds += 1

    rec.used_comm_rounds = used_rounds

    allowed = set(game.actions())

    a = agent_a.act(game, rec.messages)
    if a not in allowed:
        raise ValueError(f"agent_a returned invalid action {a!r}. Allowed: {sorted(allowed)}")
    rec.messages.append(ChatMessage(role="agent_a", content=f"ACTION: {a}"))

    b = agent_b.act(game, rec.messages)
    if b not in allowed:
        raise ValueError(f"agent_b returned invalid action {b!r}. Allowed: {sorted(allowed)}")
    rec.messages.append(ChatMessage(role="agent_b", content=f"ACTION: {b}"))

    rec.action_a = a
    rec.action_b = b

    payoff_a, payoff_b = game.payoff(a, b)
    rec.payoff_a = float(payoff_a)
    rec.payoff_b = float(payoff_b)

    rec.nash_hit = (a, b) in game.nash_equilibria()
    rec.pareto_hit = (a, b) in game.pareto_optima()

    rec.agreement_hit = agreement_hit(
        game=game,
        messages=rec.messages,
        final_action_a=a,
        final_action_b=b,
    )
    rec.rounds_to_agreement = compute_rounds_to_agreement(
        game=game,
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

    return {
        "n_episodes": float(len(recs)),
        "nash_hits": float(nash_hits),
        "pareto_hits": float(pareto_hits),
        "agreement_hits": float(agreement_hits),
        "nash_rate": nash_hits / n,
        "pareto_rate": pareto_hits / n,
        "agreement_rate": agreement_hits / n,
    }

def summarize_experiment(game: Game, records: list[EpisodeRecord]) -> ExperimentSummary:
    if not records:
        raise ValueError("No records to summarize")

    n = len(records)

    r0 = records[0]
    model_a = r0.model_a
    model_b = r0.model_b

    pairs = []
    for r in records:
        a = r.action_a or "None"
        b = r.action_b or "None"
        pairs.append(f"({a},{b})")

    counts = Counter(pairs)
    freq = {k: v / n for k, v in counts.items()}

    nash_hits = sum(1 for r in records if r.nash_hit)
    pareto_hits = sum(1 for r in records if r.pareto_hit)
    agreement_hits = sum(1 for r in records if r.agreement_hit)

    pay_a = [r.payoff_a for r in records if r.payoff_a is not None]
    pay_b = [r.payoff_b for r in records if r.payoff_b is not None]
    avg_a = float(sum(pay_a) / len(pay_a)) if pay_a else 0.0
    avg_b = float(sum(pay_b) / len(pay_b)) if pay_b else 0.0

    theory_nash = sorted(list(game.nash_equilibria()))
    theory_pareto = sorted(list(game.pareto_optima()))

    nash_rate = nash_hits / n
    pareto_rate = pareto_hits / n
    agreement_rate = agreement_hits / n

    conclusion = (
        f"Empirical outcomes over n={n} episodes show Nash-hit rate {nash_rate:.2%} "
        f"and Pareto-hit rate {pareto_rate:.2%}. "
        f"The theoretical Nash set is {theory_nash}, and the Pareto set is {theory_pareto}. "
        f"Average payoffs: A={avg_a:.3f}, B={avg_b:.3f}. "
        f"Most frequent action pairs: "
        + ", ".join([f"{k}={freq[k]:.2%}" for k, _ in counts.most_common(3)])
        + "."
    )

    return ExperimentSummary(
        game=r0.game,
        mode=r0.mode,
        n_episodes=n,
        max_comm_rounds=r0.max_comm_rounds,
        model_a=model_a,
        model_b=model_b,
        action_pair_counts=dict(counts),
        action_pair_freq=freq,
        nash_hit_rate=float(nash_rate),
        pareto_hit_rate=float(pareto_rate),
        agreement_hit_rate=float(agreement_rate),
        avg_payoff_a=avg_a,
        avg_payoff_b=avg_b,
        theory_nash=theory_nash,
        theory_pareto=theory_pareto,
        conclusion=conclusion,
    )

