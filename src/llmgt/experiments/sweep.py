from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

from llmgt.sim.runner import run_episode
from llmgt.logging.records import EpisodeRecord
from llmgt.logging.jsonl_logger import JsonlLogger
from llmgt.metrics import (
    compute_episode_comm_stats,
    regret_a,
    regret_b,
    welfare_gap,
)


def run_comm_sweep(
    *,
    game,
    agent_a,
    agent_b,
    k_values: Iterable[int],
    n_runs: int,
    mode: str = "no_workflow",
    logger: Optional[JsonlLogger] = None,
) -> list[EpisodeRecord]:
    """Run communication sweep over different K values."""
    records: list[EpisodeRecord] = []
    for k in k_values:
        for i in range(n_runs):
            rec = run_episode(
                episode_id=f"{game.name}-K{k}-run{i}",
                game=game,
                agent_a=agent_a,
                agent_b=agent_b,
                max_comm_rounds=k,
                mode=mode,
                logger=logger,
            )
            records.append(rec)
    return records


def _mean(xs: list[float]) -> float | None:
    return (sum(xs) / len(xs)) if xs else None


def _p50(xs: list[float]) -> float | None:
    if not xs:
        return None
    xs2 = sorted(xs)
    mid = len(xs2) // 2
    if len(xs2) % 2 == 1:
        return float(xs2[mid])
    return float((xs2[mid - 1] + xs2[mid]) / 2)


def summarize_by_k(records: list[EpisodeRecord], *, game=None) -> list[dict]:
    """Aggregate metrics grouped by max_comm_rounds (K)."""
    buckets: dict[int, list[EpisodeRecord]] = defaultdict(list)
    for r in records:
        buckets[r.max_comm_rounds].append(r)

    rows: list[dict] = []

    for k, recs in sorted(buckets.items()):
        n = len(recs) or 1

        agreement_rate = sum(1 for r in recs if r.agreement_hit) / n
        nash_rate = sum(1 for r in recs if r.nash_hit) / n
        pareto_rate = sum(1 for r in recs if r.pareto_hit) / n

        rounds = [r.rounds_to_agreement for r in recs if r.rounds_to_agreement is not None]
        mean_rounds = (sum(rounds) / len(rounds)) if rounds else None

        used_rounds = [float(r.used_comm_rounds) for r in recs]
        used_comm_rounds_mean = _mean(used_rounds)
        used_comm_rounds_p50 = _p50(used_rounds)

        # Частка використаних раундів від дозволених K (None для K=0)
        used_comm_rounds_over_k_mean = None
        if k > 0:
            used_comm_rounds_over_k_mean = _mean([float(r.used_comm_rounds) / float(k) for r in recs])

        # Скільки раундів було "зайвими" після фактичної угоди (рахуємо тільки там, де угода була)
        wasted_comm_rounds_mean = None
        wasted = []
        for r in recs:
            if r.rounds_to_agreement is None:
                continue
            wasted.append(float(r.used_comm_rounds) - float(r.rounds_to_agreement))
        wasted_comm_rounds_mean = _mean(wasted)

        payoff_mean = sum(((r.payoff_a + r.payoff_b) / 2) for r in recs if r.payoff_a is not None and r.payoff_b is not None) / n
        welfare_mean = sum((r.payoff_a + r.payoff_b) for r in recs if r.payoff_a is not None and r.payoff_b is not None) / n
        payoff_diff_mean = sum(abs(r.payoff_a - r.payoff_b) for r in recs if r.payoff_a is not None and r.payoff_b is not None) / n

        a_win_rate = sum(1 for r in recs if r.winner == "agent_a") / n
        b_win_rate = sum(1 for r in recs if r.winner == "agent_b") / n
        tie_rate = sum(1 for r in recs if r.winner == "tie") / n

        comm = [compute_episode_comm_stats(r) for r in recs]
        msg_count_mean = _mean([float(c.n_messages_total) for c in comm])
        words_total_mean = _mean([float(c.n_words_total) for c in comm])
        words_a_mean = _mean([float(c.n_words_agent_a) for c in comm])
        words_b_mean = _mean([float(c.n_words_agent_b) for c in comm])

        propose_rate = sum(1 for c in comm if c.has_propose) / n
        counter_rate = sum(1 for c in comm if c.has_counter) / n
        accept_rate = sum(1 for c in comm if c.has_accept) / n

        # Частка епізодів, де якщо ACCEPT був, то ACTION-и йому відповідають
        if any(c.actions_follow_accept is not None for c in comm):
            denom = sum(1 for c in comm if c.actions_follow_accept is not None)
            follow_accept_rate = sum(1 for c in comm if c.actions_follow_accept is True) / denom
        else:
            follow_accept_rate = None

        regret_a_mean = None
        regret_b_mean = None
        welfare_gap_mean = None
        if game is not None:
            ra, rb, wg = [], [], []
            for r in recs:
                if r.action_a is None or r.action_b is None:
                    continue
                ra.append(regret_a(game, r.action_a, r.action_b))
                rb.append(regret_b(game, r.action_a, r.action_b))
                wg.append(welfare_gap(game, r.action_a, r.action_b))
            regret_a_mean = _mean(ra)
            regret_b_mean = _mean(rb)
            welfare_gap_mean = _mean(wg)

        rows.append(
            {
                "game": recs[0].game,
                "K": k,
                "n_runs": n,
                "agreement_rate": agreement_rate,
                "nash_rate": nash_rate,
                "pareto_rate": pareto_rate,
                "mean_rounds_to_agreement": mean_rounds,
                "used_comm_rounds_mean": used_comm_rounds_mean,
                "used_comm_rounds_p50": used_comm_rounds_p50,
                "used_comm_rounds_over_k_mean": used_comm_rounds_over_k_mean,
                "wasted_comm_rounds_mean": wasted_comm_rounds_mean,
                "payoff_mean": payoff_mean,
                "welfare_mean": welfare_mean,
                "payoff_diff_mean": payoff_diff_mean,
                "a_win_rate": a_win_rate,
                "b_win_rate": b_win_rate,
                "tie_rate": tie_rate,
                "msg_count_mean": msg_count_mean,
                "words_total_mean": words_total_mean,
                "words_a_mean": words_a_mean,
                "words_b_mean": words_b_mean,
                "propose_rate": propose_rate,
                "counter_rate": counter_rate,
                "accept_rate": accept_rate,
                "follow_accept_rate": follow_accept_rate,
                "regret_a_mean": regret_a_mean,
                "regret_b_mean": regret_b_mean,
                "welfare_gap_mean": welfare_gap_mean,
            }
        )

    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

