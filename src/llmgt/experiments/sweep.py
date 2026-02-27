"""Communication-budget sweep and per-*k* aggregation.

The ``run_comm_sweep`` function runs multiple episodes for each value of *k*
(max communication rounds).  ``summarize_by_k`` computes aggregate metrics
grouped by *k*, and ``write_csv`` persists the result.

When ``max_workers > 1`` the episodes within each *k*-bucket are dispatched
to a thread pool, which dramatically speeds up I/O-bound LLM API calls.
"""

from __future__ import annotations

import csv
import logging
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional
import math

from llmgt.sim.runner import run_episode
from llmgt.logging.records import EpisodeRecord
from llmgt.logging.jsonl_logger import JsonlLogger
from llmgt.metrics import (
    compute_episode_comm_stats,
    regret_a,
    regret_b,
    welfare_gap,
)

# Default parallelism — override with LLMGT_MAX_WORKERS env var.
_DEFAULT_MAX_WORKERS = int(os.getenv("LLMGT_MAX_WORKERS", "8"))
_log = logging.getLogger(__name__)


def _run_one_episode(
    *,
    game,
    agent_a,
    agent_b,
    k: int,
    run_idx: int,
    mode: str,
    logger: Optional[JsonlLogger],
) -> EpisodeRecord:
    """Helper executed in a worker thread."""
    return run_episode(
        episode_id=f"{game.name}-K{k}-run{run_idx}",
        game=game,
        agent_a=agent_a,
        agent_b=agent_b,
        max_comm_rounds=k,
        mode=mode,
        logger=logger,
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
    max_workers: int = _DEFAULT_MAX_WORKERS,
) -> list[EpisodeRecord]:
    """Run communication sweep over different K values.

    Parameters
    ----------
    max_workers : int
        Number of parallel threads for episode execution.
        Set to 1 for sequential (old) behaviour.  Default is read from
        ``LLMGT_MAX_WORKERS`` env var (fallback: 8).
    """
    k_list = list(k_values)
    records: list[EpisodeRecord] = []

    if max_workers <= 1:
        # Sequential fallback — identical to old behaviour
        for k in k_list:
            for i in range(n_runs):
                rec = _run_one_episode(
                    game=game, agent_a=agent_a, agent_b=agent_b,
                    k=k, run_idx=i, mode=mode, logger=logger,
                )
                records.append(rec)
        return records

    # Parallel execution — episodes within each k-bucket run concurrently
    total_episodes = len(k_list) * n_runs
    done_count = 0

    for k in k_list:
        futures = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for i in range(n_runs):
                fut = pool.submit(
                    _run_one_episode,
                    game=game,
                    agent_a=agent_a,
                    agent_b=agent_b,
                    k=k,
                    run_idx=i,
                    mode=mode,
                    logger=logger,
                )
                futures.append(fut)

            for fut in as_completed(futures):
                done_count += 1
                try:
                    records.append(fut.result())
                except Exception as exc:  # noqa: BLE001
                    _log.error(
                        "Episode failed (K=%d): %s: %s — skipping",
                        k, type(exc).__name__, exc,
                    )
                if done_count % max(1, max_workers) == 0 or done_count == total_episodes:
                    print(f"  [{done_count}/{total_episodes}] episodes done (K={k})")

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


def _std(xs: list[float]) -> float | None:
    """Sample standard deviation (ddof=1). Returns None for <2 values."""
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var)


def summarize_by_k(records: list[EpisodeRecord], *, game=None) -> list[dict]:
    """Aggregate metrics grouped by max_comm_rounds (k)."""
    buckets: dict[int, list[EpisodeRecord]] = defaultdict(list)
    for r in records:
        buckets[r.max_comm_rounds].append(r)

    rows: list[dict] = []

    for k, recs in sorted(buckets.items()):
        n = len(recs) or 1

        agreement_rate = sum(1 for r in recs if r.agreement_hit) / n
        nash_rate = sum(1 for r in recs if r.nash_hit) / n
        pareto_rate = sum(1 for r in recs if r.pareto_hit) / n
        pareto_nash_rate = sum(1 for r in recs if r.pareto_nash_hit) / n
        theory_rate = sum(1 for r in recs if r.theory_hit) / n

        rounds = [float(r.rounds_to_agreement) for r in recs if r.rounds_to_agreement is not None]
        mean_rounds = _mean(rounds)
        mean_rounds_std = _std(rounds)

        rounds_theory = [float(r.rounds_to_theory_hit) for r in recs if r.rounds_to_theory_hit is not None]
        mean_rounds_to_theory_hit = _mean(rounds_theory)
        mean_rounds_to_theory_hit_std = _std(rounds_theory)

        used_rounds = [float(r.used_comm_rounds) for r in recs]
        used_comm_rounds_mean = _mean(used_rounds)
        used_comm_rounds_std = _std(used_rounds)
        used_comm_rounds_p50 = _p50(used_rounds)

        used_comm_rounds_over_k_mean = None
        used_comm_rounds_over_k_std = None
        if k > 0:
            used_over_k = [float(r.used_comm_rounds) / float(k) for r in recs]
            used_comm_rounds_over_k_mean = _mean(used_over_k)
            used_comm_rounds_over_k_std = _std(used_over_k)

        wasted = []
        for r in recs:
            if r.rounds_to_agreement is None:
                continue
            wasted.append(float(r.used_comm_rounds) - float(r.rounds_to_agreement))
        wasted_comm_rounds_mean = _mean(wasted)
        wasted_comm_rounds_std = _std(wasted)

        payoffs_avg = [((r.payoff_a + r.payoff_b) / 2) for r in recs if r.payoff_a is not None and r.payoff_b is not None]
        welfare_vals = [(r.payoff_a + r.payoff_b) for r in recs if r.payoff_a is not None and r.payoff_b is not None]
        payoff_diff_vals = [abs(r.payoff_a - r.payoff_b) for r in recs if r.payoff_a is not None and r.payoff_b is not None]

        payoff_mean = (sum(payoffs_avg) / n) if payoffs_avg else None
        welfare_mean = (sum(welfare_vals) / n) if welfare_vals else None
        payoff_diff_mean = (sum(payoff_diff_vals) / n) if payoff_diff_vals else None

        payoff_mean_std = _std([float(x) for x in payoffs_avg])
        welfare_mean_std = _std([float(x) for x in welfare_vals])
        payoff_diff_mean_std = _std([float(x) for x in payoff_diff_vals])

        a_win_rate = sum(1 for r in recs if r.winner == "agent_a") / n
        b_win_rate = sum(1 for r in recs if r.winner == "agent_b") / n
        tie_rate = sum(1 for r in recs if r.winner == "tie") / n

        comm = [compute_episode_comm_stats(r) for r in recs]
        msg_count = [float(c.n_messages_total) for c in comm]
        words_total = [float(c.n_words_total) for c in comm]
        words_a = [float(c.n_words_agent_a) for c in comm]
        words_b = [float(c.n_words_agent_b) for c in comm]

        msg_count_mean = _mean(msg_count)
        msg_count_std = _std(msg_count)
        words_total_mean = _mean(words_total)
        words_total_std = _std(words_total)
        words_a_mean = _mean(words_a)
        words_a_std = _std(words_a)
        words_b_mean = _mean(words_b)
        words_b_std = _std(words_b)

        propose_rate = sum(1 for c in comm if c.has_propose) / n
        counter_rate = sum(1 for c in comm if c.has_counter) / n
        accept_rate = sum(1 for c in comm if c.has_accept) / n

        if any(c.actions_follow_accept is not None for c in comm):
            denom = sum(1 for c in comm if c.actions_follow_accept is not None)
            follow_accept_rate = sum(1 for c in comm if c.actions_follow_accept is True) / denom
        else:
            follow_accept_rate = None

        regret_a_mean = None
        regret_b_mean = None
        welfare_gap_mean = None
        regret_a_std = None
        regret_b_std = None
        welfare_gap_std = None
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
            regret_a_std = _std(ra)
            regret_b_std = _std(rb)
            welfare_gap_std = _std(wg)

        rows.append(
            {
                "game": recs[0].game,
                "k": k,
                "n_runs": n,
                "agreement_rate": agreement_rate,
                "nash_rate": nash_rate,
                "pareto_rate": pareto_rate,
                "mean_rounds_to_agreement": mean_rounds,
                "mean_rounds_to_agreement_std": mean_rounds_std,
                "used_comm_rounds_mean": used_comm_rounds_mean,
                "used_comm_rounds_std": used_comm_rounds_std,
                "used_comm_rounds_p50": used_comm_rounds_p50,
                "used_comm_rounds_over_k_mean": used_comm_rounds_over_k_mean,
                "used_comm_rounds_over_k_std": used_comm_rounds_over_k_std,
                "wasted_comm_rounds_mean": wasted_comm_rounds_mean,
                "wasted_comm_rounds_std": wasted_comm_rounds_std,
                "payoff_mean": payoff_mean,
                "payoff_mean_std": payoff_mean_std,
                "welfare_mean": welfare_mean,
                "welfare_mean_std": welfare_mean_std,
                "payoff_diff_mean": payoff_diff_mean,
                "payoff_diff_mean_std": payoff_diff_mean_std,
                "a_win_rate": a_win_rate,
                "b_win_rate": b_win_rate,
                "tie_rate": tie_rate,
                "msg_count_mean": msg_count_mean,
                "msg_count_mean_std": msg_count_std,
                "words_total_mean": words_total_mean,
                "words_total_mean_std": words_total_std,
                "words_a_mean": words_a_mean,
                "words_a_mean_std": words_a_std,
                "words_b_mean": words_b_mean,
                "words_b_mean_std": words_b_std,
                "propose_rate": propose_rate,
                "counter_rate": counter_rate,
                "accept_rate": accept_rate,
                "follow_accept_rate": follow_accept_rate,
                "regret_a_mean": regret_a_mean,
                "regret_a_mean_std": regret_a_std,
                "regret_b_mean": regret_b_mean,
                "regret_b_mean_std": regret_b_std,
                "welfare_gap_mean": welfare_gap_mean,
                "welfare_gap_mean_std": welfare_gap_std,
                "pareto_nash_rate": pareto_nash_rate,
                "theory_rate": theory_rate,
                "mean_rounds_to_theory_hit": mean_rounds_to_theory_hit,
                "mean_rounds_to_theory_hit_std": mean_rounds_to_theory_hit_std,
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

