from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from llmgt.sim.runner import run_episode
from llmgt.logging.records import EpisodeRecord
from llmgt.agents.simple import FixedActionAgent


def run_comm_sweep(
    *,
    game,
    agent_a,
    agent_b,
    k_values: Iterable[int],
    n_runs: int,
    mode: str = "no_workflow",
) -> list[EpisodeRecord]:
    """
    Run communication sweep over different K values.
    """

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
            )
            records.append(rec)

    return records


def summarize_by_k(records: list[EpisodeRecord]) -> list[dict]:
    """
    Aggregate metrics grouped by max_comm_rounds (K).
    """

    buckets: dict[int, list[EpisodeRecord]] = defaultdict(list)

    for r in records:
        buckets[r.max_comm_rounds].append(r)

    rows: list[dict] = []

    for k, recs in sorted(buckets.items()):
        n = len(recs)

        agreement_rate = sum(1 for r in recs if r.agreement_hit) / n
        nash_rate = sum(1 for r in recs if r.nash_hit) / n

        rounds = [
            r.rounds_to_agreement
            for r in recs
            if r.rounds_to_agreement is not None
        ]
        mean_rounds = sum(rounds) / len(rounds) if rounds else None

        payoff_mean = sum(
            (r.payoff_a + r.payoff_b) / 2 for r in recs
        ) / n

        welfare_mean = sum(
            r.payoff_a + r.payoff_b for r in recs
        ) / n

        rows.append(
            {
                "game": recs[0].game,
                "K": k,
                "n_runs": n,
                "agreement_rate": agreement_rate,
                "nash_rate": nash_rate,
                "mean_rounds_to_agreement": mean_rounds,
                "payoff_mean": payoff_mean,
                "welfare_mean": welfare_mean,
            }
        )

    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
