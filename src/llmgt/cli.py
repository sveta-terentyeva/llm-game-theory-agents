from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from llmgt.experiments.sweep import run_comm_sweep, summarize_by_k, write_csv
from llmgt.experiments.plotting import plot_metric_by_k
from llmgt.experiments.game_configs import make_workflow_agents
from llmgt.games.prisoners_dilemma import PrisonersDilemma
from llmgt.games.stag_hunt import StagHunt
from llmgt.games.battle_of_sexes import BattleOfSexes
from llmgt.games.ultimatum import UltimatumGame
from llmgt.logging.jsonl_logger import JsonlLogger
from llmgt.sim.run_dir import make_run_dir


def _make_game(game_key: str):
    key = game_key.strip().lower()
    if key in {"pd", "prisoners_dilemma", "prisoner", "dilemma"}:
        return PrisonersDilemma()
    if key in {"stag", "stag_hunt", "sh"}:
        return StagHunt()
    if key in {"bos", "battle", "battle_of_sexes"}:
        return BattleOfSexes()
    if key in {"ult", "ultimatum", "ug"}:
        return UltimatumGame()
    raise SystemExit(
        f"Unknown game '{game_key}'. Use one of: pd, stag, bos, ultimatum."
    )


def _parse_k_values(k_args: list[str]) -> list[int]:
    if len(k_args) == 1 and ".." in k_args[0]:
        a_s, b_s = k_args[0].split("..", 1)
        a = int(a_s)
        b = int(b_s)
        step = 1 if b >= a else -1
        return list(range(a, b + step, step))
    return [int(x) for x in k_args]


def _ensure_mode(mode: str) -> str:
    m = mode.strip().lower()
    if m not in {"no_workflow", "workflow"}:
        raise SystemExit("--mode must be one of: no_workflow, workflow")
    return m


def cmd_sweep(args: argparse.Namespace) -> int:
    game = _make_game(args.game)
    mode = _ensure_mode(args.mode)

    run_dir = make_run_dir(base=Path(args.out_dir), tag=args.tag)
    logger = JsonlLogger(run_dir.logs_dir / "episodes.jsonl")

    if mode == "workflow":
        agent_a, agent_b = make_workflow_agents(game)
    else:
        agent_a, agent_b = make_workflow_agents(game)
        agent_a = agent_a.__class__(name=f"{agent_a.name}_no_wf", propose_pair=agent_a.propose_pair)
        agent_b = agent_b.__class__(
            name=f"{agent_b.name}_no_wf",
            fallback_action=agent_b.fallback_action,
            preferred_pair=agent_b.preferred_pair,
            min_payoff=agent_b.min_payoff,
        )

    k_values = _parse_k_values(args.k)

    records = run_comm_sweep(
        game=game,
        agent_a=agent_a,
        agent_b=agent_b,
        k_values=k_values,
        n_runs=int(args.n_runs),
        mode=mode,
        logger=logger,
    )

    rows = summarize_by_k(records)
    write_csv(rows, run_dir.root / "summary_by_k.csv")

    if args.plots:
        plot_metric_by_k(
            rows,
            metric="agreement_rate",
            title=f"{game.name} — agreement vs K ({mode})",
            ylabel="Agreement rate",
            out_path=run_dir.figures_dir / "agreement_rate.png",
        )
        plot_metric_by_k(
            rows,
            metric="mean_rounds_to_agreement",
            title=f"{game.name} — rounds-to-agreement vs K ({mode})",
            ylabel="Mean rounds-to-agreement",
            out_path=run_dir.figures_dir / "mean_rounds_to_agreement.png",
        )
        plot_metric_by_k(
            rows,
            metric="welfare_mean",
            title=f"{game.name} — welfare vs K ({mode})",
            ylabel="Mean welfare (A+B)",
            out_path=run_dir.figures_dir / "welfare_mean.png",
        )

    print(f"Wrote run to: {run_dir.root}")
    print(f"  logs:   {run_dir.logs_dir / 'episodes.jsonl'}")
    print(f"  csv:    {run_dir.root / 'summary_by_k.csv'}")
    if args.plots:
        print(f"  figures:{run_dir.figures_dir}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="llmgt", description="LLM game-theory experiments")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sweep", help="Run a communication sweep over K")
    s.add_argument("--game", required=True, help="pd | stag | bos | ultimatum")
    s.add_argument("--mode", default="workflow", help="workflow | no_workflow")
    s.add_argument(
        "--k",
        nargs="+",
        default=["0..6"],
        help="K values, e.g. --k 0..6 or --k 0 1 2 3",
    )
    s.add_argument("--n-runs", type=int, default=200, help="episodes per K")
    s.add_argument("--out-dir", default="data/runs", help="base output dir")
    s.add_argument("--tag", default="sweep", help="tag for run directory")
    s.add_argument("--plots", action="store_true", help="also write PNG plots (needs matplotlib)")
    s.set_defaults(func=cmd_sweep)

    return p


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
