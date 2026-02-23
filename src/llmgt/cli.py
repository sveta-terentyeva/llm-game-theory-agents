from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from llmgt.experiments.sweep import run_comm_sweep, summarize_by_k, write_csv
from llmgt.experiments.plotting import plot_metric_by_k
from llmgt.experiments.agent_factories import LLMBackendConfig, make_agents_for_mode
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

    cfg = LLMBackendConfig(
        backend=args.backend,
        temperature=float(args.temperature),
        max_output_tokens=int(args.max_output_tokens),
        # openai (optional)
        openai_model=getattr(args, "openai_model", "gpt-4o-mini"),
        base_url=getattr(args, "base_url", None),
        # ollama
        ollama_model=args.ollama_model,
        ollama_host=args.ollama_host,
        ollama_timeout_s=float(args.ollama_timeout_s),
        # hf
        hf_model=args.hf_model,
        hf_max_new_tokens=int(args.hf_max_new_tokens),
        # agent behavior
        agent_style=args.agent_style,
        workflow_level=int(args.workflow_level),
    )

    agent_a, agent_b = make_agents_for_mode(game, cfg, mode)

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

    # Pass game to summarizer so regret/welfare-gap metrics are available.
    rows = summarize_by_k(records, game=game)
    write_csv(rows, run_dir.root / "summary_by_k.csv")

    if args.plots:
        plot_metric_by_k(
            rows,
            metric="agreement_rate",
            title=f"{game.name} — agreement vs k ({mode})",
            ylabel="Agreement rate",
            out_path=run_dir.figures_dir / "agreement_rate.png",
        )
        plot_metric_by_k(
            rows,
            metric="mean_rounds_to_agreement",
            title=f"{game.name} — rounds-to-agreement vs k ({mode})",
            ylabel="Mean rounds-to-agreement",
            out_path=run_dir.figures_dir / "mean_rounds_to_agreement.png",
        )
        plot_metric_by_k(
            rows,
            metric="welfare_mean",
            title=f"{game.name} — welfare vs k ({mode})",
            ylabel="Mean welfare (A+B)",
            out_path=run_dir.figures_dir / "welfare_mean.png",
        )
        plot_metric_by_k(
            rows,
            metric="theory_rate",
            title=f"{game.name} — theory success vs k ({mode})",
            ylabel="Theory success rate",
            out_path=run_dir.figures_dir / "theory_rate.png",
        )
        plot_metric_by_k(
            rows,
            metric="mean_rounds_to_theory_hit",
            title=f"{game.name} — rounds-to-theory-hit vs k ({mode})",
            ylabel="Mean rounds-to-theory-hit",
            out_path=run_dir.figures_dir / "mean_rounds_to_theory_hit.png",
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
    s.add_argument("--backend", default="heuristic", choices=["heuristic", "ollama", "hf", "openai"])
    s.add_argument("--agent-style", default="strategic", choices=["basic", "strategic"])
    s.add_argument("--workflow-level", type=int, default=2, help="1=light, 2=standard, 3=strict (workflow mode only)")
    s.add_argument("--temperature", type=float, default=0.7)
    s.add_argument("--max-output-tokens", type=int, default=64)

    # HF
    s.add_argument("--hf-model", default="mistralai/Mistral-7B-Instruct-v0.2")
    s.add_argument("--hf-max-new-tokens", type=int, default=128)

    # Ollama
    s.add_argument("--ollama-model", default="llama3.1:8b")
    s.add_argument("--ollama-host", default="http://localhost:11434")
    s.add_argument("--ollama-timeout-s", type=float, default=120.0)

    # OpenAI (optional; you can remove later for “open-source only”)
    s.add_argument("--openai-model", default="gpt-4o-mini")
    s.add_argument("--base-url", default=None)

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
