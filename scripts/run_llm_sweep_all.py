#python scripts / run_llm_sweep_all.py
from __future__ import annotations

import os

from llmgt.experiments import run_comm_sweep, summarize_by_k, write_csv
from llmgt.experiments.plotting import plot_metric_by_k
from llmgt.experiments.agent_factories import make_llm_agents, LLMBackendConfig
from llmgt.logging.jsonl_logger import JsonlLogger
from llmgt.logging.run_meta import write_run_meta
from llmgt.sim.run_dir import make_run_dir

from llmgt.games.prisoners_dilemma import PrisonersDilemma
from llmgt.games.stag_hunt import StagHunt
from llmgt.games.battle_of_sexes import BattleOfSexes
from llmgt.games.ultimatum import UltimatumGame


def run_for_game(game, run_root, tag: str, backend_cfg: LLMBackendConfig) -> None:
    game_dir = run_root / tag
    logs_dir = game_dir / "logs"
    figs_dir = game_dir / "figures"
    logs_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    a, b = make_llm_agents(game, backend_cfg)
    logger = JsonlLogger(out_dir=logs_dir, filename="episodes.jsonl", overwrite=True)

    k_values = list(range(0, 7))
    n_runs = 200
    mode = "workflow"  # negotiation rounds used inside the run_comm_sweep episodes

    records = run_comm_sweep(
        game=game,
        agent_a=a,
        agent_b=b,
        k_values=k_values,
        n_runs=n_runs,
        mode=mode,
        logger=logger,
    )

    rows = summarize_by_k(records)
    write_csv(rows, game_dir / "summary_by_k.csv")

    plot_metric_by_k(
        rows,
        metric="agreement_rate",
        title=f"{game.name} — agreement vs K ({mode}) [{backend_cfg.backend}]",
        ylabel="Agreement rate",
        out_path=figs_dir / "agreement_rate.png",
    )
    plot_metric_by_k(
        rows,
        metric="mean_rounds_to_agreement",
        title=f"{game.name} — rounds-to-agreement vs K ({mode}) [{backend_cfg.backend}]",
        ylabel="Mean rounds-to-agreement",
        out_path=figs_dir / "mean_rounds_to_agreement.png",
    )
    plot_metric_by_k(
        rows,
        metric="welfare_mean",
        title=f"{game.name} — welfare vs K ({mode}) [{backend_cfg.backend}]",
        ylabel="Mean welfare (A+B)",
        out_path=figs_dir / "welfare_mean.png",
    )

    print(f"[{tag}] saved to: {game_dir}")


def main() -> None:
    backend = os.getenv("LLMGT_BACKEND", "heuristic").strip().lower()  # heuristic|openai
    openai_model = os.getenv("LLMGT_OPENAI_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("LLMGT_TEMPERATURE", "0.7"))
    max_out = int(os.getenv("LLMGT_MAX_OUTPUT_TOKENS", "128"))
    base_url = os.getenv("LLMGT_OPENAI_BASE_URL")  # optional

    backend_cfg = LLMBackendConfig(
        backend=backend,  # type: ignore[arg-type]
        openai_model=openai_model,
        temperature=temperature,
        max_output_tokens=max_out,
        base_url=base_url,
    )

    run = make_run_dir(tag=f"llm_{backend}_workflow_sweep_all", create_standard_dirs=False)
    run_root = run.root

    write_run_meta(
        run_root / "run_meta.json",
        {
            "tag": f"llm_{backend}_workflow_sweep_all",
            "mode": "workflow",
            "k_values": list(range(0, 7)),
            "n_runs": 200,
            "agent_backend": backend,
            "openai_model": openai_model if backend == "openai" else None,
            "temperature": temperature,
            "max_output_tokens": max_out,
            "notes": "Per-game outputs under run_root/<game>/",
        },
    )

    run_for_game(PrisonersDilemma(), run_root, "prisoners_dilemma", backend_cfg)
    run_for_game(StagHunt(), run_root, "stag_hunt", backend_cfg)
    run_for_game(BattleOfSexes(), run_root, "battle_of_sexes", backend_cfg)
    run_for_game(UltimatumGame(), run_root, "ultimatum", backend_cfg)

    print(f"\nDONE. Run directory: {run_root}")


if __name__ == "__main__":
    main()
