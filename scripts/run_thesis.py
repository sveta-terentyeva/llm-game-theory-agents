from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt

from llmgt.experiments import run_comm_sweep, summarize_by_k
from llmgt.experiments.agent_factories import LLMBackendConfig, make_agents_for_mode
from llmgt.logging.jsonl_logger import JsonlLogger
from llmgt.logging.run_meta import write_run_meta
from llmgt.sim.run_dir import make_run_dir

from llmgt.games.prisoners_dilemma import PrisonersDilemma
from llmgt.games.stag_hunt import StagHunt
from llmgt.games.battle_of_sexes import BattleOfSexes
from llmgt.games.ultimatum import UltimatumGame


# Config
MODELS: Dict[str, str] = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    # "phi2": "microsoft/phi-2",
    # "mistral7b": "mistralai/Mistral-7B-Instruct-v0.2",
}

GAMES = {
    "prisoners_dilemma": PrisonersDilemma(),
    "stag_hunt": StagHunt(),
    "battle_of_sexes": BattleOfSexes(),
    "ultimatum": UltimatumGame(),
}

MODES = ["no_workflow", "workflow"]

# Env overrides:
#   LLMGT_N_RUNS=50
#   LLMGT_K_VALUES=0,1,2,3,4,5,6
#   LLMGT_TEMPERATURE=0.7
#   LLMGT_MAX_NEW_TOKENS=64
#   LLMGT_WORKFLOW_LEVEL=2
#   LLMGT_AGENT_STYLE=strategic
N_RUNS = int(os.getenv("LLMGT_N_RUNS", "50"))
K_VALUES = [int(x.strip()) for x in os.getenv("LLMGT_K_VALUES", "0,1,2,3,4,5,6").split(",") if x.strip()]
TEMPERATURE = float(os.getenv("LLMGT_TEMPERATURE", "0.7"))
MAX_NEW_TOKENS = int(os.getenv("LLMGT_MAX_NEW_TOKENS", "64"))
WORKFLOW_LEVEL = int(os.getenv("LLMGT_WORKFLOW_LEVEL", "2"))
AGENT_STYLE = os.getenv("LLMGT_AGENT_STYLE", "strategic")

METRICS: List[str] = [
    "theory_rate",
    "mean_rounds_to_theory_hit",
    "agreement_rate",
    "mean_rounds_to_agreement",
    "welfare_mean",
]



# Helpers
def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing summary csv: {path}")
    return pd.read_csv(path)


def _plot_lines(
    *,
    df: pd.DataFrame,
    x: str,
    y: str,
    group_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
) -> None:
    plt.figure()
    for g in sorted(df[group_col].unique()):
        s = df[df[group_col] == g].sort_values(x)
        if s.empty or y not in s.columns:
            continue
        series = s[y]
        if series.isna().all():
            continue
        plt.plot(s[x], s[y], marker="o", label=str(g))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()



# Core pipeline
def run_single_experiment(
    *,
    run_root: Path,
    model_name: str,
    model_id: str,
    mode: str,
    game_name: str,
    game,
) -> pd.DataFrame:
    print(f"[run] model={model_name} mode={mode} game={game_name}")

    backend_cfg = LLMBackendConfig(
        backend="hf",
        hf_model=model_id,
        hf_max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        agent_style=AGENT_STYLE,
        workflow_level=WORKFLOW_LEVEL
    )

    agent_a, agent_b = make_agents_for_mode(game, backend_cfg, mode)

    out_dir = run_root / "raw" / model_name / mode / game_name
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_name = f"episodes_{_utc_ts()}.jsonl"
    logger = JsonlLogger(out_dir=logs_dir, filename=log_name, overwrite=False)

    records = run_comm_sweep(
        game=game,
        agent_a=agent_a,
        agent_b=agent_b,
        k_values=K_VALUES,
        n_runs=N_RUNS,
        mode=mode,
        logger=logger,
    )

    rows = summarize_by_k(records)
    df = pd.DataFrame(rows)

    df.to_csv(out_dir / "summary.csv", index=False)
    return df


def build_all_plots(run_root: Path) -> None:
    print("\n[plots] Building thesis plots...")

    raw_dir = run_root / "raw"
    plots_dir = run_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    data: List[pd.DataFrame] = []
    for model_dir in raw_dir.iterdir():
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        for mode_dir in model_dir.iterdir():
            if not mode_dir.is_dir():
                continue
            mode = mode_dir.name
            for game_dir in mode_dir.iterdir():
                if not game_dir.is_dir():
                    continue
                game = game_dir.name
                df = _safe_read_csv(game_dir / "summary.csv")
                df["model"] = model
                df["mode"] = mode
                df["game"] = game
                data.append(df)

    if not data:
        raise RuntimeError(f"No results found under: {raw_dir}")

    df_all = pd.concat(data, ignore_index=True)
    df_all.to_csv(run_root / "thesis_all_results.csv", index=False)

    for game in sorted(df_all["game"].unique()):
        for metric in METRICS:
            subset = df_all[(df_all["game"] == game) & (df_all["mode"] == "workflow")]
            if subset.empty or metric not in subset.columns:
                continue
            if subset[metric].isna().all():
                continue
            _plot_lines(
                df=subset,
                x="k",
                y=metric,
                group_col="model",
                title=f"{game} — {metric} vs K (mode=workflow)",
                xlabel="K (max communication rounds)",
                ylabel=metric,
                out_path=plots_dir / "model_comparison" / f"{game}_{metric}_models_workflow.png",
            )

    for game in sorted(df_all["game"].unique()):
        for model in sorted(df_all["model"].unique()):
            for metric in METRICS:
                subset = df_all[(df_all["game"] == game) & (df_all["model"] == model)]
                if subset.empty or metric not in subset.columns:
                    continue
                if subset[metric].isna().all():
                    continue
                _plot_lines(
                    df=subset,
                    x="k",
                    y=metric,
                    group_col="mode",
                    title=f"{game} — {metric} vs K ({model}: workflow vs no_workflow)",
                    xlabel="K (max communication rounds)",
                    ylabel=metric,
                    out_path=plots_dir / "mode_comparison" / f"{game}_{metric}_{model}_modes.png",
                )

    print(f"[plots] Wrote plots to: {plots_dir}")


def main() -> None:
    run = make_run_dir(tag="THESIS_FULL", create_standard_dirs=False)
    run_root = run.root

    write_run_meta(
        run_root / "run_meta.json",
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "models": MODELS,
            "games": list(GAMES.keys()),
            "modes": MODES,
            "k_values": K_VALUES,
            "n_runs": N_RUNS,
            "backend": "hf",
            "temperature": TEMPERATURE,
            "hf_max_new_tokens": MAX_NEW_TOKENS,
            "workflow_level": WORKFLOW_LEVEL,
            "agent_style": AGENT_STYLE,
        },
    )

    for model_name, model_id in MODELS.items():
        for mode in MODES:
            for game_name, game in GAMES.items():
                run_single_experiment(
                    run_root=run_root,
                    model_name=model_name,
                    model_id=model_id,
                    mode=mode,
                    game_name=game_name,
                    game=game,
                )

    build_all_plots(run_root)

    print("\nTHESIS PIPELINE COMPLETE")
    print(run_root)


if __name__ == "__main__":
    main()
