from __future__ import annotations

import os
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

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


MODELS = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    #"phi2": "microsoft/phi-2",
    #"mistral": "mistralai/Mistral-7B-Instruct-v0.2",
}

GAMES = {
    "prisoners_dilemma": PrisonersDilemma(),
    "stag_hunt": StagHunt(),
    "battle_of_sexes": BattleOfSexes(),
    "ultimatum": UltimatumGame(),
}

MODES = [
    "no_workflow",
    "workflow",
]

def _int_env(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))

def _list_env(name: str, default: str) -> list[int]:
    return [int(x.strip()) for x in os.getenv(name, default).split(",") if x.strip()]

N_RUNS = _int_env("LLMGT_N_RUNS", 50)

K_VALUES = _list_env("LLMGT_K_VALUES", "0,1,2,3,4,5,6")

TEMPERATURE = 0.7

MAX_NEW_TOKENS = 128

METRICS = [
    "agreement_rate",
    "mean_rounds_to_agreement",
    "welfare_mean",
]



def run_single_experiment(run_root, model_name, model_id, mode, game_name, game):

    print(f"Running: {model_name} | {mode} | {game_name}")

    backend_cfg = LLMBackendConfig(
        backend="hf",
        hf_model=model_id,
        hf_max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
    )

    agent_a, agent_b = make_llm_agents(game, backend_cfg)

    out_dir = run_root / "raw" / model_name / mode / game_name

    logs_dir = out_dir / "logs"
    figs_dir = out_dir / "figures"

    logs_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    logger = JsonlLogger(
        out_dir=logs_dir,
        filename="episodes.jsonl",
        overwrite=True,
    )

    records = run_comm_sweep(
        game=game,
        agent_a=agent_a,
        agent_b=agent_b,
        k_values=K_VALUES,
        n_runs=N_RUNS,
        mode=mode,
        logger=logger,
    )

    rows = summarize_by_k(records, game=game)

    df = pd.DataFrame(rows)

    df.to_csv(out_dir / "summary.csv", index=False)

    return df


def build_all_plots(run_root):

    print("\nBuilding ALL thesis plots...")

    raw_dir = run_root / "raw"
    plots_dir = run_root / "plots"

    plots_dir.mkdir(exist_ok=True)

    data = []

    for model_dir in raw_dir.iterdir():
        model = model_dir.name

        for mode_dir in model_dir.iterdir():
            mode = mode_dir.name

            for game_dir in mode_dir.iterdir():
                game = game_dir.name

                df = pd.read_csv(game_dir / "summary.csv")

                df["model"] = model
                df["mode"] = mode
                df["game"] = game

                data.append(df)

    df = pd.concat(data)

    df.to_csv(run_root / "thesis_all_results.csv", index=False)


    for game in df.game.unique():

        for metric in METRICS:

            plt.figure()

            for model in df.model.unique():

                subset = df[
                    (df.game == game)
                    & (df.model == model)
                    & (df.mode == "workflow")
                ]

                plt.plot(
                    subset["k"],
                    subset[metric],
                    label=model,
                    marker="o",
                )

            plt.title(f"{game} — {metric} vs communication rounds")
            plt.xlabel("K communication rounds")
            plt.ylabel(metric)
            plt.legend()

            plt.savefig(
                plots_dir / f"{game}_{metric}_model_comparison.png",
                dpi=200,
                bbox_inches="tight",
            )

            plt.close()


    for game in df.game.unique():

        for metric in METRICS:

            plt.figure()

            for mode in MODES:

                subset = df[
                    (df.game == game)
                    & (df.model == "mistral")
                    & (df.mode == mode)
                ]

                plt.plot(
                    subset["k"],
                    subset[metric],
                    label=mode,
                    marker="o",
                )

            plt.title(f"{game} — workflow vs no_workflow")
            plt.xlabel("K")
            plt.ylabel(metric)
            plt.legend()

            plt.savefig(
                plots_dir / f"{game}_{metric}_workflow_vs_no_workflow.png",
                dpi=200,
                bbox_inches="tight",
            )

            plt.close()


def main():

    run = make_run_dir(
        tag="THESIS_FULL",
        create_standard_dirs=False,
    )

    run_root = run.root

    write_run_meta(
        run_root / "run_meta.json",
        {
            "timestamp": datetime.utcnow().isoformat(),
            "models": MODELS,
            "games": list(GAMES.keys()),
            "modes": MODES,
            "k_values": K_VALUES,
            "n_runs": N_RUNS,
        },
    )

    for model_name, model_id in MODELS.items():

        for mode in MODES:

            for game_name, game in GAMES.items():

                run_single_experiment(
                    run_root,
                    model_name,
                    model_id,
                    mode,
                    game_name,
                    game,
                )

    build_all_plots(run_root)

    print("\nTHESIS PIPELINE COMPLETE")
    print(run_root)


if __name__ == "__main__":
    main()
