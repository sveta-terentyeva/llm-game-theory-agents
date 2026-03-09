from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List

from dotenv import load_dotenv
load_dotenv()  # reads .env for OPENROUTER_API_KEY, etc.

# Enable LLM response caching by default for this script.
# Precedence: shell env > .env > this default.
os.environ.setdefault("LLMGT_LLM_CACHE", "1")

import pandas as pd

# Use non-interactive backend for reliability in headless runs
import matplotlib
matplotlib.use("Agg")
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


# Config — OpenRouter model identifiers
MODELS: Dict[str, str] = {
    # --- OpenAI ---
     #"gpt-4o-mini":        "openai/gpt-4o-mini",
    #"gpt-4o":             "openai/gpt-4o",

    # --- Claude ---
    #"claude-3.5-haiku":   "anthropic/claude-3.5-haiku",
    #"claude-3.5-sonnet":  "anthropic/claude-3.5-sonnet",

    # --- Free example (OpenRouter) ---
    "llama-3.3":          "meta-llama/llama-3.3-70b-instruct",
}

GAMES = {
    #"prisoners_dilemma": PrisonersDilemma(),
    #"stag_hunt": StagHunt(),
    #"battle_of_sexes": BattleOfSexes(),
    "ultimatum": UltimatumGame(),
}

MODES = ["no_workflow", "workflow"]

# Env overrides:
#   LLMGT_N_RUNS=100
#   LLMGT_K_VALUES=0,1,2,3,4,5,6, 7,8,9
#   LLMGT_TEMPERATURE=0.7
#   LLMGT_MAX_NEW_TOKENS=64
#   LLMGT_WORKFLOW_LEVEL=2
#   LLMGT_AGENT_STYLE=strategic
N_RUNS = int(os.getenv("LLMGT_N_RUNS", "100"))
K_VALUES = [int(x.strip()) for x in os.getenv("LLMGT_K_VALUES", "0,1,2,3,4,5,6,7,8,9").split(",") if x.strip()]
TEMPERATURE = float(os.getenv("LLMGT_TEMPERATURE", "0.7"))
MAX_NEW_TOKENS = int(os.getenv("LLMGT_MAX_NEW_TOKENS", "64"))
WORKFLOW_LEVEL = int(os.getenv("LLMGT_WORKFLOW_LEVEL", "2"))
AGENT_STYLE: str = os.getenv("LLMGT_AGENT_STYLE", "strategic")  # type: ignore[assignment]


METRICS: List[str] = [
    # --- Core outcomes ---
    "agreement_rate",
    "theory_rate",
    "pareto_nash_rate",

    # --- Dynamics ---
    "mean_rounds_to_agreement",
    "mean_rounds_to_theory_hit",

    # --- Welfare / fairness ---
    "welfare_mean",
    "welfare_gap_mean",
    "payoff_diff_mean",

    # --- Communication efficiency ---
    "used_comm_rounds_mean",
    "used_comm_rounds_over_k_mean",
    "wasted_comm_rounds_mean",

    # --- Light mechanism / sanity (keep small) ---
    "accept_rate",
    "counter_rate",

    # --- Derived / summary ---
    "regret_mean",
]


# Helpers
def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing summary csv: {path}")
    return pd.read_csv(path)


def _normalize_summary_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize summary.csv schemas across versions.

    Common issues that can break/warp plots:
    - older summaries use 'K' not 'k'
    - K parsed as string/object, causing lexicographic sorting
    - duplicate K rows (e.g., concatenated runs) causing jagged lines
    """

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "k" not in df.columns and "K" in df.columns:
        df = df.rename(columns={"K": "k"})

    if "k" not in df.columns:
        raise ValueError("Summary is missing 'k' (or legacy 'K') column")

    df["k"] = pd.to_numeric(df["k"], errors="coerce")
    df = df.dropna(subset=["k"])
    df["k"] = df["k"].astype(int)

    # Ensure numeric metrics where possible
    for c in df.columns:
        if c in {"game", "mode", "model"}:
            continue
        if c == "k":
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Aggregate duplicates deterministically
    # (mean for metrics, sum for n_runs if present)
    if df.duplicated(subset=["k"]).any():
        agg: dict[str, str] = {}
        for c in df.columns:
            if c == "k":
                continue
            if c == "n_runs":
                agg[c] = "sum"
            else:
                agg[c] = "mean"
        df = df.groupby("k", as_index=False).agg(agg)

    return df.sort_values("k")


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
    for g in sorted(df[group_col].dropna().unique()):
        s = df[df[group_col] == g].copy()
        if s.empty or y not in s.columns:
            continue
        s = s.sort_values(x)

        series = pd.to_numeric(s[y], errors="coerce")
        if series.isna().all():
            continue

        # Drop NaNs to avoid broken/"teleporting" segments
        mask = ~series.isna() & ~pd.to_numeric(s[x], errors="coerce").isna()
        xs = s.loc[mask, x]
        ys = series.loc[mask]
        if len(xs) == 0:
            continue

        # Optional error bars if *_std exists
        y_std_col = f"{y}_std"
        if y_std_col in s.columns:
            yerr = pd.to_numeric(s.loc[mask, y_std_col], errors="coerce")
            if not yerr.isna().all():
                plt.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3, label=str(g))
                continue

        plt.plot(xs, ys, marker="o", label=str(g))

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def _plot_bars(
    *,
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
) -> None:
    """Simple mean bar chart.

    Notes:
    - For some runs/modes a metric can be all-NaN (e.g., failed experiments).
      If we drop NaNs naively, entire categories disappear and the plot looks
      "broken". Here we keep categories and fill missing values for *rates*
      with 0 so the absence is visible.
    """

    plt.figure()

    if x not in df.columns or y not in df.columns:
        return

    s = df[[x, y]].copy()

    # deterministic order
    order = list(sorted(s[x].dropna().unique(), key=lambda v: str(v)))
    if not order:
        return

    series = pd.to_numeric(s[y], errors="coerce")

    # Fill missing values for rates (so missing mode doesn't vanish)
    if y.endswith("_rate"):
        series = series.fillna(0.0)

    values = []
    any_filled = False
    for v in order:
        mask = s[x] == v
        sub = series[mask]
        if sub.dropna().empty:
            values.append(0.0)
            any_filled = True
        else:
            values.append(float(sub.mean(skipna=True)))

    if all(pd.isna(values)):
        return

    plot_title = title
    if any_filled and y.endswith("_rate"):
        plot_title = title + " (missing→0)"

    plt.bar([str(v) for v in order], values)
    plt.title(plot_title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def _plot_scatter(
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
    """Scatter plot for cross-metric tradeoffs."""
    plt.figure()

    for g in sorted(df[group_col].dropna().unique()):
        s = df[df[group_col] == g].copy()
        if s.empty or x not in s.columns or y not in s.columns:
            continue

        xs = pd.to_numeric(s[x], errors="coerce")
        ys = pd.to_numeric(s[y], errors="coerce")
        mask = ~xs.isna() & ~ys.isna()
        if mask.sum() == 0:
            continue

        plt.scatter(xs[mask], ys[mask], label=str(g), alpha=0.85)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def _add_derived_metrics(df_all: pd.DataFrame) -> pd.DataFrame:
    """Add derived metrics used for plotting.

    Keep derived metrics stable and interpretable across games.
    """

    df_all = df_all.copy()

    # Regret (average across players) is easier to read than two separate series.
    if "regret_a_mean" in df_all.columns and "regret_b_mean" in df_all.columns:
        ra = pd.to_numeric(df_all["regret_a_mean"], errors="coerce")
        rb = pd.to_numeric(df_all["regret_b_mean"], errors="coerce")
        df_all["regret_mean"] = (ra + rb) / 2.0

    # Equality index: 1 - (avg payoff gap / welfare). Higher => fairer split.
    # Keep as optional derived; can be unstable if welfare ~ 0.
    if "payoff_diff_mean" in df_all.columns and "welfare_mean" in df_all.columns:
        gap = pd.to_numeric(df_all["payoff_diff_mean"], errors="coerce")
        welfare = pd.to_numeric(df_all["welfare_mean"], errors="coerce")
        with pd.option_context("mode.use_inf_as_na", True):
            df_all["equality_index"] = 1.0 - (gap / welfare)

    return df_all


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
        backend="openrouter",
        openrouter_model=model_id,
        max_output_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        agent_style=AGENT_STYLE,  # type: ignore[arg-type]
        workflow_level=WORKFLOW_LEVEL,
    )

    agent_a, agent_b = make_agents_for_mode(game, backend_cfg, mode)  # type: ignore[arg-type]

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

    rows = summarize_by_k(records, game=game)
    df = pd.DataFrame(rows)

    df.to_csv(out_dir / "summary.csv", index=False)
    return df


def _is_near_constant(series: pd.Series, *, atol: float = 1e-6, rtol: float = 1e-3) -> bool:
    """Return True if the numeric series is effectively constant."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 2:
        return True

    vmin = float(s.min())
    vmax = float(s.max())
    span = vmax - vmin
    scale = max(1.0, abs(vmin), abs(vmax))
    return span <= (atol + rtol * scale)


def build_all_plots(run_root: Path) -> None:
    print("\n[plots] Building thesis plots...")

    raw_dir = run_root / "raw"
    plots_dir = run_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Clean old plots so removed metrics don't linger.
    for sub in [plots_dir / "mode_comparison", plots_dir / "model_comparison", plots_dir / "global"]:
        if sub.exists():
            for p in sub.glob("*.png"):
                try:
                    p.unlink()
                except OSError:
                    pass

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
                df = _normalize_summary_df(df)

                df["model"] = model
                df["mode"] = mode
                df["game"] = game
                data.append(df)

    if not data:
        raise RuntimeError(f"No results found under: {raw_dir}")

    df_all = pd.concat(data, ignore_index=True)
    df_all = _add_derived_metrics(df_all)
    df_all.to_csv(run_root / "thesis_all_results.csv", index=False)

    global_dir = plots_dir / "global"
    global_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Global plots (collapse across K): "overall behavior" summaries
    # ------------------------------------------------------------------

    # Overall mode comparison per game (average across K + models)
    for game in sorted(df_all["game"].unique()):
        gdf = df_all[df_all["game"] == game].copy()
        if gdf.empty:
            continue

        collapsed = (
            gdf.groupby(["mode"], as_index=False)
            .agg({m: "mean" for m in METRICS if m in gdf.columns})
        )

        for metric in ["agreement_rate", "theory_rate", "welfare_mean", "payoff_diff_mean", "mean_rounds_to_agreement"]:
            if metric not in collapsed.columns or collapsed[metric].isna().all():
                continue

            _plot_bars(
                df=collapsed,
                x="mode",
                y=metric,
                title=f"{game} — {metric} (mean over K, models)",
                xlabel="Mode",
                ylabel=metric,
                out_path=global_dir / f"{game}_{metric}_by_mode.png",
            )

    # Trade-off scatters (each point is a (k,model,mode) cell)
    for game in sorted(df_all["game"].unique()):
        gdf = df_all[df_all["game"] == game].copy()
        if gdf.empty:
            continue

        if {"welfare_mean", "theory_rate", "mode"}.issubset(gdf.columns):
            _plot_scatter(
                df=gdf,
                x="theory_rate",
                y="welfare_mean",
                group_col="mode",
                title=f"{game} — welfare vs theory_rate (all K, models)",
                xlabel="Theory success rate",
                ylabel="Mean welfare (A+B)",
                out_path=global_dir / f"{game}_scatter_welfare_vs_theory_by_mode.png",
            )

        if {"welfare_mean", "payoff_diff_mean", "mode"}.issubset(gdf.columns):
            _plot_scatter(
                df=gdf,
                x="payoff_diff_mean",
                y="welfare_mean",
                group_col="mode",
                title=f"{game} — welfare vs payoff_diff (all K, models)",
                xlabel="Mean |payoff A - payoff B|",
                ylabel="Mean welfare (A+B)",
                out_path=global_dir / f"{game}_scatter_welfare_vs_payoffdiff_by_mode.png",
            )

        # Agreement vs communication-budget usage (efficiency / stopping behavior)
        if {"agreement_rate", "used_comm_rounds_over_k_mean", "mode"}.issubset(gdf.columns):
            _plot_scatter(
                df=gdf,
                x="used_comm_rounds_over_k_mean",
                y="agreement_rate",
                group_col="mode",
                title=f"{game} — agreement_rate vs used_comm_rounds_over_k_mean (all K, models)",
                xlabel="Mean used_comm_rounds / K",
                ylabel="Agreement rate",
                out_path=global_dir / f"{game}_scatter_agreement_vs_used_over_k_by_mode.png",
            )

    # Workflow-only model comparison (average across K)
    for game in sorted(df_all["game"].unique()):
        wf = df_all[(df_all["game"] == game) & (df_all["mode"] == "workflow")].copy()
        if wf.empty:
            continue

        collapsed = (
            wf.groupby(["model"], as_index=False)
            .agg({m: "mean" for m in METRICS if m in wf.columns})
        )

        for metric in ["theory_rate", "welfare_mean", "mean_rounds_to_agreement"]:
            if metric not in collapsed.columns or collapsed[metric].isna().all():
                continue

            _plot_bars(
                df=collapsed,
                x="model",
                y=metric,
                title=f"{game} — {metric} (workflow mean over K)",
                xlabel="Model",
                ylabel=metric,
                out_path=global_dir / f"{game}_{metric}_by_model_workflow.png",
            )

    # ------------------------------------------------------------------
    # Vs-K plots (main thesis figures)
    # ------------------------------------------------------------------

    # 1) Workflow-only model comparison vs K (keep small)
    for game in sorted(df_all["game"].unique()):
        for metric in ["theory_rate", "welfare_mean", "mean_rounds_to_agreement", "payoff_diff_mean"]:
            subset = df_all[(df_all["game"] == game) & (df_all["mode"] == "workflow")]
            if subset.empty or metric not in subset.columns or subset[metric].isna().all():
                continue
            if _is_near_constant(subset[metric]):
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

    # 2) Mode comparison vs K (per model): keep only the most interpretable metrics
    for game in sorted(df_all["game"].unique()):
        for model in sorted(df_all["model"].unique()):
            for metric in [
                "agreement_rate",
                "theory_rate",
                "welfare_mean",
                "payoff_diff_mean",
                "mean_rounds_to_agreement",
                "used_comm_rounds_over_k_mean",
            ]:
                subset = df_all[(df_all["game"] == game) & (df_all["model"] == model)]
                if subset.empty or metric not in subset.columns or subset[metric].isna().all():
                    continue
                if _is_near_constant(subset[metric]):
                    continue

                _plot_lines(
                    df=subset,
                    x="k",
                    y=metric,
                    group_col="mode",
                    title=f"{game} — {metric} vs K ({model}: workflow vs no_workflow)",
                    xlabel="K (max communication rounds)",
                    ylabel=metric,
                    out_path=plots_dir
                    / "mode_comparison"
                    / f"{game}_{metric}_{model}_modes.png",
                )

    # 3) Delta plots: (workflow - no_workflow) vs K (collapsed over models)
    for game in sorted(df_all["game"].unique()):
        gdf = df_all[df_all["game"] == game].copy()
        if gdf.empty:
            continue

        wf = gdf[gdf["mode"] == "workflow"]
        nw = gdf[gdf["mode"] == "no_workflow"]
        if wf.empty or nw.empty:
            continue

        # mean across models first, then take difference per k
        wf_k = wf.groupby("k", as_index=False).agg({m: "mean" for m in wf.columns if m in METRICS})
        nw_k = nw.groupby("k", as_index=False).agg({m: "mean" for m in nw.columns if m in METRICS})

        merged = pd.merge(wf_k, nw_k, on="k", suffixes=("_wf", "_nw"))
        if merged.empty:
            continue

        merged["mode"] = "delta(workflow-no_workflow)"

        for metric in ["theory_rate", "welfare_mean", "mean_rounds_to_agreement", "payoff_diff_mean"]:
            a = f"{metric}_wf"
            b = f"{metric}_nw"
            if a not in merged.columns or b not in merged.columns:
                continue
            merged[metric] = pd.to_numeric(merged[a], errors="coerce") - pd.to_numeric(merged[b], errors="coerce")
            if merged[metric].isna().all() or _is_near_constant(merged[metric]):
                continue

            _plot_lines(
                df=merged,
                x="k",
                y=metric,
                group_col="mode",
                title=f"{game} — Δ {metric} (workflow − no_workflow) vs K (mean over models)",
                xlabel="K (max communication rounds)",
                ylabel=f"Δ {metric}",
                out_path=plots_dir / "global" / f"{game}_delta_{metric}_wf_minus_nw.png",
            )

    print(f"[plots] Wrote plots to: {plots_dir}")


def main() -> None:
    # In plots-only mode we want to *reuse* the most recent run directory,
    # not create a brand new empty one.
    if os.getenv("LLMGT_PLOTS_ONLY", "0") == "1":
        runs_base = Path(__file__).resolve().parents[1] / "data" / "runs"
        candidates = sorted(runs_base.glob("*_THESIS_FULL"), key=lambda p: p.name)
        if not candidates:
            raise RuntimeError(f"No existing THESIS_FULL runs found under: {runs_base}")
        run_root = candidates[-1]

        build_all_plots(run_root)
        print("\nPLOTS-ONLY COMPLETE")
        print(run_root)
        return

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
            "backend": "openrouter",
            "temperature": TEMPERATURE,
            "max_output_tokens": MAX_NEW_TOKENS,
            "workflow_level": WORKFLOW_LEVEL,
            "agent_style": AGENT_STYLE,
        },
    )


    failed: list[str] = []
    for model_name, model_id in MODELS.items():
        for mode in MODES:
            for game_name, game in GAMES.items():
                try:
                    run_single_experiment(
                        run_root=run_root,
                        model_name=model_name,
                        model_id=model_id,
                        mode=mode,
                        game_name=game_name,
                        game=game,
                    )
                except Exception as exc:
                    tag = f"{model_name}/{mode}/{game_name}"
                    print(f"[ERROR] {tag} failed: {exc}")
                    failed.append(tag)

    if failed:
        print(f"\n[WARN] {len(failed)} experiment(s) failed: {failed}")

    build_all_plots(run_root)

    print("\nTHESIS PIPELINE COMPLETE")
    print(run_root)


if __name__ == "__main__":
    main()
