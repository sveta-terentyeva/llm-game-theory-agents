# LLM Game-Theoretic Simulations

A research-oriented simulator for **two-player classical games** with **LLM-based agents** (plus simple baselines).
This repository is meant to support diploma/thesis work on the question:

> **How does a limited communication budget `K` (number of pre-play dialogue rounds) affect agreement, equilibrium outcomes, and welfare in classical games?**

It supports two interaction modes:

- `no_workflow`: free-form negotiation / messages (baseline)
- `workflow`: structured “workflow prompting” that nudges agents through phases like analysis → propose → counter → accept

Ukrainian translation: see [`README_UA.md`](README_UA.md).

---

## Contents

- [Supported games](#supported-games)
- [Key concepts and metrics](#key-concepts-and-metrics)
- [Installation](#installation)
- [LLM backends (configuration)](#llm-backends-configuration)
- [Running experiments](#running-experiments)
  - [Communication sweep over K (recommended)](#communication-sweep-over-k-recommended)
  - [Scripts](#scripts)
- [Outputs / artifacts](#outputs--artifacts)
  - [Run directory layout](#run-directory-layout)
  - [Episode logs (JSONL)](#episode-logs-jsonl)
  - [Aggregated metrics (CSV)](#aggregated-metrics-csv)
  - [Plots](#plots)
- [Project structure](#project-structure)
- [Testing](#testing)
- [Reproducibility notes](#reproducibility-notes)
- [Troubleshooting](#troubleshooting)

---

## Supported games

Implemented in `src/llmgt/games/`:

- **Prisoner’s Dilemma** (`pd`)
- **Stag Hunt** (`stag`)
- **Battle of the Sexes** (`bos`)
- **Ultimatum Game** (`ultimatum`)

Each game provides:

- action sets: `actions_for(player)`
- payoff function: `payoff(action_a, action_b)`
- theoretical solution sets where applicable:
  - `nash_equilibria()`
  - `pareto_optima()`

---

## Key concepts and metrics

### Episode
One simulation run of a game between **agent A** and **agent B**:

1. optional communication phase (up to `K` rounds)
2. each agent picks an action (`ACTION: ...`)
3. payoffs are computed
4. theory / agreement metrics are computed and logged

### Communication rounds `K`
`K` is the **maximum** number of chat rounds allowed *before* choosing actions.
The simulator also records:

- `used_comm_rounds` — how many rounds were actually used
- in workflow mode, it may stop early when an `ACCEPT` is detected

### Agreement
“Agreement” is computed from the dialogue + final actions. It is logged as:

- `agreement_hit: bool | None`
- `rounds_to_agreement: int | None`

### Theory hits
For each final action pair, the simulator computes:

- `nash_hit`
- `pareto_hit`
- `pareto_nash_hit`
- `theory_hit` (overall success indicator)
- `rounds_to_theory_hit`

### Welfare
From payoffs:

- `welfare = payoff_a + payoff_b`
- summary also includes `payoff_mean` and `payoff_diff_mean`

---

## Installation

Requirements:

- Python **>= 3.10**

Quickstart (editable install):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional extras from `pyproject.toml`:

- HuggingFace / Transformers backend:

```bash
pip install -e ".[hf]"
```

- Ollama backend (HTTP client):

```bash
pip install -e ".[ollama]"
```

Plotting (optional):

```bash
pip install matplotlib
```

---

## LLM backends (configuration)

The CLI supports these backends:

- `heuristic` — deterministic baseline (no external APIs)
- `openai` — OpenAI Responses API
- `ollama` — local Ollama server via HTTP
- `hf` — local HuggingFace Transformers inference

Common knobs:

- `--temperature`
- `--max-output-tokens`

### OpenAI

- Choose a model with `--openai-model` (default: `gpt-4o-mini`)
- Optional `--base-url` for compatible gateways / proxies
- Authentication: typically via environment variables used by the OpenAI SDK (commonly `OPENAI_API_KEY`)

### Ollama

- `--ollama-model` (default: `llama3.1:8b`)
- `--ollama-host` (default: `http://localhost:11434`)
- `--ollama-timeout-s` (default: `120`)

### HuggingFace Transformers

- `--hf-model` (default: `mistralai/Mistral-7B-Instruct-v0.2`)
- `--hf-max-new-tokens` (default: `128`)

Note: large models may require a GPU and substantial RAM/VRAM.

---

## Running experiments

This project exposes a CLI entrypoint:

- `llmgt` → `src/llmgt/cli.py`

### Communication sweep over K (recommended)

This is the main experiment driver for the diploma-style evaluation.
It runs `n_runs` episodes for each `K` and writes:

- `logs/episodes.jsonl` (raw episode logs)
- `summary_by_k.csv` (aggregated metrics)
- `figures/*.png` (optional plots)

Heuristic smoke test (fast):

```bash
llmgt sweep --game pd --mode workflow --backend heuristic --k 0..6 --n-runs 50 --plots
```

Ollama example:

```bash
llmgt sweep --game stag --mode workflow --backend ollama \
  --ollama-model llama3.1:8b --ollama-host http://localhost:11434 \
  --k 0..6 --n-runs 200 --plots
```

OpenAI example:

```bash
llmgt sweep --game bos --mode no_workflow --backend openai \
  --openai-model gpt-4o-mini --temperature 0.7 --max-output-tokens 64 \
  --k 0..6 --n-runs 200 --plots
```

HuggingFace example:

```bash
llmgt sweep --game ultimatum --mode workflow --backend hf \
  --hf-model mistralai/Mistral-7B-Instruct-v0.2 --hf-max-new-tokens 128 \
  --k 0..6 --n-runs 50 --plots
```

#### CLI parameters (`llmgt sweep`)

Core:

- `--game`: `pd | stag | bos | ultimatum`
- `--mode`: `workflow | no_workflow`
- `--k`: either a range `0..6` or explicit list `--k 0 1 2 3`
- `--n-runs`: episodes per K (default: `200`)
- `--out-dir`: output base directory (default: `data/runs`)
- `--tag`: a label appended to the run directory name
- `--plots`: write PNG plots (requires `matplotlib`)

Agent behavior:

- `--agent-style`: `basic | strategic`
- `--workflow-level`: `1 | 2 | 3` (workflow mode only)

Backend-specific:

- HF: `--hf-model`, `--hf-max-new-tokens`
- Ollama: `--ollama-model`, `--ollama-host`, `--ollama-timeout-s`
- OpenAI: `--openai-model`, `--base-url`

### Scripts

The `scripts/` directory contains convenience runners used for quick experiments and thesis runs, e.g.:

- `scripts/run_comm_experiment.py`
- `scripts/run_llm_sweep_all.py`
- `scripts/run_workflow_sweep_all.py`
- `scripts/run_pd.py`, `scripts/run_pd_workflow.py`, `scripts/run_pd_llm.py`
- `scripts/run_thesis.py`

Typical usage:

```bash
python scripts/run_comm_experiment.py
```

---

## Outputs / artifacts

### Run directory layout

Each run creates a new directory under `data/runs/` (or `--out-dir`).
The directory name is timestamped like:

- `YYYYMMDD_HHMMSSZ_<salt>_<tag>`

Standard subfolders:

- `logs/`
- `figures/`

### Episode logs (JSONL)

Path:

- `.../logs/episodes.jsonl`

Each line is a serialized `EpisodeRecord` (see `src/llmgt/logging/records.py`).
Important fields:

- identifiers: `episode_id`, `game`, `mode`
- communication: `max_comm_rounds`, `used_comm_rounds`
- models: `model_a`, `model_b`
- conversation: `messages[]` with `{role, content, ts_utc}`
- outcomes: `action_a`, `action_b`, `payoff_a`, `payoff_b`, `winner`
- theory hits: `nash_hit`, `pareto_hit`, `pareto_nash_hit`, `theory_hit`, `rounds_to_theory_hit`
- agreement: `agreement_hit`, `rounds_to_agreement`
- misc: `extra` (e.g., workflow can store an `accepted_pair`)
- timing: `started_at_utc`, `finished_at_utc`

### Aggregated metrics (CSV)

Path:

- `.../summary_by_k.csv`

Produced by `summarize_by_k()` (`src/llmgt/experiments/sweep.py`).
Columns include (not exhaustive):

- `K`, `n_runs`, `game`
- rates: `agreement_rate`, `nash_rate`, `pareto_rate`, `pareto_nash_rate`, `theory_rate`
- rounds: `mean_rounds_to_agreement`, `mean_rounds_to_theory_hit`
- communication usage: `used_comm_rounds_mean`, `used_comm_rounds_p50`, `used_comm_rounds_over_k_mean`, `wasted_comm_rounds_mean`
- payoffs: `payoff_mean`, `welfare_mean`, `payoff_diff_mean`
- winners: `a_win_rate`, `b_win_rate`, `tie_rate`
- text stats: `msg_count_mean`, `words_total_mean`, `words_a_mean`, `words_b_mean`
- intent markers: `propose_rate`, `counter_rate`, `accept_rate`, `follow_accept_rate`
- optional (if summarizer is given a game): `regret_a_mean`, `regret_b_mean`, `welfare_gap_mean`

### Plots

If `--plots` is enabled, the CLI writes:

- `figures/agreement_rate.png`
- `figures/mean_rounds_to_agreement.png`
- `figures/welfare_mean.png`
- `figures/theory_rate.png`
- `figures/mean_rounds_to_theory_hit.png`

Plotting is implemented in `src/llmgt/experiments/plotting.py`. If `matplotlib` is missing, plotting is skipped gracefully.

---

## Project structure

High-level map:

- `src/llmgt/cli.py` — CLI (`llmgt sweep`)
- `src/llmgt/games/` — game definitions and theoretical sets
- `src/llmgt/agents/` — agents (`LLMAgent`, strategic variants, workflow variants)
- `src/llmgt/llm/` — backend clients:
  - `heuristic.py` (baseline)
  - `openai_client.py`
  - `ollama_client.py`
  - `hf_client.py`
- `src/llmgt/sim/` — episode runner + agreement/theory logic + run directory
- `src/llmgt/experiments/` — sweeps, aggregation, plots, agent factories
- `src/llmgt/logging/` — JSONL logger + Pydantic records
- `scripts/` — convenience runners
- `tests/` — test suite (pytest)

---

## Testing

Run all tests:

```bash
pytest -q
```

---

## Reproducibility notes

For stable diploma results:

- Keep `K`, `n_runs`, `mode`, and backend configs fixed when comparing conditions.
- Record model identifiers and sampling parameters: `--temperature`, `--max-output-tokens`.
- Store raw episode logs (`episodes.jsonl`) — they contain the full dialogue + final actions.
- Prefer running multiple independent runs and reporting confidence intervals (LLM sampling variance).

---

## Troubleshooting

- **No plots / “Plotting skipped”** → install `matplotlib`.
- **Ollama connection/timeouts** → ensure the server is running and `--ollama-host` is correct.
- **HF out-of-memory** → use a smaller model, reduce `--hf-max-new-tokens`, or run on CPU.
- **OpenAI auth errors** → ensure the required API key env var is set (commonly `OPENAI_API_KEY`).
