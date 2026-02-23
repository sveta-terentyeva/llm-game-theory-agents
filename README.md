# LLM Game-Theory Agents

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-51%20passed-brightgreen.svg)](#testing)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A research-oriented simulator for **two-player classical games** with **LLM-based agents** (plus simple baselines).  
Built as the codebase for a diploma/thesis investigating:

> **How does a limited communication budget *K* (number of pre-play dialogue rounds) affect agreement, equilibrium outcomes, and welfare in classical games — and does structured "workflow prompting" improve these outcomes?**

Ukrainian translation: [`README_UA.md`](README_UA.md)

---

## Table of Contents

- [Motivation & Research Question](#motivation--research-question)
- [Architecture Overview](#architecture-overview)
- [Supported Games](#supported-games)
- [Agent Types](#agent-types)
- [Key Concepts & Metrics](#key-concepts--metrics)
- [Installation](#installation)
- [LLM Backends](#llm-backends)
- [Running Experiments](#running-experiments)
- [Output Artifacts](#output-artifacts)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Known Issues & Design Decisions](#known-issues--design-decisions)
- [Reproducibility](#reproducibility)
- [Troubleshooting](#troubleshooting)

---

## Motivation & Research Question

Classical game theory assumes perfectly rational agents.  Modern LLMs can be cast as
bounded-rational players that *communicate* before committing to actions.  This project
studies how:

1. **Communication budget `K`** (0 = no talk, 1..6 = rounds of dialogue before action) affects whether agents reach agreements, Nash equilibria, Pareto optima, and higher joint welfare.
2. **Workflow prompting** (structured reasoning: dominant strategies → best responses → Nash → Pareto → decision) compares against free-form negotiation.
3. Different **LLM backends** (heuristic baseline, Ollama local models, HuggingFace Transformers, OpenAI API) behave under the same protocol.

The simulator runs hundreds of episodes per configuration, logs full dialogues, and computes
publication-ready aggregate metrics with standard deviations.

---

## Architecture Overview

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Game        │     │  Agent A     │     │  Agent B         │
│  (payoffs,   │◄────│  (send_msg,  │────►│  (send_msg,      │
│   Nash,      │     │   act)       │     │   act)           │
│   Pareto)    │     └──────┬───────┘     └──────┬───────────┘
└──────┬───────┘            │                    │
       │           ┌───────▼────────────────────▼──────────┐
       │           │         Simulation Runner              │
       │           │  • communication loop (K rounds)       │
       └──────────►│  • action collection                   │
                   │  • payoff computation                  │
                   │  • theory-hit / agreement detection    │
                   └───────────────┬────────────────────────┘
                                   │
                   ┌───────────────▼────────────────────────┐
                   │         Logging & Metrics               │
                   │  • JSONL episode records                │
                   │  • per-K aggregation (CSV)             │
                   │  • plots (PNG)                         │
                   └────────────────────────────────────────┘
```

---

## Supported Games

Implemented in `src/llmgt/games/`:

| Game | Actions | Nash Equilibria | Pareto Optima | Key Tension |
|------|---------|-----------------|---------------|-------------|
| **Prisoner's Dilemma** (`pd`) | C, D | (D,D) | (C,C) | Individual vs. collective rationality |
| **Stag Hunt** (`stag`) | S, H | (S,S), (H,H) | (S,S) | Risk dominance vs. payoff dominance |
| **Battle of the Sexes** (`bos`) | O, F | (O,O), (F,F) | (O,O), (F,F) | Coordination with conflicting preferences |
| **Ultimatum Game** (`ultimatum`) | Proposer: L,F / Responder: A,R | (L,A) | (F,A) | Fairness vs. subgame-perfect equilibrium |

Each game class provides:
- `actions_for(role)` — role-specific action sets (important for asymmetric games like Ultimatum)
- `payoff(action_a, action_b)` → `(float, float)`
- `nash_equilibria()` → `set[tuple[str, str]]`
- `pareto_optima()` → `set[tuple[str, str]]`

---

## Agent Types

### Baseline Agents (no LLM)

| Agent | Description |
|-------|-------------|
| `FixedActionAgent` | Always plays a fixed action. No communication. |
| `EchoAgent` | Echoes last received message. Fixed final action. |
| `WorkflowProposerAgent` | Deterministic PROPOSE/ACCEPT protocol agent. |
| `WorkflowResponderAgent` | Accepts or counters based on payoff threshold. |
| `StochasticWorkflowResponderAgent` | Acceptance probability increases with round number. |

### LLM Agents

| Agent | Description |
|-------|-------------|
| `LLMAgent` | Free-form negotiation with minimal game-theory guidance. |
| `StrategicLLMAgent` | Payoff-table–aware prompts with structured protocol (PROPOSE/COUNTER/ACCEPT). |
| `WorkflowStrategicLLMAgent` | Full game-theoretic workflow reasoning (dominant strategies → best responses → Nash → Pareto → decision). Configurable depth via `workflow_level` (1–3). |

### Negotiation Protocol

All communicative agents follow the **PROPOSE / COUNTER / ACCEPT** protocol:

```
Round 1: Agent A → PROPOSE: (X,Y)     # X = A's action, Y = B's action
         Agent B → ACCEPT: (X,Y)       # or COUNTER: (X',Y')
Round 2: Agent A → PROPOSE: (X',Y')   # re-proposes
         Agent B → ACCEPT: (X',Y')     # final agreement
```

When `ACCEPT` is detected in workflow mode, communication stops early.

---

## Key Concepts & Metrics

### Episode
One simulation run: optional communication (up to *K* rounds) → actions → payoffs → metrics.

### Communication Budget `K`
Maximum chat rounds before choosing actions.  The sweep experiment varies *K* from 0 to 6 to study how more communication affects outcomes.

### Metrics (per episode)

| Metric | Description |
|--------|-------------|
| `agreement_hit` | Did agents' final actions match what they agreed on during communication? |
| `rounds_to_agreement` | First round where agreement appeared |
| `nash_hit` | Did the outcome match a Nash equilibrium? |
| `pareto_hit` | Did the outcome match a Pareto optimum? |
| `theory_hit` | Combined success: Pareto-Nash if it exists, else Nash |
| `rounds_to_theory_hit` | First round where the theory-target outcome was agreed upon |
| `welfare` | Sum of payoffs: `payoff_a + payoff_b` |
| `winner` | Which agent got a higher payoff (or tie) |

### Aggregated Metrics (per K)

The `summarize_by_k()` function computes means, standard deviations, and rates across all episodes for each *K* value.  Key columns in `summary_by_k.csv`:

- **Rates**: `agreement_rate`, `nash_rate`, `pareto_rate`, `theory_rate`
- **Rounds**: `mean_rounds_to_agreement`, `mean_rounds_to_theory_hit`  
- **Payoffs**: `welfare_mean`, `payoff_mean`, `payoff_diff_mean`
- **Communication**: `used_comm_rounds_mean`, `wasted_comm_rounds_mean`
- **Protocol**: `propose_rate`, `counter_rate`, `accept_rate`, `follow_accept_rate`
- **Game-theoretic**: `regret_a_mean`, `regret_b_mean`, `welfare_gap_mean`
- All numeric metrics include `*_std` (standard deviation) columns for error bars.

---

## Installation

**Requirements:** Python ≥ 3.10

```bash
# Clone and install
git clone <git@github.com:sveta-terentyeva/llm-game-theory-agents.git>
cd llm-game-theory-agents
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Optional: plotting
pip install matplotlib

# Optional: HuggingFace backend (GPU recommended)
pip install -e ".[hf]"

# Optional: Ollama backend
pip install -e ".[ollama]"
```

---

## LLM Backends

| Backend | Flag | Requires | Use Case |
|---------|------|----------|----------|
| `heuristic` | `--backend heuristic` | Nothing | Fast smoke tests, deterministic baseline |
| `ollama` | `--backend ollama` | Local Ollama server | Local open-source models (Llama, Mistral) |
| `hf` | `--backend hf` | `torch`, `transformers` | Direct HuggingFace inference (GPU recommended) |
| `openai` | `--backend openai` | `openai` package + API key | OpenAI API models |

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--temperature` | 0.7 | Sampling temperature |
| `--max-output-tokens` | 64 | Max tokens per LLM response |
| `--agent-style` | `strategic` | `basic` or `strategic` |
| `--workflow-level` | 2 | Workflow depth: 1=light, 2=standard, 3=strict |

Backend-specific options: see `llmgt sweep --help`.

---

## Running Experiments

### CLI: Communication Sweep (main experiment)

```bash
# Fast smoke test (heuristic, ~1 second)
llmgt sweep --game pd --mode workflow --backend heuristic \
  --k 0..6 --n-runs 50 --plots

# Ollama with local Llama 3.1
llmgt sweep --game stag --mode workflow --backend ollama \
  --ollama-model llama3.1:8b --k 0..6 --n-runs 200 --plots

# HuggingFace Transformers
llmgt sweep --game bos --mode workflow --backend hf \
  --hf-model mistralai/Mistral-7B-Instruct-v0.2 --k 0..6 --n-runs 50 --plots

# OpenAI API
llmgt sweep --game ultimatum --mode no_workflow --backend openai \
  --openai-model gpt-4o-mini --k 0..6 --n-runs 200 --plots
```

### Scripts

The `scripts/` directory contains pre-configured experiment runners:

| Script | Description |
|--------|-------------|
| `run_pd.py` | Fixed-action PD baseline |
| `run_pd_llm.py` | LLM agents on PD (heuristic backend) |
| `run_pd_workflow.py` | Stochastic workflow agents, PD sweep |
| `run_comm_experiment.py` | Fixed-agent communication sweep |
| `run_workflow_sweep_all.py` | Rule-based workflow sweep across all 4 games |
| `run_llm_sweep_all.py` | LLM sweep across all 4 games (configurable backend via env vars) |
| `run_thesis.py` | Full thesis pipeline: all models × all games × all modes → plots |

```bash
# Example: full thesis run with TinyLlama
LLMGT_N_RUNS=50 python scripts/run_thesis.py
```

---

## Output Artifacts

### Run Directory Layout

```
data/runs/20260223_082328Z_9f29a20_workflow_sweep_all/
├── run_meta.json              # Configuration snapshot
├── prisoners_dilemma/
│   ├── logs/
│   │   └── episodes.jsonl     # Raw episode records
│   ├── summary_by_k.csv       # Aggregated metrics
│   └── figures/
│       ├── agreement_rate.png
│       ├── mean_rounds_to_agreement.png
│       ├── welfare_mean.png
│       ├── theory_rate.png
│       └── mean_rounds_to_theory_hit.png
├── stag_hunt/
│   └── ...
├── battle_of_sexes/
│   └── ...
└── ultimatum/
    └── ...
```

### Episode Logs (JSONL)

Each line is a serialized `EpisodeRecord`:

```json
{
  "episode_id": "pd-K3-run42",
  "game": "prisoners_dilemma",
  "mode": "workflow",
  "max_comm_rounds": 3,
  "used_comm_rounds": 2,
  "messages": [
    {"role": "system", "content": "Episode started..."},
    {"role": "agent_a", "content": "PROPOSE: (C,C)"},
    {"role": "agent_b", "content": "ACCEPT: (C,C)"},
    {"role": "agent_a", "content": "ACTION: C"},
    {"role": "agent_b", "content": "ACTION: C"}
  ],
  "action_a": "C", "action_b": "C",
  "payoff_a": 3.0, "payoff_b": 3.0,
  "nash_hit": false, "pareto_hit": true,
  "theory_hit": false,
  "agreement_hit": true,
  "rounds_to_agreement": 1
}
```

### Plots

Generated with `--plots` flag.  Plots show metric values vs. *K* with optional error bars (from `*_std` columns):

- `agreement_rate.png` — fraction of episodes where agents agreed
- `welfare_mean.png` — mean joint payoff
- `theory_rate.png` — fraction matching theoretical solution concept
- `mean_rounds_to_agreement.png` — how quickly agents agree
- `mean_rounds_to_theory_hit.png` — rounds until theory-target agreement

---

## Project Structure

```
src/llmgt/
├── __init__.py                    # Package root
├── cli.py                         # CLI entrypoint (llmgt sweep)
│
├── games/                         # Game definitions
│   ├── base.py                    # Abstract Game base class
│   ├── prisoners_dilemma.py       # Prisoner's Dilemma
│   ├── stag_hunt.py               # Stag Hunt
│   ├── battle_of_sexes.py         # Battle of the Sexes
│   └── ultimatum.py               # Ultimatum Game
│
├── agents/                        # Agent implementations
│   ├── parsing.py                 # ★ Shared parsing utilities (refactored)
│   ├── simple.py                  # FixedActionAgent, EchoAgent
│   ├── workflow.py                # Rule-based workflow agents
│   ├── llm.py                     # Basic LLM agent
│   ├── strategic.py               # Strategic LLM agent (payoff-aware)
│   └── workflow_reasoner.py       # Workflow + game-theory reasoning agent
│
├── llm/                           # LLM backend clients
│   ├── client.py                  # LLMClient protocol + ScriptedLLMClient
│   ├── heuristic.py               # Deterministic heuristic baseline
│   ├── openai_client.py           # OpenAI Responses API
│   ├── ollama_client.py           # Ollama HTTP API
│   └── hf_client.py               # HuggingFace Transformers
│
├── sim/                           # Simulation engine
│   ├── runner.py                  # Episode runner & experiment driver
│   ├── agreement.py               # Agreement detection
│   ├── theory.py                  # Nash/Pareto hit computation
│   ├── rounds.py                  # ★ Rounds-to-agreement/theory (bug fixed)
│   ├── workflow.py                # PROPOSE/COUNTER/ACCEPT extraction
│   ├── run_dir.py                 # Timestamped output directories
│   └── utils.py                   # Episode ID generation
│
├── experiments/                   # Experiment orchestration
│   ├── sweep.py                   # Communication sweep + aggregation
│   ├── plotting.py                # ★ Metric plots with error bars
│   ├── agent_factories.py         # ★ LLM client/agent factories (refactored)
│   └── game_configs.py            # Per-game workflow baselines
│
├── logging/                       # Data persistence
│   ├── records.py                 # Pydantic models (EpisodeRecord, etc.)
│   ├── jsonl_logger.py            # Append-mode JSONL writer
│   └── run_meta.py                # Configuration snapshot writer
│
└── metrics/                       # Per-episode & aggregate metrics
    └── __init__.py                # CommStats, regret, welfare gap

scripts/                           # Convenience experiment runners
tests/                             # pytest test suite (51 tests)
```

★ = refactored or bug-fixed in this revision.

---

## Testing

```bash
# Run all 51 tests
pytest -v

# Run a specific test file
pytest tests/test_parsing.py -v

# Run with coverage (if pytest-cov installed)
pytest --cov=llmgt --cov-report=term-missing
```

Test categories:
- **Game logic**: payoffs, Nash/Pareto sets, role-specific actions
- **Agreement detection**: workflow ACCEPT matching, no-workflow fallback
- **Theory hits**: Nash/Pareto hit computation, rounds-to-theory-hit
- **Parsing**: protocol line extraction, action parsing, pair sanitisation
- **Agents**: LLM agent act/send_message, workflow agents, stochastic responder
- **Sweep**: communication sweep, summarize_by_k, CSV output
- **Plotting**: legacy/new column names, PNG generation
- **Logging**: Pydantic serialisation, JSONL output

---

## Known Issues & Design Decisions

### Fixed in This Revision

1. **`compute_rounds_to_theory_hit` bug** — Previously always returned round 1 for any theory hit because `upto is not None` is always True for a non-empty list slice. Now properly scans conversation prefixes for agreement markers.

2. **Code duplication** — `_format_history`, `_extract_accepted_pair`, `_parse_action`, regex patterns were duplicated across `llm.py`, `strategic.py`, and `workflow_reasoner.py`. Extracted to `agents/parsing.py`.

3. **Missing error bars** — `plot_metric_by_k` now automatically uses `*_std` columns for error bars when available.

4. **Factory duplication** — `_make_clients` in `agent_factories.py` had identical client-construction blocks repeated. Refactored to a single `_make_client()` function.

### Design Notes

- **Heuristic backend**: always proposes cooperative/Pareto outcomes — this is intentional as a deterministic "best-case" baseline.
- **Theory target set**: if a Pareto-optimal Nash equilibrium exists, that is the target; otherwise the Nash set is used. This follows the paper's methodology.
- **Stochastic workflow responder**: acceptance probability `p(r) = min(1, base_p + (r-1) × step_p)` — models escalating willingness to agree.

---

## Reproducibility

For stable thesis results:

1. **Fix all parameters** when comparing conditions: `K`, `n_runs`, `mode`, backend, `temperature`, `max_output_tokens`.
2. **Record model identifiers** in `run_meta.json` (done automatically).
3. **Store raw logs** (`episodes.jsonl`) — they contain full dialogues + final actions.
4. **Run multiple independent seeds** and report confidence intervals (LLM sampling has inherent variance).
5. **Use `summary_by_k.csv`** `*_std` columns for error bars in plots.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| No plots / "Plotting skipped" | `pip install matplotlib` |
| Ollama connection timeout | Ensure server is running: `ollama serve`; check `--ollama-host` |
| HuggingFace out-of-memory | Use smaller model, reduce `--hf-max-new-tokens`, or use CPU |
| OpenAI auth error | Set `OPENAI_API_KEY` environment variable |
| Flat/constant plots | Check that agents actually communicate (use `--backend heuristic` to verify); increase `--n-runs` for statistical significance |
| `theory_rate` always 0 or 1 | Expected for deterministic backends; LLM backends show variance |

---

*Built for a diploma/thesis on LLM-based game-theoretic negotiation.*
