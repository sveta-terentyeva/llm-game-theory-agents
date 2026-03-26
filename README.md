# LLM Game-Theory Agents

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-64%20passed-brightgreen.svg)](#testing)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A research-oriented simulator for **two-player classical games** with **LLM-based agents** (plus simple baselines).  
Built as the codebase for a diploma/thesis investigating:

> **How does a limited communication budget *K* (number of pre-play dialogue rounds) affect agreement, equilibrium outcomes, and welfare in classical games — and does structured "workflow prompting" improve these outcomes?**

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
- [Prompt Caching (Cost Optimization)](#prompt-caching-cost-optimization)
- [Output Artifacts](#output-artifacts)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Reproducibility](#reproducibility)
- [Troubleshooting](#troubleshooting)

---

## Motivation & Research Question

Classical game theory assumes perfectly rational agents. Modern LLMs can be cast as
bounded-rational players that *communicate* before committing to actions. This project
studies how:

1. **Communication budget `K`** (0 = no talk, 1..9 = rounds of dialogue before action) affects whether agents reach agreements, Nash equilibria, Pareto optima, and higher joint welfare.
2. **Workflow prompting** (structured reasoning: dominant strategies → best responses → Nash → Pareto → decision) compares against free-form negotiation.
3. Different **LLM models** (Claude, GPT-4o, Llama via HuggingFace or local) behave under the same protocol.

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

---

## Agent Types

### Baseline Agents

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
| `WorkflowStrategicLLMAgent` | Full game-theoretic workflow reasoning. Configurable depth via `workflow_level`. |

---

## Key Concepts & Metrics

### Episode
One simulation run: optional communication (up to *K* rounds) → actions → payoffs → metrics.

### Communication Budget `K`
Maximum chat rounds before choosing actions. The sweep experiment varies *K* from 0 to 9 to study how more communication affects outcomes.

### Key Metrics

| Metric | Description |
|--------|-------------|
| `agreement_rate` | Fraction of episodes where agents agreed on actions |
| `theory_rate` | Match rate for theoretical solution concepts (Nash/Pareto) |
| `welfare_mean` | Average joint payoff (agent A + agent B) |
| `mean_rounds_to_agreement` | Average round where agreement was reached |
| `payoff_diff_mean` | Average absolute payoff difference (fairness metric) |

For full metric descriptions, see `src/llmgt/metrics/`.

---

## Installation

**Requirements:** Python ≥ 3.10

```bash
# Clone and install
git clone https://github.com/sveta-terentyeva/llm-game-theory-agents.git
cd llm-game-theory-agents
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Optional: HuggingFace backend (GPU recommended for faster inference)
pip install -e ".[hf]"

# Optional: Ollama backend (for local models)
pip install -e ".[ollama]"
```

---

## LLM Backends

| Backend | Use Case | Requirements |
|---------|----------|--------------|
| `heuristic` | Fast smoke tests, deterministic baseline | None |
| `openrouter` | **Recommended for thesis** — Access Claude, GPT-4o, Llama via OpenRouter API | OpenRouter API key |
| `hf` | Direct HuggingFace Transformers inference (local or HF hosted) | `torch`, `transformers` (GPU recommended) |
| `ollama` | Local open-source models (Llama, Mistral) | Local Ollama server |
| `openai` | Direct OpenAI API access | OpenAI API key |

---

## Running Experiments

### Main Experiment: Full Thesis Pipeline

```bash
export LLMGT_N_RUNS=100
export LLMGT_K_VALUES=0,1,2,3,4,5,6,7,8,9
export LLMGT_TEMPERATURE=0.7
export LLMGT_MAX_NEW_TOKENS=64

# With OpenRouter backend (recommended)
export OPENROUTER_API_KEY="your-api-key"
python scripts/run_thesis.py
```

### With Prompt Caching (OpenRouter + Claude)

Prompt caching dramatically reduces costs for repeated prompts. Configuration:

```bash
# Enable prompt caching for Claude models
export LLMGT_OPENROUTER_PROMPT_CACHING=1
export LLMGT_OPENROUTER_PROMPT_CACHE_TTL=1h
export LLMGT_OPENROUTER_ANTHROPIC_ONLY=1

# Optional: control which parts of the prompt are cached
export LLMGT_OPENROUTER_CACHE_SYSTEM=1         # cache system message (always enabled)
export LLMGT_OPENROUTER_CACHE_FIRST_USER=0     # don't cache first user message
export LLMGT_OPENROUTER_PROMPT_CACHING_MODE=explicit

export LLMGT_N_RUNS=100
export LLMGT_K_VALUES=0,1,2,3,4,5,6,7,8,9
python scripts/run_thesis.py
```

### With HuggingFace Backend (Local or Free Models)

```bash
export LLMGT_BACKEND=hf
export LLMGT_HF_MODEL=meta-llama/Llama-3.3-70B-Instruct
export LLMGT_HF_MAX_NEW_TOKENS=64
export LLMGT_N_RUNS=100
python scripts/run_thesis.py
```

---

## Prompt Caching (Cost Optimization)

### Overview

When using Claude models via OpenRouter, **prompt caching** can reduce costs by up to 90%:
- First request: 1.25x cost (5-min TTL) or 2x cost (1-hour TTL) for cached tokens
- Subsequent requests: **0.1x cost** for cached tokens (10% of original price)

**Example savings**: 100 episodes × 3,000-token system prompt
- Without caching: 100 × 3,000 tokens = $0.12
- With caching: 1 write (3,750 tokens) + 99 reads (3,000 tokens each) = $0.034 **72% savings**

### Configuration

For maximum stability and cost savings with Claude 3.5 Haiku:

**`.env` file:**
```bash
OPENROUTER_API_KEY=sk-or-...

# Prompt caching settings
LLMGT_OPENROUTER_PROMPT_CACHING=1
LLMGT_OPENROUTER_PROMPT_CACHE_TTL=1h
LLMGT_OPENROUTER_ANTHROPIC_ONLY=1
LLMGT_OPENROUTER_PROMPT_CACHING_MODE=explicit
```

### How It Works

1. **Automatic system preamble**: A ~2,200 token cached preamble is prepended to system prompts for all Claude models (optional, improves cache hits)
2. **Cache breakpoint**: The system message is wrapped with `cache_control: { type: "ephemeral", ttl: "1h" }`
3. **Provider routing**: With `anthropic_only=1`, OpenRouter routes *only* to Anthropic (not Bedrock/Vertex) to avoid TTL incompatibility
4. **Sticky routing**: Within the same conversation/K-value, OpenRouter keeps you routed to the same provider to maximize cache hits

### Bedrock Compatibility Note

Amazon Bedrock (an alternate Claude provider on OpenRouter) does not support TTL fields in per-block cache control. If you don't set `anthropic_only=1`:
- Cache still works, but with default 5-minute TTL (free cache_control)
- You may see brief 400 errors on first write; client auto-retries without TTL

**Recommendation**: Use `anthropic_only=1` for 1-hour TTL + maximum stability.

---

## Output Artifacts

### Run Directory Structure

```
data/runs/20260310_142010Z_abc1234_THESIS_FULL/
├── run_meta.json                    # Configuration snapshot
├── raw/
│   └── claude-3.5-haiku/
│       ├── no_workflow/
│       │   └── prisoners_dilemma/
│       │       ├── summary.csv      # Aggregated metrics by K
│       │       └── logs/
│       │           └── episodes_*.jsonl
│       └── workflow/
│           └── ...
└── plots/
    ├── global/
    │   └── *.png                    # Overall comparison plots
    ├── mode_comparison/
    │   └── *.png                    # Workflow vs no_workflow
    └── model_comparison/
        └── *.png                    # Model-to-model comparisons
```

### Summary CSV Columns

Each `summary.csv` contains per-K aggregated metrics:

```
k,agreement_rate,nash_rate,pareto_rate,theory_rate,welfare_mean,
mean_rounds_to_agreement,payoff_diff_mean,accept_rate,counter_rate,...
0,0.15,0.22,0.20,0.22,4.1,NaN,1.8,NaN,NaN,...
1,0.35,0.40,0.38,0.40,5.2,1.2,1.5,0.28,0.05,...
...
```

All numeric metrics include `*_std` columns for standard deviations.

---

## Project Structure

```
src/llmgt/
├── games/                         # Game definitions (PD, Stag Hunt, BoS, Ultimatum)
├── agents/                        # Agent implementations
│   ├── parsing.py                 # Shared parsing utilities
│   ├── llm.py                     # Basic LLM agent
│   ├── strategic.py               # Strategic LLM agent (payoff-aware)
│   ├── workflow_reasoner.py       # Workflow + game-theory reasoning
│   └── simple.py                  # Baseline agents
├── llm/                           # LLM backend clients
│   ├── openrouter_client.py       # OpenRouter API client (supports prompt caching)
│   ├── hf_client.py               # HuggingFace Transformers
│   ├── openai_client.py           # OpenAI API
│   ├── ollama_client.py           # Ollama HTTP API
│   └── heuristic.py               # Deterministic baseline
├── sim/                           # Simulation engine
│   ├── runner.py                  # Episode runner & experiment driver
│   ├── agreement.py               # Agreement detection
│   ├── theory.py                  # Nash/Pareto hit computation
│   ├── rounds.py                  # Rounds-to-agreement/theory calculation
│   ├── workflow.py                # PROPOSE/COUNTER/ACCEPT extraction
│   └── run_dir.py                 # Timestamped output directories
├── experiments/                   # Experiment orchestration
│   ├── sweep.py                   # Communication sweep + aggregation
│   ├── agent_factories.py         # LLM client/agent factories
│   └── game_configs.py            # Per-game baseline configurations
├── logging/                       # Data persistence
│   ├── records.py                 # Pydantic models (EpisodeRecord, etc.)
│   ├── jsonl_logger.py            # Append-mode JSONL writer
│   └── run_meta.py                # Configuration snapshot
└── metrics/                       # Metric computation
    └── __init__.py                # CommStats, regret, welfare metrics

scripts/
├── run_thesis.py                  # Main thesis pipeline (all models × games × modes)
└── check_system_prompt_tokens.py  # Verify cached preamble size

tests/                             # pytest suite (64 tests)
```

---

## Testing

```bash
# Run all 64 tests
pytest -v

# Run specific test file
pytest tests/test_openrouter_prompt_caching.py -v

# Run with coverage
pytest --cov=llmgt --cov-report=term-missing
```

Test categories:
- **Game logic**: Payoff matrices, Nash equilibria, Pareto optima
- **Agents**: Message handling, action parsing, workflow protocol
- **Agreement detection**: PROPOSE/ACCEPT matching, fallback logic
- **Parsing**: Protocol lines, action tokens, pair sanitization
- **Sweep**: Communication sweeps, per-K aggregation
- **Prompt caching**: Cache control injection, TTL handling, provider routing

---

## Reproducibility

For stable thesis results:

1. **Fix all parameters**: `K` values, `n_runs`, backend, `temperature`, `max_output_tokens`
2. **Record run metadata**: `run_meta.json` stores all configuration automatically
3. **Preserve raw logs**: `episodes.jsonl` contains full dialogues + final outcomes
4. **Use confidence intervals**: Report `*_std` columns from `summary.csv` for error bars
5. **Multiple seeds**: LLM sampling has inherent variance; run multiple independent sweeps

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "OpenRouter API key not provided" | Set `OPENROUTER_API_KEY` environment variable |
| Prompt caching 400 errors | Ensure `anthropic_only=1` if using 1h TTL; client auto-retries without TTL |
| HuggingFace out-of-memory | Use smaller model or reduce `max_new_tokens`; enable GPU if available |
| Ollama connection timeout | Ensure Ollama server is running: `ollama serve` |
| No plots generated | Install matplotlib: `pip install matplotlib` |
| `theory_rate` always 0 or 1 | Expected for deterministic backends; LLMs show variance |

---

**For questions or issues**: See `tests/` for usage examples, or consult the inline docstrings in `src/llmgt/`.

*Built for a diploma/thesis on LLM-based game-theoretic negotiation.*

