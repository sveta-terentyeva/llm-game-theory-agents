# LLM Game-Theory Agents

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-51%20passed-brightgreen.svg)](#тестування)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Дослідницький симулятор для **класичних двогравцевих ігор** з **агентами на базі LLM** (та простими baseline-агентами).  
Створено як кодову базу для дипломної/кваліфікаційної роботи з дослідження:

> **Як обмежений бюджет комунікації *K* (кількість раундів діалогу до ходу) впливає на домовленість, рівноважні результати та добробут у класичних іграх — і чи покращує структуроване «workflow prompting» ці результати?**

Англійська версія: [`README.md`](README.md)

---

## Зміст

- [Мотивація та дослідницьке питання](#мотивація-та-дослідницьке-питання)
- [Огляд архітектури](#огляд-архітектури)
- [Підтримувані ігри](#підтримувані-ігри)
- [Типи агентів](#типи-агентів)
- [Ключові поняття та метрики](#ключові-поняття-та-метрики)
- [Встановлення](#встановлення)
- [LLM бекенди](#llm-бекенди)
- [Запуск експериментів](#запуск-експериментів)
- [Вихідні артефакти](#вихідні-артефакти)
- [Структура проєкту](#структура-проєкту)
- [Тестування](#тестування)
- [Відомі проблеми та проєктні рішення](#відомі-проблеми-та-проєктні-рішення)
- [Відтворюваність](#відтворюваність)
- [Усунення несправностей](#усунення-несправностей)

---

## Мотивація та дослідницьке питання

Класична теорія ігор передбачає абсолютно раціональних гравців. Сучасні LLM можна розглядати
як обмежено-раціональних гравців, що *комунікують* перед вибором дій. Цей проєкт
досліджує:

1. **Бюджет комунікації `K`** (0 = без діалогу, 1..6 = раунди діалогу до дії) — як він впливає на досягнення домовленостей, рівноваг Неша, оптимумів Парето та сумарного добробуту.
2. **Workflow prompting** (структуроване міркування: домінантні стратегії → найкращі відповіді → Неш → Парето → рішення) у порівнянні з вільною комунікацією.
3. Різні **LLM бекенди** (евристичний baseline, локальні моделі Ollama, HuggingFace Transformers, OpenAI API) під одним протоколом.

Симулятор запускає сотні епізодів на кожну конфігурацію, логує повні діалоги та обчислює
агреговані метрики зі стандартними відхиленнями, готові для публікації.

---

## Огляд архітектури

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Гра         │     │  Агент A     │     │  Агент B         │
│  (виграші,   │◄────│  (send_msg,  │────►│  (send_msg,      │
│   Неш,       │     │   act)       │     │   act)           │
│   Парето)    │     └──────┬───────┘     └──────┬───────────┘
└──────┬───────┘            │                    │
       │           ┌───────▼────────────────────▼──────────┐
       │           │         Simulation Runner              │
       │           │  • цикл комунікації (K раундів)       │
       └──────────►│  • збір дій                            │
                   │  • обчислення виграшів                 │
                   │  • theory-hit / agreement detection    │
                   └───────────────┬────────────────────────┘
                                   │
                   ┌───────────────▼────────────────────────┐
                   │         Логування та метрики            │
                   │  • JSONL записи епізодів               │
                   │  • агрегація по K (CSV)                │
                   │  • графіки (PNG)                       │
                   └────────────────────────────────────────┘
```

---

## Підтримувані ігри

Реалізовано в `src/llmgt/games/`:

| Гра | Дії | Рівноваги Неша | Оптимуми Парето | Ключова напруга |
|-----|-----|----------------|-----------------|-----------------|
| **Дилема в'язня** (`pd`) | C, D | (D,D) | (C,C) | Індивідуальна vs. колективна раціональність |
| **Полювання на оленя** (`stag`) | S, H | (S,S), (H,H) | (S,S) | Домінування ризику vs. домінування виграшу |
| **Битва статей** (`bos`) | O, F | (O,O), (F,F) | (O,O), (F,F) | Координація при конфліктних уподобаннях |
| **Гра-ультиматум** (`ultimatum`) | Пропонувач: L,F / Відповідач: A,R | (L,A) | (F,A) | Справедливість vs. рівновага, досконала для підігор |

Кожен клас гри надає:
- `actions_for(role)` — набори дій для конкретної ролі (важливо для асиметричних ігор, як Ультиматум)
- `payoff(action_a, action_b)` → `(float, float)`
- `nash_equilibria()` → `set[tuple[str, str]]`
- `pareto_optima()` → `set[tuple[str, str]]`

---

## Типи агентів

### Baseline-агенти (без LLM)

| Агент | Опис |
|-------|------|
| `FixedActionAgent` | Завжди грає фіксовану дію. Без комунікації. |
| `EchoAgent` | Повторює останнє отримане повідомлення. Фіксована фінальна дія. |
| `WorkflowProposerAgent` | Детермінований агент протоколу PROPOSE/ACCEPT. |
| `WorkflowResponderAgent` | Приймає або контрпропонує на основі порогу виграшу. |
| `StochasticWorkflowResponderAgent` | Ймовірність прийняття зростає з номером раунду. |

### LLM-агенти

| Агент | Опис |
|-------|------|
| `LLMAgent` | Вільна комунікація з мінімальним ігро-теоретичним керівництвом. |
| `StrategicLLMAgent` | Промпти з таблицею виграшів та структурованим протоколом (PROPOSE/COUNTER/ACCEPT). |
| `WorkflowStrategicLLMAgent` | Повний workflow теоретико-ігрового міркування (домінантні стратегії → найкращі відповіді → Неш → Парето → рішення). Налаштовувана глибина через `workflow_level` (1–3). |

### Протокол переговорів

Усі комунікативні агенти слідують протоколу **PROPOSE / COUNTER / ACCEPT**:

```
Раунд 1: Агент A → PROPOSE: (X,Y)     # X = дія A, Y = дія B
         Агент B → ACCEPT: (X,Y)       # або COUNTER: (X',Y')
Раунд 2: Агент A → PROPOSE: (X',Y')   # повторна пропозиція
         Агент B → ACCEPT: (X',Y')     # фінальна домовленість
```

Коли `ACCEPT` виявлено у workflow-режимі, комунікація завершується достроково.

---

## Ключові поняття та метрики

### Епізод
Один запуск симуляції: опціональна комунікація (до *K* раундів) → дії → виграші → метрики.

### Бюджет комунікації `K`
Максимальна кількість раундів діалогу до вибору дій. Sweep-експеримент варіює *K* від 0 до 6, щоб вивчити вплив обсягу комунікації.

### Метрики (на епізод)

| Метрика | Опис |
|---------|------|
| `agreement_hit` | Чи відповідають фінальні дії тому, про що домовились під час комунікації? |
| `rounds_to_agreement` | Перший раунд, де з'явилась домовленість |
| `nash_hit` | Чи відповідає результат рівновазі Неша? |
| `pareto_hit` | Чи відповідає результат оптимуму Парето? |
| `theory_hit` | Комбінований успіх: Парето-Неш, якщо існує, інакше Неш |
| `rounds_to_theory_hit` | Перший раунд, де було погоджено теоретично цільовий результат |
| `welfare` | Сума виграшів: `payoff_a + payoff_b` |
| `winner` | Хто отримав більший виграш (або нічия) |

### Агреговані метрики (на K)

Функція `summarize_by_k()` обчислює середні, стандартні відхилення та частки по всіх епізодах для кожного *K*. Ключові колонки `summary_by_k.csv`:

- **Частки**: `agreement_rate`, `nash_rate`, `pareto_rate`, `theory_rate`
- **Раунди**: `mean_rounds_to_agreement`, `mean_rounds_to_theory_hit`
- **Виграші**: `welfare_mean`, `payoff_mean`, `payoff_diff_mean`
- **Комунікація**: `used_comm_rounds_mean`, `wasted_comm_rounds_mean`
- **Протокол**: `propose_rate`, `counter_rate`, `accept_rate`, `follow_accept_rate`
- **Ігро-теоретичні**: `regret_a_mean`, `regret_b_mean`, `welfare_gap_mean`
- Усі числові метрики мають колонки `*_std` (стандартне відхилення) для побудови error bars.

---

## Встановлення

**Вимоги:** Python ≥ 3.10

```bash
# Клонувати та встановити
git clone <repo-url>
cd llm-game-theory-agents
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Опційно: побудова графіків
pip install matplotlib

# Опційно: HuggingFace бекенд (рекомендовано GPU)
pip install -e ".[hf]"

# Опційно: Ollama бекенд
pip install -e ".[ollama]"
```

---

## LLM бекенди

| Бекенд | Прапорець | Вимоги | Випадок використання |
|--------|-----------|--------|---------------------|
| `heuristic` | `--backend heuristic` | Нічого | Швидкі smoke-тести, детермінований baseline |
| `ollama` | `--backend ollama` | Локальний сервер Ollama | Локальні відкриті моделі (Llama, Mistral) |
| `hf` | `--backend hf` | `torch`, `transformers` | Пряма інференція через HuggingFace (рекомендовано GPU) |
| `openai` | `--backend openai` | пакет `openai` + API ключ | Моделі OpenAI API |

### Налаштування

| Параметр | За замовчуванням | Опис |
|----------|-----------------|------|
| `--temperature` | 0.7 | Температура семплінгу |
| `--max-output-tokens` | 64 | Макс. токенів на відповідь LLM |
| `--agent-style` | `strategic` | `basic` або `strategic` |
| `--workflow-level` | 2 | Глибина workflow: 1=легкий, 2=стандартний, 3=суворий |

Backend-специфічні опції: див. `llmgt sweep --help`.

---

## Запуск експериментів

### CLI: Communication Sweep (основний експеримент)

```bash
# Швидкий smoke-тест (heuristic, ~1 секунда)
llmgt sweep --game pd --mode workflow --backend heuristic \
  --k 0..6 --n-runs 50 --plots

# Ollama з локальною Llama 3.1
llmgt sweep --game stag --mode workflow --backend ollama \
  --ollama-model llama3.1:8b --k 0..6 --n-runs 200 --plots

# HuggingFace Transformers
llmgt sweep --game bos --mode workflow --backend hf \
  --hf-model mistralai/Mistral-7B-Instruct-v0.2 --k 0..6 --n-runs 50 --plots

# OpenAI API
llmgt sweep --game ultimatum --mode no_workflow --backend openai \
  --openai-model gpt-4o-mini --k 0..6 --n-runs 200 --plots
```

### Скрипти

Папка `scripts/` містить попередньо налаштовані запускачі експериментів:

| Скрипт | Опис |
|--------|------|
| `run_pd.py` | Фіксовані агенти для PD (baseline) |
| `run_pd_llm.py` | LLM-агенти на PD (евристичний бекенд) |
| `run_pd_workflow.py` | Стохастичні workflow-агенти, sweep по PD |
| `run_comm_experiment.py` | Sweep комунікації з фіксованими агентами |
| `run_workflow_sweep_all.py` | Rule-based workflow sweep по всіх 4 іграх |
| `run_llm_sweep_all.py` | LLM sweep по всіх 4 іграх (бекенд через змінні середовища) |
| `run_thesis.py` | Повний pipeline для диплома: всі моделі × всі ігри × всі режими → графіки |

```bash
# Приклад: повний прогон з TinyLlama
LLMGT_N_RUNS=50 python scripts/run_thesis.py
```

---

## Вихідні артефакти

### Структура папки запуску

```
data/runs/20260223_082328Z_9f29a20_workflow_sweep_all/
├── run_meta.json              # Знімок конфігурації
├── prisoners_dilemma/
│   ├── logs/
│   │   └── episodes.jsonl     # Сирі записи епізодів
│   ├── summary_by_k.csv       # Агреговані метрики
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

### Логи епізодів (JSONL)

Кожен рядок — серіалізований `EpisodeRecord`:

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

### Графіки

Генеруються з прапорцем `--plots`. Графіки показують значення метрик vs. *K* з опціональними error bars (з колонок `*_std`):

- `agreement_rate.png` — частка епізодів, де агенти домовились
- `welfare_mean.png` — середній сумарний виграш
- `theory_rate.png` — частка, що відповідає теоретичному розв'язку
- `mean_rounds_to_agreement.png` — як швидко агенти досягають домовленості
- `mean_rounds_to_theory_hit.png` — раунди до досягнення теоретичного результату

---

## Структура проєкту

```
src/llmgt/
├── __init__.py                    # Корінь пакету
├── cli.py                         # CLI точка входу (llmgt sweep)
│
├── games/                         # Визначення ігор
│   ├── base.py                    # Абстрактний базовий клас Game
│   ├── prisoners_dilemma.py       # Дилема в'язня
│   ├── stag_hunt.py               # Полювання на оленя
│   ├── battle_of_sexes.py         # Битва статей
│   └── ultimatum.py               # Гра-ультиматум
│
├── agents/                        # Реалізації агентів
│   ├── parsing.py                 # ★ Спільні утиліти парсингу (рефакторинг)
│   ├── simple.py                  # FixedActionAgent, EchoAgent
│   ├── workflow.py                # Rule-based workflow агенти
│   ├── llm.py                     # Базовий LLM агент
│   ├── strategic.py               |
