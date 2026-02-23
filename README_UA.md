# LLM Game-Theoretic Simulations

Дослідницький симулятор для **класичних двогравцевих ігор** з **агентами на базі LLM** (а також простими baseline-агентами).
Репозиторій задуманий як база для дипломної/кваліфікаційної роботи з питання:

> **Як обмежений бюджет комунікації `K` (кількість раундів діалогу до ходу) впливає на домовленість, результат з точки зору рівноваг та добробут у класичних іграх?**

Підтримуються два режими взаємодії:

- `no_workflow`: вільна/неструктурована комунікація (baseline)
- `workflow`: структуровані підказки (workflow prompting), які спрямовують агента через фази на кшталт аналіз → пропозиція → контрпропозиція → прийняття

Англійська версія: [`README.md`](README.md).

---

## Зміст

- [Підтримувані ігри](#підтримувані-ігри)
- [Ключові поняття і метрики](#ключові-поняття-і-метрики)
- [Встановлення](#встановлення)
- [LLM бекенди (налаштування)](#llm-бекенди-налаштування)
- [Запуск експериментів](#запуск-експериментів)
  - [Communication sweep по K (рекомендовано)](#communication-sweep-по-k-рекомендовано)
  - [Скрипти](#скрипти)
- [Вихідні дані / артефакти](#вихідні-дані--артефакти)
  - [Структура папки запуску](#структура-папки-запуску)
  - [Логи епізодів (JSONL)](#логи-епізодів-jsonl)
  - [Агреговані метрики (CSV)](#агреговані-метрики-csv)
  - [Графіки](#графіки)
- [Структура проєкту](#структура-проєкту)
- [Тестування](#тестування)
- [Нотатки про відтворюваність](#нотатки-про-відтворюваність)
- [Troubleshooting](#troubleshooting)

---

## Підтримувані ігри

Реалізовано в `src/llmgt/games/`:

- **Prisoner’s Dilemma** (`pd`)
- **Stag Hunt** (`stag`)
- **Battle of the Sexes** (`bos`)
- **Ultimatum Game** (`ultimatum`)

Кожна гра надає:

- множини допустимих дій: `actions_for(player)`
- функцію виграшів: `payoff(action_a, action_b)`
- теоретичні множини (де застосовно):
  - `nash_equilibria()`
  - `pareto_optima()`

---

## Ключові поняття і метрики

### Епізод
Один запуск гри між **агентом A** та **агентом B**:

1. опціональна комунікація (до `K` раундів)
2. кожен агент обирає дію (`ACTION: ...`)
3. обчислюються виграші
4. обчислюються метрики теорії ігор / домовленості та записуються в лог

### Раунди комунікації `K`
`K` — це **максимальна** кількість раундів діалогу *до* вибору дій.
Додатково логуються:

- `used_comm_rounds` — скільки раундів фактично використали
- у `workflow` режимі симуляція може завершити діалог раніше, якщо виявлено `ACCEPT`

### Домовленість (agreement)
Домовленість визначається на основі діалогу + фінальних дій і логується як:

- `agreement_hit: bool | None`
- `rounds_to_agreement: int | None`

### “Theory hits” (влучання в теоретичні розв’язки)
Для фінальної пари дій обчислюється:

- `nash_hit`
- `pareto_hit`
- `pareto_nash_hit`
- `theory_hit` (узагальнений індикатор “успіху”)
- `rounds_to_theory_hit`

### Добробут (welfare)
З виграшів:

- `welfare = payoff_a + payoff_b`
- в агрегованих метриках також є `payoff_mean` та `payoff_diff_mean`

---

## Встановлення

Вимоги:

- Python **>= 3.10**

Швидкий старт (editable install):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Опційні extras з `pyproject.toml`:

- HuggingFace / Transformers бекенд:

```bash
pip install -e ".[hf]"
```

- Ollama бекенд (HTTP-клієнт):

```bash
pip install -e ".[ollama]"
```

Побудова графіків (опційно):

```bash
pip install matplotlib
```

---

## LLM бекенди (налаштування)

CLI підтримує такі бекенди:

- `heuristic` — детермінований baseline (без зовнішніх API)
- `openai` — OpenAI Responses API
- `ollama` — локальний Ollama сервер через HTTP
- `hf` — локальне виконання через HuggingFace Transformers

Спільні параметри:

- `--temperature`
- `--max-output-tokens`

### OpenAI

- Модель вибирається через `--openai-model` (дефолт: `gpt-4o-mini`)
- Опціональний `--base-url` для сумісних gateway/proxy
- Авторизація: зазвичай через змінні середовища, які очікує OpenAI SDK (часто `OPENAI_API_KEY`)

### Ollama

- `--ollama-model` (дефолт: `llama3.1:8b`)
- `--ollama-host` (дефолт: `http://localhost:11434`)
- `--ollama-timeout-s` (дефолт: `120`)

### HuggingFace Transformers

- `--hf-model` (дефолт: `mistralai/Mistral-7B-Instruct-v0.2`)
- `--hf-max-new-tokens` (дефолт: `128`)

Примітка: великі моделі можуть потребувати GPU та значних ресурсів RAM/VRAM.

---

## Запуск експериментів

У проєкті є CLI entrypoint:

- `llmgt` → `src/llmgt/cli.py`

### Communication sweep по K (рекомендовано)

Це основний драйвер експериментів для дипломної оцінки.
Команда проганяє `n_runs` епізодів для кожного `K` і записує:

- `logs/episodes.jsonl` (сирі логи епізодів)
- `summary_by_k.csv` (агрегація метрик)
- `figures/*.png` (опційні графіки)

Швидкий smoke-test на heuristic backend:

```bash
llmgt sweep --game pd --mode workflow --backend heuristic --k 0..6 --n-runs 50 --plots
```

Приклад з Ollama:

```bash
llmgt sweep --game stag --mode workflow --backend ollama \
  --ollama-model llama3.1:8b --ollama-host http://localhost:11434 \
  --k 0..6 --n-runs 200 --plots
```

Приклад з OpenAI:

```bash
llmgt sweep --game bos --mode no_workflow --backend openai \
  --openai-model gpt-4o-mini --temperature 0.7 --max-output-tokens 64 \
  --k 0..6 --n-runs 200 --plots
```

Приклад з HuggingFace:

```bash
llmgt sweep --game ultimatum --mode workflow --backend hf \
  --hf-model mistralai/Mistral-7B-Instruct-v0.2 --hf-max-new-tokens 128 \
  --k 0..6 --n-runs 50 --plots
```

#### Параметри CLI (`llmgt sweep`)

Основні:

- `--game`: `pd | stag | bos | ultimatum`
- `--mode`: `workflow | no_workflow`
- `--k`: або діапазон `0..6`, або явний список `--k 0 1 2 3`
- `--n-runs`: епізодів на кожне K (дефолт: `200`)
- `--out-dir`: базова директорія для результатів (дефолт: `data/runs`)
- `--tag`: мітка, що додається до назви папки запуску
- `--plots`: зберегти PNG графіки (потрібен `matplotlib`)

Поведінка агента:

- `--agent-style`: `basic | strategic`
- `--workflow-level`: `1 | 2 | 3` (лише для workflow режиму)

Backend-специфічні:

- HF: `--hf-model`, `--hf-max-new-tokens`
- Ollama: `--ollama-model`, `--ollama-host`, `--ollama-timeout-s`
- OpenAI: `--openai-model`, `--base-url`

### Скрипти

Папка `scripts/` містить “convenience runners” для швидких прогонів і дипломних серій, наприклад:

- `scripts/run_comm_experiment.py`
- `scripts/run_llm_sweep_all.py`
- `scripts/run_workflow_sweep_all.py`
- `scripts/run_pd.py`, `scripts/run_pd_workflow.py`, `scripts/run_pd_llm.py`
- `scripts/run_thesis.py`

Типовий запуск:

```bash
python scripts/run_comm_experiment.py
```

---

## Вихідні дані / артефакти

### Структура папки запуску

Кожен запуск створює нову папку всередині `data/runs/` (або `--out-dir`).
Назва папки має таймстемп:

- `YYYYMMDD_HHMMSSZ_<salt>_<tag>`

Стандартні підпапки:

- `logs/`
- `figures/`

### Логи епізодів (JSONL)

Шлях:

- `.../logs/episodes.jsonl`

Кожен рядок — серіалізований `EpisodeRecord` (див. `src/llmgt/logging/records.py`).
Важливі поля:

- ідентифікатори: `episode_id`, `game`, `mode`
- комунікація: `max_comm_rounds`, `used_comm_rounds`
- моделі: `model_a`, `model_b`
- діалог: `messages[]` з `{role, content, ts_utc}`
- результат: `action_a`, `action_b`, `payoff_a`, `payoff_b`, `winner`
- theory hits: `nash_hit`, `pareto_hit`, `pareto_nash_hit`, `theory_hit`, `rounds_to_theory_hit`
- домовленість: `agreement_hit`, `rounds_to_agreement`
- інше: `extra` (наприклад, у workflow режимі може містити `accepted_pair`)
- час: `started_at_utc`, `finished_at_utc`

### Агреговані метрики (CSV)

Шлях:

- `.../summary_by_k.csv`

Генерується через `summarize_by_k()` (`src/llmgt/experiments/sweep.py`).
Колонки включають (не повний перелік):

- `k`, `n_runs`, `game`
- частки: `agreement_rate`, `nash_rate`, `pareto_rate`, `pareto_nash_rate`, `theory_rate`
- раунди: `mean_rounds_to_agreement`, `mean_rounds_to_theory_hit`
- використання комунікації: `used_comm_rounds_mean`, `used_comm_rounds_p50`, `used_comm_rounds_over_k_mean`, `wasted_comm_rounds_mean`
- виграші: `payoff_mean`, `welfare_mean`, `payoff_diff_mean`
- “перемоги”: `a_win_rate`, `b_win_rate`, `tie_rate`
- статистика тексту: `msg_count_mean`, `words_total_mean`, `words_a_mean`, `words_b_mean`
- маркери намірів: `propose_rate`, `counter_rate`, `accept_rate`, `follow_accept_rate`
- опційно (якщо передати game у summarizer): `regret_a_mean`, `regret_b_mean`, `welfare_gap_mean`

Uncertainty / variability:

- Для багатьох числових метрик `summarize_by_k()` також додає колонки `*_std` (вибіркове стандартне відхилення).
  Їх зручно використовувати для error bars і для опису розкиду результатів між запусками.

### Графіки

Якщо увімкнути `--plots`, CLI збереже:

- `figures/agreement_rate.png`
- `figures/mean_rounds_to_agreement.png`
- `figures/welfare_mean.png`
- `figures/theory_rate.png`
- `figures/mean_rounds_to_theory_hit.png`

Побудова графіків реалізована в `src/llmgt/experiments/plotting.py`. Якщо `matplotlib` не встановлений, plotting буде пропущено (без падіння).

---

## Структура проєкту

Високорівнева мапа:

- `src/llmgt/cli.py` — CLI (`llmgt sweep`)
- `src/llmgt/games/` — ігри та теоретичні множини
- `src/llmgt/agents/` — агенти (`LLMAgent`, стратегічні та workflow-варіанти)
- `src/llmgt/llm/` — LLM клієнти:
  - `heuristic.py` (baseline)
  - `openai_client.py`
  - `ollama_client.py`
  - `hf_client.py`
- `src/llmgt/sim/` — симуляція епізодів + логіка agreement/theory + run directory
- `src/llmgt/experiments/` — sweeps, агрегація, графіки, фабрики агентів
- `src/llmgt/logging/` — JSONL логер + Pydantic записи
- `scripts/` — скрипти для запусків
- `tests/` — тестовий набір (pytest)

---

## Тестування

Запуск тестів:

```bash
pytest -q
```

---

## Нотатки про відтворюваність

Щоб результати диплома були стабільнішими:

- Фіксуй `K`, `n_runs`, `mode` і параметри бекенда при порівнянні умов.
- Записуй ідентифікатори моделей та параметри семплінгу: `--temperature`, `--max-output-tokens`.
- Зберігай “сирі” логи (`episodes.jsonl`) — вони містять повний діалог + фінальні дії.
- Для LLM бажано робити кілька незалежних прогонів і будувати довірчі інтервали (варіативність семплінгу).

---

## Troubleshooting

- **Нема графіків / “Plotting skipped”** → встанови `matplotlib`.
- **Ollama connection/timeouts** → перевір, що сервер запущений, і `--ollama-host` правильний.
- **HF out-of-memory** → візьми меншу модель, зменш `--hf-max-new-tokens`, або запускай на CPU.
- **OpenAI auth errors** → перевір змінну середовища з API ключем (часто `OPENAI_API_KEY`).
