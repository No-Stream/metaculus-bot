_below description mostly by claude. it's a metaculus forecasting competition bot._

---

# Metaculus Forecasting Bot

A forecasting bot for Metaculus. It runs an ensemble of frontier LLMs, gathers evidence from several research sources in parallel, and turns the models' answers into a single prediction it publishes to Metaculus.

## Overview

What's inside:

- **LLM ensemble with a meta-stacker.** An ensemble of forecaster models runs in parallel, then the results are combined. The default aggregation is `CONDITIONAL_STACKING`: take the MEDIAN when the models agree, and let a stacker LLM rewrite the forecast when they disagree. In production the stacker is currently turned off, so prod runs effectively use the MEDIAN. The stacker chain stays live in backtests and ablations. The forecaster roster lives in `metaculus_bot/llm_configs.py` and rotates often, so read its membership there rather than from any description of it.
- **Multi-provider research, fanned out in parallel.** AskNews is the primary news source (dual-phase search, summarized by an LLM into an analyst briefing). Running alongside it: OpenAI native web search (an OpenAI model with web access via OpenRouter), Gemini grounded search (first-party Google Search via the `google-genai` SDK), a financial-data provider (yfinance + FRED), a prediction-market snapshot (Polymarket, Kalshi, Manifold, PredictIt), a resolution-source fetcher that reads the URLs cited in the question, and a time-series anchor that pulls historical base-rate data. Each source is turned on or off independently by an env flag.
- **Two gap-fill research passes.** After the first research round, the bot looks for missing facts. v1 (`research/targeted.py`) has an analyzer LLM list up to `GAP_FILL_MAX_GAPS` factual gaps and resolves each with a parallel web search. v2 (`research/agentic/`) runs an agentic tool loop: a driver LLM searches, fetches, and reads documents until it has what it needs, then appends a citation-only findings block. Both passes are on in production.
- **Numeric CDF pipeline.** Each forecaster declares the canonical percentile set (`STANDARD_PERCENTILES` in `metaculus_bot/numeric/config.py`); the bot turns them into a PCHIP CDF on the `PCHIP_CDF_POINTS` grid and enforces Metaculus's constraints (min/max step per bin, bound pinning, strictly increasing).
- **Backtest-first benchmarking.** Scores the bot's predictions against real question resolutions on binary, numeric, and multiple-choice questions (`backtest.py`).
- **Credit telemetry.** Logs OpenRouter balance and spend per run, and flags when the shared donated key drops below an early-warning floor (`metaculus_bot/credit_telemetry.py`, `make check_credits`).

## Quick Start

> **Cost note:** any run that hits live LLM or research APIs spends real money, and the live forecasting modes also publish comments to Metaculus. Never launch a paid run casually. If you're running someone else's setup, confirm before you kick off `make run` or `make backtest_*`. The free, self-contained commands are `make test`, `make lint`, `make format`, and `make check_credits`.

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for dependency management
- API keys (see [Configuration](#configuration))

### Setup

1. **Clone the repository**

   ```bash
   git clone <repo-url>
   cd metaculus-bot
   ```

2. **Install dependencies** (uv creates an in-project `.venv` from `uv.lock`)

   ```bash
   uv sync --dev
   # or: make install
   ```

3. **Configure environment**

   ```bash
   cp .env.template .env
   # Edit .env with your API keys (see Configuration below)
   ```

4. **Run the bot** (paid — asks live APIs and publishes to Metaculus)

   ```bash
   make run
   # or: uv run python main.py
   ```

   Pick a run mode with `--mode`: `tournament` (default), `minibench`, `metaculus_cup`, or `test_questions`.

   ```bash
   uv run python main.py --mode test_questions
   ```

## Core Architecture

### Main components

- **`main.py`** — thin entry shim that re-exports `TemplateForecaster` and hands off to `metaculus_bot/cli.py`.
- **`metaculus_bot/cli.py`** — CLI entry point, argument parsing, and the default bot configuration.
- **`metaculus_bot/forecaster.py`** — the `TemplateForecaster` implementation (research → forecaster fan-out → aggregation → publish).
- **`metaculus_bot/aggregation_pipeline.py`** — combines per-model forecasts (MEDIAN / stacking / conditional stacking).
- **`backtest.py`** — resolved-question backtester, the primary benchmark.
- **`community_benchmark.py`** — deprecated; the Metaculus community-prediction baseline broke when the list API dropped the `aggregations` field. `make benchmark_display` still views old results.

### Key modules

- **`llm_configs.py`** — single source of truth for the forecaster ensemble, stacker, and support models (summarizer, parser, disagreement analyzer). Rotates often, so don't hardcode model names anywhere else.
- **`research/`** — the research subpackage:
  - `providers.py` — AskNews / Exa / Perplexity / OpenAI native search
  - `orchestrator.py` — parallel fan-out, primary-provider selection, gap-fill wiring
  - `gemini_search.py` — Gemini grounded search via `google-genai`
  - `financial_data.py` — yfinance + FRED for financial and economic questions
  - `prediction_market.py` — Polymarket / Kalshi / Manifold / PredictIt snapshot
  - `resolution_source.py` — fetches and extracts the URLs cited in the question
  - `resolution_presentation.py` — renders fetched evidence with provenance and text budgets
  - `timeseries_anchor.py` — historical base-rate / time-series anchor data
  - `targeted.py` — disagreement-crux search and the v1 gap-fill pass
  - `agentic/` (+ `agentic_gap_fill.py`) — the v2 agentic gap-fill tool loop
  - `persistence.py` — writes per-provider research artifacts for backtest replay
- **`stacking.py`** / **`aggregation_strategies.py`** — the aggregation strategies (`MEAN` / `MEDIAN` / `STACKING` / `CONDITIONAL_STACKING`).
- **`spread_metrics.py`** — per-question-type disagreement metric that triggers conditional stacking.
- **`prompts.py`** — base, stacking, gap-fill, and targeted-research prompts.
- **`numeric/`** — `pipeline.py`, `pchip_cdf.py`, `tail_widening.py`: the percentile → PCHIP CDF pipeline.
- **`comment/`** — `formatting.py`, `markers.py`, `trimming.py`: assembles the comment published to Metaculus.
- **`ensemble_analysis/`** — offline correlation and ensemble-simulation tooling.
- **`probabilistic_tools/`** + **`tool_runner.py`** — deterministic probability math over structured forecaster JSON blocks. Wired but off in production (gated by `PROBABILISTIC_TOOLS_ENABLED`).

## Usage Examples

### Forecasting

```bash
# Run on tournament questions (default mode)
make run

# Run on the example test questions
uv run python main.py --mode test_questions
```

### Benchmarking

The primary benchmark replays resolved questions and scores the bot against the real outcomes (`backtest.py`). These commands spend API credits.

```bash
make backtest_smoke_test   # 4 questions
make backtest_small        # 12 questions
make backtest_medium       # 32 questions
make backtest_large        # 100 questions
```

### Correlation analysis (free, no API calls)

Analyze correlations and recompute ensembles from prior benchmark runs without re-forecasting. Model filters are case-insensitive substring matches.

Unsupported prediction types and non-finite values are excluded from correlation inputs. Numeric
ensemble scoring requires a usable CDF from every member; it can rebuild a CDF from declared
percentiles but does not substitute a uniform forecast when reconstruction fails. Fallback
model identifiers use a deterministic digest so repeated analyses keep the same labels.

```bash
# Analyze the most recent benchmark file
make analyze_correlations_latest

# Analyze a file or directory, excluding some models
uv run python analyze_correlations.py "$(ls -t benchmarks/benchmarks_*.jsonl | head -1)" \
  --exclude-models grok gemini
```

### Testing (free, self-contained)

```bash
make test                          # full suite
uv run pytest tests/test_foo.py    # a single file
```

## Configuration

Copy `.env.template` to `.env` and fill in your keys. Never commit `.env`.

### API keys

```bash
# Metaculus (required)
METACULUS_TOKEN=...

# LLMs via OpenRouter
OPENROUTER_API_KEY=...        # your personal OpenRouter key
OAI_ANTH_OPENROUTER_KEY=...   # Metaculus-donated key: covers OpenAI, Anthropic,
                              # and Google models. Other providers (e.g. Grok)
                              # 404 on it and fall back to OPENROUTER_API_KEY.

# Research
ASKNEWS_CLIENT_ID=...         # primary news source
ASKNEWS_SECRET=...
GOOGLE_API_KEY=...            # Gemini grounded search (personal Google AI Studio
                              # key; in CI it's the GEMINI_API_KEY secret)
FRED_API_KEY=...              # for the financial-data provider
EXA_API_KEY=...               # optional research fallback
PERPLEXITY_API_KEY=...        # optional research fallback
```

### Feature flags

Each research source is turned on independently. The four production workflows set these; for local runs add whichever you want to `.env`.

- `NATIVE_SEARCH_ENABLED` — OpenAI native web search via OpenRouter (default model `NATIVE_SEARCH_DEFAULT_MODEL` in `constants.py`; `NATIVE_SEARCH_MODEL` overrides it).
- `GEMINI_SEARCH_ENABLED` — Gemini grounded search via Google AI Studio (needs `GOOGLE_API_KEY`).
- `FINANCIAL_DATA_ENABLED` — yfinance + FRED for financial/economic questions (needs `FRED_API_KEY`).
- `PREDICTION_MARKETS_ENABLED` — prediction-market snapshot (suppressed during backtests to avoid leakage).
- `RESOLUTION_SOURCE_ENABLED` — fetch the URLs cited in the question's resolution criteria.
- `TS_ANCHOR_ENABLED` — time-series / base-rate anchor data. `TS_ANCHOR_CHART_ENABLED` adds a chart-image side channel (off by default).
- `GAP_FILL_ENABLED` — v1 gap-fill: an analyzer lists factual gaps, then parallel native searches resolve them.
- `GAP_FILL_V2_ENABLED` — v2 gap-fill: an agentic tool loop that searches, fetches, and reads documents.
- `PROBABILISTIC_TOOLS_ENABLED` — deterministic probability-math post-processor (off in production).

A few knobs also route Gemini traffic and control timeouts. `GEMINI_USE_DONATED_OPENROUTER_KEY` (default on) sends most OpenRouter Gemini calls through the donated key. See `.env.template` and `metaculus_bot/constants.py` for the full list of tunables and their defaults.

### Model and aggregation configuration

- **Models:** `metaculus_bot/llm_configs.py` is the single source of truth for the forecaster roster, the stacker and its fallback, the disagreement analyzer, the summarizer/researcher, and the parser. It rotates frequently.
- **Aggregation:** `CONDITIONAL_STACKING` by default (set in `metaculus_bot/cli.py`). Disagreement thresholds live in `metaculus_bot/constants.py`, one per question type: `CONDITIONAL_STACKING_BINARY_PROB_RANGE_THRESHOLD` (a probability range), `CONDITIONAL_STACKING_MC_MAX_OPTION_THRESHOLD` (a max per-option spread), and `CONDITIONAL_STACKING_NUMERIC_NORMALIZED_THRESHOLD` (a normalized percentile spread). Stacking is disabled across all production workflows via `*_STACKING_ENABLED=false`, so prod effectively runs the MEDIAN.
- **Provider:** OpenRouter, with automatic fallback from the donated key to the personal key on credit or routing errors.

## Development

```bash
make install      # uv sync --dev
make test         # pytest suite
make lint         # ruff check
make format       # ruff format + autofix
make typecheck    # basedpyright (standard mode, must stay at 0 errors)
make all          # format + lint + typecheck + verbose tests
make cov          # coverage report
make audit        # osv-scanner over uv.lock
```

Dependencies are managed with uv: `uv add <pkg>` for runtime, `uv add --dev <pkg>` for dev tools, then commit the updated `pyproject.toml` and `uv.lock`. Do not use `pip` or `poetry`; both are blocked here.

Use uv 0.12.10 or newer (`uv self update` for a standalone installation). To refresh
dependencies, run `uv lock --upgrade` followed by `uv sync --dev --frozen`. The project
excludes packages uploaded within the last week. Validate updates with the build,
offline test suite, lint, and type checks before committing.

### Testing philosophy

- Favor end-to-end integration tests of the forecasting pipeline over narrow unit tests.
- The whole suite is self-contained — no API keys, no network, no cost.
- All tests must pass before a PR; CI runs lint plus tests.

## Repository Structure

```
metaculus-bot/
├── main.py                     # Thin CLI shim (re-exports TemplateForecaster, calls cli.py)
├── backtest.py                 # Resolved-question backtester (primary benchmark)
├── community_benchmark.py      # DEPRECATED community-prediction benchmarker
├── analyze_correlations.py     # Offline correlation / ensemble analysis (free)
├── metaculus_bot/
│   ├── cli.py                      # CLI entry point + default config
│   ├── forecaster.py               # TemplateForecaster: research → fan-out → aggregation
│   ├── aggregation_pipeline.py     # MEDIAN / stacking / conditional stacking
│   ├── aggregation_strategies.py   # Strategy enum
│   ├── llm_configs.py              # Forecasters + stacker + support models
│   ├── constants.py                # Env flags, thresholds, timeouts
│   ├── credit_telemetry.py         # OpenRouter balance/spend logging
│   ├── spread_metrics.py           # Per-type disagreement metric
│   ├── prompts.py                  # Base / stacking / gap-fill / targeted prompts
│   ├── stacking.py                 # Stacker meta-prompts
│   ├── research/                   # providers, orchestrator, gemini_search, financial_data,
│   │                               #   prediction_market, resolution_source, timeseries_anchor,
│   │                               #   targeted (v1 gap-fill), agentic/ (v2 gap-fill), persistence
│   ├── numeric/                    # pipeline, pchip_cdf, tail_widening — percentiles → PCHIP CDF
│   ├── comment/                    # formatting, markers, trimming — published-comment assembly
│   ├── ensemble_analysis/          # offline correlation + ensemble simulation
│   ├── probabilistic_tools/        # deterministic prob math (gated off in prod)
│   └── tool_runner.py              # runs probabilistic_tools over structured blocks
├── tests/                      # Pytest suite (self-contained)
├── docs/                       # Detailed docs (see below)
├── .github/workflows/          # CI + scheduled bot runs
├── AGENTS.md                   # Repo-specific agent/coding guidelines (CLAUDE.md symlinks to it)
└── Makefile                    # Development commands
```

## Detailed docs

- **[docs/architecture.md](docs/architecture.md)** — the end-to-end pipeline: research, forecaster fan-out, aggregation, publishing.
- **[docs/research.md](docs/research.md)** — every research provider, what it does, and how the flags gate it.
- **[docs/agentic_gap_fill.md](docs/agentic_gap_fill.md)** — the v2 agentic gap-fill loop and its tools.
- **[docs/numeric_pipeline.md](docs/numeric_pipeline.md)** — percentiles → PCHIP CDF and the Metaculus constraints.
- **[docs/value_extraction.md](docs/value_extraction.md)** — how a forecast value is read out of a model's rationale.
- **[docs/prompts.md](docs/prompts.md)** — every forecasting-prompt rule and the failure it corrects.
- **[docs/roster_history.md](docs/roster_history.md)** — the ensemble roster, its history, and the dormant subsystems.
- **[docs/performance_analysis.md](docs/performance_analysis.md)** — residual-analysis conventions: eras, cohorts, scoring.
- **[docs/operations.md](docs/operations.md)** — running the bot, workflows, cost discipline, and credit telemetry.

## Framework Integration

This project builds on the [`forecasting-tools`](https://github.com/Metaculus/forecasting-tools) framework:

- `GeneralLlm` for model interfaces
- `MetaculusApi` for platform integration
- Question types: `BinaryQuestion`, `NumericQuestion`, `MultipleChoiceQuestion`
- Prediction types: `ReasonedPrediction`, `BinaryPrediction`, and so on
- Research helpers: `AskNewsSearcher`, `SmartSearcher`

## Additional Resources

- **[AGENTS.md](AGENTS.md)** — the terse agent-facing starting point: the cost gate, repo overrides, layout, pipeline outline, standing rules, and an index into `docs/`.
- **[metac-bot-template](https://github.com/Metaculus/metac-bot-template)** — the upstream starter template this repo forked from.
- **[forecasting-tools](https://github.com/Metaculus/forecasting-tools)** — framework documentation.

## Environment Notes

- **Python:** 3.12+
- **Package manager:** uv (`uv sync`, `uv run`, `uv add`); the lockfile is `uv.lock`.
- **Build backend:** `uv_build`, flat layout (`metaculus_bot/` at the repo root).
- **Formatting:** Ruff, 120-character line length.
- **Type checking:** basedpyright, standard mode, kept at zero errors.
- **Testing:** Pytest with `pytest-asyncio`.
