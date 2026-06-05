_below description mostly by claude. it's a metaculus forecasting competition bot._

---

# Metaculus Forecasting Bot

A forecasting bot for Metaculus using ensemble learning with several frontier LLMs and multiple research approaches to predict future events.

## Overview

includes:

- **Model ensembling with meta-stacker**: 6-LLM forecaster fan-out → `CONDITIONAL_STACKING` by default (MEDIAN when models agree, stacker LLM rewrite when they disagree). Mean / median / always-stack strategies also available.
- **Multi-provider research**: AskNews + Gemini grounded search (first-party Google Search) + optional Grok native search + Perplexity / Exa fallbacks + yfinance/FRED for financial questions, all fanned out in parallel.
- **Gap-fill second pass**: after first-pass research, a Gemini analyzer identifies factual gaps and resolves each via parallel grounded searches.
- **Targeted disagreement research**: when base models disagree, a cheap LLM extracts the crux and Grok native search resolves it before the stacker runs.
- **Numeric CDF pipeline**: PCHIP interpolation (thanks Panshul) + tail widening, with aggressive enforcement of Metaculus CDF constraints (201 pts, min/max step, bound pinning).
- **Backtest-first benchmarking** on binary, numeric, and MC resolved questions (`backtest.py`).

## Quick Start

### Prerequisites

- Python 3.11+ with conda and poetry
- Required API keys (see Configuration section)

### Setup

1. **Clone and navigate to the repository**

```bash
git clone <repo-url>
cd metaculus-bot
```

1. **Set up conda environment**

```bash
conda create -n metaculus-bot python=3.11
conda activate metaculus-bot
```

1. **Install dependencies**

```bash
make install
# or: conda run -n metaculus-bot poetry install
```

1. **Configure environment**

```bash
cp .env.template .env
# Edit .env with your API keys (see Configuration section)
```

1. **Run the bot**

```bash
make run
# or: conda run -n metaculus-bot poetry run python main.py
```

## Core Architecture

### Main Components

- **`main.py`**: Thin CLI entry shim — re-exports `TemplateForecaster` and dispatches to `metaculus_bot/cli.py`
- **`metaculus_bot/forecaster.py`**: Primary `TemplateForecaster` implementation (research → forecaster fan-out → aggregation)
- **`metaculus_bot/aggregation_pipeline.py`**: Aggregation pipeline (MEDIAN / stacking / conditional-stacking, Platt calibration hook)
- **`backtest.py`**: Primary benchmarking system — scores predictions against actual resolutions
- **`community_benchmark.py`**: DEPRECATED benchmarking CLI (community prediction baseline broken)
- **`metaculus_bot/`**: Core utilities and configurations

### Key modules

- **`llm_configs.py`**: LLM ensemble + stacker + support-model configuration (single source of truth; rotates frequently)
- **`research/`**: research-provider subpackage — `providers.py` (AskNews / Perplexity / Exa / native-search), `orchestrator.py` (parallel fan-out + fallback), `gemini_search.py` (first-party Google Search grounding), `financial_data.py` (yfinance + FRED), `prediction_market.py` (Polymarket / Kalshi / Manifold), `targeted.py` (disagreement-crux search + gap-fill second pass), `persistence.py`
- **`stacking.py`** and **`aggregation_strategies.py`**: CONDITIONAL_STACKING / STACKING / MEAN / MEDIAN
- **`spread_metrics.py`**: per-type spread computation that triggers CONDITIONAL_STACKING
- **`prompts.py`**: prompts for base forecasting, stacking, gap-fill, targeted research
- **`numeric/pipeline.py`** / **`numeric/pchip_cdf.py`** / **`numeric/tail_widening.py`**: percentile → 201-point CDF pipeline
- **`comment/`**: published-comment assembly — `formatting.py`, `markers.py`, `trimming.py`
- **`ensemble_analysis/`**: offline correlation + ensemble-simulation tooling — `correlation_analysis.py`, `ensemble_simulator.py`, `cdf_cache.py`, `benchmark_identity.py`, `types.py`
- **`probabilistic_tools/`** and **`tool_runner.py`**: deterministic probability math for structured forecaster JSON blocks (active, gated by `PROBABILISTIC_TOOLS_ENABLED`)

## Usage Examples

### Basic Forecasting

```bash
# Run the bot on current Metaculus questions
make run

# Run with specific question filtering
python main.py --filter-type binary --max-questions 10
```

### Benchmarking

The primary benchmarking approach uses **resolved-question backtesting** (`backtest.py`), which scores bot predictions against actual question resolutions:

```bash
# Smoke test (4 resolved questions)
make backtest_smoke_test

# Small backtest (12 questions)
make backtest_small

# Medium backtest (32 questions)
make backtest_medium

# Large backtest (100 questions)
make backtest_large
```

<details>
<summary>DEPRECATED: Community prediction benchmark</summary>

The community benchmark (`community_benchmark.py`) scored bot predictions against the Metaculus community prediction as a proxy for ground truth. Metaculus removed the `aggregations` field from their list API, so `expected_baseline_score` is broken for newly-fetched questions.

```bash
make benchmark_run_smoke_test_binary   # deprecated
make benchmark_run_small               # deprecated
make benchmark_display                 # still works for viewing old results
```

</details>

### Correlation Analysis & Model Filtering

You can analyze correlations and recompute ensembles from prior runs without re-forecasting. Simple substring-based filters let you include or exclude models in the analysis.

Examples:

```bash
# Analyze the most recent benchmark file, excluding Grok and Gemini
PYTHONPATH=. ~/miniconda3/envs/metaculus-bot/bin/python analyze_correlations.py "$(ls -t benchmarks/benchmarks_*.jsonl | head -1)" \
  --exclude-models grok-4 gemini-2.5-pro

# Analyze a directory while excluding models
python analyze_correlations.py benchmarks/ --exclude-models grok-4 gemini-2.5-pro

# Include-only a subset (mutually exclusive with --exclude-models)
python analyze_correlations.py benchmarks/ --include-models qwen3-235b o3

# Apply filters to the built-in post-run analysis
python community_benchmark.py --mode run --num-questions 30 --mixed \
  --exclude-models grok-4 gemini-2.5-pro
```

Notes:

- Matching is substring-only, case-insensitive (no regex or space/hyphen normalization). For example, `grok-4` matches `openrouter/x-ai/grok-4`, but `grok 4` will not.
- Filters apply before computing correlation matrices, model stats, and ensemble search. The generated report includes a “Filters Applied” section.

### Testing

```bash
# Run all tests
make test

# Run specific test file
conda run -n metaculus-bot PYTHONPATH=. poetry run pytest tests/test_specific.py
```

## Configuration

### Required Environment Variables

Create a `.env` file based on `.env.template`:

```bash
# Metaculus API
METACULUS_TOKEN=your_metaculus_token

# Research APIs
ASKNEWS_CLIENT_ID=your_asknews_client_id
ASKNEWS_CLIENT_SECRET=your_asknews_secret
PERPLEXITY_API_KEY=your_perplexity_key
EXA_API_KEY=your_exa_key

# Optional: Gemini grounded search (Google AI Studio key, billing enabled for
# Gemini 3 Flash grounding; falls back to gemini-2.5-flash on free tier)
GOOGLE_API_KEY=your_google_ai_studio_key

# LLM APIs (via OpenRouter)
OPENROUTER_API_KEY=your_openrouter_key
```

### Optional feature flags

Opt-in research sources, each independently enabled:

- `NATIVE_SEARCH_ENABLED=true` — Grok 4.1-fast native web search via OpenRouter
- `GEMINI_SEARCH_ENABLED=true` — Gemini 3 Flash with first-party Google Search grounding (requires `GOOGLE_API_KEY`)
- `FINANCIAL_DATA_ENABLED=true` — yfinance + FRED for economic/market questions (requires `FRED_API_KEY`)
- `GAP_FILL_ENABLED=true` — always-on second-pass that identifies gaps in first-pass research and resolves each via a parallel grounded Gemini search (requires `GOOGLE_API_KEY`)

### Model configuration

- `metaculus_bot/llm_configs.py` is the single source of truth for the 6-model forecaster ensemble, stacker + cross-provider fallback stacker, disagreement analyzer, summarizer/researcher, and parser. Rotates frequently — don't hardcode model names elsewhere.
- **Aggregation**: `CONDITIONAL_STACKING` by default (CLI: `metaculus_bot/cli.py`). MEDIAN when base models agree, stacker LLM rewrite when they disagree. Thresholds in `metaculus_bot/constants.py` (binary 0.15 probability range, MC 0.20 max option, numeric 0.15 normalized percentile spread).
- **Research**: AskNews + Gemini grounded search + optional Grok native search + Perplexity + Exa fallbacks, with yfinance/FRED for financial questions. All opt-in independently via env flags; Gemini grounded search uses the Google AI Studio SDK directly for first-party Google Search results.
- **Provider**: OpenRouter with automatic key fallback for LLMs.

## Development

### Code Quality

```bash
# Lint code
make lint

# Format code
make format

# Install pre-commit hooks
make precommit_install

# Run pre-commit on all files
make precommit_all
```

### Makefile Commands

- `make install` - Install dependencies via conda + poetry
- `make test` - Run pytest suite
- `make run` - Run the forecasting bot
- `make lint` - Run Ruff linting
- `make format` - Format code with Ruff
- `make benchmark_*` - Various benchmarking options

### Testing Philosophy

- Focus on end-to-end integration tests for the forecasting pipeline
- Test core aggregation logic and API integrations
- All tests must pass before PRs
- Use `pytest` with async support for LLM testing

## Repository Structure

```
metaculus-bot/
├── main.py                     # Thin CLI entry shim (re-exports TemplateForecaster, dispatches to cli.py)
├── backtest.py                 # Resolved-question backtester (primary benchmarking)
├── community_benchmark.py      # DEPRECATED community-prediction benchmarker
├── metaculus_bot/              # Core utilities
│   ├── forecaster.py               # Primary TemplateForecaster implementation
│   ├── aggregation_pipeline.py     # Aggregation pipeline (MEDIAN / stacking / conditional, Platt hook)
│   ├── cli.py                      # CLI entry point + default config
│   ├── llm_configs.py              # Forecaster + stacker + support models
│   ├── research/                   # providers.py, orchestrator.py, gemini_search.py, financial_data.py, prediction_market.py, targeted.py, persistence.py
│   ├── stacking.py                 # Stacker LLM meta-prompts
│   ├── aggregation_strategies.py   # MEAN / MEDIAN / STACKING / CONDITIONAL_STACKING
│   ├── spread_metrics.py           # Per-type disagreement metric
│   ├── prompts.py                  # Base / stacking / gap-fill / targeted prompts
│   ├── numeric/                    # pipeline.py, pchip_cdf.py, tail_widening.py — percentile → 201pt CDF
│   ├── comment/                    # formatting.py, markers.py, trimming.py — published-comment assembly
│   ├── ensemble_analysis/          # correlation_analysis, ensemble_simulator, cdf_cache, benchmark_identity, types (offline analysis)
│   ├── probabilistic_tools/        # active (gated by PROBABILISTIC_TOOLS_ENABLED) Bayesian / survival / fit helpers
│   └── tool_runner.py              # active (gated by PROBABILISTIC_TOOLS_ENABLED) deterministic math over structured blocks
├── tests/                      # Pytest suite
├── .github/workflows/          # CI + scheduled bot runs
├── AGENTS.md                   # Repo-specific agent/coding guidelines (CLAUDE.md is a symlink)
└── Makefile                    # Development commands
```

## Framework Integration

This project heavily uses the [`forecasting-tools`](forecasting_tools_readme.md) framework:

- `GeneralLlm` for model interfaces
- `MetaculusApi` for platform integration
- Question types: `BinaryQuestion`, `NumericQuestion`, `MultipleChoiceQuestion`
- Prediction types: `ReasonedPrediction`, `BinaryPrediction`, etc.
- Research: `AskNewsSearcher`, `SmartSearcher`

## Additional Resources

- **[AGENTS.md](AGENTS.md)**: Comprehensive coding guidelines and repository standards
- **[starter_guide.md](starter_guide.md)**: Original template setup instructions
- **[forecasting_tools_readme.md](forecasting_tools_readme.md)**: Framework documentation

## Environment Notes

- **Conda environment**: `metaculus-bot`
- **Python version**: 3.11+
- **Code formatting**: Ruff with 120-character line length
- **Testing**: Pytest with async support
