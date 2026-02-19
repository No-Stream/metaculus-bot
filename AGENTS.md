# Metaculus Forecasting Bot — Agent Guidelines

General coding guidelines (style, testing, error handling, etc.) are in `~/.claude/CLAUDE.md`.
This file covers **repo-specific** context only.

## Repo-Specific Overrides
- **Python**: 3.11+ (see `pyproject.toml`)
- **Formatter**: Ruff with 120-char line length (not Black)
- **Testing**: Pytest + `pytest-asyncio`; all tests are self-contained (no API keys needed in CI)

## Project Overview
This is a Metaculus forecasting bot forked from the metaculus starter template. It uses model ensembling, plus research integration through AskNews, native search (Grok), and fallback providers.

## Core Architecture
- `main.py`: Primary bot implementation using `forecasting-tools` framework
- `main_with_no_framework.py`: Minimal dependencies variant 
- `backtest.py`: Primary benchmarking system — scores bot predictions against actual resolutions
- `community_benchmark.py`: **DEPRECATED** benchmarking CLI (community prediction baseline broken)
- `metaculus_bot/`: Core utilities including LLM configs, prompts, and research providers
- `REFERENCE_COPY_OF_forecasting_tools*/`: Read-only reference copy of the forecasting-tools framework source. Edits here won't affect the installed package. Path varies by machine — search for `REFERENCE_COPY_OF_forecasting_tools*` in the repo root or workspace if not found.
- A reference copy of the Q2 2025 competition winner (panchul) may exist in the workspace (`REFERENCE_COPY_OF_panchul*`). Good ideas for comparison.
- A Metaculus API doc (`metaculus_api_doc_LARGE_FILE.yml`) may exist in `scratch_docs_and_planning/`. Large file — use offset/limit when reading.

The bot architecture follows these key components:
- **Model Ensembling**: Multiple LLMs configured in `metaculus_bot/llm_configs.py` with aggregation strategies
- **Research Integration**: AskNews, native search (Grok), and fallback providers through `research_providers.py`
- **Forecasting Pipeline**: Question ingestion → research → reasoning → prediction extraction → aggregation

## Project Structure & Module Organization
- `tests/`: Pytest suite (`tests/test_*.py`).
- `.github/workflows/`: CI (lint + test on PRs) and scheduled bot runs.
- `.env.template`: Reference for required environment variables.

## Configuration & Environment
- Copy `.env.template` to `.env` for local development
- See `.env.template` for required API keys. Never commit secrets to repository.

### Python Environment
- **Conda environment**: `metaculus-bot`
- **Python binary**: `~/miniconda3/envs/metaculus-bot/bin/python`
- **Direct execution**: Use the full python path when conda commands fail
- Example: `~/miniconda3/envs/metaculus-bot/bin/python script.py` instead of `conda run -n metaculus-bot python script.py`
- **NEVER use pip directly** — dependencies are managed by conda + poetry. Use `make install` or `poetry install` within the conda env.

## Key Framework Integration
The project heavily uses `forecasting-tools` framework:
- `GeneralLlm` for model interfaces
- `MetaculusApi` for platform integration  
- Question types: `BinaryQuestion`, `NumericQuestion`, `MultipleChoiceQuestion`
- Prediction types: `ReasonedPrediction`, `BinaryPrediction`, etc.
- Research: `AskNewsSearcher`, `SmartSearcher` for information gathering

## Model Configuration
LLM ensemble configured in `metaculus_bot/llm_configs.py` — see that file for current models.
Models rotate frequently; do not hardcode model names outside of `llm_configs.py`.
Provider: OpenRouter (with automatic key fallback).

## Development Commands

### Environment Setup
- **Install**: `conda run -n metaculus-bot poetry install` (or `make install`)
- **Activate environment**: `conda activate metaculus-bot`

### Core Operations
- **Run bot**: `conda run -n metaculus-bot poetry run python main.py` (or `make run`)
- **Run tests**: `conda run -n metaculus-bot poetry run pytest` (or `make test`)

### Benchmarking

**Primary approach — resolved-question backtest** (`backtest.py`):
Scores bot predictions against actual question resolutions. This is the preferred benchmarking method.
- **Smoke test (4 questions)**: `make backtest_smoke_test`
- **Small (12 questions)**: `make backtest_small`
- **Medium (32 questions)**: `make backtest_medium`
- **Large (100 questions)**: `make backtest_large`

**DEPRECATED — community benchmark** (`community_benchmark.py`): Baseline scoring broken (Metaculus removed aggregations from list API). `make benchmark_display` still works for viewing old results.

### Code Quality
- **Lint**: `make lint` (Ruff check)
- **Format**: `make format` (Ruff format + autofix)
- **Pre-commit**: `make precommit_install` then `make precommit` or `make precommit_all`
- **Test single file**: `conda run -n metaculus-bot PYTHONPATH=. poetry run pytest tests/test_specific.py`

### Important commands
The **Makefile** has most commands — e.g. `make test`, `make format`, `make run`. In agentic CLIs you may need to use the full python path (`~/miniconda3/envs/metaculus-bot/bin/python`) since conda activation can be unreliable.

## Commit & Pull Request Guidelines
- Commits: concise, imperative subject (e.g., "fix test cmd", "add conda to make"). Add a short body when context helps.
- PRs: clear description, link issues, include config/docs updates, and screenshots/logs for behavior changes.
- CI: all checks pass; code formatted and imports sorted.

## Security & Configuration Tips
- Copy `.env.template` to `.env`; never commit secrets.
- Use GitHub Actions secrets for `METACULUS_TOKEN` and API keys (AskNews, Perplexity, Exa, etc.).
- Limit changes to workflow files unless CI behavior is intended to change.

