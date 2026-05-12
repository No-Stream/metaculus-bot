# Metaculus Forecasting Bot — Agent Guidelines

General coding guidelines (style, testing, error handling, etc.) are in `~/.claude/CLAUDE.md`.
This file covers **repo-specific** context only.

## Repo-Specific Overrides

- **Python**: 3.11+ (see `pyproject.toml`)
- **Formatter**: Ruff with 120-char line length (not Black)
- **Testing**: Pytest + `pytest-asyncio`; all tests are self-contained (no API keys needed in CI)

## Project Overview

Fork of the Metaculus starter template. Runs a multi-LLM ensemble with a meta-stacker, multi-provider research, and question-type-specific post-processing. Aggregation defaults to `CONDITIONAL_STACKING` — MEDIAN when base models agree, a stacker LLM rewrites the forecast when they disagree.

## Core architecture

- `main.py`: primary bot implementation using `forecasting-tools` framework.
- `backtest.py`: primary benchmarking — scores bot predictions against actual resolutions.
- `community_benchmark.py`: **DEPRECATED** (Metaculus removed `aggregations` from list API; `make benchmark_display` still works for old runs).
- `metaculus_bot/`: core utilities — LLM configs, prompts, research providers, aggregation, numeric CDF pipeline, probabilistic_tools (dormant), stacking.
- `REFERENCE_COPY_OF_forecasting_tools*/`: read-only reference copy of the framework source (edits here don't affect installed package).
- `REFERENCE_COPY_OF_panchul*/`: Q2 2025 competition winner, present for comparison.
- `scratch_docs_and_planning/`: plans and audits (including `probabilistic_tools_activation.md` — activation pending, see below).
- `scratch_docs_and_planning/metaculus_api_doc_LARGE_FILE.yml`: full Metaculus API spec (use offset/limit).

## Forecasting pipeline (current)

Per question (`main.py:_research_and_make_predictions`):

1. **Research** — `run_research` (`main.py:459`) fans out providers in parallel via `_select_research_providers` / `_run_providers_parallel`. Always-on **gap-fill second pass** (`targeted_research.run_gap_fill_pass`, `main.py:478`) identifies factual gaps and resolves them via parallel grounded Gemini searches.
2. **Forecaster fan-out** — N forecaster LLMs run in parallel via `_forecaster_with_soft_deadline` (10-min cap each) → `_make_prediction` → type-specific runner (binary/MC/numeric).
3. **Min-forecasters guard** (`main.py:738`) drops the question if fewer than `MIN_FORECASTERS_TO_PUBLISH` returned a valid prediction.
4. **Aggregation** — see CONDITIONAL_STACKING below.

### Ensemble (6 forecasters)

See `metaculus_bot/llm_configs.py` for the authoritative list (rotates frequently). As of this writing: gpt-5.4, gpt-5.5, claude-opus-4.7, claude-opus-4.6, gemini-3.1-pro-preview, grok-4.1-fast. Do NOT hardcode model names outside `llm_configs.py`. Provider: OpenRouter with automatic key fallback.

Support models (also in `llm_configs.py`):

- **Stacker**: `claude-opus-4.5` (primary), `gpt-5.5` (cross-provider fallback for independence if Anthropic thrashes). Both `allowed_tries=1` — on stall, we fall back rather than burn budget retrying the same provider.
- **Disagreement analyzer**: `gpt-5-mini` (cheap crux extractor for targeted search).
- **Summarizer / researcher**: `gemini-3-flash-preview`.
- **Parser**: `gpt-5-mini` (low effort, deterministic).

### CONDITIONAL_STACKING (default)

`AggregationStrategy.CONDITIONAL_STACKING` (set in `metaculus_bot/cli.py:62`). Behavior:

- Compute spread across the N forecasters via `spread_metrics.compute_spread`.
- If spread ≤ threshold → return **MEDIAN** of raw per-model predictions (base-combine via `_aggregate_predictions`, `main.py:1037`).
- If spread > threshold → extract the **disagreement crux**, run **targeted search** (Grok native search), then invoke the **stacker LLM** with the full base-model reasonings + targeted research (`stacking.run_stacking_{binary,mc,numeric}`).
- Stacker fallback chain: primary `STACKER_LLM` under `STACKER_SOFT_DEADLINE` → `STACKER_FALLBACK_LLM` under `STACKER_FALLBACK_SOFT_DEADLINE` → MEDIAN. (`main.py:1148-1223`.)

Thresholds (`metaculus_bot/constants.py:166-175`):

- Binary: probability range (max − min) ≥ **0.15**.
- MC: max per-option spread ≥ **0.20**.
- Numeric: normalized percentile spread ≥ **0.15**.

### Clamps / bounds

- **Binary**: `[BINARY_PROB_MIN=0.01, BINARY_PROB_MAX=0.99]` (`constants.py:149-150`). Applied per-model at `main.py:1381` and on stacker output at `stacking.py:92`. Median/mean of already-clamped values stays in-bounds, so no post-aggregation clip needed.
- **MC**: `[0.005, 0.995]`, clamp-then-renormalize via `clamp_and_renormalize_mc` (`numeric_utils.py:206`).
- **Numeric CDF**: 201 points, `min_step ≥ 5e-5`, `max_step ≤ 0.2`, open bounds clipped to `[0.001, 0.999]`, closed bounds pinned to `{0.0, 1.0}`. Enforced in `pchip_cdf.generate_pchip_cdf` with aggressive repair + `safe_cdf_bounds` redistribution on max-step violations.

### Numeric pipeline (percentiles → PCHIP CDF)

Each forecaster emits the 11 standard percentiles `{2.5, 5, 10, 20, 40, 50, 60, 80, 90, 95, 97.5}` as plain text (prompt example at `prompts.py:478-488`). Per-model stages:

1. Parser LLM extracts `list[Percentile]` (`main.py:1487`).
2. `sanitize_percentiles` (`numeric_pipeline.py:52`): filter to the 11, validate, sort, spread count-like clusters, jitter duplicates, clamp to bounds, ensure strictly increasing, optionally widen tails.
3. `widen_declared_percentiles` (`tail_widening.py:95`, enabled by default): bound-aware stretch of distance-from-median by `k_tail=1.25`, only outside the central 60%. Enforces span floors on outer tails.
4. `build_numeric_distribution` → `generate_pchip_cdf_with_smoothing` produces 201-point PCHIP CDF → ramp smoothing for min-step → validation. On failure: `create_fallback_numeric_distribution` delegates CDF build to forecasting-tools.
5. **Discrete integer snapping** (`main.py:1515`): if a majority of forecasters vote DISCRETE, snap the distribution to integers.
6. **Unit-mismatch guard** (`numeric_validation.detect_unit_mismatch`): withholds the prediction if values look off by orders of magnitude.

**Ensemble aggregation** (`numeric_utils.aggregate_numeric:140`): pointwise **in CDF space** — concatenate each model's 201-point CDF, groupby value, mean or median the probabilities, then `_postprocess_ensemble_cdf` re-pins endpoints, enforces monotonic + min-step, resamples via PCHIP for discrete questions. Not percentile-space averaging.

### Research providers

Orchestration in `main.py:_select_research_providers:561-606`.

**Primary provider** — exactly one, chosen by priority in `research_providers.choose_provider_with_name:352-415`:

1. **AskNews** if `ASKNEWS_CLIENT_ID` + `ASKNEWS_SECRET` are set (the prod case): dual-phase search (HOT + HISTORICAL), rate-limited with retry/dedup (`research_providers.py:71-197`).
2. **Exa.ai SmartSearcher** if `EXA_API_KEY` set (fallback when AskNews absent): generic rundown (`research_providers.py:252-269`).
3. **Perplexity direct** if `PERPLEXITY_API_KEY` set: `research_providers.py:272-289`. Prompt explicitly requests prediction-market consideration unless benchmarking.
4. **Perplexity via OpenRouter** if `OPENROUTER_API_KEY` set: same function, `use_open_router=True`.
5. Empty stub.

In production (AskNews creds present) Exa/Perplexity/OpenRouter do NOT run. They're priority-ordered fallbacks, not peers. `RESEARCH_PROVIDER=<name>` forces a specific one.

**Additional providers run in parallel on top of the primary** (each independently gated):

- **Grok / xAI native search** (OpenRouter web plugin, `research_providers.py:292-344`): gated by `NATIVE_SEARCH_ENABLED`.
- **Gemini grounded search** (`gemini_search_provider.py`): real Google Search grounding via `google-genai` SDK (not OpenRouter) + `url_context` tool for specific URL reads. Gated by `GEMINI_SEARCH_ENABLED` + `GOOGLE_API_KEY`.
- **Financial data** (`financial_data_provider.py`): LLM classifier routes to yfinance + FRED for financial/economic questions. Gated by `FINANCIAL_DATA_ENABLED` + `FRED_API_KEY`.

**Second-pass gap-fill** (`targeted_research.run_gap_fill_pass`): always-on when `GAP_FILL_ENABLED` + `GOOGLE_API_KEY` are set. Two stages: Gemini analyzer identifies up to `GAP_FILL_MAX_GAPS` factual gaps → parallel grounded-Gemini searches resolve each. Soft-fails (returns `""`) on any error.

**All 4 production workflows** (`.github/workflows/run_bot_on_{tournament,metaculus_cup,minibench}.yaml`, `test_bot.yaml`) set `NATIVE_SEARCH_ENABLED=true`, `GEMINI_SEARCH_ENABLED=true`, `FINANCIAL_DATA_ENABLED=true`, `GAP_FILL_ENABLED=true`. So in prod the stack is AskNews + Grok native + Gemini grounded + financial-data (when classified as financial) + always-on Gemini gap-fill.

No dedicated prediction-market provider (Polymarket / Kalshi / Manifold) — yet. Prompt-level nudge only, in `prompts.web_research_prompt:113-115` (outside benchmark runs) and the Perplexity prompt at `research_providers.py:278`. Benchmarking mode actively forbids prediction-market search to avoid data leakage (`_benchmarking_warning`, `prompts.py:29-46`). See `scratch_docs_and_planning/atlas_inspired_improvements.md` Workstream G for the planned first-class provider.

### Prompts (`metaculus_bot/prompts.py`, ~900 lines)

- `_benchmarking_warning` (L29), `_forecasting_window_str` (L49), `web_research_prompt` (L86).
- Base: `binary_prompt` (L144), `multiple_choice_prompt` (L241), `numeric_prompt` (L334).
- Stacking: `stacking_binary_prompt` (L493), `stacking_multiple_choice_prompt` (L563), `stacking_numeric_prompt` (L641).
- Conditional-stacking support: `disagreement_crux_prompt` (L744), `targeted_search_prompt` (L768).
- Gap-fill: `gap_fill_analyzer_prompt` (L786), `gap_fill_search_prompt` (L865).

## Dormant scaffolding (important)

### `metaculus_bot/probabilistic_tools/`

Reusable probability math — pooling, Beta-Binomial Bayes, percentile → parametric fits (normal/lognormal/Student-t), declared-vs-math consistency checks, Dirichlet CIs, Neg-Bin/Poisson discrete percentiles, exponential/Weibull survival, Gamma-conjugate hazard. `prob_event_before`, `poisson_at_least_one`, `linear_pool` / `log_pool` / `satopaa_extremize`, `beta_binomial_update`, `cdf_at_threshold`, `dirichlet_with_other` are wired into `tool_runner` dispatch.

### `metaculus_bot/tool_runner.py`

Despite the name, **not** an LLM tool-calling harness. A **deterministic probability-math post-processor** that would run on structured JSON blocks emitted by each forecaster (priors, base rates, hazards, percentiles, scenarios) and inject a "Computed quantities" section into the stacker prompt. Entry points `run_tools_for_forecaster` (L360) and `build_cross_model_aggregation` (L554). Gated by `PROBABILISTIC_TOOLS_ENABLED` env var (unset) and **not imported anywhere in production code** — only tests.

### Activation plan

`scratch_docs_and_planning/probabilistic_tools_activation.md` details the three pending edits: (1) append structured-block JSON instructions to `binary_prompt` / `multiple_choice_prompt` / `numeric_prompt`, (2) call `run_tools_for_forecaster` in `_make_prediction`, (3) plumb `build_cross_model_aggregation` output into the stacker prompts. All library code and unit tests exist; none of the three wiring edits have been shipped.

**Coverage gaps vs. desirable patterns**: noisy-OR for rare-binary decomposition (`1 − ∏(1−pᵢ)`) is not implemented; mixture-of-normals CDF builder is not implemented (schema slot exists in `structured_output_schema.NumericStructured.mixture_components` but no evaluator); Gamma-waiting-time fitter is missing (have exponential survival, Weibull unconditional, and Gamma-conjugate hazard, but not Gamma waiting-time with conditional-given-survival).

## Project structure

- `tests/`: Pytest suite (`tests/test_*.py`).
- `.github/workflows/`: CI (lint + test on PRs) and scheduled bot runs.
- `.env.template`: reference for required environment variables.

## Configuration & environment

- Copy `.env.template` to `.env` for local development. Never commit secrets.

### Python environment

- **Conda environment**: `metaculus-bot`
- **Python binary**: `~/miniconda3/envs/metaculus-bot/bin/python`
- **Direct execution**: use the full python path when conda commands fail (`~/miniconda3/envs/metaculus-bot/bin/python script.py` instead of `conda run -n metaculus-bot python script.py`).
- **NEVER use pip directly** — dependencies managed by conda + poetry. Use `make install` or `poetry install` within the conda env.

## Framework integration (`forecasting-tools`)

- `GeneralLlm` for model interfaces (wrapper around litellm).
- `MetaculusApi` for platform integration.
- Question types: `BinaryQuestion`, `NumericQuestion`, `MultipleChoiceQuestion`.
- Prediction types: `ReasonedPrediction`, `BinaryPrediction`, etc.
- Research helpers: `AskNewsSearcher`, `SmartSearcher`.
- Numeric: `NumericDistribution`, `Percentile`. We subclass `NumericDistribution` as `PchipNumericDistribution` (`pchip_processing.py:234`) to override `.cdf` with our pre-computed 201-point PCHIP CDF; forecasting-tools' built-in CDF builder is only used on the fallback path.

## Model configuration

LLM ensemble lives in `metaculus_bot/llm_configs.py` — single source of truth. Rotates frequently; don't hardcode model names elsewhere. Provider: OpenRouter with automatic key fallback.

## Development commands

### Environment setup

- **Install**: `conda run -n metaculus-bot poetry install` (or `make install`).
- **Activate environment**: `conda activate metaculus-bot`.

### Core operations

- **Run bot**: `conda run -n metaculus-bot poetry run python main.py` (or `make run`).
- **Run tests**: `conda run -n metaculus-bot poetry run pytest` (or `make test`).

### Benchmarking

**Primary approach — resolved-question backtest** (`backtest.py`):

- **Smoke (4)**: `make backtest_smoke_test`
- **Small (12)**: `make backtest_small`
- **Medium (32)**: `make backtest_medium`
- **Large (100)**: `make backtest_large`

**DEPRECATED — community benchmark** (`community_benchmark.py`): baseline scoring broken (Metaculus removed aggregations from list API). `make benchmark_display` still works for viewing old results.

### Code quality

- **Lint**: `make lint` (Ruff check).
- **Format**: `make format` (Ruff format + autofix).
- **Pre-commit**: `make precommit_install` then `make precommit` or `make precommit_all`.
- **Test single file**: `conda run -n metaculus-bot PYTHONPATH=. poetry run pytest tests/test_specific.py`.

### Important commands

The **Makefile** has most commands (`make test`, `make format`, `make run`, etc.). In agentic CLIs, prefer the full python path (`~/miniconda3/envs/metaculus-bot/bin/python`) since conda activation can be unreliable.

## Commit & pull request guidelines

- Commits: concise, imperative subject (e.g., "fix test cmd", "add conda to make"). Short body when context helps.
- PRs: clear description, link issues, include config/docs updates, screenshots/logs for behavior changes.
- CI: all checks pass; code formatted and imports sorted.

## Metaculus API reference

- **API docs**: <https://www.metaculus.com/api/> (Swagger UI).
- **Backend source**: <https://github.com/Metaculus/metaculus> (open-source, validation in `questions/serializers/common.py`).
- **CDF constraints** (server-side, for `continuous_cdf` submissions):
  - Length: `inbound_outcome_count + 1` (default 201).
  - Min step per bin: `round(0.01 / N, 9)` (default 5e-5) — no flat segments allowed.
  - Max step per bin: `0.2 * 200 / N` (default 0.2) — spikiness cap.
  - Closed bounds: `cdf[0] == 0.0`, `cdf[-1] == 1.0`.
  - Open bounds: `cdf[0] >= 0.001`, `cdf[-1] <= 0.999`.
  - Strictly increasing (implied by min step > 0).
- **Comments API restriction**: `/api/comments/?author=X` returns only the caller's own comments (or staff authors). Dozens-of-comments analysis for other bots requires either a Metaculus support exemption, manual browsing, or a browser-driven scrape.

## Security & configuration tips

- Copy `.env.template` to `.env`; never commit secrets.
- Use GitHub Actions secrets for `METACULUS_TOKEN` and API keys (AskNews, Perplexity, Exa, OpenRouter, Google AI Studio, FRED, etc.).
- Limit changes to workflow files unless CI behavior is intended to change.
