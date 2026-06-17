# Metaculus Forecasting Bot — Agent Guidelines

General coding guidelines (style, testing, error handling, etc.) are in `~/.claude/CLAUDE.md`.
This file covers **repo-specific** context only.

## ⚠️ Cost discipline — never run a paid run without asking

**Any command that hits live LLM / research APIs spends real money (OpenRouter
credits, AskNews, Exa, Perplexity, Google). NEVER launch one autonomously — ask
first, every time, even after a clean build / passing tests / clean `/forge`.**
This is a hard gate, not a courtesy: a single broad run can burn meaningful
credits, and `--mode test_questions` also **publishes comments to Metaculus**
(a visible external action that pings nothing but is hard to retract).

Paid / external-effecting commands (ask before each):

- `python main.py` / `make run` in any live mode (`--mode test_questions`,
  `tournament`, `metaculus_cup`, `minibench`) — spends API credits AND publishes.
- `make backtest_*` (`smoke`/`small`/`medium`/`large`) — spends API credits on
  every forecaster + research call (no publish, but real money; `large`=100 Qs).
- Anything invoking research providers or the ensemble against real questions.

Free / safe (run freely): `make test`, `make lint`, `make format`,
`make check_credits`, `make benchmark_display` (views old results), and any
unit/integration test — the suite is self-contained and hits no paid APIs.

When verification needs a paid run, surface the exact command + rough cost and
let the user decide. Unit/integration coverage is the default proof of
correctness; live runs are opt-in.

## Repo-Specific Overrides

- **Python**: 3.11+ (see `pyproject.toml`)
- **Formatter**: Ruff with 120-char line length (not Black)
- **Testing**: Pytest + `pytest-asyncio`; all tests are self-contained (no API keys needed in CI)

## Project Overview

Fork of the Metaculus starter template. Runs a multi-LLM ensemble with a meta-stacker, multi-provider research, and question-type-specific post-processing. Aggregation defaults to `CONDITIONAL_STACKING` — MEDIAN when base models agree, a stacker LLM rewrites the forecast when they disagree.

## Core architecture

- `main.py`: thin entrypoint shim that re-exports `TemplateForecaster` from `metaculus_bot/forecaster.py` (the primary bot implementation, using the `forecasting-tools` framework) and invokes the CLI.
- `backtest.py`: primary benchmarking — scores bot predictions against actual resolutions.
- `community_benchmark.py`: **DEPRECATED** (Metaculus removed `aggregations` from list API; `make benchmark_display` still works for old runs).
- `metaculus_bot/`: core utilities — LLM configs, prompts, research providers, aggregation, numeric CDF pipeline, probabilistic_tools (dormant), stacking.
- `REFERENCE_COPY_OF_forecasting_tools*/`: read-only reference copy of the framework source (edits here don't affect installed package).
- `REFERENCE_COPY_OF_panchul*/`: Q2 2025 competition winner, present for comparison.
- `scratch_docs_and_planning/`: plans and audits (including `probabilistic_tools_activation.md` — activation pending, see below).
- `scratch_docs_and_planning/metaculus_api_doc_LARGE_FILE.yml`: full Metaculus API spec (use offset/limit).

## Forecasting pipeline (current)

Per question (`forecaster.py:_research_and_make_predictions`):

1. **Research** — `run_research` (`forecaster.py:406`) fans out providers in parallel via `_select_research_providers` / `_run_providers_parallel`. Always-on **gap-fill second pass** (`research/targeted.py` `run_gap_fill_pass`, `research/orchestrator.py:89`) identifies factual gaps and resolves them via parallel grounded Gemini searches.
2. **Forecaster fan-out** — N forecaster LLMs run in parallel via `_forecaster_with_soft_deadline` (10-min cap each) → `_make_prediction` → type-specific runner (binary/MC/numeric).
3. **Min-forecasters guard** (`forecaster.py:580`) drops the question if fewer than `MIN_FORECASTERS_TO_PUBLISH` returned a valid prediction.
4. **Aggregation** — see CONDITIONAL_STACKING below.

### Ensemble (6 forecasters)

See `metaculus_bot/llm_configs.py` for the authoritative list (rotates frequently). As of this writing: gpt-5.4, gpt-5.5, claude-opus-4.8, claude-opus-4.6, gemini-3.1-pro-preview, grok-4.3. Do NOT hardcode model names outside `llm_configs.py`. Provider: OpenRouter with automatic key fallback.

Support models (also in `llm_configs.py`):

- **Stacker**: `claude-opus-4.5` (primary), `gpt-5.5` (cross-provider fallback for independence if Anthropic thrashes). Both `allowed_tries=1` — on stall, we fall back rather than burn budget retrying the same provider.
- **Disagreement analyzer**: `gpt-5.5` (medium-effort crux extractor; quality drives targeted-search query, with a 180s soft deadline).
- **Summarizer / researcher**: `gpt-5.4-mini` (low effort, deterministic). Migrated 2026-05-17 from gemini-3-flash-preview for consistency with OpenAI-based support stack and to dodge donated-key Google rate limits; bills to personal `OPENROUTER_API_KEY` until OpenAI data-policy block is lifted.
- **Parser**: `gpt-5-mini` (low effort, deterministic).

### CONDITIONAL_STACKING (default)

`AggregationStrategy.CONDITIONAL_STACKING` (set in `metaculus_bot/cli.py:62`). Behavior:

- Compute spread across the N forecasters via `spread_metrics.compute_spread`.
- If spread ≤ threshold → return **MEDIAN** of raw per-model predictions (base-combine via `_aggregate_predictions`, `aggregation_pipeline.py:203`).
- If spread > threshold → extract the **disagreement crux**, run **targeted search** (OpenAI native search via `gpt-5.5` with `reasoning={"effort":"medium"}` + `verbosity="low"`, 360s timeout), then invoke the **stacker LLM** with the full base-model reasonings + targeted research (`stacking.run_stacking_{binary,mc,numeric}`).
- Stacker fallback chain: primary `STACKER_LLM` under `STACKER_SOFT_DEADLINE` → `STACKER_FALLBACK_LLM` under `STACKER_FALLBACK_SOFT_DEADLINE` → MEDIAN. (`aggregation_pipeline.py:274-357`.)

Thresholds (`metaculus_bot/constants.py:166-175`):

- Binary: probability range (max − min) ≥ **0.15**.
- MC: max per-option spread ≥ **0.20**.
- Numeric: normalized percentile spread ≥ **0.15**.

### Clamps / bounds

- **Binary**: `[BINARY_PROB_MIN=0.02, BINARY_PROB_MAX=0.98]` (`constants.py`). Applied per-model in `forecaster_runners.py` and on stacker output in `stacking.py`. Median/mean of already-clamped values stays in-bounds, so no post-aggregation clip needed.
- **MC**: `[0.005, 0.995]`, clamp-then-renormalize via `clamp_and_renormalize_mc` (`numeric/utils.py`).
- **Numeric CDF**: 201 points, `min_step ≥ 5e-5`, `max_step ≤ 0.2`, open bounds clipped to `[0.001, 0.999]`, closed bounds pinned to `{0.0, 1.0}`. Enforced in `numeric/pchip_cdf.py` `generate_pchip_cdf` with aggressive repair + `safe_cdf_bounds` redistribution on max-step violations.

### Numeric pipeline (percentiles → PCHIP CDF)

Each forecaster emits the 11 standard percentiles `{2.5, 5, 10, 20, 40, 50, 60, 80, 90, 95, 97.5}` as plain text (prompt example in `prompts.numeric_prompt`). Per-model stages:

1. Parser LLM extracts `list[Percentile]`.
2. `sanitize_percentiles` (`numeric/pipeline.py`): filter to the 11, validate, sort, spread count-like clusters, jitter duplicates, clamp to bounds, ensure strictly increasing, optionally widen tails.
3. `widen_declared_percentiles` (`numeric/tail_widening.py`): bound-aware stretch of distance-from-median by `k_tail=1.0` (identity by default; widening only kicks in when callers raise `k_tail` above 1.0) with `span_floor_gamma=0.0` (no span-floor enforcement by default). Both knobs are configurable per-call.
4. `build_numeric_distribution` → `generate_pchip_cdf_with_smoothing` produces 201-point PCHIP CDF → ramp smoothing for min-step → validation. On failure: `create_fallback_numeric_distribution` delegates CDF build to forecasting-tools.
5. **Discrete integer snapping**: if a majority of forecasters vote DISCRETE, snap the distribution to integers.
6. **Unit-mismatch guard** (`numeric/validation.py` `detect_unit_mismatch`): withholds the prediction if values look off by orders of magnitude.

### Numeric format router (`numeric_format_router.py`)

The router decides whether the LLM's numeric output is in OPTION A (the default 11 trailing `Percentile X.X: ...` lines) or OPTION B (a `mixture_components` list inside the JSON block). It always returns a 201-point Metaculus CDF and records which branch produced it for residual analysis (logged as `numeric_format=...`). If both formats are present, the mixture wins deterministically and a WARNING is logged so the frequency is auditable. The mixture branch flows through `percentiles_to_metaculus_cdf_via_mixture` (constraint-enforced grid evaluation of the mixture CDF).

**Ensemble aggregation** (`numeric/utils.py` `aggregate_numeric:140`): pointwise **in CDF space** — concatenate each model's 201-point CDF, groupby value, mean or median the probabilities, then `_postprocess_ensemble_cdf` re-pins endpoints, enforces monotonic + min-step, resamples via PCHIP for discrete questions. Not percentile-space averaging.

### Research providers

Orchestration in `research/orchestrator.py:_select_research_providers:196-240`.

**Primary provider** — exactly one, chosen by priority in `research/providers.py` `choose_provider_with_name:405-475`:

1. **AskNews** if `ASKNEWS_CLIENT_ID` + `ASKNEWS_SECRET` are set (the prod case): dual-phase search (HOT + HISTORICAL), rate-limited with retry/dedup (`research/providers.py:82-210`).
2. **Exa.ai SmartSearcher** if `EXA_API_KEY` set (fallback when AskNews absent): generic rundown (`research/providers.py:263-281`).
3. **Perplexity direct** if `PERPLEXITY_API_KEY` set: `research/providers.py:283-301`. Prompt explicitly requests prediction-market consideration unless benchmarking.
4. **Perplexity via OpenRouter** if `OPENROUTER_API_KEY` set: same function, `use_open_router=True`.
5. Empty stub.

In production (AskNews creds present) Exa/Perplexity/OpenRouter do NOT run. They're priority-ordered fallbacks, not peers. `RESEARCH_PROVIDER=<name>` forces a specific one.

**Additional providers run in parallel on top of the primary** (each independently gated):

- **OpenAI native search** (OpenRouter web plugin, `research/providers.py`): default model `openai/gpt-5.5` with `reasoning={"effort":"medium"}` + `extra_body={"verbosity":"low"}` and a 360s timeout (`NATIVE_SEARCH_DEFAULT_MODEL` / `NATIVE_SEARCH_TIMEOUT` / `NATIVE_SEARCH_REASONING_EFFORT_DEFAULT` / `NATIVE_SEARCH_VERBOSITY_DEFAULT`; overridable via the matching `NATIVE_SEARCH_*` env vars). Migrated 2026-05-17 from deprecated `x-ai/grok-4.1-fast` (initial flip to `gpt-5.4-mini`, then v3 bench at `scratch/native_search_bench_2026-05-17/comparison_v3.md` showed `gpt-5.5` medium-effort fits in ~230s under a 360s cap and produces materially deeper research; supersedes the v2 mini verdict). Gated by `NATIVE_SEARCH_ENABLED`. **Note**: donated key currently blocks OpenAI native search via data-policy guardrail; calls bill to personal `OPENROUTER_API_KEY` until resolved (see `FUTURE.md` "Resolve OAI_ANTH_OPENROUTER_KEY data-policy block"). `FallbackOpenRouterLlm` handles the fallback transparently.
- **Gemini grounded search** (`research/gemini_search.py`): real Google Search grounding via `google-genai` SDK (not OpenRouter) + `url_context` tool for specific URL reads. Gated by `GEMINI_SEARCH_ENABLED` + `GOOGLE_API_KEY`.
- **Financial data** (`research/financial_data.py`): LLM classifier routes to yfinance + FRED for financial/economic questions. Gated by `FINANCIAL_DATA_ENABLED` + `FRED_API_KEY`.
- **Prediction-market snapshot** (`research/prediction_market.py`): fans out to Polymarket Gamma, Kalshi (prefetch + local rapidfuzz match), and Manifold concurrently; aggregates the top matches into a benchmarking-safe research blurb. Gated by `PREDICTION_MARKETS_ENABLED` + a benchmarking guard (the snapshot is suppressed in `is_benchmarking=True` runs to avoid data leakage). **OFF by default in prod workflows** — flip on after a smoke + medium backtest gate.

**Second-pass gap-fill** (`research/targeted.py` `run_gap_fill_pass`): always-on when `GAP_FILL_ENABLED` + `GOOGLE_API_KEY` are set. Two stages: Gemini analyzer identifies up to `GAP_FILL_MAX_GAPS` factual gaps → parallel grounded-Gemini searches resolve each. Soft-fails (returns `""`) on any error.

**Production workflows** (`.github/workflows/run_bot_on_{tournament,metaculus_cup,minibench}.yaml`, `test_bot.yaml`) set `NATIVE_SEARCH_ENABLED=true`, `GEMINI_SEARCH_ENABLED=true`, `FINANCIAL_DATA_ENABLED=true`, `GAP_FILL_ENABLED=true`. `PREDICTION_MARKETS_ENABLED` is not yet on by default. So in prod the active stack is AskNews + OpenAI native search (`gpt-5.5` medium-effort + verbosity=low, 360s) + Gemini grounded + financial-data (when classified as financial) + always-on Gemini gap-fill, with the prediction-market provider available behind its env flag.

### Prompts (`metaculus_bot/prompts.py`)

- `_benchmarking_warning`, `_forecasting_window_str`, `web_research_prompt`.
- Base: `binary_prompt`, `multiple_choice_prompt`, `numeric_prompt`. Each base prompt now embeds the STRUCTURED FORECAST JSON-block schema instruction (Workstream C activation) so forecaster rationales emit machine-readable blocks for `tool_runner` to consume.
- Stacking: `stacking_binary_prompt`, `stacking_multiple_choice_prompt`, `stacking_numeric_prompt`. The stacker prompts include a "Cross-model aggregation (deterministic math)" block at the top when `build_cross_model_aggregation` returns markdown.
- Conditional-stacking support: `disagreement_crux_prompt`, `targeted_search_prompt`.
- Gap-fill: `gap_fill_analyzer_prompt`, `gap_fill_search_prompt`.

## Probabilistic tools (ACTIVE)

### `metaculus_bot/probabilistic_tools/`

Reusable probability math — pooling, Beta-Binomial Bayes, percentile → parametric fits (normal/lognormal/Student-t), declared-vs-math consistency checks, Dirichlet CIs, Neg-Bin/Poisson discrete percentiles, exponential/Weibull survival, Gamma-conjugate hazard. `prob_event_before`, `poisson_at_least_one`, `linear_pool` / `log_pool` / `satopaa_extremize`, `beta_binomial_update`, `cdf_at_threshold`, `dirichlet_with_other` are wired into `tool_runner` dispatch.

Newly-added math (Workstreams D1-D3):

- **Noisy-OR** (`noisy_or.py`): rare-binary decomposition `1 − ∏(1 − pᵢ)` for combining independent failure-mode probabilities. Wired into `tool_runner`.
- **Mixture-of-normals** (`mixtures.py`): `MixtureOfNormals` / `MixtureComponent` types, `mixture_cdf`, `fit_mixture_from_percentiles` (multi-start L-BFGS-B with single-normal fallback), and `percentiles_to_metaculus_cdf_via_mixture` (constraint-enforced 201-point CDF). Schema slot lives in `structured_output_schema.NumericStructured.mixture_components`; the numeric-format router branches on it.
- **Gamma waiting-time, conditional-given-survival**: `gamma_prob_event_before` with elapsed-window split (`survival_distributions.py`) — covers the missing waiting-time fitter alongside the existing exponential / Weibull / Gamma-hazard variants.

### `metaculus_bot/tool_runner.py`

Despite the name, **not** an LLM tool-calling harness. A **deterministic probability-math post-processor** that runs on structured JSON blocks emitted by each forecaster (priors, base rates, hazards, percentiles, scenarios) and injects a "Computed quantities" section into per-forecaster rationales plus a cross-model aggregation block into the stacker prompt. Entry points `run_tools_for_forecaster` and `build_cross_model_aggregation`. Gated by `PROBABILISTIC_TOOLS_ENABLED`; both entry points no-op when the flag is unset. **Wired into production code**: `run_tools_for_forecaster` runs from `_make_prediction`, and `build_cross_model_aggregation` feeds the stacker prompts in both the STACKING and CONDITIONAL_STACKING paths. A `TOOLS_USED` marker is emitted in the comment trailer alongside the `STACKER_OUTCOME` marker so residual analysis can bucket tool-augmented vs. vanilla runs.

## Project structure

- `tests/`: Pytest suite (`tests/test_*.py`).
- `.github/workflows/`: CI (lint + test on PRs) and scheduled bot runs.
- `.env.template`: reference for required environment variables.

## Configuration & environment

- Copy `.env.template` to `.env` for local development. Never commit secrets.

### API keys & secrets — what's shared vs. personal

The bot uses several API keys; they fall into two buckets and the names don't always make this obvious. Be explicit when reasoning about routing:

- **`OAI_ANTH_OPENROUTER_KEY` — Metaculus-donated OpenRouter key (SHARED).** Despite the name, this is the *only* shared/donated credential in the bot. Metaculus provides credits to bot operators on this key for OpenAI, Anthropic, and Google models routed via OpenRouter. It has server-side allowed-providers preferences locked to `{openai, anthropic, google}`; non-listed providers (e.g. `x-ai` for Grok) 404 on it. Wrapped by `FallbackOpenRouterLlm` (`metaculus_bot/fallback_openrouter.py`) which falls back to `OPENROUTER_API_KEY` on credential / credit / allowed-providers errors.
- **`OPENROUTER_API_KEY` — operator's personal OpenRouter key.** Pays for everything the donated key can't (Grok via x-ai, Qwen, Perplexity-via-OpenRouter) plus serves as the fallback when the donated key fails.
- **`GOOGLE_API_KEY` — operator's personal Google AI Studio key.** In CI it's stored as `secrets.GEMINI_API_KEY` and surfaced as `GOOGLE_API_KEY` in the workflow env so the `google-genai` SDK picks it up. **There is NO Metaculus-donated Google AI Studio key** — Google AI Studio doesn't offer one. The grounded-search side (`research/gemini_search.py`, `research/targeted.py`) always uses this personal key. Don't confuse the OpenRouter Gemini path (which DOES have a donated route via `OAI_ANTH_OPENROUTER_KEY`) with the google-genai grounded-search path (which doesn't).
- **`METACULUS_TOKEN`, `ASKNEWS_*`, `EXA_API_KEY`, `PERPLEXITY_API_KEY`, `FRED_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` — all personal.** No shared variants. The two direct provider keys (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`) are only used if you bypass OpenRouter; most flows route through OpenRouter and don't need them.

**Toggle: `GEMINI_USE_DONATED_OPENROUTER_KEY`** (default `false`). Only affects OpenRouter Gemini routing; does NOT touch grounded search. Off by default because the donated OpenRouter key **cannot** serve Gemini: the donated account has a free-tier Google AI Studio BYOK key attached, OpenRouter forces all `google/*` traffic through that BYOK key first (not overridable per-request — confirmed against current OpenRouter docs), and `gemini-3.x-pro` has no Google free tier, so the BYOK quota is structurally 0 → every donated-key Gemini call 429s (`is_byok:true` + `FreeTier limit: 0`) and falls back to the personal key anyway (spamming the personal-key-fallback alert counter → reddening CI every run). So Gemini routes through the operator's personal `OPENROUTER_API_KEY` directly. This is a Metaculus-account-side fix (enable Cloud billing on the BYOK key's GCP project, or remove the Google AI Studio BYOK integration so native OpenRouter Google credits are used) — flip this to `true` only after that's resolved. The prod workflow yamls (`.github/workflows/run_bot_on_*.yaml`, `test_bot.yaml`) pin it to `'false'` explicitly. See `metaculus_bot/fallback_openrouter.py:should_route_via_donated_key` and `FUTURE.md` "Gemini on the donated OpenRouter key blocked by free-tier BYOK".

**Diagnosing auth / credit errors**:

- OpenRouter 401/402/credit error on a Gemini call → if the toggle is ON, suspect `OAI_ANTH_OPENROUTER_KEY`; if OFF (default), suspect `OPENROUTER_API_KEY`.
- OpenRouter 401/402 on an OpenAI or Anthropic call → suspect `OAI_ANTH_OPENROUTER_KEY` first (donated key is always tried first for those providers), then `OPENROUTER_API_KEY` if the wrapper doesn't fall back.
- OpenRouter 401/402 on Grok / Qwen / Perplexity → always `OPENROUTER_API_KEY` (donated key 404s on these providers).
- `google-genai` 401 / quota / API-key-invalid error → always `GOOGLE_API_KEY` (no donated path).
- `403 forbidden / moderation` or `429 rate limit` → not a key issue; the wrapper deliberately doesn't fall back on these. See `should_retry_with_general_key`.

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
- Numeric: `NumericDistribution`, `Percentile`. We subclass `NumericDistribution` as `PchipNumericDistribution` (`numeric/pchip_processing.py:217`) to override `.cdf` with our pre-computed 201-point PCHIP CDF; forecasting-tools' built-in CDF builder is only used on the fallback path.

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

### Checking OpenRouter credits

The donated Metaculus OpenRouter key (`OAI_ANTH_OPENROUTER_KEY`) is shared and rate-limited; check burn-rate periodically:

- **`make check_credits`** — prints `limit / limit_remaining / usage` for both `OAI_ANTH_OPENROUTER_KEY` (donated) and `OPENROUTER_API_KEY` (personal). Pass `ARGS="--key donated"` to check just one.
- **Raw curl backup** (avoid putting the key on disk; pull from `.env`):

  ```bash
  curl -s -H "Authorization: Bearer $OAI_ANTH_OPENROUTER_KEY" \
    https://openrouter.ai/api/v1/auth/key | jq
  ```

- Never paste the full key into chat or commit it. `.env` is gitignored.

### Function-scoped imports in `forecaster.py`

`forecaster.py` keeps a handful of `from x import y` statements inside functions instead of at module scope, each tagged `# noqa: PLC0415  # function-scoped: see AGENTS.md`. Two reasons drive this:

1. **Optional dependency loading.** `prediction_market_provider` pulls in `rapidfuzz`; `tool_runner`, `numeric_format_router`, etc. only matter when their corresponding feature flag is on. Importing them at function scope keeps the cold-start path lean and avoids surprising errors when an optional dep isn't installed.
2. **Ruff auto-formatter behavior.** When a usage edit is staged separately from the import edit (common during refactors and subagent dispatches), Ruff's auto-formatter strips the now-unused top-level import between cycles. Function-scoped imports survive this because the symbol is referenced in the same statement block.

Don't hoist these to the top of `forecaster.py` without first checking that both reasons no longer apply.

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
