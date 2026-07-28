# Metaculus Forecasting Bot — Agent Guidelines

General coding guidelines (style, testing, error handling, etc.) are in `~/.claude/CLAUDE.md`.
This file covers **repo-specific** context only.

## ⚠️ Cost discipline — every credit spend goes through the operator

**Any command that hits live LLM / research APIs spends real money (OpenRouter
credits, AskNews, Exa, Perplexity, Google). NEVER launch one autonomously — ask
first, every time, even after a clean build / passing tests / clean `/forge`.**
This is a hard gate, not a courtesy: a single broad run can burn meaningful
credits, and `--mode test_questions` also **publishes comments to Metaculus**
(a visible external action that pings nothing but is hard to retract).

**The gate is on the SPEND, not on the mechanism.** It covers anything that
causes a paid call, whoever or whatever ultimately makes it — a local command, a
GitHub Actions dispatch, an edit that flips a schedule, or a script that wraps
any of those. An agent proposes and prices the run; the operator decides. There
is no clean-gates exemption and no "small enough to skip asking" threshold: a
$2-3 run still goes through the operator. The rule holds even when a paid run is
the only way to verify the work, in which case say so and stop.

**What the gate forbids is an agent DECIDING to spend.** An explicit instruction
from the operator IS the approval: when they say to fire a run already discussed
("fire off the benchmark run we agreed to"), run it and don't re-ask. Approval is
per-run and doesn't carry forward — one "go" is not standing authorization for the
next run, nor for re-running the same one after further changes.

**Paid runs are a final pre-merge check, not part of the verification loop.** The
smoke test exists to be fired ONCE, deliberately, when a change is otherwise
finished and about to merge. Its small per-run cost is exactly the trap: an agent
that folds it into a normal check-my-work loop fires it several times a session
and spends real money for no added signal. The free gates are the loop —
`make test`, `make lint`, `make typecheck` — and unit or integration coverage is
how an agent earns confidence. A paid run is the operator's last step, not an
agent's reassurance.

Paid / external-effecting — ask before each:

- `uv run python main.py` / `make run` in any live mode (`--mode test_questions`,
  `tournament`, `metaculus_cup`, `minibench`) — spends API credits AND publishes.
- `make backtest_smoke_test` / `_small` / `_medium` / `_large` — spends on every
  forecaster + research call, plus the `LEAKAGE_DETECTOR_MODEL` screen. No
  publish, but real money; the question counts are the `--num-questions` values
  in the Makefile and scale with the target name.
- `make backtest_with_cache` — `--research-dir` replays archived research instead
  of fetching it (`_load_research_from_archive`, `backtest.py`), and the live
  ensemble still forecasts every question, so forecaster spend is real. A
  question with no archived record falls back to live research, which the run
  logs as a warning.
- `make ablation_*` (`qa_research`/`smoke`/`small`/`medium`) — real research +
  forecaster spend. `ablation_score` is the free exception: `--stages score`
  hydrates every artifact off disk and makes no provider call.
- `make benchmark_run_*` — deprecated (`community_benchmark.py` baseline scoring
  is broken), but the `run` / `custom` modes still fan the real ensemble over
  real questions. Use `make backtest_*` instead; `make benchmark_display` is the
  free view-only mode.
- `make test_live` — the only suite that leaves the network. It pins a `:free`
  OpenRouter slug, so dollar spend is near-zero, but the calls are real and need
  a key. Ask anyway.
- **GitHub Actions runs of any bot workflow**, which spend exactly as a local run
  does and publish to Metaculus. `test_bot_basic.yaml` (one numeric question,
  ~$2.60) and `test_bot.yaml` are `workflow_dispatch`-only, so they fire when
  somebody chooses to fire them — that choice is the operator's. The three
  `run_bot_on_*.yaml` prod workflows are additionally on `schedule:` crons.
  Never dispatch one, and never edit a `schedule:` block or a research/model flag
  in a way that adds runs or raises per-run cost, without the operator's say-so.
  (`gh` needs `--repo No-Stream/metaculus-bot` here: `origin` is the fork,
  `upstream` is the Metaculus template, and no default repo is set, so a bare
  `gh workflow` command silently targets upstream. Dispatch mechanics for the
  smoke test are in `docs/operations.md`.)
- Anything invoking research providers or the ensemble against real questions,
  including one-off scripts an agent writes to do so.

Free / safe — run freely:

- Gates and formatting: `make test`, `make test_fast`, `make test_e2e`,
  `make lint`, `make format`, `make typecheck`, `make typecheck_ty`, `make cov`,
  `make audit`, `make precommit*`. The whole suite is self-contained: the `e2e`
  marker means full-pipeline with MOCKED LLMs, and the autouse
  `_block_network_egress` fixture (`tests/conftest.py`) raises on any non-loopback
  connect, so no selected test can reach a paid API. `addopts` deselects only the
  `live` marker.
- Read-only Metaculus / artifact pulls: `make sync_all` and its parts
  (`sync_research`, `sync_telemetry`, `sync_raw_research`, `download_*`,
  `backfill_*`), the `performance_analysis` package, `make score_ghosts` (its
  `--tournament` pull is Metaculus-only), `make close_margin_watch`,
  `make ablation_score`, `make benchmark_display` (views old runs). These hit only
  the Metaculus API and GitHub artifacts.
- `make check_credits` — reads both OpenRouter key balances.

When verification needs a paid run, surface the exact command + rough cost and
let the operator decide. Unit/integration coverage is the default proof of
correctness; live runs are opt-in.

## Repo-Specific Overrides

- **Python**: 3.12+ (see `pyproject.toml` `requires-python`)
- **Package manager**: **uv** (migrated off Poetry 2026-06). `uv.lock` is the lockfile; there is no `poetry.lock`. The package is installed editable via `uv sync`, so no `PYTHONPATH=.` is needed.
- **Build backend**: `uv_build`, flat layout (`metaculus_bot/` at repo root, `module-root = ""`).
- **Formatter**: Ruff with 120-char line length (not Black)
- **Type checker**: **basedpyright** (standard mode, must stay at 0 errors); `ty` is a secondary/advisory checker. Promoting the core pipeline to strict is a tracked follow-on (see `FUTURE.md`).
- **Testing**: Pytest + `pytest-asyncio`; all tests are self-contained (no API keys needed in CI)
- **NEVER use `pip` or `poetry`** — both are blocked in this environment. Use `uv add <pkg>` / `uv sync` / `uv run <cmd>`.

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

1. **Research** — `run_research` (`forecaster.py`) fans out providers in parallel via `_select_research_providers` / `_run_providers_parallel`. Always-on **gap-fill second pass** (`run_gap_fill_pass` in `research/targeted.py`, called from `run_research` in `research/orchestrator.py`) identifies factual gaps and resolves them via parallel OpenAI native web searches (`GAP_FILL_RESOLVER_MODEL` at `GAP_FILL_RESOLVER_REASONING_EFFORT`, via OpenRouter on the donated key).
2. **Forecaster fan-out** — N forecaster LLMs run in parallel via `_forecaster_with_soft_deadline` (a per-forecaster cap of `FORECASTER_SOFT_DEADLINE`) → `_make_prediction` → type-specific runner (binary/MC/numeric).
3. **Min-forecasters guard** (in `_research_and_make_predictions`, `forecaster.py`) drops the question if fewer than `MIN_FORECASTERS_TO_PUBLISH` returned a valid prediction. A **single-forecaster short-circuit** just below the guard skips spread computation + stacking when exactly one forecaster survived (the `spread_metrics` helpers require ≥2 predictions and raise otherwise) and hands the lone prediction to the aggregator. Past the guard, every question logs `FORECASTERS_SURVIVED: question=... survived=n/N models=...` at INFO — the positive counterpart to the per-run `FORECASTER_DROPS` marker, and the only place a run log states the survivor count. It is load-bearing precisely because the floor is low: a degraded publish exits zero and the failure-path "Only n/N forecasters succeeded" line never fires, while the comment-side `FORECASTERS_USED` marker never reaches stdout, so without this line a thinned ensemble reads identically to a full one. `models=` names the survivors (read off each prediction's own `Model:` prefix, not the configured roster) so survivors can be diffed against drops from the log alone. Harvested into the telemetry archive as `forecasters_survived` (`scripts/telemetry/markers.py`).
4. **Aggregation** — see CONDITIONAL_STACKING below.

### Ensemble

`FORECASTER_LLMS` in `metaculus_bot/llm_configs.py` is the authoritative roster. It rotates frequently, so read it rather than trusting any copy — this file deliberately does not name the current members outside the dated history below. The standing design is latest-per-vendor: the newest frontier reasoning model from each major vendor, one slot each. Do NOT hardcode model names outside `llm_configs.py`. Provider: OpenRouter with automatic key fallback. Dropping grok (x-ai) ended routine personal-key forecaster spend: the only forecaster still billing `OPENROUTER_API_KEY` is the Google slot pinned by the `DONATED_KEY_BLOCKED_GOOGLE_MODELS` blocklist (`fallback_openrouter.py`); the rest route via the donated key. Alongside the roster shrink, `MIN_FORECASTERS_TO_PUBLISH` (constants.py) was lowered 3→2→1 over 2026-07-20; the operator accepts publishing on a single surviving forecaster (median-of-1 = that forecast), with `_research_and_make_predictions` short-circuiting the n==1 case before spread computation + stacking (the `spread_metrics` helpers require ≥2 predictions and raise otherwise). Exception-driven drops stay CI-visible via the alertable counter, so a degraded single-forecaster publish still reddens CI rather than silently withholding the question.

Roster-change history: 2026-07-15 — Fable-5 joined the forecaster roster (was stacker-only; stacking disabled in prod made it idle) and opus-4.6 retired, keeping n=6 at a 2 Anthropic / 2 OpenAI / 1 Google / 1 xAI balance. 2026-07-20 (first change) — Fable-5 PULLED from the forecaster roster and the stacker after it returned `message.content=None` on 4/4 attempts for Q14333's numeric forecast + a truncated no-JSON-block output on Q578 in the 2026-07-19 test_bot run (suspected content classifiers refusing certain question content — fast deterministic empty completions, not timeouts); opus-4.7 took the slot, keeping n=6. Reconsidering fable-5 is a tracked follow-up (FUTURE.md). 2026-07-20 (second change, current) — dropped from 6 to the 3-member latest-per-vendor triple, removing gpt-5.5, opus-4.7, and grok-4.5. Two adversarially-verified analyses (`scratch/ensemble_3member_audit_2026-07-20/` + `scratch/ensemble_power_model_2026-07-20/`) found the triple non-inferior on binary/MC and only a fragile numeric lean toward the full roster (+3.24, 95% CI [-2.5, +9.1], P(loss>1pt/Q)=0.80, driven by 2 questions); accepted as a ship-and-watch bet — see FUTURE.md "Frozen-triple numeric watch". This second change supersedes the first as the config-era boundary for residual analysis.

Support models (also in `llm_configs.py`):

- **Stacker**: `STACKER_LLM` — the Anthropic slot, since 2026-07-20 when fable-5 was pulled from both roles (see the 2026-07-20 roster-change note above; it uses effort-based adaptive thinking rather than a max_tokens budget; the stacker chain remains configured but prod-disabled. Was `claude-fable-5` from 2026-07-07 to 2026-07-20). Falls back to `STACKER_FALLBACK_LLM`, deliberately a different vendor so an Anthropic stall doesn't take both attempts down; it stays one effort tier below the primary because it fires late on the critical path under `STACKER_FALLBACK_SOFT_DEADLINE`. Both are `allowed_tries=1` — on stall, we fall back rather than burn budget retrying the same provider. OpenRouter's full effort enum (live-verified 2026-07-15): none/minimal/low/medium/high/xhigh/max; the xhigh slots are the Anthropic forecaster and the stacker. The OpenAI forecaster dropped xhigh→high on 2026-07-20 evening (it is ~70% of forecaster reasoning spend and the high→xhigh premium is unmeasured — the Anthropic slot keeps xhigh as the remaining premium bet; see FUTURE.md "Price the high→xhigh reasoning-effort premium"). The Google forecaster has no xhigh tier and runs at provider defaults. "max" is Anthropic-only (one tier above xhigh; OpenAI's ceiling is xhigh and rejects max upstream) — held back on the Anthropic slots for latency.
- **Disagreement analyzer**: `DISAGREEMENT_ANALYZER_LLM` (low-effort crux extractor; quality drives targeted-search query, running under `CRUX_SOFT_DEADLINE`; sol→terra 2026-07-17 per the blind role audit — terra 2nd, sol 3rd, at −49% cost, and the role fires rarely with stacking disabled in prod).
- **Summarizer / researcher**: `SUMMARIZER_LLM` (aliased as `RESEARCHER_LLM`; low effort, deterministic; sol→terra 2026-07-18 operator decision — AskNews is an auxiliary source per the content audit (16% unique content), the role audit had sol over terra only at "MARGINAL EDGE", and 4/5 audited briefing failures were prompt-era not model-tier; terra −43% cost, ~50s vs ~118s wall). The summarizer prompt carries the 2026-07-18 AskNews-audit rules: a hard per-article relevance gate (off-topic articles DROPPED to a one-line "Screened out as not decision-relevant" list), recency-first ordering (lead with the newest resolution-relevant facts, don't mirror the raw Historical/Recent input structure), supersession + quote-the-deadline-inputs arithmetic transparency, an evidence-age disclosure opening the briefing ("Newest directly-relevant article: ..."), and a proportionality rule (length tracks decision-relevant content, not article count).
- **Parser**: `PARSER_LLM` (low effort, deterministic; a capability-saturated extraction task, so it sits on the cheapest tier that saturates it — the per-token comparison behind that choice is in the `llm_configs.py` comment).

### CONDITIONAL_STACKING (default)

`AggregationStrategy.CONDITIONAL_STACKING` (set in `main`, `metaculus_bot/cli.py`). Note: this describes the code default — stacking is currently DISABLED in prod, since all four prod workflow yamls set `BINARY_STACKING_ENABLED`/`MC_STACKING_ENABLED`/`NUMERIC_STACKING_ENABLED` to `'false'` (the chain stays live in backtests/ablation). Behavior:

- Compute spread across the N forecasters via `spread_metrics.compute_spread`.
- If spread ≤ threshold → return **MEDIAN** of raw per-model predictions (base-combine via `_base_combine`, `aggregation_pipeline.py`).
- If spread > threshold → extract the **disagreement crux**, run **targeted search** (OpenAI native search on the same `NATIVE_SEARCH_*` model / effort / verbosity / timeout settings as the native-search provider, described under Research providers below), then invoke the **stacker LLM** with the full base-model reasonings + targeted research (`stacking.run_stacking_{binary,mc,numeric}`).
- Stacker fallback chain: primary `STACKER_LLM` under `STACKER_SOFT_DEADLINE` → `STACKER_FALLBACK_LLM` under `STACKER_FALLBACK_SOFT_DEADLINE` → MEDIAN. (`_stacking_aggregate`, `aggregation_pipeline.py`.)

Thresholds (all in `metaculus_bot/constants.py`):

- Binary: probability range (max − min) ≥ `CONDITIONAL_STACKING_BINARY_PROB_RANGE_THRESHOLD`.
- MC: max per-option spread ≥ `CONDITIONAL_STACKING_MC_MAX_OPTION_THRESHOLD`.
- Numeric: normalized percentile spread ≥ `CONDITIONAL_STACKING_NUMERIC_NORMALIZED_THRESHOLD`.

### Clamps / bounds

- **Binary**: `[BINARY_PROB_MIN, BINARY_PROB_MAX]` (`constants.py`). Applied per-model in `forecaster_runners.py` and on stacker output in `stacking.py`. Median/mean of already-clamped values stays in-bounds, so no post-aggregation clip needed.
- **MC**: `[MC_PROB_MIN, MC_PROB_MAX]` (`constants.py`), set to match ft 0.2.92's `PredictedOptionList` validator, which clamps every option on construction — matching bounds makes it a no-op and removes publish-time `ValueError` risk on many-option ballots. Drift-free clamp-then-renormalize via `clamp_and_renormalize_probs` (`mc_processing.py`), applied BEFORE every `PredictedOptionList` construction and re-applied idempotently by `clamp_and_renormalize_mc` (`numeric/utils.py`); the repair pass keeps floored options from dividing back below the floor.
- **Numeric CDF**: a `PCHIP_CDF_POINTS`-point grid with every bin's step held between `MIN_CDF_PROB_STEP` and `MAX_CDF_PROB_STEP` (`numeric/config.py`; `grid_step_constraints` rescales both for off-standard grid sizes, relaxing the max on coarse discrete grids). Open bounds are a **one-sided** constraint per tail, not a box: open lower bound → `cdf[0]` is floored at a *minimum* of 0.001 (a required minimum positive mass, NOT a cap on below-bound mass); open upper bound → `cdf[-1]` is ceilinged at 0.999. There is no cap on out-of-bound mass — a distribution can legitimately place e.g. 78% of its mass below an open lower bound. Out-of-bound mass is expressed by placing percentile values beyond the displayed range (values are not clamped on open bounds) so `F(bound)` interpolates to the intended fraction; it's bounded only by min/max-step feasibility (~0.99). Closed bounds are pinned to `{0.0, 1.0}`. Enforced in `numeric/pchip_cdf.py` `generate_pchip_cdf` with aggressive repair + `safe_cdf_bounds` redistribution on max-step violations.
  - **Repair-tier WARN signals are effectively dead code on real forecasts.** The five logged numeric repair signals — `pchip_aggressive`/aggressive enforcement, `clamp_frac`/`clamp_dist` (bounds-clamp corrections), `violated_steps_frac`, and `ramp_smoothing_delta` — never fire on real model output: `generate_pchip_cdf`'s uniform-mixture construction pre-enforces the min-step before any of those repair tiers is reached (0 of 1182 archived numeric forecasts fired any of them; the one degenerate case raises `pchip_failed` instead). Keep the guards — they defend against pathological inputs — but their absence in logs carries no information, and they must NOT be used as model-quality features (verified 2026-07-15; receipts in `scratch/coherence_2026-07-15/synthesis.md`).

### Numeric pipeline (percentiles → PCHIP CDF)

Each forecaster emits the `STANDARD_PERCENTILES` set as plain text (`numeric/config.py` owns the list and derives both `EXPECTED_PERCENTILE_COUNT` and the `STANDARD_PERCENTILES_CSV` label string the prompts interpolate — never restate the percentiles elsewhere; prompt example in `prompts.numeric_prompt`). Per-model stages:

1. Value extraction ladder (`value_extraction.py` `extract_numeric`): deterministic block-first parse of the fenced ```json STRUCTURED FORECAST block into `list[Percentile]`, with json-repair salvage and the parser LLM (`parse_structured`) only as rung-3 salvage. Returns exactly the canonical percentile set (or raises `ValueExtractionError`).
2. `sanitize_percentiles` (`numeric/pipeline.py`): filter to the canonical set, validate, sort, spread count-like clusters, jitter duplicates, clamp to bounds, ensure strictly increasing, optionally widen tails.
3. `widen_declared_percentiles` (`numeric/tail_widening.py`): bound-aware stretch of distance-from-median by `TAIL_WIDEN_K_TAIL`, whose default is the identity value — widening only kicks in when a caller raises it. `TAIL_WIDEN_SPAN_FLOOR_GAMMA` gates span-floor enforcement on a positive value and defaults off. Both knobs are configurable per-call.
4. `build_numeric_distribution` → `generate_pchip_cdf_with_smoothing` produces the `PCHIP_CDF_POINTS`-point PCHIP CDF → ramp smoothing for min-step → validation. On failure: `create_fallback_numeric_distribution` delegates CDF build to forecasting-tools.
5. **Discrete integer snapping**: if a majority of forecasters vote DISCRETE, snap the distribution to integers.
6. **Unit-mismatch guard** (`numeric/validation.py` `detect_unit_mismatch`): withholds the prediction if values look off by orders of magnitude.

### Value extraction ladder (`value_extraction.py`)

Since 2026-07-10, forecast values (binary probability, MC option probabilities, numeric percentiles) come out of the fenced ```json STRUCTURED FORECAST block via a deterministic four-rung ladder — the block is the authoritative source, and the parser LLM survives only as a salvage rung. `extract_binary`, `extract_mc`, and `extract_numeric` share the same driver and are called by both `forecaster_runners.py` (per-model forecasts) and `stacking.py` (stacker output), so both flows share one extraction path:

1. **block** — deterministic parse of the fenced ```json block (`parse_structured_block`: `json.loads` + Pydantic validation against `BinaryStructured` / `MultipleChoiceStructured` / `NumericStructured`). This absorbs the old numeric_format_router F5 block-lift: when the schema-valid numeric block carries a `declared_percentiles` dict, it is lifted directly into `Percentile(percentile=..., value=...)` objects.
2. **repair** — deterministic JSON repair (`json_repair`) of a malformed fenced block, or a balanced-braces scan of the rationale's trailing `_TAIL_SCAN_CHARS` when no fence survived.
3. **llm** — the existing LLM parser (`parse_structured`) run over the full rationale as a last-resort salvage. Post-rung validation stays strict (bounds, the full canonical percentile set for numeric, option-set match for MC) so the parser can never smuggle in a fabricated value.
4. Raise `ValueExtractionError` — the numeric forecaster (or the stacker) has no usable value source; the caller drops or soft-fails the forecaster exactly as parser failures propagated before the ladder.

Every successful extraction emits one `EXTRACTION_RUNG: question=... model=... qtype=... rung=... block_present=...` INFO log line — this is the drift-signal telemetry that superseded the deleted shadow-divergence comparison. Watch `rung=llm` salvages and `block_present=false` in prod logs; those are the two symptoms that indicate a forecaster stopped emitting a well-formed block. A sibling WARN, `OPEN_BOUND_PILING: question=... model=... bound=... bin_mass=... declared_edge=... bound_value=...` (`numeric/diagnostics.py`, threshold `OPEN_BOUND_PILING_THRESHOLD` in `numeric/config.py`), fires when a model piles at least that fraction of mass on the terminal displayed bin of an *open*-bound numeric question without declaring any percentile beyond the edge — the "crammed the open ceiling" failure mode fixed 2026-07-12 by rendering nominal/displayed bounds in the numeric prompts (`nominal_bounds` in `numeric/utils.py`). It takes the pre-resample model-declared percentiles explicitly (the discrete resample overwrites `prediction.declared_percentiles` with a grid pinned to the raw bounds, which would defeat the above-edge exemption). Since 2026-07-12 all four workflow runs tee stdout+stderr to a `run_logs/` artifact (`research-<run_id>` for the three prod workflows, `logs-<run_id>` for test_bot), so this telemetry is durably grep-able after the fact. Runs also emit OpenRouter credit telemetry (`metaculus_bot/credit_telemetry.py`, wired in `cli.py`): `CREDIT_BALANCE: key=... phase=start|end remaining=... usage=...` per key, `CREDIT_SPEND: key=... run_delta_usd=... remaining=...` at end of run, and a `CREDIT_FLOOR_BREACH` WARN when the donated key drops below `OPENROUTER_CREDIT_FLOOR_USD` (constants.py) — a breach lets the run complete/publish normally, then exits non-zero as a refill reminder ONLY when credit alerting is active (see below).

**Credit alerting is suppressed until 2026-09-10.** The operator is funding the rest of the season out of pocket, so a drained donated key is expected rather than broken, and until `CREDIT_ALERT_RESUME_DATE` (constants.py, `2026-09-10`, a few days past `TOURNAMENT_END_DATE`) a credit shortfall must not redden CI. Both paths that turned a shortfall into `sys.exit(1)` are gated on `credit_alerts_active()` (reads the system clock at CALL time, accepts an injected `today` for tests, so the resume needs no redeploy): (1) the floor breach in `cli.py` skips the exit and logs an INFO line naming the resume date; (2) the credit-caused donated→personal key fallbacks are tracked in `_credit_key_fallback_count`, a strict subset of the all-causes `_generic_key_fallback_count` (same relationship `_donated_404_fallback_count` has), and `cli.py` subtracts that subset from `alertable` while suppressed. All three counters are incremented in one place, `record_donated_key_fallback` (`fallback_openrouter.py`), and each event is counted exactly once: generic adds it, at most one subset subtracts it. The accounting block there must stay free of any `await` after the threaded probe — `+=` on a module global is interruptible between bytecodes, so an await would let N forecasters failing on one dry key race the increment and take a degraded run green. **Every other fallback cause stays fully alertable** — 401 invalid/disabled key, 404 "no allowed providers", 429 rate limit, guardrail/data-policy — as does all bot-side degradation. No log line is silenced: all `CREDIT_*` markers and every `PAID PERSONAL-KEY FALLBACK` WARN still fire, and only the exit status plus the alertable arithmetic change. Credit classification lives in one place (`_is_credit_failure` / its public form `is_credit_caused_error` in `fallback_openrouter.py`) so the retry decision and the counter classification can't drift apart. After the resume date, behavior is exactly what it was before the suppression.

**A dry donated key returns 403, not 402, and only a genuinely DRAINED one is suppressed (2026-07-26).** OpenRouter reports a breached per-key spend cap as HTTP 403 with the message `Key limit exceeded (total limit)`; litellm has no 403 branch for OpenRouter, so it always arrives as a bare `APIError` carrying `"code":403` in the body. The old negative rule vetoed any message containing "403" (written for content moderation), so a drained donated key never fell back — a tournament run lost 2 of 3 forecasters, native search, the AskNews summarizer, the financial classifier, prediction-market keyword extraction, and both gap-fill passes while the operator's funded personal key sat idle. The classifier now matches `key limit exceeded`; the **full phrase is load-bearing** because the shorter `limit exceeded` is a substring of `rate limit exceeded: free-models-per-day` and would exempt every 429 from alerting. Because a key Metaculus revoked or re-capped to zero produces the identical 403 text, `credit_telemetry.classify_donated_key_state` probes the free read-only `/auth/key` endpoint once per run (cached; bounded by `DONATED_KEY_PROBE_TIMEOUT_S`, which httpx applies PER NETWORK OPERATION rather than as a cap on elapsed time — the hard total cap is the `asyncio.wait_for` that `record_donated_key_fallback` wraps around the threaded probe, which unblocks the caller on schedule but leaves a trickling probe running orphaned) and `is_suppressible_credit_error` exempts ONLY the `drained` verdict — `zeroed` (cap set to 0), `revoked` (401/404), `funded` (money remains, so the failure wasn't about credit), and `unknown` (probe failed) all keep CI red, and the verdict rides the end-of-run summary as `donated_key=<state>`. That probe belongs to the AMBIGUOUS spend-cap 403; the unambiguous 402 / insufficient-credit family is suppressed without probing at all (deliberate — it predates the discriminator, so an unreachable `/auth/key` can't change its long-standing behavior). Fallback ROUTING reads the status the provider reported (`llm_retry.llm_status_code`, an int already on the exception) and never a live balance: a reported 403 falls back only on the spend-cap phrase or route-scoped wording, a reported 402 always falls back, and a statusless exception falls back on text alone. The probe is consulted for ALERTING only, so a stale `funded` read can never strand the ensemble on a dry key. Related hardening: a bare `402` substring is now trusted only when the body doesn't look like a moderation refusal, since OpenRouter replays up to ~100 chars of our own prompt as `flagged_input` and forecasting prompts routinely contain dollar figures. Full detail in `docs/operations.md` "What a dry donated key actually returns".

Discrete-integer snapping is decided separately by the C3 block-read in `forecaster_runners.run_numeric_forecast` — it reads `NumericStructured.outcome_type` directly before falling back to a parser LLM call for `OutcomeTypeResult`, and the ladder does not touch that path.

**Ensemble aggregation** (`aggregate_numeric` in `numeric/utils.py`): pointwise **in CDF space** — concatenate each model's `PCHIP_CDF_POINTS`-point CDF, groupby value, mean or median the probabilities, then `_postprocess_ensemble_cdf` re-pins endpoints, enforces monotonic + min-step, resamples via PCHIP for discrete questions. Not percentile-space averaging.

### Research providers

Orchestration in `_select_research_providers` (`research/orchestrator.py`).

**Primary provider** — exactly one, chosen by priority in `choose_provider_with_name` (`research/providers.py`):

1. **AskNews** if `ASKNEWS_CLIENT_ID` + `ASKNEWS_SECRET` are set (the prod case): dual-phase search (HOT + HISTORICAL), rate-limited with retry/dedup (`_asknews_provider`, `research/providers.py`).
2. **Exa.ai SmartSearcher** if `EXA_API_KEY` set (fallback when AskNews absent): generic rundown (`_exa_provider`, `research/providers.py`).
3. **Perplexity direct** if `PERPLEXITY_API_KEY` set: `_perplexity_provider` in `research/providers.py`. Prompt explicitly requests prediction-market consideration unless benchmarking.
4. **Perplexity via OpenRouter** if `OPENROUTER_API_KEY` set: same function, `use_open_router=True`.
5. Empty stub.

In production (AskNews creds present) Exa/Perplexity/OpenRouter do NOT run. They're priority-ordered fallbacks, not peers. `RESEARCH_PROVIDER=<name>` forces a specific one.

**Additional providers run in parallel on top of the primary** (each independently gated):

- **OpenAI native search** (OpenRouter web plugin, `research/providers.py`): configured entirely by `NATIVE_SEARCH_DEFAULT_MODEL`, `NATIVE_SEARCH_REASONING_EFFORT_DEFAULT` (passed as `reasoning={"effort": ...}`), `NATIVE_SEARCH_VERBOSITY_DEFAULT` (as `extra_body={"verbosity": ...}`), and `NATIVE_SEARCH_TIMEOUT` — each overridable via the matching `NATIVE_SEARCH_*` env var. Model migrated 2026-07-09 to `gpt-5.6-sol`, then 2026-07-17 to `gpt-5.6-terra` per the blind research-role audit (`scratch/research_role_audit_2026-07-17/` — terra 1st, sol 2nd, luna 3rd; verdict "MARGINAL EDGE", terra at −42% cost); effort has been low since 2026-05-20 (latency; see constants.py). Gated by `NATIVE_SEARCH_ENABLED`. **Note**: the donated-key data-policy block has been RESOLVED (verified 2026-06-25 by a live call returning 200 with grounding). OpenAI native search now routes through and bills the donated `OAI_ANTH_OPENROUTER_KEY`; `FallbackOpenRouterLlm` still falls back to personal `OPENROUTER_API_KEY` on credential / credit errors.
- **Gemini grounded search** (`research/gemini_search.py`): real Google Search grounding via `google-genai` SDK (not OpenRouter) + `url_context` tool for specific URL reads. Gated by `GEMINI_SEARCH_ENABLED` + `GOOGLE_API_KEY`.
- **Financial data** (`research/financial_data.py`): LLM classifier routes to yfinance + FRED for financial/economic questions. Gated by `FINANCIAL_DATA_ENABLED` + `FRED_API_KEY`.
- **Prediction-market snapshot** (`research/prediction_market.py`): fans out to Polymarket Gamma, Kalshi (prefetch + local rapidfuzz match), Manifold, and PredictIt (prefetch + local fuzzy match with query-aware contract selection; added 2026-07-12, folded under the same flag — no separate PredictIt toggle) concurrently; aggregates the top matches into a benchmarking-safe research blurb. Each match carries liquidity signals (total volume, open interest, Manifold bettor count) rendered in the snapshot table with a thin/decent/deep (or thin/decent/high) signal label; PredictIt exposes no liquidity fields and renders `no-liquidity-data`. The prompt-side "strong evidence" market clause lives in one shared helper (`prompts.py` `_strong_evidence_market_clause`) used by all three question types and instructs forecasters to weight market signals by that liquidity label. Gated by `PREDICTION_MARKETS_ENABLED` + a benchmarking guard (the snapshot is suppressed in `is_benchmarking=True` runs to avoid data leakage). **ON in all prod workflows** as of commit `3c12dbe` (the `is_benchmarking=False` prod path means the benchmarking guard doesn't suppress it there). Because that guard hard-disables the provider during backtests, its forecasting value can't be measured by the standard `make backtest_*` gate — it was validated via manual `test_bot.yaml` prod-mode runs + opt-in live integration tests instead. Research section header: `## Prediction Market Snapshot`.
- **Resolution-source fetcher** (`research/resolution_source.py`): deterministically extracts URLs from a question's resolution_criteria + fine_print (markdown + bare URLs, order-preserving dedup), skip-filters Metaculus self-refs / FRED (financial_data owns) / Yahoo-ticker (yfinance owns) URLs, caps the post-filter list at `RESOLUTION_SOURCE_MAX_URLS`, fetches in parallel (per-host politeness serialization, browser-like headers via shared `research/http_fetch.py`) and extracts main content with trafilatura — zero LLM calls. SSRF-hardened: `is_public_http_url` preflight plus a connect-time `FilteringResolver` (the resolver, not the preflight, is the real DNS-rebinding boundary) and a bounded manual redirect loop re-guarding each hop. Per-URL `FetchStatus` retains blocked/js_wall as the seam for a future Tier-2 LLM fetch pass. Per-URL truncation appends a `[truncated at N chars — full source at URL]` marker so forecasters can tell the snapshot is partial; the section formatter appends an analogous `[N additional source(s) omitted — section budget]` note when later sections are dropped. Hard `""` when `is_benchmarking=True` (same leakage rationale as prediction_market). Gated by `RESOLUTION_SOURCE_ENABLED` — **ON in all four workflows** (test_bot.yaml plus the three `run_bot_on_*.yaml` prod workflows) as of 2026-07-10 after live validation. Research section header: `## Resolution Source Snapshot`.

**Second-pass gap-fill** (`research/targeted.py` `run_gap_fill_pass`): always-on when `GAP_FILL_ENABLED` is set (the resolver migrated off Google as of 2026-06-25, so `GOOGLE_API_KEY` is no longer required). Two stages: the non-grounded OpenRouter analyzer `GAP_FILL_ANALYZER_MODEL` identifies up to `GAP_FILL_MAX_GAPS` factual gaps, ranked by decision-relevance so the trailing slot holds the least forecast-moving gap → parallel OpenAI native web searches (`GAP_FILL_RESOLVER_MODEL` at `GAP_FILL_RESOLVER_REASONING_EFFORT`, via OpenRouter on the donated key; sol→terra 2026-07-20 — terra preferred-or-within-noise in all three 2026-07 blind role audits at ~40-50% lower cost, and these searches are the single biggest research line item ~44% of spend) resolve each. Soft-fails (returns `""`) on any error.

**Gap-fill v2 (agentic research loop)** (`metaculus_bot/research/agentic/`, entry `research/agentic_gap_fill.py` `run_gap_fill_v2`): a bounded tool loop run by a driver LLM (`GAP_FILL_V2_DRIVER_MODEL` / `GAP_FILL_V2_DRIVER_EFFORT` in `constants.py`, both picked by the 2026-07-17 blind driver eval — `scratch/driver_replay_2026-07-17/blind_judge_report.md`). The driver is briefed with the forecaster prompt template, privately dry-runs the forecast to find fill/verify targets, then iterates over four tools (`research/agentic/tools.py`): `search_news` (AskNews via the existing rate-gate), `search_web` (Exa direct), `fetch` (auto-escalating ladder plain → headless Chromium → `read_document`), and `read_document` (`GAP_FILL_V2_READER_MODEL` via Gemini url_context). Output is a detached citation-only findings artifact appended to the bundle under `## Agentic Research Findings`, with a `### ⚠ Corrections to the briefing` priority block first; a ghost forecast is logged for telemetry only (`GHOST_FORECAST` marker, never published). Anytime output under `GAP_FILL_V2_WALL_DEADLINE` / `GAP_FILL_V2_MAX_TOOL_CALLS` (constants.py); soft-fails to `""` at every boundary; benchmarking-guarded OFF (`is_benchmarking=True` returns `""`). Gated by `GAP_FILL_V2_ENABLED` — **ON in all four workflows since 2026-07-17**; v1 gap-fill stays ON alongside for an overlap window. Per-run telemetry marker in the run logs: `GAP_FILL_V2: model=... steps=... tool_calls=... searches= fetches= rendered= reads= dup_tool_calls= deadline_hit= concluded_early= wall_s= findings= pending_leads= lint_rejections= provenance_rejections= quote_mismatch_warnings= plan_gaps= plan_skipped= conclude_gate_rejections= error=` (emitted by `_log_completion`; `docs/agentic_gap_fill.md` reads the fields). `error=` is the one field separating a step-zero crash from an idle run — both otherwise emit `steps=0 tool_calls=0 findings=0`. The turn-one plan and the concluding ghost each emit a `GHOST_PRE` / `GHOST_PRE_JSON` and `GHOST_FORECAST` / `GHOST_FORECAST_JSON` pair, the `_JSON` half carrying the full forecast for `scripts/score_ghosts.py`.

**Production workflows** (`.github/workflows/run_bot_on_{tournament,metaculus_cup,minibench}.yaml`, `test_bot.yaml`) set `NATIVE_SEARCH_ENABLED=true`, `GEMINI_SEARCH_ENABLED=true`, `FINANCIAL_DATA_ENABLED=true`, `GAP_FILL_ENABLED=true`, `PREDICTION_MARKETS_ENABLED='true'`, `RESOLUTION_SOURCE_ENABLED='true'` (the last flipped in all three prod yamls 2026-07-10 after a live-output eyeball). So in prod the active stack is AskNews + OpenAI native search (on the `NATIVE_SEARCH_*` settings above) + Gemini grounded + financial-data (when classified as financial) + always-on OpenAI native-search gap-fill + the prediction-market snapshot + the Tier-1 resolution-source fetcher.

### Prompts (`metaculus_bot/prompts.py`)

- `_benchmarking_warning`, `_forecasting_window_str`, `web_research_prompt`.
- Base: `binary_prompt`, `multiple_choice_prompt`, `numeric_prompt`. Each base prompt embeds the STRUCTURED FORECAST JSON-block schema instruction and requires the fenced ```json block to be the **last** output — no trailing prose value lines. The value extraction ladder relies on this: rung 1 parses the block deterministically, and the tail-scan repair rung reads only the rationale's trailing `_TAIL_SCAN_CHARS`.
- Stacking: `stacking_binary_prompt`, `stacking_multiple_choice_prompt`, `stacking_numeric_prompt`. Same block-last schema instruction as the base prompts (the stacker output flows through the same ladder). The stacker prompts also include a "Cross-model aggregation (deterministic math)" block at the top when `build_cross_model_aggregation` returns markdown.
- Conditional-stacking support: `disagreement_crux_prompt`, `targeted_search_prompt`.
- Gap-fill: `gap_fill_analyzer_prompt`, `gap_fill_search_prompt`.

## Probabilistic tools (wired, DISABLED in prod)

### `metaculus_bot/probabilistic_tools/`

Reusable probability math — pooling, Beta-Binomial Bayes, percentile → parametric fits (normal/lognormal/Student-t), declared-vs-math consistency checks, Dirichlet CIs, Neg-Bin/Poisson discrete percentiles, exponential/Weibull survival, Gamma-conjugate hazard. `prob_event_before`, `linear_pool` / `log_pool` / `satopaa_extremize`, `beta_binomial_update`, `cdf_at_threshold`, `dirichlet_with_other` are wired into `tool_runner` dispatch. (`poisson_at_least_one` is exported and used inside `mc_discrete.py` / `survival.py`, but is NOT itself dispatched by `tool_runner`.)

Newly-added math (Workstreams D1-D3):

- **Noisy-OR** (`aggregation.py` `noisy_or`): rare-binary decomposition `1 − ∏(1 − pᵢ)` for combining independent failure-mode probabilities. Exported from the package, but NOT currently dispatched by `tool_runner` (no references in `tool_runner.py`) — it is a callable available for future wiring, not an active dispatch path. `TODO(noisy-or-wiring)`: either add a binary Noisy-OR dispatch (when a forecaster declares independent sub-event probabilities) or leave as a library-only helper.
- **Mixture-of-normals** (`mixtures.py`): `MixtureOfNormals` / `MixtureComponent` types, `mixture_cdf`, `fit_mixture_from_percentiles` (multi-start L-BFGS-B with single-normal fallback), and `percentiles_to_metaculus_cdf_via_mixture` (constraint-enforced `PCHIP_CDF_POINTS`-point CDF). The library itself is preserved but currently dormant — the `NumericStructured.mixture_components` schema slot and the router branch that consumed it were removed 2026-07-08 after zero prod fires; percentiles+PCHIP outperformed the mixture path in every benchmark.
- **Gamma waiting-time, conditional-given-survival**: `gamma_prob_event_before` with elapsed-window split (`survival_distributions.py`) — covers the missing waiting-time fitter alongside the existing exponential / Weibull / Gamma-hazard variants.

### `metaculus_bot/tool_runner.py`

Despite the name, **not** an LLM tool-calling harness. A **deterministic probability-math post-processor** that runs on structured JSON blocks emitted by each forecaster (priors, base rates, hazards, percentiles, scenarios) and injects a "Computed quantities" section into per-forecaster rationales plus a cross-model aggregation block into the stacker prompt. Entry points `run_tools_for_forecaster` and `build_cross_model_aggregation`. Gated by `PROBABILISTIC_TOOLS_ENABLED`; both entry points no-op when the flag is unset. **Wired but DORMANT in prod**: all three prod workflows (`.github/workflows/run_bot_on_{tournament,minibench,metaculus_cup}.yaml`) pin `PROBABILISTIC_TOOLS_ENABLED: 'false'` (retired via Workstream C2, which also removed the tier-2 scaffold from the prompts; tool_runner + probabilistic_tools stay behind the flag). The wiring remains live: `run_tools_for_forecaster` runs from `_make_prediction`, and `build_cross_model_aggregation` feeds the stacker prompts in both the STACKING and CONDITIONAL_STACKING paths. A `TOOLS_USED` marker is emitted in the comment trailer alongside the `STACKER_OUTCOME` marker so residual analysis can bucket tool-augmented vs. vanilla runs (always `false` in prod while the flag is off; see `metaculus_bot/comment/markers.py` for the marker-dormancy details).

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
- **`GOOGLE_API_KEY` — operator's personal Google AI Studio key on a BILLING-ENABLED (paid-tier, prepaid-credit) project; near-zero marginal cost at current usage.** In CI it's stored as `secrets.GEMINI_API_KEY` and surfaced as `GOOGLE_API_KEY` in the workflow env so the `google-genai` SDK picks it up. **There is NO Metaculus-donated Google AI Studio key.** Billing mechanics (verified against ai.google.dev pricing/billing/google-search docs 2026-07-17 — don't re-litigate without fetching them): Gemini 3.x grounding is paid-tier-ONLY (free-tier column: "Not available") and includes **5,000 free grounded prompts/month shared across all Gemini 3 models per project, then $14/1k individual search queries** (multi-query prompts bill per QUERY on overage — deep-research prompts fire several). Current usage ≈ 100-200 grounded prompts/month ≈ 3% of the allowance. The spring-2026 billing arc, explained: gap-fill's 5x grounded-call multiplier + backtest volume (backtest_large = 600 grounded prompts/run) blew past 5,000/month → per-query overage → prepaid-credit top-up debits ("started getting billed"); the 2026-06-25 resolver migration (`a51617e`) cut the multiplier and new charges stopped, with residual ~$1/month token spend silently drawing down the prepaid balance. **Watch item: prepaid-balance exhaustion produces 429s, not surprise charges** — if Gemini grounded search starts soft-failing across a run, check the AI Studio credit balance FIRST. Also: any future feature multiplying grounded-call counts (or Gemini-grounded backtests at scale) re-eats the same monthly pool. Model note: `gemini-3.5-flash` shares the same grounding pool (grounding = $0 to switch) but its tokens are 3x 3-flash-preview ($1.50/$9 vs $0.50/$3 per M) — a few dollars/month, from prepaid credits. The grounded-search side (`research/gemini_search.py`) and v2's `read_document`/url_context both use this personal key (url_context: no per-request fee; retrieved documents bill as input tokens). Don't confuse the OpenRouter Gemini path (donated route via `OAI_ANTH_OPENROUTER_KEY`, minus whatever `DONATED_KEY_BLOCKED_GOOGLE_MODELS` excepts) with this google-genai path — separate keys, separate billing.
- **`METACULUS_TOKEN`, `ASKNEWS_*`, `EXA_API_KEY`, `PERPLEXITY_API_KEY`, `FRED_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` — all personal.** No shared variants. The two direct provider keys (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`) are only used if you bypass OpenRouter; most flows route through OpenRouter and don't need them.

**Toggle: `GEMINI_USE_DONATED_OPENROUTER_KEY`** (default `true`, since 2026-06-16). Only affects OpenRouter Gemini routing; does NOT touch grounded search. On by default because Metaculus raised the Google rate limits — the donated OpenRouter key now serves most Gemini models (verified by live call: `gemini-3.5-flash` and `gemini-3.1-flash-lite` both succeed on the donated key). **Known exception:** the Gemini Pro forecaster slot is **PINNED to the personal key** via the `DONATED_KEY_BLOCKED_GOOGLE_MODELS` blocklist in `fallback_openrouter.py` (read the blocklist for which models it currently covers) — `should_route_via_donated_key` returns `False` for anything on it even with the toggle ON, so there's no donated attempt, no 429, and no personal-key-fallback-counter bump (which would otherwise redden CI on every question). It's pinned (not "falls back") because that model routes through a free-tier Google AI Studio BYOK key on the donated account with no Pro free tier (quota 0 → `is_byok:true` + `FreeTier limit: 0`). This is a **temporary workaround** tagged `TODO(gemini-3.1-pro-donated)` in code — remove the blocklist entry once Metaculus fixes the BYOK routing (enable Cloud billing on the BYOK key's GCP project, remove the Google AI Studio BYOK integration so native OpenRouter Google credits are used, or disable "Always use for this provider" on that BYOK key), then re-verify with one live call. Set this toggle to a false-y value (`false`/`0`/`no`) to force personal-key-only routing for ALL Gemini. The prod workflow yamls (`.github/workflows/run_bot_on_*.yaml`, `test_bot.yaml`) pin it to `'true'` explicitly. See `metaculus_bot/fallback_openrouter.py:should_route_via_donated_key` and `FUTURE.md` "Gemini on the donated OpenRouter key".

**Diagnosing auth / credit errors**:

- OpenRouter 401/402/credit error on a Gemini call → OpenRouter Gemini routes donated-first by default (with personal fallback), so suspect `OAI_ANTH_OPENROUTER_KEY` first; if the toggle has been forced OFF, suspect `OPENROUTER_API_KEY`. (Note: anything on `DONATED_KEY_BLOCKED_GOOGLE_MODELS` is pinned to the personal `OPENROUTER_API_KEY` with no donated attempt, so a credit error on one of those models is always a personal-key issue.)
- OpenRouter 401/402 on an OpenAI or Anthropic call → suspect `OAI_ANTH_OPENROUTER_KEY` first (donated key is always tried first for those providers), then `OPENROUTER_API_KEY` if the wrapper doesn't fall back.
- OpenRouter 401/402 on Grok / Qwen / Perplexity → always `OPENROUTER_API_KEY` (donated key 404s on these providers).
- `google-genai` 401 / quota / API-key-invalid error → always `GOOGLE_API_KEY` (no donated path).
- **403 splits three ways — the reported status decides the branch, and the spend-cap phrase outranks it.** (1) A `403` whose body says **`Key limit exceeded`** IS a key issue: that is how OpenRouter reports a breached per-key SPEND CAP (not the 402 its docs promise), so the wrapper falls back to the personal key and classifies it as credit-caused. That phrase fires on any reported status or none. (2) A `403` carrying `no allowed providers` / `guardrail` / `data policy` is scoped to the donated key's ROUTING, so it falls back too, but it is NOT credit-caused. (3) Every other `403 forbidden / moderation` is not a key issue and deliberately does not fall back — both keys would refuse the same prompt. Where a status is reported it is the only numeric evidence consulted, so credit wording on a non-402 reported status (a 403 body saying "insufficient credit") no longer classifies as credit or falls back. See `should_retry_with_general_key` / `_is_credit_failure`, and `docs/operations.md` "What a dry donated key actually returns". Whether a spend-cap 403 is SUPPRESSED from CI alerting is a separate question answered by the `/auth/key` probe (`credit_telemetry.classify_donated_key_state`): only a genuinely `drained` key is suppressed; `zeroed`, `revoked`, `funded`, and `unknown` all stay red.
- `429 rate limit` → not a key-scoped defect, but the wrapper DOES fall back (BYOK quotas are per-key, so a throttled donated key doesn't imply a throttled personal one) and the event stays fully alertable.

### Python environment

- **Managed by uv**: `uv sync` creates/updates the in-project `.venv` from `uv.lock`. Run commands via `uv run <cmd>` (e.g. `uv run python`, `uv run pytest`).
- **Adding dependencies**: `uv add <pkg>` (runtime) or `uv add --dev <pkg>` (dev group), then commit the updated `pyproject.toml` + `uv.lock`. NEVER use `pip` or `poetry` — both are blocked.
- **Supply-chain freshness**: lock-time `exclude-newer` (≈1 week) is enforced via the operator's global uv config, so `uv lock` avoids packages published in the last week.

## Framework integration (`forecasting-tools`)

- `GeneralLlm` for model interfaces (wrapper around litellm).
- `MetaculusApi` for platform integration.
- Question types: `BinaryQuestion`, `NumericQuestion`, `MultipleChoiceQuestion`.
- Prediction types: `ReasonedPrediction`, `BinaryPrediction`, etc.
- Research helpers: `AskNewsSearcher`, `SmartSearcher`.
- Numeric: `NumericDistribution`, `Percentile`. We subclass `NumericDistribution` as `PchipNumericDistribution` (`numeric/pchip_processing.py`) to override `get_cdf()` (the method ft 0.2.92's publish/aggregate paths call; `.cdf` is a deprecated property delegating to it) with our pre-computed `PCHIP_CDF_POINTS`-point PCHIP CDF; forecasting-tools' built-in CDF builder is only used on the fallback path.

## Model configuration

LLM ensemble lives in `metaculus_bot/llm_configs.py` — single source of truth. Rotates frequently; don't hardcode model names elsewhere. Provider: OpenRouter with automatic key fallback.

## Development commands

### Environment setup

- **Install**: `uv sync --dev` (or `make install`).
- **Run any command in the env**: `uv run <cmd>` — no manual activation needed.

### Core operations

- **Run bot**: `uv run python main.py` (or `make run`).
- **Run tests**: `uv run pytest` (or `make test`).

### Benchmarking

**Primary approach — resolved-question backtest** (`backtest.py`):

- **Smoke (4)**: `make backtest_smoke_test`
- **Small (12)**: `make backtest_small`
- **Medium (32)**: `make backtest_medium`
- **Large (100)**: `make backtest_large`

**DEPRECATED — community benchmark** (`community_benchmark.py`): baseline scoring broken (Metaculus removed aggregations from list API). `make benchmark_display` still works for viewing old results.

### Residual / performance analysis (read-only, FREE — not gated)

The `metaculus_bot/performance_analysis/` package evaluates the live bot's calibration against actual resolutions. Entry point: `uv run python -m metaculus_bot.performance_analysis --tournament <slug> --output <path>` (defaults to `spring-aib-2026`; pass `--tournament` explicitly). The **pull is read-only and free** — it hits only the Metaculus API (resolved questions + the bot's own comments, user id 275109, auth via `METACULUS_TOKEN`), no LLM/research calls and no publishing, so it is **NOT subject to the cost gate above** (unlike `make backtest_*` and live runs). Full era-bucketed methodology (Recon → Pull → Analyze → Synthesize, config-era bucketing keyed on submission time) in `scratch_docs_and_planning/residual_rerun_workflow.js`; dated outputs land under `scratch/residual_<date>/` (gitignored). Before analysis, run `make sync_all` (read-only/free) — "residual analysis" implies `make sync_all` first, always. It pulls **everything** sync-shaped in one command: the research archive (`backtests/research_archive/latest/<qid>.json`, per-question post-summarizer research), the run-log telemetry archive (`backtests/telemetry_archive/`, the `EXTRACTION_RUNG` / `GAP_FILL_V2` / `GHOST_FORECAST` / `OPEN_BOUND_PILING` / `CREDIT_*` markers), AND the raw research-provider payload archive (`backtests/research_archive/raw/<run_id>.jsonl`, one file per run — each research provider's RAW return before formatting: AskNews article dicts per HOT/HISTORICAL phase, native-search + Gemini raw responses with grounding, prediction-market contracts, resolution-source per-URL fetches, gap-fill v1 search results; written by `metaculus_bot.research.raw_log` when `RAW_RESEARCH_LOG_ENABLED` is set, so the raw evidence behind every forecast is auditable without depending on published comments — financial_data is deliberately not captured, its raw series live only inside `to_thread` workers). GHA artifacts expire at 90 days, so a single-source pull silently drops whatever it didn't fetch — `make sync_all` is the standing rule so a future source is never forgotten. See `scripts/research_sync/` for the weekly launchd job (now wired to `sync_all`). The telemetry archive also feeds `make score_ghosts` (gap-fill v2 ghost-vs-published log-score gate, ~0 scoreable until v2-era questions resolve).

### Recovering per-model forecasts

The bot's published Metaculus comments are the durable per-model record: on non-stacked questions the summary carries one `*Forecaster N (model)*: value` bullet per ensemble member (post-clamp values). `performance_analysis/collector.py` `build_performance_dataset` already parses these into `per_model_forecasts` / `per_model_mc_option_probs` / `per_model_numeric_percentiles` / `per_base_model_forecasts`; the bullet regex and `Model:`-prefix attribution live in `performance_analysis/parsing.py`. Gotchas:

- Comments longer than `COMMENT_CHAR_LIMIT` are middle-trimmed (`comment/trimming.py`) — summary bullets survive, but rationale-body percentile detail may not.
- Stacked-era questions publish only the stacker's aggregate bullet; base values are recoverable only from self-declared rationale text (the `## Base Model Reasoning` sub-blocks).
- Soft-deadline drops mean some questions have fewer than N bullets.
- Old-era (May–June 2026) blocks carry retired tier-2 fields (`mixture_components`, `tails`, `distribution_family_hint`) or edge values (`concentration: 0.0`) that the strict `parse_structured_block` schemas reject wholesale — a tolerant raw-JSON fallback rung in `parsing.py` (strict block → prose regex → tolerant salvage, added 2026-07-15) recovers the declared values from block-only rationales that would otherwise vanish (the false "gemini missed 5/45" screening artifact).
- Roster drift makes era-conditioning mandatory — see the era-bucketing paragraph below.
- **Per-model cuts run on a filtered cohort, aggregates don't.** When no `Model:` line identifies a bullet, `parsing.py` keys it by position instead (`anonymous_model_key` → `Forecaster N`), and on a stacker-fired question that positional bucket holds the stacker's aggregate — so pooling it across questions produces a stacker-vs-base-model mixture posing as one model (measured: 50 such forecasts in the 2026-04 data). Every per-model cut in `analysis.py` (`per_model_binary_scores`, `stacking_effectiveness`, `disagreement_predicts_error`) therefore goes through `per_model_cohort`, which drops anonymous keys and drops records whose stacker is *confirmed* fired, logging both counts at INFO under `PER_MODEL_COHORT`. Only the confirmed verdict excludes: `likely_stacker` is a high-spread-plus-large-delta heuristic that also matches an ordinary MEAN-era aggregate, so honoring it would delete the high-disagreement questions those cuts exist to measure. Aggregate/overall calibration paths are deliberately untouched — they still count every record.

**The attribution parsers are guarded on two cohorts, and only one of them runs in CI.** `tests/data/performance_comments_mini.jsonl` is a checked-in miniature — one real comment per distinct SHAPE (attributable vs not, trimmed vs intact, with vs without the `### Research Summary` boundary marker, named vs anonymized, all four question types), redacted down to the structural skeleton the parsers key on. It is the deterministic CI floor: `TestMiniFixtureAttribution` and `TestAgainstCheckedInMiniComments` are not skip-gated, so a parse or trim regression reddens every PR. The broad sweep over `scratch/performance_data.json` (283 records, every era) still runs locally and catches shapes the miniature hasn't been taught, but that file is gitignored and rewritten by each collector run, so it can never be the only guard — a parse regression hid behind exactly that gap until 2026-07-27. Regenerate the miniature with `uv run python scripts/derive_mini_comment_fixture.py` when a pull introduces a genuinely new shape; the derivation only admits a record whose miniature parses IDENTICALLY to its full-size source, and the shape-coverage test fails loudly if the set ever narrows.

**Era-bucketing is mandatory for calibration claims.** Any calibration, aggregation, or bias claim computed on pooled resolved data is suspect until split by config/roster era (proxy: `source_tournament`, or `bot_comment_created_at` versus config-flip dates). Three separate conclusions have flipped under era-bucketing: the numeric "too wide" verdict (2026-06, pre-flip-only data), the "current pipeline too narrow" verdict (2026-07, softened then reversed as post-flip n grew), and the YES-side overconfidence finding (2026-07-08, which turned out to be spring-2026-era-local — fall was well-calibrated, and a pooled fit would have degraded fall out-of-sample). Bucket by major config/roster changes (model swaps, aggregation changes, widening flips, research-stage changes — e.g. 2026-07-17: gap-fill v2 enabled in prod, a new config era; same-era: native-search + crux-analyzer models sol→terra per blind role audit; 2026-07-20 saw two forecaster-roster changes the same day — first the fable-5 → opus-4.7 forecaster + opus-4.8 stacker swap, then the drop from 6 to the 3-member latest-per-vendor triple (gpt-5.6-sol / opus-4.8 / gemini-3.1-pro-preview), which supersedes the first as the current config-era boundary; a later same-day tweak — sol forecaster effort xhigh→high plus `MIN_FORECASTERS_TO_PUBLISH` 2→1 — is a config tweak WITHIN that triple era, not a new era, since roster membership is unchanged), NOT by every git hash — a small prompt tweak does not start a new era; a forecaster-roster or pipeline-behavior change does. Judgment call: "would this change plausibly shift the forecast distribution?" If unsure, run the analysis both ways. Fitted calibration layers (shrinks, clamps, haircuts) require a decisive out-of-sample era test before shipping (fit on eras 1..k-1, must improve era k), else they are drift bombs.

### Code quality

- **Lint**: `make lint` (Ruff check).
- **Format**: `make format` (Ruff format + autofix).
- **Pre-commit**: `make precommit_install` then `make precommit` or `make precommit_all`.
- **Typecheck**: `make typecheck` (basedpyright; `make typecheck_ty` for the secondary ty checker).
- **Coverage**: `make cov`. **Audit**: `make audit` (osv-scanner over `uv.lock`; requires `brew install osv-scanner` locally — CI runs it via `google/osv-scanner-action`).
- **Test single file**: `uv run pytest tests/test_specific.py`.

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

1. **Optional dependency loading.** `prediction_market_provider` pulls in `rapidfuzz`; `tool_runner` only matters when `PROBABILISTIC_TOOLS_ENABLED` is on. Importing them at function scope keeps the cold-start path lean and avoids surprising errors when an optional dep isn't installed.
2. **Ruff auto-formatter behavior.** When a usage edit is staged separately from the import edit (common during refactors and subagent dispatches), Ruff's auto-formatter strips the now-unused top-level import between cycles. Function-scoped imports survive this because the symbol is referenced in the same statement block.

Don't hoist these to the top of `forecaster.py` without first checking that both reasons no longer apply.

### Important commands

The **Makefile** has most commands (`make test`, `make format`, `make typecheck`, `make run`, etc.). In agentic CLIs, invoke through `uv run` (e.g. `uv run python script.py`); uv resolves the in-project `.venv` automatically.

## Commit & pull request guidelines

- Commits: concise, imperative subject (e.g., "fix test cmd", "migrate to uv"). Short body when context helps.
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
