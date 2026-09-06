# Architecture overview

This is the map a new contributor should read first. It walks through what happens
end to end when the bot forecasts one question, names the entry points, and points
you at the deeper docs for each subsystem. Read this, then dive into the specific
page you need.

The bot is a fork of the Metaculus starter template built on the `forecasting-tools`
framework. The core idea: for each question, gather research from several providers,
run an ensemble of large language models to produce independent forecasts, then
combine them into one prediction and publish it as a comment on Metaculus.

## Entry points

Three files form the startup chain:

- `main.py` — a thin shim. It re-exports `TemplateForecaster` (for anything that
  imports it) and, when run directly, calls `cli.main()`.
- `metaculus_bot/cli.py` — the command-line entry point. It parses `--mode`
  (`tournament`, `minibench`, `metaculus_cup`, `quarterly_cup`, `test_questions`),
  builds the LLM roster dict from `llm_configs.py`, constructs a `TemplateForecaster`
  with `aggregation_strategy=CONDITIONAL_STACKING`, and runs the mode-specific
  forecast loop. It also wires credit telemetry and decides the process exit code:
  the run exits non-zero when any degradation counter fired (`alertable_count` on
  `TemplateForecaster` sums them: dropped forecasters, questions that failed to
  publish, stacker fallbacks, research-provider and summarizer failures, gap-fill
  v2 errors, and prediction-market degradation) or the donated OpenRouter key
  dropped below the $100 early-warning floor (`OPENROUTER_CREDIT_FLOOR_USD`, sized
  so the reminder to ask Metaculus for a top-up arrives with runway left).
  Credit-caused alerts are live again as of 2026-09-03 and are suppressed only
  inside a dated window — see "The credit-alert suppression window" in
  `docs/operations.md`. See `main` in `cli.py`.
- `metaculus_bot/forecaster.py` — the bot itself. `TemplateForecaster` subclasses the
  framework's `ForecastBot` and owns the per-question pipeline. The method to read
  first is `_research_and_make_predictions`.

Publication happens inside the framework's forecast loop, not in `cli.py`. Every
question that clears the min-forecasters guard is already on Metaculus by the time
`cli.py` decides the exit code.

## The per-question pipeline

Everything below runs once per question inside `_research_and_make_predictions`,
under a shared per-question wall-clock budget (`PER_QUESTION_WALL_CLOCK_DEADLINE`,
sized to finish just inside the 60-minute Metaculus close window). Research,
forecaster fan-out, aggregation, and publish all draw from that one budget.

```
                        one Metaculus question
                                 │
                                 ▼
        ┌────────────────────────────────────────────────┐
        │  1. RESEARCH  (ResearchOrchestrator.run_research)│
        │  Providers fan out in parallel:                  │
        │    - AskNews (primary, summarized to a briefing) │
        │    - OpenAI native search                        │
        │    - Gemini grounded search                      │
        │    - financial data (yfinance / FRED)            │
        │    - prediction-market snapshot                  │
        │    - resolution-source fetcher                   │
        │  Each is independently env-gated.                │
        └────────────────────────────────────────────────┘
                                 │  research bundle
                                 ▼
        ┌────────────────────────────────────────────────┐
        │  2. GAP-FILL  (two passes, run concurrently)     │
        │    v1: analyzer finds gaps → parallel searches   │
        │    v2: agentic tool loop (bounded, driver LLM)   │
        │  Each appends its own section; both soft-fail.   │
        └────────────────────────────────────────────────┘
                                 │  enriched bundle
                                 ▼
        ┌────────────────────────────────────────────────┐
        │  3. FORECASTER FAN-OUT                           │
        │  N forecaster LLMs run in parallel, each capped  │
        │  by FORECASTER_SOFT_DEADLINE. Type-specific      │
        │  runner per question (binary / MC / numeric).    │
        └────────────────────────────────────────────────┘
                                 │  N reasoned predictions
                                 ▼
        ┌────────────────────────────────────────────────┐
        │  4. MIN-FORECASTERS GUARD                        │
        │  Fewer than MIN_FORECASTERS_TO_PUBLISH valid     │
        │  → skip this question, keep the batch going.     │
        └────────────────────────────────────────────────┘
                                 │
                                 ▼
        ┌────────────────────────────────────────────────┐
        │  5. AGGREGATION  (CONDITIONAL_STACKING)          │
        │  Measure spread across forecasters.              │
        │  Low spread OR stacking disabled → MEDIAN.       │
        │  High spread + stacking on → crux + targeted     │
        │  search + stacker LLM rewrite.                   │
        └────────────────────────────────────────────────┘
                                 │  one aggregated prediction
                                 ▼
        ┌────────────────────────────────────────────────┐
        │  6. PUBLISHED COMMENT                            │
        │  Per-model bullets + full research + diagnostics.│
        └────────────────────────────────────────────────┘
```

### 0. Close-derived time budget

The budget is granted at intake by `metaculus_bot/time_budget.py`, before any spend: `total_s = min(PER_QUESTION_WALL_CLOCK_DEADLINE, close_time − now − PUBLISH_RESERVE_SECONDS)`, so the static 3510 s deadline is now only the UPPER bound on a question's budget (non-publishing runs — backtests/ablations — keep exactly the static budget; `close_aware` gates on `publish_reports_to_metaculus`). Three consequences: (a) **intake skip** — a question whose budget is non-positive, or close-limited below `TIME_BUDGET_MIN_VIABLE_S`, is skipped before any research or forecaster spend (counted under `publish_skipped_closed`: latency cost us the question, however early we noticed); (b) **fast path** — below `TIME_BUDGET_FAST_PATH_THRESHOLD` (= the full pipeline's configured worst case) the slow optional search providers and BOTH gap-fill passes are dropped, and the resolution-source fetcher's two expensive escalation rungs (the Chromium render, the paid `url_context` read) decline with a `fast_path` skip while its direct fetch and cheap rungs still run, counted by the alertable `time_budget_fast_path`; (c) **research-phase deadline** — the provider phase and each gap-fill pass are bounded by `RESEARCH_PHASE_BUDGET_SHARE` of the remaining budget, cancelling stragglers (`RESEARCH_PHASE_DEADLINE` WARN; off the fast path such cuts count under the alertable `research_budget_cuts`). Every question logs a `TIME_BUDGET` marker; the loud markers (`TIME_BUDGET_FAST_PATH`, `GAP_FILL_SKIPPED_FOR_BUDGET`, `GAP_FILL_V1/V2_CUT_FOR_BUDGET`) all have telemetry-archive specs.


### 1. Research fan-out

`run_research` (`forecaster.py`) delegates to a `ResearchOrchestrator`
(`research/orchestrator.py`). It picks one primary provider by priority (AskNews in
prod, then Exa, then Perplexity, then a stub) and runs several additional providers
alongside it in parallel, each behind its own env flag. AskNews returns raw article
text and gets summarized into an analyst briefing inline; the other providers write
their own prose and pass through as-is. See [research.md](research.md) for provider
selection, gating, and the shared-vs-personal API-key routing.

### 2. Gap-fill (two passes)

After the first-pass bundle is assembled, two gap-fill passes run concurrently in one
`asyncio.gather` inside `run_research` (`orchestrator.py`), so the research phase costs `max(v1, v2)`
in wall-clock, not the sum:

- **v1** (`research/targeted.py`): an analyzer LLM reads the bundle, names up to a few
  factual gaps, then resolves each with a parallel web search. Appends a
  `## Targeted Gap-Fill (second pass)` section.
- **v2** (`research/agentic/`): a bounded agentic tool loop. A driver LLM privately
  dry-runs the forecast to find what to verify, then iterates over search/fetch/read
  tools within a wall deadline and tool-call cap. Appends an
  `## Agentic Research Findings` section, led by a corrections block.

Both are wrapped so a failure in one never zeroes the other or kills the forecast.
Both are on in prod. See [agentic_gap_fill.md](agentic_gap_fill.md) for the v2 loop,
tools, and telemetry.

The orchestrator also builds a provider-diagnostics block that is deliberately
withheld from the forecaster-facing text (so it never pollutes prompts) but is
re-attached to the published comment later. This is the "diagnostics seam": the
orchestrator's `pop_provider_diagnostics`, which `_research_and_make_predictions` in
`forecaster.py` drains once the research phase is done.

### 3. Forecaster fan-out

Each forecaster LLM runs through `_forecaster_with_soft_deadline` (`forecaster.py`),
which caps a single model at `FORECASTER_SOFT_DEADLINE` so one stuck
model can't hold the whole question. `_make_prediction` dispatches to the
type-specific runner (`forecaster_runners.py`) for binary, multiple-choice, or numeric
questions. The N coroutines are gathered under the shared wall-clock budget by
`_gather_predictions_with_wall_clock` (`forecaster.py`), which cancels any
forecaster still pending at the deadline and counts the drop.

The ensemble is a handful of forecaster LLMs, one per vendor. The exact roster
rotates often, so **read `metaculus_bot/llm_configs.py` for the current list** rather
than trusting any names written here. Support models (summarizer, parser, stacker,
disagreement analyzer) live in the same file. The standing design rule, each support
model's role, and the dated history of every roster change with the merge that made it
live are in [roster_history.md](roster_history.md).

Each forecaster emits its answer inside a fenced ```json STRUCTURED FORECAST block,
which is parsed by a deterministic extraction ladder (`value_extraction.py`). Numeric
questions produce the canonical percentile set (`STANDARD_PERCENTILES` in
`numeric/config.py`), turned into a PCHIP CDF on the `PCHIP_CDF_POINTS` grid. See
[numeric_pipeline.md](numeric_pipeline.md) for the percentile-to-CDF machinery and its
bound/step constraints.

### 4. Min-forecasters guard

If fewer than `MIN_FORECASTERS_TO_PUBLISH` (`constants.py`) forecasters
returned a valid prediction, the ensemble is too degraded to trust. The question is
skipped and a counter bumps for end-of-run alerting, but the rest of the batch and all
other publications continue. The guard lives in `_research_and_make_predictions`
(`forecaster.py`).

When the threshold is 1, a lone survivor publishes: the median of one forecast is
that forecast. Because the spread metrics in `spread_metrics.py` require at least two
predictions and raise otherwise, `route_after_forecasts` (`stacking_route.py`)
short-circuits the n == 1 case before spread computation and stacking and hands the
single prediction straight to the aggregator. Exception-driven drops still bump the
degradation counters, so a run thinned to one model reddens CI rather than silently
withholding the question.

#### Survivor and extreme-call telemetry

Past the guard, every question logs `FORECASTERS_SURVIVED: question=... survived=n/N models=...` at INFO — the positive counterpart to the per-run `FORECASTER_DROPS` marker, and the only place a run log states the survivor count. It is load-bearing precisely because the floor is low: a degraded publish exits zero and the failure-path "Only n/N forecasters succeeded" line never fires, while the comment-side `FORECASTERS_USED` marker never reaches stdout, so without this line a thinned ensemble reads identically to a full one. `models=` names the survivors (read off each prediction's own `Model:` prefix, not the configured roster) so survivors can be diffed against drops from the log alone. Harvested into the telemetry archive as `forecasters_survived` (`scripts/telemetry/markers.py`). Immediately after that line, a BINARY question also logs one `EXTREME_CALL: question=... model=... p=... side=low|high lone=... survivors=...` INFO line **per surviving member whose probability sat at or past an edge of the extreme band** (`format_extreme_call_markers`, `metaculus_bot/extreme_call.py`; band `EXTREME_CALL_LOW` / `EXTREME_CALL_HIGH` in `constants.py` beside `BINARY_PROB_MIN`/`BINARY_PROB_MAX`, inclusive at both edges). It is pure measurement — the module reads probabilities and returns strings, and nothing clamps or gates on this membership check (step 4's single-survivor publish clamp reuses the same two constants by aliasing them, but it is a separate rule keyed on the survivor count). A member inside the band leaves NO line, so `FORECASTERS_SURVIVED` in the same run log is the denominator for any rate. `lone=true` means no other survivor was extreme **on the same side**, which is the cut worth having: the 2026-08-31 gemini-slot review found lone extremes right 4 of 9 against 21 of 23 for accompanied ones. Two scope facts keep the numerator honest — binary only (MC concentration is a different measurement and was not adopted), and `lone` is vacuous at `survivors=1`, which is why the survivor count rides the same line. **Do not pool these counts with the memo's**: the memo's own scripts implement the looser "no other member extreme at all", which disagrees on 4 of 570 archived member-calls and reads pre_flip lone as 48 where this marker reads 52 (post_flip and triple_era agree exactly). Harvested as `extreme_call`.


### 5. Aggregation: CONDITIONAL_STACKING

The default strategy is `CONDITIONAL_STACKING` (set in `cli.py`'s `main`). Conceptually:

- Compute the spread across the N forecasts (`spread_metrics.compute_spread`).
- **Low spread**: return the MEDIAN of the raw per-model predictions.
- **High spread**: extract the disagreement crux with the analyzer LLM (under
  `CRUX_SOFT_DEADLINE`), run a targeted search on it — OpenAI native search on the same
  `NATIVE_SEARCH_*` model, effort, verbosity and timeout settings the native-search provider
  uses — then hand the full base-model reasonings plus that research to a stacker LLM that
  rewrites the forecast (`stacking.run_stacking_binary` / `_mc` / `_numeric`). The fallback
  ladder is primary `STACKER_LLM` under `STACKER_SOFT_DEADLINE` → `STACKER_FALLBACK_LLM` under
  `STACKER_FALLBACK_SOFT_DEADLINE` → MEDIAN, driven by `stack_predictions`
  (`aggregation_pipeline.py`).

Spread thresholds live in `constants.py`, one per question type:
`CONDITIONAL_STACKING_BINARY_PROB_RANGE_THRESHOLD` (a probability range),
`CONDITIONAL_STACKING_MC_MAX_OPTION_THRESHOLD` (a max per-option spread), and
`CONDITIONAL_STACKING_NUMERIC_NORMALIZED_THRESHOLD` (a normalized percentile spread).

**Stacking is disabled in production.** All five workflow YAMLs set
`BINARY_STACKING_ENABLED`, `MC_STACKING_ENABLED`, and `NUMERIC_STACKING_ENABLED` to
`false`, so even when spread exceeds the threshold, the per-type gate in
`route_after_forecasts` (`stacking_route.py`) bypasses the stacker and forces the
MEDIAN path. In effect, **prod runs MEDIAN of the raw forecasts.** The stacker chain
stays fully wired and is exercised in
backtests and ablation runs. `AggregationPipeline` owns the aggregation configuration,
per-question metadata, and counters. Its explicit operations are `base_combine`,
`stack_predictions`, and `simple_combine`. The framework-required
`TemplateForecaster._aggregate_predictions` hook selects the appropriate operation;
internal callers state that choice directly. Routing and stacked-result finalization
live in `stacking_route.py`, which receives the pipeline rather than the whole bot.
The conditional-stacking path runs the combined result
through a Platt-calibration hook (`_apply_platt_calibration` in
`aggregation_pipeline.py`), but that hook is gated by `PLATT_CALIBRATION_ENABLED`,
which is unset in every workflow, so in prod `apply_platt_calibration`
(`post_processing.py`) is a passthrough.

Multiple research reports for one question share the pipeline's existing
question-id maps. These remain separate because sibling reports can leave a skip
reason that must survive a later failed stack attempt. Stacked-result finalization
consumes meta reasoning; comment construction consumes outcome and skip metadata
after the parent comment builder returns successfully. The expected-combine set
is consumed by the framework's final combine. Moving ownership does not change
those read, write, or consumption points.

`tests/test_aggregation_lifecycle_e2e.py` covers the real framework lifecycle with
failed sibling reports, raw versus pre-stacked singles, and stacker fallback.
`tests/test_aggregation_report_e2e.py` drives the public entrypoint through numeric
and MC report construction with two reports and four model results. These tests
replace research and model calls while retaining fan-out, routing, aggregation,
and comment construction. Focused pipeline tests also pin cancellation and
validation before state consumption. `tests/test_aggregation_failure_lifecycle.py`
checks real timeout expiry and the state retained when the parent comment builder
fails after aggregation. Both cases passed unchanged against the implementation
before the ownership refactor.

#### The thin-publish floor

One survivor-conditional rule sits on top: when exactly ONE forecaster survived a BINARY question, the published probability is clamped into `[THIN_PUBLISH_BINARY_FLOOR, THIN_PUBLISH_BINARY_CEIL]` (`constants.py`, 0.05/0.95 — defined by aliasing `EXTREME_CALL_LOW` / `EXTREME_CALL_HIGH` so the extreme band has one definition, and narrower than the per-model `[BINARY_PROB_MIN, BINARY_PROB_MAX]` = [0.02, 0.98] clamp the member already passed) by `apply_thin_publish_floor` in `AggregationPipeline.base_combine`, triggered by the `single_forecaster` skip reason rather than the prediction count (a fired stacker's lone output shares that branch and is never floored); the per-model summary bullet keeps the raw value, an actual move logs `THIN_PUBLISH_FLOOR: question=... raw=... clamped=... survivors=1` (harvested as `thin_publish_floor`), and a multi-member median is never floored — median-of-1 has no variance reduction, which is the whole justification (q44874, −105.27 spot peer on a lone 0.03; receipt in `scratch/residual_2026-08-31/gemini_review/RECOMMENDATION.md` §2).

#### An unmeasurable spread is its own case

An UNMEASURABLE spread (non-positive normalizing denominator) reports `inf` and logs `SPREAD_UNDEFINED`, and `route_after_forecasts` treats it as its own case: MEDIAN with skip reason `spread_undefined`, spending no crux extraction / targeted search / stacker call on a question where nothing was measured. It used to report `0.0`, which read as an affirmative "the models agree" and published the marker `spread_below_threshold` — a measurement failure disguised as agreement. Latent in prod (the per-type gates are off) but live in backtests and ablation.


### 6. Published comment

The framework assembles the comment: per-model forecast bullets (annotated with model
names so per-model attribution survives comment trimming), the full research bundle,
the targeted-research section if stacking fired, and the provider-diagnostics block
re-attached via the seam. These published comments are also the durable per-model
record the performance-analysis tooling later parses.

### 7. Publish, behind a close-time gate

The gate lives in `publish_gate.py`, wired as layer 4 of `publish_hardening.py`'s patch of ft's `publish_report_to_metaculus`. Immediately before the POSTs, the question's `close_time` is compared to now; if the window has passed — or the question's cached `state` is already CLOSED/RESOLVED — the whole publish is SKIPPED (prediction and comment together, since a comment for a forecast Metaculus never accepted would seed `performance_analysis` with a forecast that doesn't exist on the platform). The skip emits one `PUBLISH_SKIPPED_CLOSED: question=... reason=... close_time=... now=... overdue_s=... state=...` WARN, bumps `publish_skipped_closed` on the degradation line, and counts as ALERTABLE — a skip means latency cost us the question, which is exactly what should redden CI. The run continues with every other question. Deliberately **no safety margin**: ft's publish body sleeps 3.5-4.5s twice, so a question with seconds left can still 405 after passing the gate, but widening it would start skipping publishes that would have landed, and a forfeited question costs far more than a rejected POST. That residual 405 now costs ONE attempt, not two — `publish_hardening` no longer retries a 4xx outside {408, 429}, since a second identical POST cannot fix a 405/401/400. Shipped 2026-08-25 as the root-cause fix for q45085 (2026-08-03: forecast at full 3/3 strength, submitted 12:05 against a 12:00 close, `405 "already closed to forecasting"`, whose crash also took out that run's end-of-run alertable summary).


## Framework integration (`forecasting-tools`)

What the bot takes from the framework, and the one place it overrides it:

- `GeneralLlm` for model interfaces (a wrapper around litellm).
- `MetaculusApi` for platform integration.
- Question types: `BinaryQuestion`, `NumericQuestion`, `MultipleChoiceQuestion`.
- Prediction types: `ReasonedPrediction`, `BinaryPrediction`, and friends.
- Research helpers: `AskNewsSearcher`, `SmartSearcher`.
- Numeric: `NumericDistribution`, `Percentile`. We subclass `NumericDistribution` as
  `PchipNumericDistribution` (`numeric/pchip_processing.py`) to override `get_cdf()` — the
  method ft 0.2.92's publish and aggregate paths call, with `.cdf` a deprecated property that
  delegates to it — so it returns our pre-computed `PCHIP_CDF_POINTS`-point PCHIP CDF. The
  framework's own CDF builder is used only on the fallback path.

## Import conventions

Imports go at module top, and `forecaster.py` has none inside functions. A
function-scoped import needs one of exactly three real justifications, and its
`# noqa: PLC0415` comment must name which:

1. **Genuinely optional dependency** — matplotlib behind an `ImportError` guard
   (`research/timeseries_anchor.py`, `calibration/fit_platt_cli.py`). matplotlib is in
   the dev group and prod installs `uv sync --no-dev`, so it is the one package that is
   genuinely absent at runtime, and the `DEP004` entry in `pyproject.toml` is where that
   exemption is declared to deptry. `rapidfuzz`, `yfinance` and `asknews` are all declared
   runtime dependencies, so a function-scoped import never protected against their
   absence.
2. **Late binding for a patch surface** — a test patches the name on its SOURCE module
   and the consumer must resolve it at call time. Hoisting a `from x import y` here binds
   the unpatched object at import time and silently defeats the test; this repo has
   shipped that bug. Live cases: `numeric.pipeline.sanitize_percentiles` from
   `ablation/run_stacker.py`; `numeric.pchip_cdf.*` from `numeric/pchip_processing.py` and
   `scripts/score_ghosts.py`; `asknews_sdk.AsyncAskNewsSDK` and
   `constants.NATIVE_SEARCH_WALL_TIMEOUT` from `research/providers.py`;
   `constants.FETCH_GET_RETRIES` from `fetch_hardening.py`;
   `fallback_openrouter.build_llm_with_openrouter_fallback` from `research/targeted.py`;
   and `ablation/forecasters.py`'s deliberate self-import (tests rebind
   `run_forecasters_for_question` on the module).
3. **A real circular import** — verify it by hoisting and importing, do not assume.
   Prefer fixing the module layout over keeping the lazy import.

Cold start is not a justification on its own. `import metaculus_bot.forecaster` costs
about 4.2 s, nearly all of it `forecasting_tools` and `litellm`; the four hoists done in
2026-08 added about 124 ms (2.7%), of which 105 ms is yfinance arriving via
`timeseries_anchor` → `ts_fetch`. matplotlib does not come along, because
`timeseries_anchor` imports `ts_chart` inside its own render guard. Re-measure with
`python -X importtime` if you add a top-level import that pulls scipy-, matplotlib- or
browser-weight machinery onto this path; the bar for a lazy import is a couple of hundred
milliseconds, not ten.

"The formatter would strip it" is also not a justification. Ruff only strips an import
with no usage, so add the import and its usage in the SAME edit and it survives.

Whichever applies, keep the `# noqa: PLC0415`, state the reason inline, and never delete a
`HARNESS-SCAN-EXEMPT-function-level-import` marker.

## Where the pieces live

| Concern | Module |
|---|---|
| Startup / CLI | `main.py`, `metaculus_bot/cli.py` |
| Per-question orchestration | `metaculus_bot/forecaster.py` |
| Post-fan-out aggregation routing | `metaculus_bot/stacking_route.py` |
| Drop attribution / degradation counters | `metaculus_bot/drop_telemetry.py`; `degradation_counters.py` formats immutable snapshots built by `forecaster.py` |
| Research fan-out | `metaculus_bot/research/orchestrator.py`, `research/providers.py` |
| Outbound fetch transports | `research/http_fetch.py` (plain HTTP, SSRF guards, redirects, per-host gates), `research/impersonated_fetch.py` (the `curl_cffi` TLS-impersonating retry of a 403, with its own DNS pin and per-hop re-guard), `research/rendered_fetch.py` (headless Chromium), `research/url_context_reader.py` (one paid Gemini `url_context` read), `research/robots_policy.py` (the `Google-Extended` pre-check in front of that read) |
| Resolution-source fetcher and its escalation rungs | `research/resolution_source.py`, `research/resolution_fetch_result.py` (the status, reason and route vocabularies), `research/derived_api.py`, `research/wayback.py` |
| Resolution-source text and section budgets | `research/resolution_presentation.py` |
| Datawrapper response classification, freshness and dataset ordering | `research/resolution_datawrapper.py`; requests and question budgets remain in `research/resolution_source.py` |
| Gap-fill v1 / v2 | `research/targeted.py`, `research/agentic/` |
| Forecaster runners | `metaculus_bot/forecaster_runners.py` |
| Value extraction | `metaculus_bot/value_extraction.py` |
| Numeric CDF | `metaculus_bot/numeric/` |
| Aggregation + stacking | `metaculus_bot/aggregation_pipeline.py`, `stacking.py` |
| Model roster (source of truth) | `metaculus_bot/llm_configs.py` |
| Prompts | `metaculus_bot/prompts.py` |
| Constants / thresholds / env flags | `metaculus_bot/constants.py` |

## Related docs

- [research.md](research.md) — research providers, gating, API-key routing.
- [numeric_pipeline.md](numeric_pipeline.md) — percentiles to PCHIP CDF, bounds, steps.
- [value_extraction.md](value_extraction.md) — the extraction ladder and its fidelity rules.
- [prompts.md](prompts.md) — every forecasting-prompt rule and why it is there.
- [agentic_gap_fill.md](agentic_gap_fill.md) — the v2 agentic research loop.
- [roster_history.md](roster_history.md) — the ensemble roster, its history, dormant paths.
- [performance_analysis.md](performance_analysis.md) — residual-analysis conventions.
- [operations.md](operations.md) — running the bot, workflows, cost discipline, credits.

## A note on cost

Any command that hits live LLM or research APIs spends real money and, in live modes,
publishes comments to Metaculus. Do not launch one without the operator's approval.
The free, self-contained paths (`make test`, `make lint`, `make format`,
`make check_credits`) are safe to run anytime. Details in
[operations.md](operations.md) and the repo's `AGENTS.md`.
