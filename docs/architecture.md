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

- `main.py` — a six-line shim. It re-exports `TemplateForecaster` (for anything that
  imports it) and, when run directly, calls `cli.main()`.
- `metaculus_bot/cli.py` — the command-line entry point. It parses `--mode`
  (`tournament`, `minibench`, `metaculus_cup`, `quarterly_cup`, `test_questions`),
  builds the LLM roster dict from `llm_configs.py`, constructs a `TemplateForecaster`
  with `aggregation_strategy=CONDITIONAL_STACKING`, and runs the mode-specific
  forecast loop. It also wires credit telemetry and decides the process exit code:
  the run exits non-zero when any degradation counter fired (dropped forecasters,
  stacker fallbacks, research timeouts) or the donated OpenRouter key dropped below
  the refill floor. See `cli.py:41` (`main`).
- `metaculus_bot/forecaster.py` — the bot itself. `TemplateForecaster` subclasses the
  framework's `ForecastBot` and owns the per-question pipeline. The method to read
  first is `_research_and_make_predictions` (`forecaster.py:548`).

Publication happens inside the framework's forecast loop, not in `cli.py`. Every
question that clears the min-forecasters guard is already on Metaculus by the time
`cli.py` decides the exit code.

## The per-question pipeline

Everything below runs once per question inside `_research_and_make_predictions`,
under a shared per-question wall-clock budget (`PER_QUESTION_WALL_CLOCK_DEADLINE`,
3510s — about 58.5 minutes of the 60-minute Metaculus close window). Research,
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
        │  N forecaster LLMs run in parallel, each under   │
        │  a 10-min soft deadline. Type-specific runner    │
        │  per question (binary / MC / numeric).           │
        └────────────────────────────────────────────────┘
                                 │  N reasoned predictions
                                 ▼
        ┌────────────────────────────────────────────────┐
        │  4. MIN-FORECASTERS GUARD                        │
        │  Fewer than MIN_FORECASTERS_TO_PUBLISH (3) valid │
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

### 1. Research fan-out

`run_research` (`forecaster.py:414`) delegates to a `ResearchOrchestrator`
(`research/orchestrator.py`). It picks one primary provider by priority (AskNews in
prod, then Exa, then Perplexity, then a stub) and runs several additional providers
alongside it in parallel, each behind its own env flag. AskNews returns raw article
text and gets summarized into an analyst briefing inline; the other providers write
their own prose and pass through as-is. See [research.md](research.md) for provider
selection, gating, and the shared-vs-personal API-key routing.

### 2. Gap-fill (two passes)

After the first-pass bundle is assembled, two gap-fill passes run concurrently in one
`asyncio.gather` (`orchestrator.py:144`), so the research phase costs `max(v1, v2)`
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
re-attached to the published comment later. This is the "diagnostics seam" you'll see
referenced in `forecaster.py:571`.

### 3. Forecaster fan-out

Each forecaster LLM runs through `_forecaster_with_soft_deadline` (`forecaster.py:869`),
which caps a single model at `FORECASTER_SOFT_DEADLINE` (600s / 10 min) so one stuck
model can't hold the whole question. `_make_prediction` dispatches to the
type-specific runner (`forecaster_runners.py`) for binary, multiple-choice, or numeric
questions. The N coroutines are gathered under the shared wall-clock budget by
`_gather_predictions_with_wall_clock` (`forecaster.py:430`), which cancels any
forecaster still pending at the deadline and counts the drop.

The ensemble is six forecaster LLMs balanced across providers. The exact roster
rotates often, so **read `metaculus_bot/llm_configs.py` for the current list** rather
than trusting any names written here. Support models (summarizer, parser, stacker,
disagreement analyzer) live in the same file.

Each forecaster emits its answer inside a fenced ```json STRUCTURED FORECAST block,
which is parsed by a deterministic extraction ladder (`value_extraction.py`). Numeric
questions produce 13 percentiles that get turned into a 201-point PCHIP CDF. See
[numeric_pipeline.md](numeric_pipeline.md) for the percentile-to-CDF machinery and its
bound/step constraints.

### 4. Min-forecasters guard

If fewer than `MIN_FORECASTERS_TO_PUBLISH` (default 3, `constants.py:469`) forecasters
returned a valid prediction, the ensemble is too degraded to trust. The question is
skipped and a counter bumps for end-of-run alerting, but the rest of the batch and all
other publications continue (`forecaster.py:613`).

### 5. Aggregation: CONDITIONAL_STACKING

The default strategy is `CONDITIONAL_STACKING` (`cli.py:120`). Conceptually:

- Compute the spread across the N forecasts (`spread_metrics.compute_spread`).
- **Low spread**: return the MEDIAN of the raw per-model predictions.
- **High spread**: extract the disagreement crux with the analyzer LLM, run a targeted
  search on it, then hand the full base-model reasonings plus that research to a
  stacker LLM that rewrites the forecast. If the stacker fails, it falls back to a
  second stacker LLM, then to MEDIAN.

Spread thresholds live in `constants.py`: binary 0.15, MC 0.20, numeric 0.15.

**Stacking is disabled in production.** All four workflow YAMLs set
`BINARY_STACKING_ENABLED`, `MC_STACKING_ENABLED`, and `NUMERIC_STACKING_ENABLED` to
`false`, so even when spread exceeds the threshold, the per-type gate bypasses the
stacker and forces the MEDIAN path (`forecaster.py:689`). In effect, **prod runs
MEDIAN of the raw forecasts.** The stacker chain stays fully wired and is exercised in
backtests and ablation runs. The aggregation dispatch, base-combine, and stacker
fallback ladder all live in `metaculus_bot/aggregation_pipeline.py`
(`AggregationPipeline`). The conditional-stacking path runs the combined result
through a Platt-calibration hook (`aggregation_pipeline.py:276`), but that hook is
gated by `PLATT_CALIBRATION_ENABLED`, which is unset in every workflow, so in prod
it is a passthrough (`post_processing.py:34`).

### 6. Published comment

The framework assembles the comment: per-model forecast bullets (annotated with model
names so per-model attribution survives comment trimming), the full research bundle,
the targeted-research section if stacking fired, and the provider-diagnostics block
re-attached via the seam. These published comments are also the durable per-model
record the performance-analysis tooling later parses.

## Where the pieces live

| Concern | Module |
|---|---|
| Startup / CLI | `main.py`, `metaculus_bot/cli.py` |
| Per-question orchestration | `metaculus_bot/forecaster.py` |
| Research fan-out | `metaculus_bot/research/orchestrator.py`, `research/providers.py` |
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
- [agentic_gap_fill.md](agentic_gap_fill.md) — the v2 agentic research loop.
- [operations.md](operations.md) — running the bot, workflows, cost discipline, credits.

## A note on cost

Any command that hits live LLM or research APIs spends real money and, in live modes,
publishes comments to Metaculus. Do not launch one without the operator's approval.
The free, self-contained paths (`make test`, `make lint`, `make format`,
`make check_credits`) are safe to run anytime. Details in
[operations.md](operations.md) and the repo's `AGENTS.md`.
