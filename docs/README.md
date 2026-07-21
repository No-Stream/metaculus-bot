# Documentation

This directory holds the human-facing guides for the Metaculus forecasting bot.
Each file covers one area of the system. Start with whichever matches what you
need; the repo-root pointers at the bottom cover quick-start and design history.

## Guides

| Doc | What it covers |
| --- | --- |
| [architecture.md](architecture.md) | How a question flows through the bot end to end: research fan-out, the 6-model forecaster ensemble, the min-forecasters guard, and aggregation (CONDITIONAL_STACKING by default, MEDIAN in prod since stacking is disabled). |
| [research.md](research.md) | The research providers that build each question's briefing: AskNews (primary), OpenAI native search, Gemini grounded search, financial data, the prediction-market snapshot, and the resolution-source fetcher, plus the always-on gap-fill passes. All fan out in parallel and are independently env-gated. |
| [agentic_gap_fill.md](agentic_gap_fill.md) | Gap-fill v2, the bounded agentic research loop: a driver LLM dry-runs the forecast, picks fill/verify targets, then iterates over search/fetch/read tools to produce a citation-only findings artifact. Runs alongside the v1 gap-fill pass. |
| [numeric_pipeline.md](numeric_pipeline.md) | How numeric forecasts turn 13 declared percentiles into a 201-point PCHIP CDF: value extraction, sanitizing, tail widening, min/max-step enforcement, bound pinning, discrete-integer snapping, and the unit-mismatch guard. |
| [operations.md](operations.md) | Running the bot: environment setup with `uv`, the shared vs. personal API keys, the production workflows and their env flags, credit telemetry, backtesting, and the cost gate (any live or paid run needs operator approval). |

## Repo-root references

- [README.md](../README.md) — human quick-start: install, configure, run.
- [AGENTS.md](../AGENTS.md) — detailed, current-state guidelines for coding agents
  (also symlinked as `CLAUDE.md`). The most complete map of the codebase.
- [FUTURE.md](../FUTURE.md) — the branch design log. Records intent and history,
  including shipped, planned, and rejected ideas. Read it for the "why," not for
  the current state of the code.

For the authoritative model roster, see `metaculus_bot/llm_configs.py`. It is the
single source of truth for the ensemble and rotates often, so the guides describe
it rather than pin exact model names.
