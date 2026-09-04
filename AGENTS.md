# Metaculus Forecasting Bot — Agent Guidelines

General coding guidelines (style, testing, error handling, etc.) live in the operator's private
global config; this file covers **repo-specific** context only.

This file is a starting point, not the manual. It carries the cost gate, the repo's overrides,
the layout, the pipeline outline, and the standing rules whose violation is silent and
expensive. Everything else lives in `docs/`, indexed at the bottom. When a task needs depth,
open the doc rather than expecting it here.

## ⚠️ Cost discipline — every credit spend goes through the operator

**Any command that hits live LLM or research APIs spends real money (OpenRouter credits,
AskNews, Exa, Perplexity, Google). NEVER launch one autonomously — ask first, every time, even
after clean gates and a clean `/forge`.** `--mode test_questions` also publishes comments to
Metaculus, which is hard to retract.

Four clarifications, each load-bearing:

- **The gate is on the SPEND, not the mechanism.** A local command, a GitHub Actions dispatch,
  an edit that flips a schedule, a script that wraps any of those — all the same rule. An agent
  proposes and prices the run; the operator decides. There is no clean-gates exemption and no
  "small enough to skip asking" threshold; a $2-3 run still goes through the operator. If a paid
  run is the only way to verify the work, say so and stop.
- **What the gate forbids is an agent DECIDING to spend.** An explicit instruction from the
  operator IS the approval: when they say to fire a run already discussed, run it and don't
  re-ask. Approval is per-run and does not carry forward — one "go" is not standing
  authorization for the next run, nor for re-running the same one after further changes.
- **Paid runs are a final pre-merge check, not part of the verification loop.** The smoke test
  exists to be fired ONCE, deliberately, when a change is otherwise finished. An agent that
  folds it into a check-my-work loop spends real money for no added signal. The free gates are
  the loop; unit and integration coverage is how an agent earns confidence.
- **When verification needs a paid run, surface the exact command and rough cost** and let the
  operator decide.

Paid or externally visible — ask before each:

- `uv run python main.py` / `make run` in any live mode (`--mode test_questions`, `tournament`,
  `metaculus_cup`, `minibench`): spends credits AND publishes.
- `make backtest_smoke_test` / `_small` / `_medium` / `_large`: every forecaster and research
  call, plus the `LEAKAGE_DETECTOR_MODEL` screen. No publish, real money. Question counts are
  the `--num-questions` values in the Makefile.
- `make backtest_with_cache`: `--research-dir` replays archived research
  (`_load_research_from_archive`, `backtest.py`), but the live ensemble still forecasts every
  question, so forecaster spend is real. A question with no archived record falls back to live
  research and the run logs a warning.
- `make ablation_*` (`qa_research` / `smoke` / `small` / `medium`): real research and forecaster
  spend. `ablation_score` is the free exception — `--stages score` hydrates every artifact off
  disk and makes no provider call.
- `make benchmark_run_*`: deprecated (`community_benchmark.py` baseline scoring is broken), but
  the `run` / `custom` modes still fan the real ensemble over real questions. Use
  `make backtest_*` instead; `make benchmark_display` is the free view-only mode.
- `make test_live`: the only suite that leaves the network. It pins a `:free` OpenRouter slug so
  dollar spend is near-zero, but the calls are real and need a key. Ask anyway.
- **GitHub Actions runs of any bot workflow**, which spend exactly as a local run does and
  publish to Metaculus. `test_bot_basic.yaml` (one numeric question, ~$2.60) and `test_bot.yaml`
  are `workflow_dispatch`-only. The three `run_bot_on_*.yaml` prod workflows are additionally on
  `schedule:` crons. Never dispatch one, and never edit a `schedule:` block or a research/model
  flag in a way that adds runs or raises per-run cost, without the operator's say-so. `gh` needs
  `--repo No-Stream/metaculus-bot` here: `origin` is the fork, `upstream` is the Metaculus
  template, and no default repo is set, so a bare `gh workflow` command silently targets
  upstream. Two standing facts about enablement, both confirmed 2026-09-03:
  **`run_bot_on_minibench.yaml` is `disabled_manually` on GitHub by operator design and has
  NEVER been enabled**, so a supply-probe row showing minibench with closed questions and zero
  bot forecasts is expected rather than a forfeit or a token problem — do not ask the operator
  about it again. **The Metaculus Cup workflow is being ENABLED for the fall 2026 season**
  (Metaculus granted $1,500 of API credits on 2026-09-03; the repo-side configuration landed the
  same day), but until the operator flips it on in the GitHub UI it is still `disabled_manually`
  and fires nothing, so a cup row with no bot forecasts also stays expected for now.
  `docs/operations.md` tracks what landed and what is left.
- `fetch_diagnostic.yaml` is NOT a bot workflow — `workflow_dispatch`-only, no secrets,
  structurally incapable of spending or publishing — but a dispatch burns Actions minutes and
  probes federal hosts from the runner IP, so ask before dispatching it too.
- `uv run python scripts/probes/gemini_verify.py --i-accept-spend`: three live google-genai calls
  on the operator's personal AI Studio key (one grounded search plus two `url_context` reads).
  Cents, plus one prompt off the 5,000/month grounded allowance. It refuses without the flag and
  prints a cost estimate; the ask-first gate still applies.
- `RESOLUTION_SOURCE_URL_CONTEXT_ENABLED` is the one flag that turns a free provider into a paid
  one. It is off by default and set in no workflow yaml, so the resolution-source fetcher spends
  nothing today; turning it on adds a Gemini `url_context` read, billed to the operator's
  personal `GOOGLE_API_KEY`, as the last rung of its fetch ladder. Flipping it on anywhere is the
  operator's cost decision. Detail: `docs/operations.md`.
- Anything invoking research providers or the ensemble against real questions, including a
  one-off script an agent writes to do so.

Free and safe — run freely:

- Gates and formatting: `make test`, `make test_fast`, `make test_e2e`, `make lint`,
  `make format`, `make typecheck`, `make typecheck_ty`, `make cov`, `make audit`, `make deps`
  (deptry), `make lint_imports` (import-linter), `make precommit*`. The whole suite is
  self-contained: the `e2e` marker means full-pipeline with MOCKED LLMs, and the autouse
  `_block_network_egress` fixture (`tests/conftest.py`) raises on any non-loopback connect, so no
  selected test can reach a paid API. `addopts` deselects only the `live` marker.
- Read-only Metaculus and artifact pulls: `make sync_all` and its parts (`sync_research`,
  `sync_telemetry`, `sync_raw_research`, `download_*`, `backfill_*`), the `performance_analysis`
  package, `make score_ghosts`, `make close_margin_watch`, `make ablation_score`,
  `make supply_probe`, `make benchmark_display`. These hit only the Metaculus API and GitHub
  artifacts.
- `make check_credits` — reads both OpenRouter key balances.
- `uv run --with curl_cffi python scripts/probes/fetch_diagnostic.py` — public GETs only, no LLM
  and no key.

## Repo overrides

- **Python 3.12+** (`pyproject.toml` `requires-python`).
- **Package manager: uv.** `uv.lock` is the lockfile; there is no `poetry.lock`. `uv sync --dev`
  installs the package editable, so no `PYTHONPATH=.`. Add deps with `uv add <pkg>` /
  `uv add --dev <pkg>`, then commit `pyproject.toml` + `uv.lock`. **NEVER use `pip` or
  `poetry`** — both are blocked here. Lock-time `exclude-newer` (about a week) comes from the
  operator's global uv config, so `uv lock` avoids just-published packages.
- **Build backend `uv_build`**, flat layout (`metaculus_bot/` at repo root, `module-root = ""`).
- **Formatter: Ruff**, 120-char lines (not Black).
- **Type checker: basedpyright** in standard mode, must stay at 0 errors. `ty` is secondary and
  advisory. Promoting the core pipeline to strict is tracked in `FUTURE.md`.
- **Testing: pytest + `pytest-asyncio`**, self-contained, no API keys in CI.
- Copy `.env.template` to `.env` for local dev. Never commit secrets; `.env` is gitignored, and
  CI keys come from GitHub Actions secrets.

## What this repo is

A fork of the Metaculus starter template, built on the `forecasting-tools` framework. For each
question it gathers research from several providers in parallel, runs a small ensemble of
frontier LLMs to produce independent forecasts, combines them, and publishes the result as a
Metaculus comment. Aggregation defaults to `CONDITIONAL_STACKING` (MEDIAN when the base models
agree, a stacker LLM rewrite when they disagree), but **stacking is disabled in production, so
prod publishes the MEDIAN of the raw forecasts.**

## Layout

Top level:

- `main.py` — thin shim: re-exports `TemplateForecaster` from `metaculus_bot/forecaster.py` and
  invokes the CLI.
- `backtest.py` — primary benchmarking; scores bot predictions against real resolutions.
- `community_benchmark.py` — **DEPRECATED** (Metaculus removed `aggregations` from the list API).
  `make benchmark_display` still works for old runs.
- `FUTURE.md` — the design log: intent and history, including rejected ideas. Read it for the
  why, not for current state.
- `docs/` — the current-state guides (index at the bottom of this file).
- `scratch_docs_and_planning/` — plans and audits, including
  `residual_analysis_playbook.md` (the per-round residual procedure),
  `probabilistic_tools_activation.md` (activation pending) and
  `fetch_escalation_ladder_design.md` (DESIGN ONLY, nothing in it implemented, its open
  questions are the operator's). `metaculus_api_doc_LARGE_FILE.yml` there is the full Metaculus
  API spec — read it with offset/limit.
- `REFERENCE_COPY_OF_forecasting_tools*/` — read-only copy of the framework source; edits do not
  affect the installed package. `REFERENCE_COPY_OF_panchul*/` — a Q2 2025 competition winner,
  for comparison.
- `tests/`, `.github/workflows/` (CI on PRs, plus the scheduled bot runs), `scripts/`.

Inside `metaculus_bot/`:

| Concern | Where |
|---|---|
| Per-question orchestration | `forecaster.py` (`_research_and_make_predictions`), `cli.py` |
| Close-derived time budget | `time_budget.py` |
| Research fan-out and providers | `research/` (`orchestrator.py`, `providers.py`, one module per provider) |
| Outbound fetch transports (never hand-rolled) | `research/http_fetch.py`, `rendered_fetch.py` (headless Chromium), `url_context_reader.py` (the paid Gemini read), `robots_policy.py` |
| Resolution-source fetcher and its escalation rungs | `research/resolution_source.py`, `resolution_fetch_result.py`, `derived_api.py`, `wayback.py` |
| Gap-fill v1 / v2 | `research/targeted.py`, `research/agentic/` |
| Model roster (source of truth) | `llm_configs.py` |
| Prompts | `prompts.py` |
| Constants, thresholds, env flags | `constants.py` |
| Per-type forecaster runners | `forecaster_runners.py` |
| Value extraction ladder | `value_extraction.py`, `structured_parse.py`, `structured_output_schema.py` |
| Numeric percentiles → CDF | `numeric/` |
| MC clamp / renormalize | `mc_processing.py` |
| Aggregation routing and stacking | `stacking_route.py`, `aggregation_pipeline.py`, `stacking.py`, `spread_metrics.py` |
| Publish hardening and close gate | `publish_hardening.py`, `publish_gate.py` |
| Comment assembly and trimming | `comment/` |
| Telemetry and degradation counters | `drop_telemetry.py`, `degradation_counters.py`, `credit_telemetry.py`, `extreme_call.py`, `member_forecast.py`, `close_margin.py` |
| OpenRouter key fallback and retries | `fallback_openrouter.py`, `llm_retry.py`, `llm_setup.py` |
| Residual / calibration analysis | `performance_analysis/`, `calibration/` |
| Backtest, ablation, benchmark harnesses | `backtest/`, `ablation/`, `benchmark/`, `ensemble_analysis/` |
| Probability math, dormant in prod | `probabilistic_tools/`, `tool_runner.py` |

`scripts/` holds the read-only sync and analysis tooling: `sync_all.py`, `download_*.py`,
`backfill_*.py`, `supply_probe.py`, `score_ghosts.py`, `reconcile_credit_spend.py`,
`derive_mini_comment_fixture.py`, the `telemetry/` marker registry, `probes/`, and the
`research_sync/` launchd job.

## Pipeline outline

Per question, inside `forecaster.py:_research_and_make_predictions`. Detail:
`docs/architecture.md`.

0. **Close-derived time budget** (`time_budget.py`), granted at intake before any spend:
   `min(PER_QUESTION_WALL_CLOCK_DEADLINE, close_time − now − PUBLISH_RESERVE_SECONDS)`. Three
   consequences: an intake skip when the budget is non-viable, a fast path that drops the slow
   optional providers and both gap-fill passes, and a research-phase deadline that cancels
   stragglers. Every question logs a `TIME_BUDGET` marker.
1. **Research** — `run_research` (`research/orchestrator.py`) picks exactly one primary provider
   by priority and runs the add-on providers alongside it in parallel, each independently
   env-gated, then two gap-fill passes append their own sections. Detail: `docs/research.md`,
   `docs/agentic_gap_fill.md`.
2. **Forecaster fan-out** — N forecaster LLMs in parallel via `_forecaster_with_soft_deadline`
   (capped per model by `FORECASTER_SOFT_DEADLINE`) → `_make_prediction` → the type-specific
   runner in `forecaster_runners.py`. Each forecaster emits its value inside a fenced ```json
   STRUCTURED FORECAST block, read by the extraction ladder in `value_extraction.py`. Detail:
   `docs/value_extraction.md`, `docs/prompts.md`.
3. **Min-forecasters guard** — drops the question below `MIN_FORECASTERS_TO_PUBLISH`
   (`constants.py`). With the floor at 1 a lone survivor publishes, and `route_after_forecasts`
   (`stacking_route.py`) short-circuits n == 1 before spread computation, because the
   `spread_metrics` helpers require two predictions and raise otherwise. Every question past the
   guard logs `FORECASTERS_SURVIVED`.
4. **Aggregation** — `aggregation_pipeline.py`. Spread below the per-type threshold gives the
   MEDIAN; above it, a crux extraction plus targeted search plus a stacker rewrite, with a
   second stacker and then MEDIAN as fallbacks. In prod the per-type gates are off, so this is
   always MEDIAN. Numeric aggregation happens pointwise in CDF space
   (`aggregate_numeric`, `numeric/utils.py`), not in percentile space. Detail:
   `docs/architecture.md`, `docs/numeric_pipeline.md`.
5. **Publish, behind a close-time gate** (`publish_gate.py`, layer 4 of `publish_hardening.py`).
   A question whose window has passed is skipped entirely, prediction and comment together, and
   the skip is alertable. Detail: `docs/architecture.md`.

## Standing rules

Each of these has cost real work at least once. The pointer is where the reasoning lives.

**Config and models**

- **Model names live only in `llm_configs.py`.** Never hardcode one elsewhere, and never state
  the current roster from memory — "latest per vendor" resolves only from a live model-list read.
  Detail: `docs/roster_history.md`, `docs/operations.md` "Season-start checklist".
- **A roster or pipeline-behaviour change is a config-era boundary.** It ships in one merge,
  before the season's first question, not piecemeal.
- **Never restate `STANDARD_PERCENTILES`** (or its count, or its CSV label) anywhere; derive from
  `numeric/config.py`.
- **Constants belong in `constants.py`** (or `numeric/config.py` for grid-scoped ones), not
  inline in a function.
- **Only one API key in this repo is shared**: `OAI_ANTH_OPENROUTER_KEY`, the Metaculus-donated
  OpenRouter key, despite the name. Everything else is the operator's personal key, including
  `GOOGLE_API_KEY`, which is a billing-enabled AI Studio key with no donated equivalent. Which
  key pays for what, the fallback rules, and every auth-error diagnosis are in
  `docs/operations.md` "API keys and the shared-vs-personal key model".

**Telemetry and logs**

- **Every marker and status string is a data contract, not just a log line.** The archive keys
  off the exact spelling (`scripts/telemetry/markers.py` is the registry), so ADD a marker or a
  field; never rename or re-spell one, and never change a field's meaning in place. A status
  string (`FetchStatus` values, provider `sources` tokens, skip reasons) is the same kind of
  token.
- **A new signal needs a marker spec, or it is invisible after 90 days**, when the GitHub Actions
  logs expire.
- **Prose must never stand in for an absent section.** Any non-empty provider return flips the
  orchestrator's status to `ok`, counts the provider as succeeded, and defeats every downstream
  empty guard at once. A provider with nothing to say returns `""` and records a loss token;
  a count (`details["counts"]`) is how "ran and found none" stays distinguishable from "never
  ran". Detail: `docs/research.md`.

**Guards and safety**

- **A guard fails SHUT.** The unit-mismatch guard is the worked example: wrapping it in
  try/except made a guard crash byte-identical to a passing check, so it published the
  order-of-magnitude error the guard exists to block. Let it raise.
- **Any new outbound HTTP path goes through `research/http_fetch.py`.** It owns the SSRF
  invariants: the `is_public_http_url` preflight, the connect-time `FilteringResolver` (the
  resolver, not the preflight, is the real DNS-rebinding boundary), a bounded manual redirect
  loop that re-guards every hop, the meta-refresh hop that no HTTP status announces, and the
  per-host politeness semaphores. Two transports sit beside it and are SHARED with gap-fill v2
  rather than copied: `research/rendered_fetch.py` owns the headless-Chromium render, including
  the DNS pin, the per-request route guard and the process-global launch cap, and
  `research/url_context_reader.py` owns the one paid Gemini `url_context` read, with
  `research/robots_policy.py` the per-host `Google-Extended` pre-check in front of it. Do not
  hand-roll a fetch, a render or a reader.
- **Timing, deadline and fallback code gets strictly-safer changes only.** A tidy-up in a
  soft-deadline, retry or key-fallback path can silently thin the ensemble or strand it on a dead
  key. If a change is not obviously safer, leave it and note it in `FUTURE.md`.
- **Benchmarking leakage guards are hard returns.** `prediction_market` and `resolution_source`
  return `""` when `is_benchmarking=True`. Never soften one to make a backtest measure more.

**Prompts**

- **A prompt rule stays only if the pipeline requires it, it scaffolds the model's reasoning, or
  it corrects a measured failure with no shorter form.** State it once, with its one-clause
  reason attached, as a named constant or a named template step. Every surviving rule has a
  presence pin and every removed one an absence pin in `tests/prompts/`; no base-prompt rule may
  appear in the three stacking prompts. Detail: `docs/prompts.md`.

**Analysis**

- **Run `make sync_all` before any residual analysis.** "Residual analysis" implies it, always.
  A single-source pull silently drops whatever it did not fetch, and GitHub Actions artifacts
  expire at 90 days.
- **Era boundaries are merge-to-main committer timestamps, never authoring dates.** Prod runs
  from `main`, so a config change is live only from the moment its merge lands there. Keying on
  the authoring date has produced a phantom era and a wrong telemetry-presence rate.
- **Era-bucket every calibration, aggregation or bias claim.** Three separate conclusions have
  flipped under bucketing. A fitted calibration layer ships only after a decisive out-of-sample
  era test.
- **Rank and aggregate on SPOT PEER, never on the coverage-scaled `peer_score`.** Use
  `spot_peer_score()` / `ranking_score()` in `performance_analysis/platform_scores.py` rather
  than indexing `metaculus_scores`, and price a counterfactual with `spot_peer_delta` rather than
  by hand — the continuous-question halving gets applied twice or not at all otherwise.
- **The exclusion cohorts have one home: `performance_analysis/cohorts.py`** (`KNOWN_BUG_QIDS`,
  `DEGRADED_RUN_QIDS`, `PARTIAL_DEGRADED_QIDS`, `EXCLUSION_COHORTS`). Import them; every private
  copy has drifted at least once. These are QUESTION ids — translate through
  `performance_analysis/id_mapping`, never match raw integers against post ids.
- **Never pool the three research-archive record classes** (`artifact`, `comment_backfill`,
  `log_backfill`) for a presence, provider-mix or length claim, and read the `source` field
  rather than inferring the class from `run_id`.
- **A published comment carrying fewer than N per-model bullets is ambiguous.** It can mean a
  soft-deadline drop, a stacked-era question that published only the aggregate, or a roster
  change. Never read a missing bullet as a model that declined to forecast.
- All of the above, with the receipts: `docs/performance_analysis.md`. The per-round procedure:
  `scratch_docs_and_planning/residual_analysis_playbook.md`.

**Code structure**

- **A function-scoped import needs one of exactly three justifications**, named in its
  `# noqa: PLC0415` comment: a genuinely optional dependency, late binding for a patch surface,
  or a real circular import. Cold start and "the formatter would strip it" are not
  justifications. Never delete a `HARNESS-SCAN-EXEMPT-function-level-import` marker. Detail:
  `docs/architecture.md` "Import conventions".
- **Patch a name on the module where it is USED, not where it is defined.** Two live traps:
  every `Fred` / `fetch_series` patch target is `metaculus_bot.research.fred_rendering`, not
  `financial_data` (fredapi's real class carries the identical literals, so a patch at the wrong
  module stays green while proving nothing); and hoisting a late-bound
  `from x import y` binds the unpatched object at import time and silently defeats the test.
- **Fix all occurrences.** If you found a bug pattern, grep for its siblings.

## Development

- **Install**: `uv sync --dev` (or `make install`). Run anything with `uv run <cmd>`; no manual
  activation.
- **Run the bot**: `uv run python main.py` (or `make run`) — paid, see the cost gate.
- **Tests**: `make test` (full), `make test_fast`, `make test_e2e`, or
  `uv run pytest tests/test_specific.py`.
- **Lint and format**: `make lint` (Ruff check), `make format` (Ruff format + autofix).
- **Typecheck**: `make typecheck` (basedpyright), `make typecheck_ty` (secondary).
- **Coverage**: `make cov` (branch coverage is on). **Audit**: `make audit` (osv-scanner over
  `uv.lock`; `brew install osv-scanner` locally, CI runs it via `google/osv-scanner-action`).
- **Dependency hygiene**: `make deps` (deptry). **Import contracts**: `make lint_imports`
  (import-linter; the contracts live in `pyproject.toml` `[tool.importlinter]`). Both are free
  and both run in CI's lint job and in `make all`.
- **Pre-commit**: `make precommit_install` installs both hook types — the Ruff hooks on commit,
  plus a `pytest-full-suite` pre-push hook running the same command CI runs. Per-push rather than
  per-commit because the suite takes about 105 s.
- **Benchmarking** (all paid except `benchmark_display`): `make backtest_smoke_test` (4),
  `make backtest_small` (12), `make backtest_medium` (32), `make backtest_large` (100).
- **Residual analysis** (read-only, free):
  `uv run python -m metaculus_bot.performance_analysis --tournament <slug> --output <path>`,
  and always pass `--prior <previous round's dataset>`. Then
  `python -m metaculus_bot.performance_analysis.width_monitor`,
  `metaculus_bot.performance_analysis.clip_threshold`, and `make supply_probe`. Detail:
  `docs/performance_analysis.md`.

**CI green is the gate, not a local green run.** A test that depends on the developer's
environment — a hardcoded absolute path, the checkout location, `$HOME`, gitignored local data —
passes locally by construction, so CI is the first place it can fail. This has shipped: a test
asserted a launchd plist's `ProgramArguments` equalled a path derived from `__file__`, which
holds only on the machine whose absolute path the plist hardcodes. Never assert an absolute path
derived from the developer's environment; assert a repo-relative suffix
(`Path.relative_to(_REPO_ROOT)`), or skip when the artifact is inherently machine-specific.
After pushing, check the run:
`gh run list --repo No-Stream/metaculus-bot --branch <branch>` (`--repo` is required, since
`origin` is the fork and `upstream` is the template).

## Commits and pull requests

- Commits: concise, imperative subject ("fix test cmd", "migrate to uv"). Short body when
  context helps.
- PRs: clear description, link issues, include config and docs updates, logs or screenshots for
  behaviour changes.
- CI: all checks pass, code formatted, imports sorted.
- Limit changes to workflow files unless CI behaviour is meant to change.

## Metaculus API notes

- API docs: <https://www.metaculus.com/api/> (Swagger UI). Backend source (open, and where the
  validation lives): <https://github.com/Metaculus/metaculus>, `questions/serializers/common.py`.
- Server-side CDF constraints for `continuous_cdf` submissions are in
  `docs/numeric_pipeline.md` "Server-side constraints".
- `/api/comments/?author=X` returns only the caller's own comments (or staff authors). Analysing
  other bots' comments at scale needs a Metaculus support exemption, manual browsing, or a
  browser-driven scrape.

## Docs

| Doc | What it covers |
|---|---|
| `docs/architecture.md` | End-to-end map: entry points, the per-question pipeline stage by stage, the time budget, aggregation, the publish gate, import conventions, where every module lives. |
| `docs/research.md` | Every research provider: AskNews and its summarizer, OpenAI native search, Gemini grounded search (citation strip, attribution check), financial data (pegs, variance ratio, FRED), the prediction-market snapshot, the resolution-source fetcher and its rungs, the time-series anchor, and both gap-fill passes. |
| `docs/agentic_gap_fill.md` | Gap-fill v2: the bounded agentic tool loop, its four tools, the fetch ladder, the findings artifact, the ghost forecast, and its telemetry. |
| `docs/numeric_pipeline.md` | Percentiles to PCHIP CDF: sanitizing, tail widening, min/max-step enforcement, bound pinning, the repair tiers, discrete snapping, CDF-space aggregation, the unit-mismatch guard, and the binary and MC clamps. |
| `docs/value_extraction.md` | The four-rung extraction ladder, its fidelity checks, and the `EXTRACTION_RUNG` / `MEMBER_FORECAST` markers with the raw-versus-published convention. |
| `docs/prompts.md` | Every forecasting-prompt rule, the constant that carries it, its receipt, and the size accounting from the 2026-09 de-bloat. |
| `docs/roster_history.md` | The roster design rule, the dated roster-change history and its era boundary, the support-model roles, and the dormant `probabilistic_tools` / `tool_runner` paths. |
| `docs/operations.md` | Running the bot: setup, the season-start checklist, the shared-versus-personal API keys, Google AI Studio billing, credit telemetry and the refill floor, the dry-donated-key semantics, the workflows and their env flags, backtesting, and reading run logs. |
| `docs/performance_analysis.md` | Residual-analysis conventions: era bucketing and the merge-date rule, the exclusion cohorts, the archive's record classes, the PIT and spot-peer conventions, `spot_peer_delta`, the starved outer tail, per-model recovery, and the clip-threshold sweep. |
| `README.md` | Human quick-start: install, configure, run. |
| `FUTURE.md` | The design log — intent, history, and rejected ideas. Read it for the why, not for current state. |
| `scratch_docs_and_planning/residual_analysis_playbook.md` | The per-round residual procedure. |
