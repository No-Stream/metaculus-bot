# Operations & configuration

How to set up, configure, and run the Metaculus forecasting bot. This is the
reference for a human operating the bot: local setup, API keys, the environment
flags, the GitHub Actions workflows, cost discipline, and the telemetry you can
grep after a run.

For a code-level map of the pipeline, read `AGENTS.md` at the repo root. This
doc points at code with `file:line` references so you can click through.

## Setup

Prerequisites: Python 3.12+ and [uv](https://docs.astral.sh/uv/). The project
uses uv for everything. There is no poetry, conda, or pip in this repo.

```bash
uv sync --dev        # create/update .venv from uv.lock (or: make install)
cp .env.template .env
```

Then fill in `.env` with your keys (see below). Run any command inside the
project environment with `uv run <cmd>` — uv resolves the in-project `.venv`
automatically, so you never activate it by hand.

Quick sanity checks (all free, no paid APIs):

```bash
make test            # full pytest suite (self-contained, needs no keys)
make lint            # ruff check
make typecheck       # basedpyright (must stay at 0 errors)
make check_credits   # print OpenRouter balances for both keys
```

## API keys and the shared-vs-personal key model

The bot needs several credentials. `.env.template` lists them with inline
notes; copy it and fill in real values. The one piece of routing that trips
people up is the two OpenRouter keys.

- **`OAI_ANTH_OPENROUTER_KEY` — donated / shared.** Metaculus provides credits
  on this key for OpenAI, Anthropic, and Google models routed via OpenRouter.
  Its server-side allowed-providers list is locked to those three, so anything
  else (Grok via x-ai, Qwen, Perplexity) returns 404 on this key. This is the
  only shared credential in the bot; despite the name it covers all three
  providers, not just OpenAI and Anthropic.
- **`OPENROUTER_API_KEY` — personal.** Pays for what the donated key can't
  (Grok, Qwen, Perplexity-via-OpenRouter) and serves as the fallback when the
  donated key hits a credential, credit, or allowed-providers error. The
  fallback wrapper is `metaculus_bot/fallback_openrouter.py`.
- **`GOOGLE_API_KEY` — personal.** The operator's Google AI Studio key on a
  billing-enabled project. Powers the Gemini grounded-search provider and gap-
  fill v2's document reads. There is no donated Google AI Studio path. In CI
  this is stored as the `GEMINI_API_KEY` secret and surfaced to the workflow as
  `GOOGLE_API_KEY` so the `google-genai` SDK picks it up.

Gemini has two separate routes, which is the other easy thing to confuse:

- **OpenRouter Gemini** (forecaster / stacker / summarizer slots) routes
  donated-key-first with personal-key fallback, controlled by
  `GEMINI_USE_DONATED_OPENROUTER_KEY` (default `true`). One model,
  `gemini-3.1-pro-preview`, is pinned to the personal key via a blocklist in
  `fallback_openrouter.py` (no donated attempt), so a credit error on a Pro call
  is always a personal-key issue.
- **Gemini grounded search** (`research/gemini_search.py`) always uses the
  personal `GOOGLE_API_KEY`. The donated toggle does not touch it.

Other keys, all personal, no shared variants: `METACULUS_TOKEN`, `ASKNEWS_CLIENT_ID`
+ `ASKNEWS_SECRET`, `EXA_API_KEY`, `PERPLEXITY_API_KEY`, `FRED_API_KEY`,
`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`. The two direct provider keys only matter
if you bypass OpenRouter; most flows route through OpenRouter and don't need
them.

Diagnosing auth errors: an OpenRouter 401/402 on an OpenAI or Anthropic call
means suspect the donated key first (it's always tried first for those
providers). A 401/402 on Grok, Qwen, or Perplexity is always the personal key
(the donated key 404s on those). A `google-genai` 401 or quota error is always
`GOOGLE_API_KEY`. A `403 moderation` or `429 rate limit` is not a key problem;
the wrapper deliberately does not fall back on those.

## Environment flags

Flags are read at call time via `env_flag_enabled` (`constants.py:136`), which
treats `true`/`1`/`yes` as on and `false`/`0`/`no` as off (case-insensitive).
When a flag is unset it takes the code default shown below. The four workflow
YAMLs set these explicitly, so the "prod value" column is what actually runs in
CI.

### Research providers

| Flag | Code default | Prod value | What it gates |
|---|---|---|---|
| `NATIVE_SEARCH_ENABLED` | off | `true` | OpenAI native web search (`gpt-5.6-terra`, low effort, via OpenRouter) running in parallel with the primary provider |
| `GEMINI_SEARCH_ENABLED` | off | `true` | First-party Google grounded search via the `google-genai` SDK |
| `FINANCIAL_DATA_ENABLED` | off | `true` | yfinance + FRED data for questions an LLM classifier tags as financial |
| `PREDICTION_MARKETS_ENABLED` | off | `true` | Polymarket / Kalshi / Manifold / PredictIt snapshot (suppressed under `is_benchmarking=True`) |
| `RESOLUTION_SOURCE_ENABLED` | off | `true` | Tier-1 fetcher of URLs cited in the resolution criteria (plain HTTP + trafilatura, no LLM) |
| `TS_ANCHOR_ENABLED` | off | `true` | Time-series empirical P10/P50/P90 band from a question's own resolution series |
| `TS_ANCHOR_CHART_ENABLED` | off | `false` | Chart-image side-channel for the anchor (vision message to base models); held off pending a text-vs-image A/B |
| `RESEARCH_PROVIDER` | `auto` | unset | Forces one primary provider (`asknews`/`exa`/`perplexity`/`openrouter`) instead of the priority order |

The primary provider is chosen by priority: AskNews (when
`ASKNEWS_CLIENT_ID` + `ASKNEWS_SECRET` are set, the prod case), then Exa, then
Perplexity, then Perplexity-via-OpenRouter. The flags above run on top of the
primary, each independently gated.

### Gap-fill

| Flag | Code default | Prod value | What it gates |
|---|---|---|---|
| `GAP_FILL_ENABLED` | off | `true` | v1 gap-fill: analyzer finds up to 5 factual gaps, parallel native searches resolve each |
| `GAP_FILL_V2_ENABLED` | off | `true` | v2 agentic research loop (`research/agentic/`); runs concurrently with v1 during the overlap window |

Both gap-fill passes run in prod as of 2026-07-17. Each soft-fails to an empty
string on any error, and both are suppressed under `is_benchmarking=True`. v2's
driver model and effort default to `gpt-5.6-terra` at `effort=low`
(`GAP_FILL_V2_DRIVER_MODEL` / `GAP_FILL_V2_DRIVER_EFFORT`); its wall deadline is
540s and it caps at 20 tool calls (`constants.py:405-444`).

### Stacking

| Flag | Code default | Prod value | What it gates |
|---|---|---|---|
| `BINARY_STACKING_ENABLED` | off | `false` | Stacker LLM on binary questions |
| `MC_STACKING_ENABLED` | off | `false` | Stacker LLM on multiple-choice questions |
| `NUMERIC_STACKING_ENABLED` | off | `false` | Stacker LLM on numeric questions |

The aggregation strategy is `CONDITIONAL_STACKING` (set in `cli.py:120`), but
all three stacking flags are `false` in every workflow, so prod effectively runs
MEDIAN aggregation. The stacker chain stays live for backtests and ablation. The
disable is evidence-backed: an n=88 ablation found the stacker hurts numeric
CRPS and is no better than median on binary.

### Other flags

| Flag | Code default | Prod value | What it gates |
|---|---|---|---|
| `PROBABILISTIC_TOOLS_ENABLED` | off | `false` | The deterministic probability-math post-processor (`tool_runner.py`); wired but dormant |
| `PERSIST_RESEARCH_ENABLED` | off | `true` (prod runs; off in test_bot) | Writes per-question research to JSONL for offline backtest replay |
| `PLATT_CALIBRATION_ENABLED` | off | unset | Post-hoc logistic recalibration of the final published probability |
| `GEMINI_USE_DONATED_OPENROUTER_KEY` | on | `true` | Route OpenRouter Gemini calls through the donated key with personal fallback |
| `OPENROUTER_CREDIT_FLOOR_USD` | `1.0` | unset (uses default) | Donated-key remaining-balance floor for the end-of-run refill reminder |
| `OPENROUTER_CREDIT_ALERT_RESUME_DATE` | `2026-09-10` | unset (uses default) | Date the credit alerts start reddening CI again; before it, credit shortfalls log but exit zero |

## GitHub Actions workflows

Four workflows live in `.github/workflows/`. All four share the same setup
(checkout, `uv sync --no-dev --frozen`, install Playwright Chromium for gap-fill
v2's rendered-fetch rung), the same env block, and a 300-minute job timeout that
is a backstop for a wedged run, not a normal duration. Each tees stdout and
stderr to a `run_logs/` file and uploads it as an artifact with 90-day
retention.

| Workflow | Trigger | Mode | What it does |
|---|---|---|---|
| `run_bot_on_tournament.yaml` | cron at :03/:23/:43 hourly, plus manual | `tournament` | Forecasts new questions in the current AI benchmark tournament (`TOURNAMENT_ID` in `constants.py`); publishes to Metaculus |
| `run_bot_on_minibench.yaml` | cron at :08/:38 hourly, plus manual | `minibench` | Forecasts the current MiniBench question set; publishes |
| `run_bot_on_metaculus_cup.yaml` | cron 00:03 every 2 days, plus manual | `metaculus_cup` | Forecasts open Metaculus Cup questions (human + bot competition); publishes |
| `test_bot.yaml` | manual only (`workflow_dispatch`) | `test_questions` | Runs a fixed handful of example questions end-to-end in prod mode; publishes comments |

All four skip already-forecasted questions (`skip_previously_forecasted_questions`)
so a re-run never double-spends or re-publishes. The three scheduled workflows
split their cron across offset entries because GitHub silently drops `*/N`
schedules under runner load, and a `concurrency` group prevents overlapping
runs of the same workflow.

The prod workflows (tournament / minibench / cup) upload their artifact as
`research-<run_id>` and include `research_outputs/`. `test_bot.yaml` uploads
`logs-<run_id>` with only `run_logs/`, so the research-archive sync script never
picks up test runs. `test_bot.yaml` also does not set `PERSIST_RESEARCH_ENABLED`.

`ci.yaml` is the pull-request check (lint + tests); the `gemini-*` and
`claude.yml` workflows are repo automation unrelated to forecasting.

## Cost discipline

Any command that hits live LLM or research APIs spends real money and, for the
run modes, publishes to Metaculus. Never launch one without operator approval,
even after a clean build. This is a hard rule, documented at the top of
`AGENTS.md`.

Paid and externally-visible (approve before each):

- `uv run python main.py` / `make run` in any live mode — spends API credits
  and publishes to Metaculus.
- `make backtest_smoke_test` / `_small` / `_medium` / `_large` — spends credits
  on every forecaster and research call (no publish, but real money;
  `_large` is 100 questions).
- The ablation targets (`make ablation_*`) and anything invoking research
  providers or the ensemble against real questions.

Free and safe (run freely): `make test`, `make lint`, `make format`,
`make typecheck`, `make check_credits`, `make benchmark_display` (views old
results), the performance-analysis and width-monitor tooling below, and any
unit or integration test. The suite is self-contained and hits no paid APIs.

When you need a paid run to verify something, surface the exact command and a
rough cost and let the operator decide.

## Credit telemetry and the refill floor

Every run logs OpenRouter balances for both keys at start and end, and computes
per-run spend. The code is `metaculus_bot/credit_telemetry.py`, wired into
`cli.py:130`; balances come from the `/auth/key` endpoint via
`check_openrouter_credits.py`.

Marker lines land in the `run_logs/` artifact (all four workflows tee stdout +
stderr), so per-run spend is durably grep-able:

- `CREDIT_BALANCE: key=<donated|personal> phase=<start|end> remaining=... usage=...`
- `CREDIT_SPEND: key=... run_delta_usd=... remaining=...` at end of run.
- `CREDIT_FLOOR_BREACH: key=donated remaining=... floor=...` when the donated
  key's remaining balance drops below `OPENROUTER_CREDIT_FLOOR_USD` (default $1,
  `constants.py`).

A floor breach does not abort the run. Forecasting and publishing complete
normally, and outside the suppression window below `cli.py` then exits non-zero
so the GitHub Actions check turns red as a reminder to top up the donated key.
The floor is only checked against the donated key (the personal key reports no
`limit_remaining`). Per-run spend prefers the `limit_remaining` drop because the
donated key routes nearly all spend through BYOK provider integrations, which
leaves the plain `usage` field frozen while real money burns.

### Credit alerting is suppressed until 2026-09-10

The operator is funding the rest of the season out of pocket, so an empty donated
key is the expected state rather than a defect. Until
`CREDIT_ALERT_RESUME_DATE` (`2026-09-10` in `constants.py`, a few days after the
tournament closes on `TOURNAMENT_END_DATE`), credit shortfalls no longer redden
CI. Two paths are gated, because either one alone would keep the check red:

1. The floor breach. `cli.py` skips the `sys.exit(1)` and logs an INFO line
   saying the breach was observed but alerting is suppressed until the resume
   date.
2. The credit-caused donated-to-personal key fallbacks. When the donated key
   actually runs dry it returns 402 / insufficient credit, which is a
   fallback-worthy key-scoped error, and each fallback normally counts toward
   `alertable`. `fallback_openrouter.py` now tracks those credit-caused
   fallbacks in `_credit_key_fallback_count`, a subset of the all-causes
   `_generic_key_fallback_count`, and `cli.py` subtracts the subset back out
   while alerting is suppressed.

Non-credit fallback causes still alert in full, since each means real breakage
rather than an empty wallet: 401 invalid or disabled key, 404 "no allowed
providers", 429 rate limit, and the guardrail / data-policy block. Bot-side
degradation (forecaster drops, stacker fallbacks, research timeouts) is
untouched by the suppression.

Nothing is silenced. Every `CREDIT_*` marker line, `CREDIT_FLOOR_BREACH`
included, and every `PAID PERSONAL-KEY FALLBACK` warning fires exactly as
before; only the process exit status and the `alertable` arithmetic change. The
end-of-run summary renders the breakdown, including how many credit events were
suppressed and until when. The window is read from the system clock at call
time, so alerting resumes on the resume date with no redeploy, and behavior from
that date on is what it was before the suppression. `credit_alerts_active()` in
`constants.py` takes an optional `today` so tests pin both sides of the
boundary.

Balances outside a run:

```bash
make check_credits                    # both keys
make check_credits ARGS="--key donated"
```

## Backtesting

The primary benchmark scores bot predictions against actual question
resolutions. It spends API credits (it runs the real ensemble and research), so
it is gated by the cost rule above.

```bash
make backtest_smoke_test   # 4 questions
make backtest_small        # 12
make backtest_medium       # 32
make backtest_large        # 100
```

The prediction-market snapshot and the resolution-source fetcher are hard-off
under `is_benchmarking=True` to avoid leaking post-resolution data, so their
forecasting value cannot be measured by these targets. They were validated via
manual `test_bot.yaml` prod-mode runs and opt-in live integration tests instead.

To backtest against cached, non-leaky research from the archive:

```bash
make backtest_with_cache   # uses backtests/research_archive/latest
```

The old `community_benchmark.py` path is deprecated: Metaculus removed the
`aggregations` field from the list API, so baseline scoring is broken.
`make benchmark_display` still views old results.

## Performance analysis and the width monitor (read-only, free)

`metaculus_bot/performance_analysis/` evaluates the live bot's calibration
against actual resolutions. The pull hits only the Metaculus API (resolved
questions plus the bot's own comments, user id 275109, auth via
`METACULUS_TOKEN`). It makes no LLM or research calls and does not publish, so
it is not subject to the cost gate.

```bash
uv run python -m metaculus_bot.performance_analysis --tournament <slug> --output <path>
```

The `--tournament` default is `spring-aib-2026`, so pass the current slug
explicitly. Pass `--cached <path>` to re-analyze a saved dataset without
re-fetching.

The width monitor (`performance_analysis/width_monitor.py`) tracks how wide the
published numeric distributions are and how well that width is calibrated, split
by config era. Era-bucketing is mandatory for any calibration claim: the bot's
roster and pipeline change often enough that pooled calibration numbers are
misleading. The monitor reports central-80% and central-50% coverage with
Jeffreys-prior CIs, tail coverage (cov@10/50/90), PIT std, and median relative
band width per era.

```bash
uv run python -m metaculus_bot.performance_analysis.width_monitor --tournament <slug>
# or against a cached dataset:
uv run python -m metaculus_bot.performance_analysis.width_monitor --cached <path>
```

Before either analysis, run `make sync_research` (also read-only and free) so
the per-provider research artifacts in `backtests/research_archive/latest/` are
fresh. GHA artifacts expire at 90 days, so this local archive is the only
durable copy; `scripts/research_sync/` has a weekly launchd job.

## Reading run logs

Each run tees to `run_logs/run_<run_id>_<timestamp>.log`, uploaded as a workflow
artifact (`research-<run_id>` for the three prod workflows, `logs-<run_id>` for
`test_bot`). Grep these for the telemetry markers:

- `EXTRACTION_RUNG: question=... model=... qtype=... rung=... block_present=...`
  — one line per forecast value extraction. Watch for `rung=llm` (LLM salvage
  fired) and `block_present=false` (a forecaster stopped emitting a well-formed
  structured block). Emitted in `value_extraction.py:90`.
- `OPEN_BOUND_PILING: question=... model=... bound=... bin_mass=... ...` — a
  forecaster piled 10%+ of its mass on the terminal displayed bin of an
  open-bound numeric question without declaring any percentile beyond the edge.
  `numeric/diagnostics.py`, threshold `OPEN_BOUND_PILING_THRESHOLD` in
  `numeric/config.py`.
- `GAP_FILL_V2: model=... steps=... tool_calls=... searches=... fetches=...
  rendered=... reads=... dup_tool_calls=... deadline_hit=... concluded_early=...
  wall_s=... findings=... pending_leads=... lint_rejections=...` — one summary
  line per gap-fill v2 loop (`research/agentic/loop.py:468`). A companion
  `GHOST_FORECAST` line logs the loop's private dry-run forecast for telemetry
  only; it is never published.
- `CREDIT_BALANCE` / `CREDIT_SPEND` / `CREDIT_FLOOR_BREACH` — credit telemetry,
  described above. `CREDIT_FLOOR_BREACH` keeps firing during the credit-alert
  suppression window, so seeing one on a green run is expected until 2026-09-10;
  the adjacent INFO line names the resume date.

A run can also exit non-zero for degradation alerts (forecaster drops, personal-
key fallbacks, model deprecations) even when every question that met the
minimum-forecaster threshold was published. The non-zero exit is the CI red-check
signal to investigate; it does not mean publishing failed. Credit-caused
shortfalls are exempt until 2026-09-10 (see the suppression section above); every
other cause still alerts. See the alert block near the end of `cli.py` for the
exact conditions.
