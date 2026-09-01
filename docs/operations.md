# Operations & configuration

How to set up, configure, and run the Metaculus forecasting bot. This is the
reference for a human operating the bot: local setup, API keys, the environment
flags, the GitHub Actions workflows, cost discipline, and the telemetry you can
grep after a run.

For a code-level map of the pipeline, read `AGENTS.md` at the repo root. This
doc points at code by file and symbol name, so `rg <symbol>` takes you there.

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

### Git hooks

`make precommit_install` installs both hook types (`pre-commit` alone installs only
the first):

- **at commit** — the ruff hooks (check with `--fix --unsafe-fixes`, then format), plus
  `no-commit-to-main`, which refuses a commit whose HEAD is `main`. `main` is
  ruleset-protected on GitHub (PR required, `lint` + `test` required), so a direct push is
  rejected — but only at push time, once the commits already sit on local `main` and have
  to be replayed onto a branch. The guard moves that refusal to commit time, where the fix
  is one `git switch -c`. Its message names the recovery command and the `git commit
  --no-verify` bypass; `scripts/hooks/no_commit_to_main.sh` is a `language: script` hook,
  so it must stay executable in the index (mode `100755` — `tests/test_no_commit_to_main_hook.py`
  asserts that, along with the behavior on a feature branch and on a detached HEAD).
- **at push** — the full pytest suite, the same command CI runs. ~105s, too much friction
  per commit but the right price on the thing reviewers see.

Two things can block or stale the install on a checkout that predates the uv migration:

- **A pre-existing `core.hooksPath` makes pre-commit refuse to install** ("Cowardly
  refusing to install hooks with `core.hooksPath` set"). Check where it points with
  `git config --show-origin --get core.hooksPath` against `git rev-parse --git-path hooks`.
  If they match, the setting is redundant — it names git's own default hooks directory —
  so `git config --unset-all core.hooksPath` unblocks the install and changes nothing
  about where git looks for hooks.
- **The generated `.git/hooks/pre-commit` can be stale.** Hooks installed before the uv
  migration hardcode an `INSTALL_PYTHON` under the old conda env, which still resolves on
  disk and so fails confusingly rather than obviously. `make precommit_install` regenerates
  the file against the current interpreter.

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
  `GEMINI_USE_DONATED_OPENROUTER_KEY` (default `true`). The Gemini Pro
  forecaster slot is pinned to the personal key by the
  `DONATED_KEY_BLOCKED_GOOGLE_MODELS` blocklist in `fallback_openrouter.py` (no
  donated attempt), so a credit error on a Pro call is always a personal-key
  issue.
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
`GOOGLE_API_KEY`. A `403` splits three ways:

- Body says `Key limit exceeded` — a drained spend cap. Falls back to the
  personal key and is credit-classified, whatever status came with it.
- Body says `no allowed providers`, `guardrail`, or `data policy` — scoped to
  the donated key's routing, so the personal key genuinely can serve the call.
  Falls back, but is NOT credit-classified.
- Anything else — a moderation or permission refusal. Does not fall back, since
  both keys would refuse the same prompt. Those two phrasings are the only ways
  out of this branch, so it holds even when the body happens to contain ordinary
  credit English like "insufficient funds": on a reported 403 the body is the
  least trustworthy input we have (see the `flagged_input` prompt replay below),
  and credit wording there classifies as neither credit nor a key issue.

A `429 rate limit` is not a key defect but does fall back, since BYOK quotas are
per-key. See "What a dry donated key actually returns" below.

## Environment flags

Flags are read at call time via `env_flag_enabled` in `constants.py`, which
treats `true`/`1`/`yes` as on and `false`/`0`/`no` as off (case-insensitive).
When a flag is unset it takes the code default shown below. The bot workflow
YAMLs set these explicitly, so the "prod value" column is what actually runs in
CI.

### Research providers

| Flag | Code default | Prod value | What it gates |
|---|---|---|---|
| `NATIVE_SEARCH_ENABLED` | off | `true` | OpenAI native web search via OpenRouter (model and reasoning effort from `NATIVE_SEARCH_DEFAULT_MODEL` / `NATIVE_SEARCH_REASONING_EFFORT_DEFAULT`), running in parallel with the primary provider |
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
| `GAP_FILL_ENABLED` | off | `true` | v1 gap-fill: analyzer finds up to `GAP_FILL_MAX_GAPS` factual gaps, parallel native searches resolve each |
| `GAP_FILL_V2_ENABLED` | off | `true` | v2 agentic research loop (`research/agentic/`); runs concurrently with v1 during the overlap window |

Both gap-fill passes run in prod as of 2026-07-21 (v2 was authored 2026-07-17 but reached
`main` in merge `b4e9df0`; era analysis keys on the latter). Each soft-fails to an empty
string on any error, and both are suppressed under `is_benchmarking=True`. v2's
driver model and reasoning effort come from `GAP_FILL_V2_DRIVER_MODEL` /
`GAP_FILL_V2_DRIVER_EFFORT`; its wall deadline is `GAP_FILL_V2_WALL_DEADLINE` and
its tool-call budget is `GAP_FILL_V2_MAX_TOOL_CALLS`. Every `GAP_FILL_V2_*`
setting is defined in `constants.py`, which is the only place their values are
worth reading; `docs/agentic_gap_fill.md` has the full env-var table.

### Stacking

| Flag | Code default | Prod value | What it gates |
|---|---|---|---|
| `BINARY_STACKING_ENABLED` | off | `false` | Stacker LLM on binary questions |
| `MC_STACKING_ENABLED` | off | `false` | Stacker LLM on multiple-choice questions |
| `NUMERIC_STACKING_ENABLED` | off | `false` | Stacker LLM on numeric questions |

The aggregation strategy is `CONDITIONAL_STACKING` (set in `cli.py`'s `main`), but
all three stacking flags are `false` in every workflow, so prod effectively runs
MEDIAN aggregation. The stacker chain stays live for backtests and ablation. The
disable is evidence-backed: an n=88 ablation found the stacker hurts numeric
CRPS and is no better than median on binary.

### Other flags

| Flag | Code default | Prod value | What it gates |
|---|---|---|---|
| `PROBABILISTIC_TOOLS_ENABLED` | off | `false` | The deterministic probability-math post-processor (`tool_runner.py`); wired but dormant |
| `PERSIST_RESEARCH_ENABLED` | off | `true` (every bot workflow, test ones included since 2026-08-03) | Writes per-question research to JSONL for offline backtest replay |
| `PLATT_CALIBRATION_ENABLED` | off | unset | Post-hoc logistic recalibration of the final published probability |
| `GEMINI_USE_DONATED_OPENROUTER_KEY` | on | `true` | Route OpenRouter Gemini calls through the donated key with personal fallback |
| `OPENROUTER_CREDIT_FLOOR_USD` | see `constants.py` | unset (uses default) | Donated-key remaining-balance floor for the end-of-run refill reminder |
| `OPENROUTER_CREDIT_ALERT_RESUME_DATE` | `2026-09-10` | unset (uses default) | Date the credit alerts start reddening CI again; before it, credit shortfalls log but exit zero |

## GitHub Actions workflows

Five bot workflows live in `.github/workflows/`. They share the same setup
(checkout, `uv sync --no-dev --frozen`, install Playwright Chromium for gap-fill
v2's rendered-fetch rung), the same env block, and a `timeout-minutes` job cap
that is a backstop for a wedged run, not a normal duration. Each tees stdout and
stderr to a `run_logs/` file and uploads it as an artifact with 90-day
retention.

| Workflow | Trigger | Mode | What it does |
|---|---|---|---|
| `run_bot_on_tournament.yaml` | cron at :03/:23/:43 hourly, plus manual | `tournament` | Forecasts new questions in the current AI benchmark tournament (`TOURNAMENT_ID` in `constants.py`); publishes to Metaculus |
| `run_bot_on_minibench.yaml` | cron at :08/:38 hourly, plus manual | `minibench` | Forecasts the current MiniBench question set; publishes |
| `run_bot_on_metaculus_cup.yaml` | cron 00:03 every 2 days, plus manual | `metaculus_cup` | Forecasts open Metaculus Cup questions (human + bot competition); publishes |
| `test_bot.yaml` | manual only (`workflow_dispatch`) | `test_questions` | Runs a fixed handful of example questions end-to-end in prod mode; publishes comments |
| `test_bot_basic.yaml` | manual only (`workflow_dispatch`) | `test_questions` | One-question smoke test; publishes one comment. See below |

The three prod workflows are the only ones with a `schedule:` block; both test
workflows are `workflow_dispatch` and never fire on their own.

All five skip already-forecasted questions
(`skip_previously_forecasted_questions`) except in `test_questions` mode, where
`cli.py` deliberately turns that off so a re-run re-forecasts the same test
question. The three scheduled workflows split their cron across offset entries
because GitHub silently drops `*/N` schedules under runner load, and a
`concurrency` group prevents overlapping runs of the same workflow.
`test_bot_basic.yaml` has its own group, so a smoke run never contends with a
full `test_bot` run.

All five bot workflows (the three prod tournaments plus `test_bot` and
`test_bot_basic`) upload their artifact as `research-<run_id>` with both
`research_outputs/` and `run_logs/`, and all five set
`PERSIST_RESEARCH_ENABLED`. The two test workflows joined that shape on
2026-08-03: they previously uploaded `logs-<run_id>` with only `run_logs/` and
set no persist flag, which was framed as keeping test runs out of the research
archive but in practice just discarded their research. Three runs' worth of
assembled per-question research is gone that way — we still hold their raw
provider payloads and telemetry markers, but not the briefing the forecasters
read. Test runs now contribute to the archive on purpose; they forecast the
evergreen questions, so their records are the ones backtest replay wants most.

`ci.yaml` is the pull-request check (lint + tests); the `claude.yml` workflow is
repo automation unrelated to forecasting.

### The one-question smoke test (`test_bot_basic.yaml`)

This is the cheapest way to exercise the whole live pipeline end to end. It
forecasts exactly one question — Q14333, "Age of Oldest Human as of 2100", a
plain-continuous numeric — chosen because numeric carries the deepest
type-specific pipeline and is the likeliest thing to break. Every research flag
matches `test_bot.yaml`, so a run touches AskNews, native search, Gemini
grounded search, financial data, both gap-fill passes, prediction markets, the
resolution-source fetcher, and the time-series anchor. It runs in prod mode
(`is_benchmarking=False`), which means it **publishes a comment to Metaculus**.

Cost is about $2.60 per run at current config. That is real OpenRouter and
research-API money plus a published comment, so firing it is the operator's
call under the cost rule above. An agent may propose and price it; it does not
dispatch it.

The single question comes from `TEST_QUESTIONS_OVERRIDE` (the env var named by
`TEST_QUESTIONS_OVERRIDE_ENV` in `constants.py`), which `cli.py`'s
`test_questions` path reads as a whitespace- or comma-separated URL list. Unset,
the same mode would forecast the full `EXAMPLE_QUESTIONS` set, which is what
`test_bot.yaml` does.

Firing it, from the Actions UI or the CLI:

```bash
gh workflow run test_bot_basic.yaml --repo No-Stream/metaculus-bot --ref <branch>
```

The workflow has no inputs, so the only choice is which ref to run. Two things
about the plumbing are easy to get wrong.

First, pass `--repo`. This checkout has two remotes — `origin` is the operator's
fork `No-Stream/metaculus-bot`, `upstream` is the Metaculus template it was
forked from — and no `gh` default repo is configured, so a bare
`gh workflow run` or `gh workflow list` resolves against the *upstream* template
and reports a workflow list that does not include this one.

Second, and the yaml header calls this out: a `workflow_dispatch` workflow only
appears in the Actions "Run workflow" UI once its file exists on the **default**
branch. A brand-new dispatch-only workflow on a feature branch is invisible
until it merges to `main`. That is already satisfied here — the file is on
`origin/main` and `gh workflow list --repo No-Stream/metaculus-bot` shows "Test
Bot Basic (1 numeric Q smoke)" as active — so the `--ref` argument can point at
any branch you want to test.

Afterward, the log is in the `research-<run_id>` artifact (90-day retention),
tee'd from `run_logs/` during the run, alongside the run's
`research_outputs/` JSONL. Worth grepping in the downloaded log:

- `PAID PERSONAL-KEY FALLBACK` (`fallback_openrouter.py`) — a call fell off the
  donated key onto the operator's personal one.
- `DONATED_KEY_STATE:` (`credit_telemetry.py`) — the `/auth/key` probe's verdict
  on why a credit-shaped failure happened (`drained`, `zeroed`, `revoked`,
  `funded`, `unknown`).
- `CREDIT_BALANCE:` / `CREDIT_SPEND:` / `CREDIT_FLOOR_BREACH:` — the per-key
  balances at start and end, the run's spend delta, and the refill warning. Read
  `CREDIT_SPEND`'s `source=` field before trusting the number:
  - `source=remaining_delta` (the donated key) is reliable.
  - `source=usage_delta_unsettled` (the personal key, which reports no
    `limit_remaining`) is a **lower bound**, and frequently `0.00` on a run that
    spent real money — OpenRouter has usually not settled the spend by the time
    the end snapshot fires. A `CREDIT_SPEND_UNSETTLED` warning accompanies it.
    **Do not read `0.00` here as "this run was free."** Measured over 178
    archived personal-key runs: the markers captured 58% of true spend and 160 of
    178 read exactly `0.00`.
  - For the settled per-run figure, run
    `uv run python scripts/reconcile_credit_spend.py` (free, offline, reads the
    telemetry archive). It differences each run's start usage against the next
    run's, which is the only place the lagged spend is observable. The most
    recent run has no successor yet, so it shows as unsettled until another runs.
- `FORECASTERS_SURVIVED:` (`forecaster.py`) — the answer to "did every forecaster
  survive?", as `survived=n/N models=...`. Check it rather than inferring: the
  minimum to publish is low enough that a thinned ensemble still exits zero, and
  the failure-path "Only n/N forecasters succeeded" line stays silent on a
  degraded-but-published question. Anything below `n == N` means a model dropped,
  and `FORECASTER_DROPS` names which and why.

The general telemetry markers under "Reading run logs" below apply too; those
are just the money-shaped ones.

## Cost discipline

Every credit spend goes through the operator. Anything that hits a live LLM or
research API spends real money, and the run modes also publish comments to
Metaculus, which is a visible external action that is hard to retract. Nothing
in that class launches without the operator saying yes first. `AGENTS.md` at the
repo root carries the terse agent-facing version of the same rule.

The gate is on the **spend**, not on the mechanism. It covers anything that
causes a paid call no matter who or what finally makes it: a local `make`
target, a GitHub Actions dispatch of a bot workflow, an edit that adds cron
entries to a `schedule:` block, a flag change that raises per-run cost, or a
one-off script that wraps any of those. There is no clean-gates exemption and no
threshold below which a run is small enough to skip asking. A two- or
three-dollar smoke run still goes through the operator. When a paid run is the
only way to verify a change, the right move is to name the exact command, price
it, and stop there.

What the gate forbids is an agent *deciding* to spend. An explicit instruction is
the approval: told to fire a run already discussed, an agent should run it and not
re-ask. That approval is per-run. One go-ahead is not standing authorization for
the next run, or for re-running the same one after further changes.

Paid runs are a final pre-merge check rather than part of the verification loop.
The one-question smoke test below exists to be fired once, deliberately, when a
change is otherwise finished and about to merge. Its small per-run cost is the
trap: an agent that treats it as a normal check-my-work step fires it several
times in a session and spends real money for no added signal, since the run tells
it nothing the free gates did not. The loop is `make test`, `make lint`, and
`make typecheck`, with unit and integration coverage as the proof of correctness.
The paid run is the operator's last step.

### Paid and externally visible

- `uv run python main.py` / `make run` in any live mode (`tournament`,
  `minibench`, `metaculus_cup`, `test_questions`) — spends credits and publishes
  to Metaculus. `cli.py` builds the bot with `publish_reports_to_metaculus=True`
  in every mode.
- `make backtest_smoke_test` / `_small` / `_medium` / `_large` — spends on every
  forecaster and research call, plus one `LEAKAGE_DETECTOR_MODEL` call per
  question for the leakage screen. No publish (the benchmark config sets
  `publish_reports_to_metaculus=False` and `is_benchmarking=True`), but real
  money. The per-target question counts are the `--num-questions` values in the
  Makefile.
- `make backtest_with_cache` — the `--research-dir` flag replays archived
  research instead of fetching it, so the research and leakage-screen calls go
  away. The live ensemble still forecasts every question, so forecaster spend is
  real. A question with no archived record falls back to live research and the
  run logs a warning saying so.
- `make ablation_qa_research` / `ablation_smoke` / `ablation_small` /
  `ablation_medium` — real research plus forecaster spend.
- `make benchmark_run_*` — deprecated, since `community_benchmark.py` baseline
  scoring broke when Metaculus dropped `aggregations` from the list API, but the
  `run` and `custom` modes still fan the real ensemble over real questions.
  Prefer `make backtest_*`.
- `make test_live` — the only test target that leaves the network. It pins a
  `:free` OpenRouter model slug so the dollar figure is near zero, but the calls
  are real and need a live key, so it still goes through the operator.
- GitHub Actions runs of any bot workflow. A dispatched run spends exactly what
  the same mode spends locally and publishes to Metaculus the same way. See the
  workflow table above for triggers, and the smoke-test subsection there for the
  one-question variant.
- Any script that invokes a research provider or the ensemble against real
  questions, including one an agent writes on the spot.

### Free and safe

- Gates and formatting: `make test`, `make test_fast`, `make test_e2e`,
  `make lint`, `make format`, `make typecheck`, `make typecheck_ty`, `make cov`,
  `make audit`, `make precommit*`.
- Read-only Metaculus and GitHub-artifact pulls: `make sync_all` and its parts
  (`sync_research`, `sync_telemetry`, `sync_raw_research`, the `download_*` and
  `backfill_*` targets), the `performance_analysis` package and its width
  monitor, `make score_ghosts`, and `make close_margin_watch`.
- `make ablation_score` — `--stages score` hydrates every artifact off disk
  (`_hydrate_working_set_from_cache`) and makes no provider call.
- `make benchmark_display` — views saved benchmark results, no forecasting.
- `make check_credits` — reads the `/auth/key` balance for both OpenRouter keys.

The test suite is safe by construction, not by convention. The `e2e` marker
means a full-pipeline test with mocked LLMs, and `tests/conftest.py` installs an
autouse `_block_network_egress` fixture that raises on any AF_INET connect to a
non-loopback host. `addopts` deselects only the `live` marker, which is the one
suite that opts out of the egress guard because real calls are its whole point.
So a plain `make test` cannot reach a paid API even if a new test tries to.

`make score_ghosts ARGS="--tournament <slug>"` is worth calling out because
"live pull" reads like spend: it is a Metaculus-only fetch through
`build_performance_dataset`, with no LLM or research provider in the path.

## Credit telemetry and the refill floor

Every run logs OpenRouter balances for both keys at start and end, and computes
per-run spend. The code is `metaculus_bot/credit_telemetry.py`, whose
`CreditTelemetry` is wired into `cli.py`'s `main`; balances come from the
`/auth/key` endpoint via `check_openrouter_credits.py`.

Marker lines land in the `run_logs/` artifact (every bot workflow tees stdout +
stderr), so per-run spend is durably grep-able:

- `CREDIT_BALANCE: key=<donated|personal> phase=<start|end> remaining=... usage=...`
- `CREDIT_SPEND: key=... run_delta_usd=... remaining=... source=...` at end of
  run. `source` is `remaining_delta` (reliable), `usage_delta_unsettled` (a lower
  bound — see the smoke-test grep list above), or `unavailable`.
- `CREDIT_SPEND_UNSETTLED: key=... run_delta_usd=... is a LOWER BOUND ...` beside
  every `usage_delta_unsettled` figure, so a `0.00` is never mistaken for
  no-spend. `scripts/reconcile_credit_spend.py` recovers the settled number.
- `CREDIT_FLOOR_BREACH: key=donated remaining=... floor=...` when the donated
  key's remaining balance drops below `OPENROUTER_CREDIT_FLOOR_USD`
  (`constants.py`).

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
2. The credit-caused donated-to-personal key fallbacks. Each fallback normally
   counts toward `alertable`. `record_donated_key_fallback` tracks the
   suppressible subset in `_credit_key_fallback_count`, a subset of the
   all-causes `_generic_key_fallback_count`, and `cli.py` subtracts the subset
   back out while alerting is suppressed. Every event is counted exactly once:
   generic adds it, at most one subset subtracts it. That is why the whole
   accounting block in `record_donated_key_fallback` has to contain no `await`
   after the threaded probe — `+=` on a module global is interruptible between
   bytecodes, so an await there would let N forecasters failing on one dry key
   race the increment, undercount the generic total, and take a degraded run
   green.

Non-credit fallback causes still alert in full, since each means real breakage
rather than an empty wallet: 401 invalid or disabled key, 404 "no allowed
providers", 429 rate limit, and the guardrail / data-policy block. Bot-side
degradation is untouched by the suppression too: every counter in the
`Degradation counters:` summary still alerts in full (they are enumerated under
"Reading run logs" below).

### What a dry donated key actually returns (and the drained-vs-revoked probe)

A breached per-key spend cap does **not** come back as the 402 OpenRouter's
error docs describe. It comes back as HTTP **403** with the message
`Key limit exceeded (total limit)`, and litellm has no 403 branch for
OpenRouter, so it always surfaces as a bare `litellm.APIError` whose body
carries a `"code":403` field. On 2026-07-26 that cost a tournament run two of
three forecasters, native search, the AskNews summarizer, the financial-data
classifier, prediction-market keyword extraction, and both gap-fill passes: the
wrapper's negative rule vetoed any message containing "403" (written for content
moderation, where both keys really would refuse), so the operator's funded
personal key was never tried. The classifier now matches the phrase
`key limit exceeded`, which flips both the fallback decision and the credit
classification through the single shared helper (`_is_credit_failure`).

The cue has to be the full phrase. `limit exceeded` alone is a substring of
`rate limit exceeded: free-models-per-day`, so the short form would classify
every 429 as an empty wallet and silently exempt real rate-limit breakage from
alerting for the whole suppression window.

Text alone cannot tell a genuinely **drained** key from one Metaculus
**revoked** or **re-capped to zero** — all three produce that same 403 — and the
operator wants opposite CI colors for them. So on the first spend-cap failure of
a run, `credit_telemetry.classify_donated_key_state` reads the free, read-only
`/auth/key` endpoint once (verdict cached for the process) and classifies:

| `/auth/key` says | State | Alerting |
| --- | --- | --- |
| 200, cap > 0, nothing remaining | `drained` | suppressed — the expected empty wallet |
| 200, cap == 0 | `zeroed` | **red** — Metaculus cut us off, never an "empty wallet" |
| 401 / 404 | `revoked` | **red** — key is gone, not empty |
| 200, money remaining | `funded` | **red** — the failure was not about credit |
| probe failed, or no donated key configured | `unknown` | **red** — fail safe |

Only `drained` is subtracted from `alertable`. A probe that errors or times out
classifies as `unknown` and stays red, so a broken probe can never silently turn
a red run green.

The probe is what the *ambiguous* spend-cap 403 needs, so it is the only path that
pays for one. A documented 402 or plain insufficient-credit response says the
wallet is empty and nothing else, so `is_suppressible_credit_error` suppresses that
family before reaching the probe at all — deliberately, since it predates the
discriminator and an unreachable `/auth/key` must not change long-standing
behavior. Read the table above as the verdict on a spend-cap 403 specifically, not
on every credit failure (`test_documented_402_needs_no_probe` in
`tests/test_fallback_openrouter.py` pins the carve-out).

`DONATED_KEY_PROBE_TIMEOUT_S` bounds the probe, but read what shape of promise
that is: httpx applies a bare float **per network operation** — connect, read,
write and pool each get the full budget independently — so it is not a cap on
elapsed time. A server trickling bytes slower than the read timeout resets the
clock on every chunk, and a probe can run many multiples of the nominal budget
(measured against a local trickling server, a one-second timeout took ten
seconds to return twenty bytes). The hard total cap therefore lives at the one
latency-sensitive call site rather than in the timeout: on the fallback path
`record_donated_key_fallback` runs the probe on `asyncio.to_thread` under an
`asyncio.wait_for`, so the awaiting coroutine gives up on schedule however long
the socket takes. `wait_for` doesn't kill the worker thread, so a trickling probe
outlives that cap — orphaned, holding a socket and (under the probe's lock)
writing the cache — while the fallback proceeds without it. Callers outside that
path (the CLI, the start/end telemetry) run outside the forecasting window and
take the per-operation budget only. The state is logged as
`DONATED_KEY_STATE: state=<state>` (INFO for `drained`, WARNING for everything
else) and is echoed in the end-of-run summary as `donated_key=<state>` whenever a
probe actually ran.

Fallback **routing** reads the status the provider reported
(`llm_retry.llm_status_code`, an int already on the exception) and never a live
balance. A reported 403 falls back only on the spend-cap phrase or route-scoped
wording; a reported 402 always falls back; an exception carrying no status falls
back on text alone. The
`/auth/key` probe is consulted for alerting only (`is_suppressible_credit_error`),
so a stale or cached read reporting `funded` can never strand the ensemble on a
dry key — that is the exact failure this change exists to fix.

Two related hardenings ride along, both about how little the body can be trusted.

First, "was this about money?" has exactly one arbiter, `_is_credit_failure`
(in `fallback_openrouter.py`, whose docstring is the canonical version of this),
which both the routing decision and the alerting counter reach through. It reads
three tiers in a fixed order:

1. The spend-cap phrase `key limit exceeded` outranks everything, including the
   moderation veto below, and fires on any status or none. The production body
   renders as `403 Forbidden: Key limit exceeded`, and `forbidden` is both a
   moderation cue and generic HTTP boilerplate, so gating the phrase behind the
   veto would keep the dry key from falling back all over again.
2. Otherwise, a reported status decides alone: credit means exactly 402. So a
   reported 402 outranks moderation wording — `APIError(status_code=402,
   message="Blocked by moderation policy")` both falls back and is
   credit-classified — and credit English on any other reported status does not
   classify.
3. With no status reported, moderation wording (`moderation`, `forbidden`,
   `flagged_input`, `flagged for`) vetoes; failing that, a bare `402` or one of
   `payment required` / `insufficient credit` / `out of credits` /
   `insufficient funds` classifies.

That last ordering is why `insufficient credit` alone classifies as credit while
`blocked by moderation: insufficient credit` does not.

Second, OpenRouter moderation 403 bodies include `flagged_input`, up to ~100
characters of our own prompt replayed back, and a forecasting prompt full of
dollar figures and bill numbers can easily contain the token `402`. A bare `402`
substring match therefore read an ordinary moderation refusal as an empty wallet
— billing the personal key for a call that would refuse again, and exempting a
real moderation block from alerting. Everything after a prompt-echo marker is now
stripped before any word cue reads the body, and the bare digits are only trusted
when nothing in what remains looks like a moderation refusal. Word cues only,
deliberately: a genuine 402 links to a key hash with a small but non-negligible
chance of containing the substring `403` somewhere in it, and reading that as
moderation would break the long-standing 402 fallback. The odds are derived (and
pinned as bands) by `test_key_hash_status_collision_is_small_but_nonnegligible`
in `tests/test_llm_retry.py`, which is the only place that arithmetic lives.

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

The `--tournament` default is `DEFAULT_TOURNAMENT` (`performance_analysis/cli.py`)
and lags the live season, so pass the current slug explicitly. Pass
`--cached <path>` to re-analyze a saved dataset without re-fetching.

The width monitor (`performance_analysis/width_monitor.py`) tracks how wide the
published numeric distributions are and how well that width is calibrated, split
by config era. Era-bucketing is mandatory for any calibration claim: the bot's
roster and pipeline change often enough that pooled calibration numbers are
misleading. The monitor reports central-80% and central-50% coverage with
Jeffreys-prior CIs, tail coverage (cov@10/50/90), PIT std, median relative
band width, and `band_miss (lo/hi)` per era. That last one is the out-of-band
rate split by tail: it distinguishes a band that is too tight (both tails
elevated) from one of roughly the right width that is mis-centered (misses piled
in one tail), which `cov80` cannot express and which call for opposite
corrections.

Its era boundaries are **merge-to-main timestamps** (`WIDENING_FLIP`,
`TS_ANCHOR_ENABLE`), not authoring dates — prod runs from `main`, so a change is
live only once its merge commit lands there, and keying on the authoring date
files every run in the author-to-merge gap under the wrong config. Empty eras are
omitted, so while no post-july15-bundle numeric has resolved the `ts_anchor` row
is absent from the table rather than present-and-empty.

```bash
uv run python -m metaculus_bot.performance_analysis.width_monitor --tournament <slug>
# or against a cached dataset:
uv run python -m metaculus_bot.performance_analysis.width_monitor --cached <path>
# drop a standing exclusion cohort from every row; the excluded count is rendered
# in the table, so the exclusion is never silent. Three shorthands — known_bug
# (since-fixed pipeline defects), degraded_run (dry-key 1-of-3 publishes) and
# partial_degraded (2-of-3) — compose with each other and with explicit ids; the
# id sets live in performance_analysis/cohorts.py (EXCLUSION_COHORTS):
uv run python -m metaculus_bot.performance_analysis.width_monitor --cached <path> --exclude-qids known_bug,degraded_run
```

Before either analysis, run `make sync_all` (also read-only and free) so the
local archives are fresh: the per-provider research archive
(`backtests/research_archive/latest/`), the run-log telemetry archive
(`backtests/telemetry_archive/`), and the raw research-provider payload archive
(`backtests/research_archive/raw/`). Use `sync_all` rather than one of the
narrower `sync_*` targets — it is a single download pass over the union of
artifact families, so it is cheaper than running them in sequence, and GHA
artifacts expire at 90 days, which makes anything a partial pull skipped
permanently unrecoverable. The twice-weekly launchd job in
`scripts/research_sync/` is wired to `sync_all` for the same reason.

### The persisted artifact store, and re-parsing for free

`sync_all` downloads each artifact into `backtests/gha_artifact_store/<artifact-name>/`
and leaves it there — the extracted contents as `gh run download` unzipped them,
plus a `_meta.json` holding `artifact_id` / `name` / `created_at` / `run_id`. All
three archives are parsed FROM that store, never from a self-destructing temp
dir, which is the point: 90 days is a hard ceiling for this repo
(`{"days":90,"maximum_allowed_days":90}`), so GHA is a staging area and local
disk is the source of truth the moment an artifact is grabbed. An artifact
already in the store is never re-downloaded — uploads are immutable, so only
absent or half-extracted dirs are fetched.

```bash
make resync_from_store    # rebuild all three archives from local disk, zero network
```

Reach for that after fixing an ingest or parse bug: the bytes are already on
disk, so a corrected harvest costs nothing and still works on artifacts GitHub
has since deleted. Each sync script also accepts `--from-store` / `--store-dir`.
In `download_research.py` the two offline flags differ in an important way —
`--rebuild-only` re-merges the records already in `by_qid/`, while `--from-store`
re-reads the persisted JSONL and so can RECOVER records a past ingest bug
dropped. The offline path cannot ask GitHub which workflow a run belonged to, so
it recovers that from the telemetry archive's own `runs.jsonl`; a run entering the
store for the first time during an offline re-parse reads `workflow: unknown`
until the next online sync.

Storage is not a concern at this scale: 859 artifacts occupy 38 MB (median 4.4
KiB, mean 44 KiB, largest under 1 MB), and at ~13 artifacts/day that is roughly
17 MB/month, so about 210 MB after a year. Nothing needs compression, and
nothing is pruned on purpose — permanence is the whole point.

`uv run python -m scripts.research_sync.verify_completeness` checks store
coverage as its own FAIL condition (a live artifact missing from the store is
research one clock-tick from unrecoverable), separately from archive coverage.
Read the two signals differently: most artifacts legitimately hold no research at
all — 632 of the 859 carry only `run_logs/`, which is why the archive holds
artifact records from 227 runs rather than 859.

## Reading run logs

Each run tees to `run_logs/run_<run_id>_<timestamp>.log`, uploaded as a workflow
artifact (`research-<run_id>` for every bot workflow; the two test
workflows used `logs-<run_id>` before 2026-08-03, and those older artifacts are
still harvested — `RUN_LOG_ARTIFACT_PREFIXES` covers both names). Grep these for
the telemetry markers:

- `EXTRACTION_RUNG: question=... model=... qtype=... rung=... block_present=...`
  — one line per forecast value extraction. Watch for `rung=llm` (LLM salvage
  fired) and `block_present=false` (a forecaster stopped emitting a well-formed
  structured block). Emitted by `_log_extraction` in `value_extraction.py`.
- `OPEN_BOUND_PILING: question=... model=... bound=... bin_mass=... ...` — a
  forecaster put enough mass on the terminal displayed bin of an open-bound
  numeric question, without declaring any percentile beyond the edge, to trip
  `OPEN_BOUND_PILING_THRESHOLD` (`numeric/config.py`). Emitted by
  `numeric/diagnostics.py`.
- `GAP_FILL_V2: model=... steps=... tool_calls=... searches=... fetches=...
  rendered=... reads=... dup_tool_calls=... deadline_hit=... concluded_early=...
  wall_s=... findings=... pending_leads=... lint_rejections=...
  provenance_rejections=... quote_mismatch_warnings=... plan_gaps=...
  plan_skipped=... conclude_gate_rejections=... error=...` — one summary line per
  gap-fill v2 loop, emitted by `_log_completion` in `research/agentic/loop.py`.
  `error=` is what separates a step-zero crash from an idle run; both otherwise
  emit `steps=0 tool_calls=0 findings=0`. Companion `GHOST_PRE` /
  `GHOST_PRE_JSON` and `GHOST_FORECAST` / `GHOST_FORECAST_JSON` lines log the
  loop's pre- and post-research private forecasts for telemetry only; neither is
  ever published. `docs/agentic_gap_fill.md` reads the fields in full.
- `CREDIT_BALANCE` / `CREDIT_SPEND` / `CREDIT_FLOOR_BREACH` — credit telemetry,
  described above. `CREDIT_FLOOR_BREACH` keeps firing during the credit-alert
  suppression window, so seeing one on a green run is expected until 2026-09-10;
  the adjacent INFO line names the resume date.
- `TIME_BUDGET: question=... budget_s=... close_time=... close_limited=...
  fast_path=...` — one line per question, emitted by `time_budget.py` before any
  research runs. Emitted even on roomy questions on purpose: `CLOSE_MARGIN` fires
  only after a SUCCESSFUL submission, so it is censored on exactly the thin-window
  questions this budget exists for. `close_limited=true` means the question's own
  close time, not the static `PER_QUESTION_WALL_CLOCK_DEADLINE`, set the budget.
  `fast_path=true` means it fell below `TIME_BUDGET_FAST_PATH_THRESHOLD`, so the
  optional research stages were dropped to protect the prediction POST — companion
  `TIME_BUDGET_FAST_PATH` and `GAP_FILL_SKIPPED_FOR_BUDGET` WARNs say so too, and
  `RESEARCH_PHASE_DEADLINE` names any provider cancelled at the phase deadline.
  A question with no publishable budget at all (close already passed, or so near
  that the prediction POST cannot fit) is skipped before any spend and bumps
  `questions_failed_to_publish`.
- `Degradation counters: forecasters_dropped=..., questions_failed_to_publish=...,
  stacker_primary_failed=..., stacker_fallback_used=...,
  stacker_fallback_failed=..., research_provider_failures=...,
  summarizer_failures=..., gap_fill_v2_errors=...,
  prediction_market_degraded=..., prediction_market_source_losses=...,
  provider_degradation=..., publish_attempt_failures=...,
  publish_skipped_closed=..., time_budget_fast_path=...` — the
  end-of-run summary from `forecaster.py`'s `forecast_questions`, and the line
  that decides CI color: these are exactly the counters `alertable_count` sums, so
  any one of them non-zero exits the run non-zero.
  `time_budget_fast_path` is the earliest-firing member of the publish-side family:
  the other three fire once a publish has already failed or been withheld, while
  this one fires while the question is still savable and says latency is closing in
  on a close deadline.
  `research_provider_failures` counts any provider exception, not only timeouts —
  it was named `research_provider_timeouts` until 2026-07-26, when
  `prediction_market_platform_failures` also became
  `prediction_market_source_losses`. `scripts/telemetry/markers.py` matches both
  spellings, so archived pre-rename logs still harvest.
  `prediction_market_degraded` kept its name when the counter behind it moved off
  the retired Kalshi `/series` index onto the full events-catalogue pull, so the
  field name is stable across that change while what it guards got strictly more
  load-bearing — the catalogue feeds both the settlement-source join and the fuzzy
  channel. Note that a lost catalogue pull bumps BOTH this counter and
  `prediction_market_source_losses`, so one outage adds 2 to `alertable_count`;
  that is deliberate over-counting (the two carry different marker fields) and not
  two separate failures.

**One analysis hazard from ranked market retrieval, worth knowing before you diff
`providers_used` across eras.** The ranker may legitimately return zero rows, in
which case the provider renders nothing and the `## Prediction Market Snapshot`
header never appears. An ARTIFACT record still lists the provider under
`providers_attempted` (it ran, it just had nothing to say), but a COMMENT- or
LOG-backfilled record reconstructs `providers_used` by scanning for that header,
so the provider simply vanishes from it. So a drop in prediction-market presence
across backfilled records can mean "the ranker declined" rather than "the provider
broke", and the two are only distinguishable from an artifact record or from the
`MARKET_RANKING:` line's `outcome=` field. No code change: the header-scan
reconstruction is lossy by construction and always was.

`outcome=` alone does not say WHY a question fell back, so read the sibling
`MARKET_RANKING_DEGRADED:` line beside it: `reason=shape_regression` means a
well-formed but non-empty ranking array yielded no usable row — a renamed index key,
or every index outside the pool — i.e. OUR prompt/parser contract broke, and before
2026-08-25 that case was reported as `ok(0)` and rendered the deliberate-empty
sentence ("prediction markets were retrieved and reviewed… none was judged to bear on
it") to forecasters. `reason=unreadable` means the completion was not a ranking array
at all. Both are harvested as `market_ranking_degraded`, so the split survives the
90-day GHA log expiry; a `MARKET_RANKING` line with `outcome=failopen` and no
degraded sibling in the archive predates this marker.

A run can also exit non-zero for degradation alerts — the counters above,
personal-key fallbacks, or the model-deprecation tripwire — even when every
question that met the minimum-forecaster threshold was published. The non-zero
exit is the CI red-check signal to investigate; it does not mean publishing
failed. Credit-caused shortfalls are exempt until 2026-09-10 (see the suppression
section above); every other cause still alerts. See the alert block near the end
of `cli.py` for the exact conditions.
