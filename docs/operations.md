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
| `GAP_FILL_ENABLED` | off | `true` | v1 gap-fill: analyzer finds up to `GAP_FILL_MAX_GAPS` factual gaps, parallel native searches resolve each |
| `GAP_FILL_V2_ENABLED` | off | `true` | v2 agentic research loop (`research/agentic/`); runs concurrently with v1 during the overlap window |

Both gap-fill passes run in prod as of 2026-07-17. Each soft-fails to an empty
string on any error, and both are suppressed under `is_benchmarking=True`. v2's
driver model and reasoning effort come from `GAP_FILL_V2_DRIVER_MODEL` /
`GAP_FILL_V2_DRIVER_EFFORT`; its wall deadline is `GAP_FILL_V2_WALL_DEADLINE` and
its tool-call budget is `GAP_FILL_V2_MAX_TOOL_CALLS`. All four are defined in
`constants.py`, which is the only place their values are worth reading.

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
degradation (forecaster drops, stacker fallbacks, research timeouts) is
untouched by the suppression.

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
(`fallback_openrouter.py:349`, whose docstring is the canonical version of this),
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

Before either analysis, run `make sync_all` (also read-only and free) so the
local archives are fresh: the per-provider research archive
(`backtests/research_archive/latest/`), the run-log telemetry archive
(`backtests/telemetry_archive/`), and the raw research-provider payload archive
(`backtests/research_archive/raw/`). Use `sync_all` rather than one of the
narrower `sync_*` targets — it is a single download pass over the union of
artifact families, so it is cheaper than running them in sequence, and GHA
artifacts expire at 90 days, which makes anything a partial pull skipped
permanently unrecoverable. The weekly launchd job in `scripts/research_sync/` is
wired to `sync_all` for the same reason.

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
