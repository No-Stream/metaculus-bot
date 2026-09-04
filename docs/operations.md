# Operations & configuration

How to set up, configure, and run the Metaculus forecasting bot. This is the
reference for a human operating the bot: local setup, API keys, the environment
flags, the GitHub Actions workflows, cost discipline, and the telemetry you can
grep after a run.

For a code-level map of the pipeline, read `docs/architecture.md`; `AGENTS.md` at
the repo root is the terse agent-facing starting point and indexes both. This doc
points at code by file and symbol name, so `rg <symbol>` takes you there.

## Setup

Prerequisites: Python 3.12+ and [uv](https://docs.astral.sh/uv/). The project
uses uv for everything. There is no poetry, conda, or pip in this repo.

```bash
uv sync --dev        # create/update .venv from uv.lock (or: make install)
cp .env.template .env
```

Then fill in `.env` with your keys (see below). Never commit secrets: `.env` is
gitignored and is the only place real keys should live locally. Run any command
inside the project environment with `uv run <cmd>` — uv resolves the in-project
`.venv` automatically, so you never activate it by hand.

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
- **at push** — the full pytest suite, `uv run --frozen pytest --cov=metaculus_bot`, which
  is the command `.github/workflows/ci.yaml` runs. ~105s, too much friction per commit but
  the right price on the thing reviewers see.

Run the hooks by hand with `make precommit` (staged files) or `make precommit_all` (the
whole tree).

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

## Season-start checklist

Before a new tournament or Cup season opens its first question. These are
operator steps: the reads below are free metadata pulls with no inference spend,
but they are network calls and they feed a roster decision, so an implementing
session does not run them — it proposes, the operator runs and decides.

- **Resolve "latest per vendor" from a LIVE model-list read, never from memory.**
  The roster design (`FORECASTER_LLMS` in `metaculus_bot/llm_configs.py`) is the
  newest frontier reasoning model from each vendor, one slot each, and nothing in
  the repo can say what that currently resolves to — the 2026-08-31 gemini-slot
  review found that a roster decision needs this one read before anything else.
  OpenRouter's public models endpoint lists every slug with `created`, its
  listing time as a Unix timestamp (per the endpoint's OpenAPI schema):

  ```bash
  curl -s https://openrouter.ai/api/v1/models \
    | jq -r '.data[] | [.id, .created] | @tsv' | sort
  ```

  Filter per vendor prefix — `openai/`, `anthropic/`, `google/`, `x-ai/` — and
  read the newest `created` per vendor:

  ```bash
  curl -s https://openrouter.ai/api/v1/models \
    | jq -r '.data[] | [.id, (.created | todate)] | @tsv' \
    | grep -E '^(openai|anthropic|google|x-ai)/' | sort -t$'\t' -k2
  ```

  Then check, before touching the roster: the slug is a reasoning model, not a
  mini/flash/fast tier or a `:free` route; its provider is on the donated key's
  allowed list (`DONATED_KEY_PROVIDERS` in `fallback_openrouter.py`), or the slot
  knowingly bills the personal key like the pinned Google Pro slot
  (`DONATED_KEY_BLOCKED_GOOGLE_MODELS`); and its reasoning-effort enum accepts
  the tier the slot is configured for (the OpenAI ceiling is `xhigh`; `max` is
  Anthropic-only).
- **A roster change is a config-era boundary.** Residual analysis buckets by the
  merge-to-main timestamp, so make any swap once, in the same merge as everything
  else that shifts the forecast distribution, before the first question — never
  mid-window (`FUTURE.md`, "FREEZE the triple").
- **Refresh the tournament constants** (`TOURNAMENT_ID`, the date checks in
  `constants.py`) and flip the cup reminder off once configured
  (`FALL_CUP_CONFIGURED`), or every scheduled run reddens on the reminder.
  `TOURNAMENT_END_DATE` is the project's `forecasting_end_date`, not its
  `close_date` — on `summer-futureeval-2026` those sit two months apart.
- **Read the project object before editing a slug, and note the route.** A project
  is served at `/api/projects/tournaments/<slug-or-id>/`; a bare
  `/api/projects/<id>/` 404s for every id, which reads exactly like "this project
  does not exist" rather than like a wrong route. Slug and numeric id both resolve
  on the working route. The list endpoint `/api/projects/tournaments/` omits
  anything whose `visibility` is `unlisted`, which is the state a new season sits
  in before its first question, so ABSENCE from that list is not evidence a
  project does not exist — fetch the candidate directly, or walk the id space.
- **Take a question-supply census with `make supply_probe`.** It counts posts and
  questions at each status per tournament slug, and unlike the two scratch probes
  it replaces it counts post status `closed` — closed to forecasting but not yet
  resolved — which is what made two consecutive residual rounds' supply
  projections miss. It also lists the backlog of unresolved questions already past
  their own `scheduled_resolve_time`, worst overdue first, which is how you tell
  "Metaculus is late resolving" from "our pull is missing questions". It also sweeps
  FORFEITS: every question on a `closed` or `resolved` post that the bot never
  forecast at all, newest window first, with each window's length in hours. That
  sweep exists because a forfeited question never enters the performance dataset, so
  nothing downstream of the scoring pull can see one — the 2026-09-01 residual round
  found six lost to delivery (a cron gap, a late submit, three cancelled runs, one
  retroactive close) where the prior sweep had found one. Resolving "did we forecast
  this" needs `my_forecasts`, which the posts list does not reliably carry, so the
  sweep issues one extra read-only detail GET per closed/resolved post that the list
  page did not already answer for; pass `ARGS="--no-forfeits"` to skip that, at the
  cost of every question's state reading `unknown`. A question whose state stays
  unreadable is reported as `unknown` rather than filed as a forfeit, and a slug
  where NOTHING carries a bot forecast prints a warning to check that
  `METACULUS_TOKEN` is the bot's own token before believing the number. Default slugs
  come from the repo's own constants (`TOURNAMENT_ID`, `METACULUS_CUP_ID`,
  `FALL_CUP_SLUG`, plus minibench off `MetaculusApi.CURRENT_MINIBENCH_ID`), so it
  needs no arguments; scope or redirect it with
  `ARGS="--slugs metaculus-cup-fall-2026 --output /tmp/supply.json"`. Read-only and
  free — the Metaculus posts list and post detail only, no LLM, research, or publish
  call — so it sits outside the cost gate. A dead slug renders as one error row and the rest
  report normally, which makes this the cheapest way to watch for the fall cup
  opening: the `metaculus-cup-fall-2026` row goes from zero posts to non-zero on
  the day it does.

### Fall 2026 season: what was done on 2026-09-03, and what is left

Metaculus granted $1,500 of API credits for the bot to compete in both the fall
Metaculus Cup and the fall bot tournament. Landed in the repo:

- `METACULUS_CUP_ID` now holds `metaculus-cup-fall-2026`. It used to hold the
  undated `metaculus-cup` slug and rely on Metaculus redirecting it; Metaculus
  rejects that slug now (the posts list answers HTTP 400 for
  `tournaments=metaculus-cup`, re-verified 2026-09-03 with `make supply_probe`), so
  a cup run under it would have found no questions and forfeited the season with
  nothing in the log saying why. Verified read-only against
  `/api/projects/tournaments/metaculus-cup-fall-2026/`: project id 33108, name
  "Metaculus Cup Fall 2026", `start_date` 2026-08-28T12:00:00Z,
  `forecasting_end_date` 2027-01-01T00:00:00Z, `close_date` 2027-01-04T00:00:00Z,
  `score_type` `peer_tournament`, `visibility` `unlisted`,
  `bot_leaderboard_status` `exclude_and_show`, `questions_count` 0. The cup is
  open but had published nothing yet, and bots forecast on it outside the human
  leaderboard — `exclude_and_show` is what every recent cup season carries (fall
  2025, spring 2026 and summer 2026 read identically), so it is the cup's normal
  setting rather than anything fall-specific. `visibility` reads `unlisted` where
  those older seasons read `normal`, which is the pre-first-question state.
  **One analysis consequence: the cup scores on `peer_tournament`, not
  `spot_peer_tournament` like the bot tournament.** Cup records therefore carry a
  coverage-scaled `peer_score` and no spot peer, so they cannot be pooled with
  tournament records on one score field. `performance_analysis/platform_scores.py`
  already handles that — `RankingScore.tier` keeps spot-scored and peer-only records
  in separate sort tiers — but any new cut written over fall data has to respect it.
- `FALL_CUP_CONFIGURED` is True, so the dated reminder that would have reddened
  every run from 2026-09-15 is discharged, and its CI time bomb in
  `tests/test_tournament_dates.py` is now a pin that the cup stays pointed at a
  dated slug. Re-arming it for the spring 2027 cup means re-dating
  `FALL_CUP_REMINDER_DATE` and setting the flag back to False.
- `run_bot_on_metaculus_cup.yaml` is at full parity with
  `run_bot_on_tournament.yaml` — same env block, step caps, Playwright install and
  artifact upload — and moved from `3 0 */2 * *` (00:03 every second day) to
  hourly at :13/:33/:53. Hourly costs nothing when nothing is new, because
  `skip_previously_forecasted_questions` is on, and it removes most of the
  open-to-forecast latency that forfeited six triple-era questions. The minutes
  are staggered off the tournament's :03/:23/:43 and minibench's :08/:38 because
  the three workflows are in separate concurrency groups, so a shared minute means
  simultaneous runs rather than a queue.
- Research records are labelled by run mode (`cli.persisted_tournament_id`), so
  cup runs archive under the cup slug instead of the bot tournament's.

One adjacent thing the grant settles: credit alerting is back ON.
`make check_credits` on 2026-09-03 reads the donated key at **$1,449.19 remaining
of a $2,300 limit**, so a credit shortfall is real news again rather than the
expected state, and `CREDIT_ALERT_RESUME_DATE` was moved up from 2026-09-10 to
**2026-09-03** rather than left to expire. `OPENROUTER_CREDIT_FLOOR_USD` moved with
it, from $1.00 to **$100.00**: the operator cannot refill this key — Metaculus does
— so the warning has to arrive with runway left to ask, and $100 is roughly 250
questions at the measured $0.38-0.41 each. A $1 floor would have fired only once
the key was already dry, which on an hourly cup cron is an hourly red check that
arrives too late to act on.

Still the operator's, and not doable from a merge:

- **Enable the workflow on GitHub.** `run_bot_on_metaculus_cup.yaml` is
  `disabled_manually` there, which no file in this repo can change, so the crons
  above do not fire until it is enabled. Nothing in the repo will warn about this;
  the way to notice is a supply-probe row showing cup questions with no bot
  forecasts.
- **The fall bot tournament does not exist yet.** No successor to
  `summer-futureeval-2026` had been published as of 2026-09-03 (the id space above
  the summer tournament is empty, the four plausible slugs 404, and
  forecasting-tools still points `CURRENT_AI_COMPETITION_ID` at the summer id), so
  `TOURNAMENT_ID` stays on the summer season deliberately rather than being
  guessed. Consequence: from 2026-09-20 (`TOURNAMENT_END_DATE` plus
  `TOURNAMENT_HARD_STOP_WEEKS`) `check_tournament_dates` raises and both
  `--mode tournament` runs and the CI freshness test go red. That is the intended
  reminder, and it does not touch the cup — the cup mode never calls that check.
  The cheapest watch is `make supply_probe`: re-run it, and when a fall bot
  tournament appears, point the constants at it.

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
  fallback wrapper is `FallbackOpenRouterLlm` in
  `metaculus_bot/fallback_openrouter.py`.
- **`GOOGLE_API_KEY` — personal.** The operator's Google AI Studio key on a
  billing-enabled project. Powers the Gemini grounded-search provider and gap-
  fill v2's document reads. There is no donated Google AI Studio path. In CI
  this is stored as the `GEMINI_API_KEY` secret and surfaced to the workflow as
  `GOOGLE_API_KEY` so the `google-genai` SDK picks it up.

Gemini has two separate routes, which is the other easy thing to confuse:

- **OpenRouter Gemini** (forecaster / stacker / summarizer slots) routes
  donated-key-first with personal-key fallback, controlled by
  `GEMINI_USE_DONATED_OPENROUTER_KEY` (default `true`, since 2026-06-16). It is on
  by default because Metaculus raised the Google rate limits, so the donated key
  now serves most Gemini models — verified by live call, `gemini-3.5-flash` and
  `gemini-3.1-flash-lite` both succeed on it. Setting the toggle to a false-y
  value (`false`/`0`/`no`) forces personal-key-only routing for ALL Gemini; the
  three prod workflow YAMLs and `test_bot.yaml` pin it to `'true'` explicitly.

  **Known exception:** the Gemini Pro forecaster slot is PINNED to the personal
  key by the `DONATED_KEY_BLOCKED_GOOGLE_MODELS` blocklist in
  `fallback_openrouter.py` (read the blocklist for which models it currently
  covers). `should_route_via_donated_key` returns `False` for anything on it even
  with the toggle ON, so there is no donated attempt, no 429, and no
  personal-key-fallback-counter bump — which would otherwise redden CI on every
  question — and a credit error on one of those models is always a personal-key
  issue. It is pinned rather than falling back because that model routes through a
  free-tier Google AI Studio BYOK key on the donated account with no Pro free tier
  (quota 0 → `is_byok:true` + `FreeTier limit: 0`). This is a temporary workaround
  tagged `TODO(gemini-3.1-pro-donated)` in code: remove the blocklist entry once
  Metaculus fixes the BYOK routing — enable Cloud billing on the BYOK key's GCP
  project, remove the Google AI Studio BYOK integration so native OpenRouter
  Google credits are used, or disable "Always use for this provider" on that BYOK
  key — then re-verify with one live call. See
  `metaculus_bot/fallback_openrouter.py:should_route_via_donated_key` and
  `FUTURE.md` "Gemini on the donated OpenRouter key".
- **Gemini grounded search** (`research/gemini_search.py`) always uses the
  personal `GOOGLE_API_KEY`. The donated toggle does not touch it, and neither
  does anything else on the OpenRouter side — what that key costs is its own
  subsection below.

Other keys, all personal, no shared variants: `METACULUS_TOKEN`, `ASKNEWS_CLIENT_ID`
+ `ASKNEWS_SECRET`, `EXA_API_KEY`, `PERPLEXITY_API_KEY`, `FRED_API_KEY`,
`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`. The two direct provider keys only matter
if you bypass OpenRouter; most flows route through OpenRouter and don't need
them.

Diagnosing auth errors: an OpenRouter 401/402 on an OpenAI or Anthropic call
means suspect the donated key first (it's always tried first for those
providers). A 401/402/credit error on an OpenRouter Gemini call also means the
donated key first, since OpenRouter Gemini routes donated-first by default with
personal fallback — unless `GEMINI_USE_DONATED_OPENROUTER_KEY` has been forced
OFF, in which case suspect `OPENROUTER_API_KEY`; and anything on
`DONATED_KEY_BLOCKED_GOOGLE_MODELS` is pinned to the personal key with no donated
attempt, so a credit error on one of those models is always a personal-key issue.
A 401/402 on Grok, Qwen, or Perplexity is always the personal key
(the donated key 404s on those). A `google-genai` 401 or quota error is always
`GOOGLE_API_KEY`. A `403` splits three ways, with the reported status deciding the
branch and the spend-cap phrase outranking it:

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

The routing half of that decision — fall back to the personal key, or don't —
lives in `should_retry_with_general_key` (`fallback_openrouter.py`); the credit
classification is `_is_credit_failure`. Whether a spend-cap 403 is additionally
SUPPRESSED from CI alerting is a third, separate question, answered by the
`/auth/key` probe described below.

A `429 rate limit` is not a key defect but does fall back, since BYOK quotas are
per-key. See "What a dry donated key actually returns" below.

### Google AI Studio billing and the grounded-search allowance

`GOOGLE_API_KEY` is the operator's personal Google AI Studio key on a
BILLING-ENABLED (paid-tier, prepaid-credit) project, and marginal cost at current
usage is near zero. In CI it is stored as `secrets.GEMINI_API_KEY` and surfaced to
the workflow env as `GOOGLE_API_KEY` so the `google-genai` SDK picks it up. There
is NO Metaculus-donated Google AI Studio key.

Billing mechanics, verified against the ai.google.dev pricing / billing /
google-search docs on 2026-07-17 — don't re-litigate without fetching them again:
Gemini 3.x grounding is paid-tier-ONLY (the free-tier column reads "Not
available") and includes **5,000 free grounded prompts/month shared across all
Gemini 3 models per project, then $14/1k individual search queries**. Multi-query
prompts bill per QUERY on overage, and deep-research prompts fire several.

**Count queries, never prompts.** Current usage is ≈ 70-110 grounded PROMPTS per
month, but the pool is counted per search QUERY and the measured profile is 12.7
queries per prompt (`usage_metadata` plus
`grounding_metadata.web_search_queries` over 113 archived calls, 2026-07-20 →
08-28), so the real draw is ≈ 850-1,400 queries/month, ≈ 17-28% of the allowance.

The spring-2026 billing arc, explained: gap-fill's 5x grounded-call multiplier
plus backtest volume (`backtest_large` = 600 grounded prompts/run) blew past
5,000/month → per-query overage → prepaid-credit top-up debits ("started getting
billed"); the 2026-06-25 resolver migration (`a51617e`) cut the multiplier and new
charges stopped, with a residual ~$1/month of token spend silently drawing down
the prepaid balance. A reconstruction of the whole summer season
(`scratch/fetch_ladder_2026-09-03/`, plan doc
`scratch_docs_and_planning/fetch_ladder_plan_2026-09-03.md`) puts June 2026 alone
at ≈ 6,600 queries, because the pre-06-25 gap-fill resolver added ≈ 4 grounded
calls per question. Any future feature that multiplies grounded-call counts — or
Gemini-grounded backtests at scale — re-eats the same monthly pool.

**Watch item: prepaid-balance exhaustion produces 429s, not surprise charges.** If
Gemini grounded search starts soft-failing across a run, check the AI Studio credit
balance FIRST (`docs/research.md` § Gemini grounded search says the same from the
provider side).

**A third native surface is live.** The resolution-source fetcher's last escalation
rung is a `url_context` read on this same key, gated by
`RESOLUTION_SOURCE_URL_CONTEXT_ENABLED`, which defaults off in code and is set to `true` in every
bot workflow yaml since 2026-09-04. It
shares the reader (`research/url_context_reader.py`) and the model with v2's `read_document`, so
it adds token spend on the operator's personal key rather than a new billing
relationship, and it draws no grounded queries because it retrieves a URL instead of searching.
What it costs is bounded twice over. First by how often the free rungs fail, since a read
happens only for a cited URL no free rung could read and the free `Google-Extended` robots
pre-check declines the hosts that would refuse Gemini anyway. Second by a per-question cap:
`RESOLUTION_SOURCE_URL_CONTEXT_MAX_ATTEMPTS` allows at most two paid reads per question across all
of its cited URLs, the analogue of the Wayback per-question cap. The cap slot is claimed LAST of
all the gates, so a read the wall budget or the robots pre-check declined spends no slot, and a
read the cap itself declines is recorded as a `url_context_cap` skip. That is a different quantity
from `RESOLUTION_SOURCE_URL_CONTEXT_ATTEMPTS`, which is the SDK retry count for one read.

One change on 2026-09-03 widened the population that reaches this rung, and it is worth knowing
now that the flag is on. The extractor policy now withholds a page whose extraction clears the
400-character chrome floor on short lines alone, and that withhold is `no_resolving_content` with
reason `thin_page`, which is one of the paid rung's trigger statuses. On the calibration corpus
(`scratch/fetch_ladder_2026-09-03/chrome_calibration.md`) that is 9 of the 59 labelled bodies, 6
chrome and 3 ambiguous, which the previous default-only extraction published, against a census of
68 cited successes; each of them reaches the paid rung only when the free rendered rung fails to
rescue it first. The sharper consequence is crowd-out rather than volume: the two slots go to
whichever of a question's concurrently fetched URLs reaches the gate first, so these new candidates
can take both of them ahead of the `blocked` pages whose Google-egress advantage is the rung's whole
reason to exist. If that ordering turns out to matter in the archive, the
honest fix is to prefer `blocked` and `error` over `thin_page` when the cap binds, not to drop the
population, because the withhold came from our own extractor's output rather than from the page and
Gemini reading the same URL is a different extractor.

Turning it off again, or widening its trigger population, is the operator's cost decision (see
`AGENTS.md` "Cost discipline"). Its spend is separable in the archive by the `GEMINI_USAGE` role
`resolution_source`, and its three own markers (`RESOLUTION_SOURCE_URLCONTEXT_ROBOTS_SKIP`,
`RESOLUTION_SOURCE_URLCONTEXT_UNGROUNDED_SUPPRESSED` and
`RESOLUTION_SOURCE_URLCONTEXT_NOT_ADDRESSED`, under "Reading run logs") say how often the free
pre-check saved a read and how often a paid read served nothing.

Model prices, read from ai.google.dev on 2026-09-03: the two live native surfaces —
grounded search and gap-fill v2's `read_document` — run `gemini-3.8-flash` since
2026-09-03, verified live on the google-genai SDK. It draws the same grounding
pool (grounding costs $0 to switch) and its tokens are $0.75/$3.75 per M through
2026-12-31, then $1.50/$7.50 — against $0.50/$3.00 for the
`gemini-3-flash-preview` it replaced on search and $1.50/$9.00 for the
`gemini-3.5-flash` it replaced on the reader. Either way that is a few dollars a
month, from prepaid credits. `url_context` carries no per-request fee; retrieved
documents bill as input tokens.

Don't confuse the OpenRouter Gemini path (donated route via
`OAI_ANTH_OPENROUTER_KEY`, minus whatever `DONATED_KEY_BLOCKED_GOOGLE_MODELS`
excepts) with this google-genai path: separate keys, separate billing. The
grounded-search side is `research/gemini_search.py`; v2's `read_document` /
`url_context` path is `research/agentic/tool_backends.py`. What a run actually drew
on this key is readable from the `GEMINI_USAGE` marker in its log (see "Reading run
logs").

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
| `RESOLUTION_SOURCE_ENABLED` | off | `true` | Tier-1 fetcher of URLs cited in the resolution criteria (plain HTTP + trafilatura, plus the free escalation rungs; no LLM call and no spend of its own) |
| `RESOLUTION_SOURCE_URL_CONTEXT_ENABLED` | off | `true` | The one PAID rung of that fetcher's escalation ladder: when every free rung has failed to read a cited page, Gemini's `url_context` reader is asked to read it, billed to the operator's personal `GOOGLE_API_KEY`. ON in every bot workflow since 2026-09-04, so the resolution-source provider is a paid surface; changing it anywhere is a cost-gate decision for the operator, not an agent |
| `TS_ANCHOR_ENABLED` | off | `true` | Time-series empirical P10/P50/P90 band from a question's own resolution series |
| `TS_ANCHOR_CHART_ENABLED` | off | `false` | Chart-image side-channel for the anchor (vision message to base models); held off pending a text-vs-image A/B |
| `RESEARCH_PROVIDER` | `auto` | unset | Forces one primary provider (`asknews`/`exa`/`perplexity`/`openrouter`) instead of the priority order |

The primary provider is chosen by priority: AskNews (when
`ASKNEWS_CLIENT_ID` + `ASKNEWS_SECRET` are set, the prod case), then Exa, then
Perplexity, then Perplexity-via-OpenRouter. The flags above run on top of the
primary, each independently gated.

The resolution-source escalation ladder is otherwise tuned by constants rather than flags, all
in `constants.py` and all read at call time. Each rung has a minimum-wall-budget floor below
which it is skipped: `RESOLUTION_SOURCE_META_REFRESH_MIN_BUDGET_S`,
`RESOLUTION_SOURCE_PDF_MIN_BUDGET_S`, `RESOLUTION_SOURCE_DERIVED_API_MIN_BUDGET_S`,
`RESOLUTION_SOURCE_RENDER_MIN_BUDGET_S` (12 s, far above the others, because the rung launches a
browser and the launch slot is contended process-wide; it is the pre-gate floor, and the
transport's post-gate need of `RENDER_MIN_GOTO_MS` plus `RENDER_POST_GOTO_TAIL_MS` plus
`RENDER_EXIT_RESERVE_MS` is 15 s, so a render admitted with 12 to 15 s left declines at the gates
with a `wall_budget` skip; that band is deliberate and the operator's to change),
`RESOLUTION_SOURCE_WAYBACK_MIN_BUDGET_S` and `RESOLUTION_SOURCE_URL_CONTEXT_MIN_BUDGET_S`. Every
floor is measured against the remaining provider wall less
`RESOLUTION_SOURCE_RUNG_WALL_MARGIN_S`. Three more bound the two rungs that serve something
other than the live page: `RESOLUTION_SOURCE_WAYBACK_MAX_AGE_DAYS` (an archived capture older
than this is withheld as `stale_data` rather than served, matching the Datawrapper freshness
bound), `RESOLUTION_SOURCE_WAYBACK_MAX_ATTEMPTS` (snapshot fetches per question, since every
snapshot contends on one host gate) and `RESOLUTION_SOURCE_URL_CONTEXT_ATTEMPTS` (the SDK retry
count for one paid read, deliberately fewer than gap-fill v2 allows its reader). One more bounds
the paid rung's spend per question: `RESOLUTION_SOURCE_URL_CONTEXT_MAX_ATTEMPTS` caps how many
paid reads one question may pay for across its cited URLs (the analogue of the Wayback cap, a
distinct quantity from the SDK retry count above, recorded as a `url_context_cap` skip when it
binds). One floor bounds CPU rather than a rung:
`RESOLUTION_SOURCE_PRECISION_RETRY_MIN_BUDGET_S` (5 s) is what the extractor policy needs left on
the wall to run its second, `favor_precision` pass over a body already in hand, and below it the
page is withheld exactly as a failed precision pass would withhold it. `docs/research.md` has the
reasoning behind each. Read the values off `constants.py`; that
is the only authoritative copy.

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
| `OPENROUTER_CREDIT_FLOOR_USD` | `100.0` (see `constants.py`) | unset (uses default) | Donated-key remaining-balance level for the end-of-run early warning to ask Metaculus for a top-up |
| `OPENROUTER_CREDIT_ALERT_RESUME_DATE` | `2026-09-03` | unset (uses default) | Date the credit alerts start reddening CI again; before it, credit shortfalls log but exit zero. Push it forward to re-arm a suppression window |

## GitHub Actions workflows

Five bot workflows live in `.github/workflows/`. They share the same setup
(checkout, `uv sync --no-dev --frozen`, install Playwright Chromium), the same env
block, and a `timeout-minutes` job cap
that is a backstop for a wedged run, not a normal duration. Each tees stdout and
stderr to a `run_logs/` file and uploads it as an artifact with 90-day
retention.

That Chromium install now serves two rendered-fetch rungs rather than one: gap-fill v2's fetch
ladder and the Tier-1 resolution-source ladder, which share the transport in
`research/rendered_fetch.py`. The step is `continue-on-error` in every workflow, with a following
step that raises a GitHub warning annotation when it failed, because both callers degrade
gracefully without a browser. The cost of a missing browser is visible per question in the
resolution-source provider's `renderer_unavailable_skips` count, so a run whose install failed
is readable from the archive rather than only from the annotation. That count excludes a URL an
earlier question already rendered to nothing this run (its own `rendered_no_text_skips`) and the
three declines that are facts about the page rather than the runner, each under its own key: a
render the transport's DOM-read bound cut off (`render_timeout_skips`), a browser answered a
non-200 where the direct GET got 200 (`render_non_200_skips`) and a rendered DOM over
`RENDERED_DOM_MAX_CHARS` (`render_dom_too_large_skips`). Neither a memo hit nor a hostile page
can inflate the install-failed signal.

| Workflow | Trigger | Mode | What it does |
|---|---|---|---|
| `run_bot_on_tournament.yaml` | cron at :03/:23/:43 hourly, plus manual | `tournament` | Forecasts new questions in the current AI benchmark tournament (`TOURNAMENT_ID` in `constants.py`); publishes to Metaculus |
| `run_bot_on_minibench.yaml` | cron at :08/:38 hourly in the YAML, but the workflow is disabled on GitHub — see below | `minibench` | Forecasts the current MiniBench question set; publishes |
| `run_bot_on_metaculus_cup.yaml` | cron at :13/:33/:53 hourly, plus manual | `metaculus_cup` | Forecasts open Metaculus Cup questions (`METACULUS_CUP_ID` in `constants.py`, the season's dated slug); publishes |
| `test_bot.yaml` | manual only (`workflow_dispatch`) | `test_questions` | Runs a fixed handful of example questions end-to-end in prod mode; publishes comments |
| `test_bot_basic.yaml` | manual only (`workflow_dispatch`) | `test_questions` | One-question smoke test; publishes one comment. See below |

The three prod workflows are the only ones with a `schedule:` block; both test
workflows are `workflow_dispatch` and never fire on their own.

**A `schedule:` block in the YAML is not the same as a workflow that runs.**
GitHub carries a per-workflow enabled/disabled state that no file in this repo can
set, and `run_bot_on_minibench.yaml` is `disabled_manually` there by operator
design — it has NEVER been enabled (confirmed 2026-09-03), so despite the :08/:38
crons above the bot does not forecast MiniBench at all. Read the row above as
"would fire hourly if enabled". The practical consequence is that a
`make supply_probe` row showing minibench posts closed with zero bot forecasts is
the EXPECTED state, not a forfeit and not a `METACULUS_TOKEN` problem. The same
mechanism currently applies to `run_bot_on_metaculus_cup.yaml` (see the
season-start checklist above), with the difference that the cup one is meant to be
enabled and is waiting on the operator; minibench is off on purpose. To check the
live state rather than the YAML:

```bash
gh workflow list --repo No-Stream/metaculus-bot --all
```

`--repo` is required: `origin` is the fork, `upstream` is the Metaculus template,
and no default repo is configured, so a bare `gh workflow` command silently
targets upstream.

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
- `CREDIT_ROLE_SPEND:` (`credit_telemetry.py`) — one line per (role, key): which
  pipeline stage spent what, off OpenRouter's own per-call accounting. On a
  one-question smoke run expect three `forecaster:<vendor>` rows plus the
  research roles; `usd=n/a` means no cost data, `role=untagged` means a builder
  call site is missing its `role=`. Described under "Credit telemetry" below.
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
- `CREDIT_ROLE_SPEND: role=... key=... usd=... calls=... costed_calls=...
  byok_usd=...` — one line per (role, key) at end of run, saying WHERE the
  OpenRouter dollars went. See "Per-role spend" below.
- `CREDIT_FLOOR_BREACH: key=donated remaining=... floor=...` when the donated
  key's remaining balance drops below `OPENROUTER_CREDIT_FLOOR_USD`
  (`constants.py`, $100). That level is an early warning, not an empty tank — read
  it as "ask Metaculus for a top-up", not "the key is dry".

### Per-role spend (`CREDIT_ROLE_SPEND`)

The per-key deltas above say what a run cost; the role lines say which part of
the pipeline spent it: `forecaster:openai` / `forecaster:anthropic` /
`forecaster:google` (the vendor slot, so the series survives a model swap),
`stacker`, `stacker_fallback`, `parser`, `summarizer`, `crux_analyzer`,
`native_search`, `targeted_search`, `gap_fill_analyzer`, `gap_fill_resolver`,
`gap_fill_v2_driver`, `market_query_author`, `market_ranker`,
`financial_classifier`, `perplexity_research`. The list lives in
`credit_telemetry.llm_call_metadata`.

How the number is produced, because it decides how to read it:

- Every LLM built through `build_llm_with_openrouter_fallback(..., role=...)`
  (and the raw-`acompletion` gap-fill v2 driver) stamps a litellm `metadata=`
  tag with its role and the key it bills (`donated` for the wrapper's primary,
  `personal` for its fallback or a personal-key-pinned model, `direct` for a
  non-OpenRouter slug). A litellm success callback (`RoleSpendTracker`) reads
  the tag back together with **OpenRouter's own per-call usage accounting** off
  the response: `usage.cost` (credits drawn from the key) plus
  `usage.cost_details.upstream_inference_cost` (the provider's charge on a BYOK
  route). `usd` is their sum; `byok_usd` is the upstream part on its own. The
  donated key routes through Metaculus's BYOK integrations, so on that key nearly
  everything is `byok_usd` — the same money `/auth/key` books as `byok_usage` and
  subtracts from `limit_remaining`; the personal key is not BYOK, so its rows read
  `byok_usd=0.0000`. This is the provider's figure, not litellm's price table.
- `usd=n/a` means none of that row's calls carried cost data. It is never a
  fabricated zero; `costed_calls` says how many of `calls` the sum covers.
- `role=untagged` means a completion nobody stamped — forecasting-tools' own
  helpers, or a builder call site that forgot its `role=`. `key=unknown` is the
  same for the key.
- Not on OpenRouter, so never in this ledger: Gemini grounded search and gap-fill
  v2's `read_document` (google-genai on the personal Google AI Studio key), the
  AskNews subscription, Exa. The ledger is therefore an OpenRouter-only figure,
  like the `$0.38–0.41/question` in `FUTURE.md`.
- The lines are logged from the same `finally` as `CREDIT_SPEND`, after the
  forecast loop has drained litellm's callback queue (`cli.py`
  `_forecast_with_callback_drain`), so a crashed run still reports what it booked.
  That drain is bounded at `LITELLM_CALLBACK_DRAIN_TIMEOUT_S` (10s) and swallows
  its own timeout, because telemetry must never be able to fail a run that already
  published. When the bound trips, the run logs one
  `LITELLM_CALLBACK_DRAIN_TIMEOUT` WARNING and the rows below it may be missing
  the last few completions. Treat that warning as "this run's ledger is a lower
  bound"; without it, the ledger covers every completion of the run.

Harvested as `credit_role_spend.jsonl` in the telemetry archive.
`uv run python scripts/reconcile_credit_spend.py --roles` (free, offline) prints
each run's role-ledger total beside its settled per-key spend — the two measure
the same money from opposite ends, so their ratio is the ledger's own coverage
check — plus a per-role table over the selected runs.

**The per-question spend figure to quote is `$0.38–0.41`**, measured over 29
triple-era runs across 33 questions, and it is an OpenRouter-only LOWER bound: it
excludes Google AI Studio prepaid (Gemini grounded search and gap-fill v2 document
reads), the AskNews subscription, and Exa. The older "~$3.05 → ~$1.65 after the
6→3 roster drop" estimate was never measured, is an order of magnitude too high,
and is superseded — it must not appear in a roster re-add decision. `FUTURE.md`'s
"Cost context for the re-add decision" holds the receipt path, and
`CREDIT_ROLE_SPEND` plus `scripts/reconcile_credit_spend.py --roles` is how a
re-add gets priced per role rather than estimated.

A floor breach does not abort the run. Forecasting and publishing complete
normally, and outside a suppression window `cli.py` then exits non-zero so the
GitHub Actions check turns red as a reminder to ask Metaculus to top the donated
key up. The floor is an EARLY-WARNING level ($100, roughly 250 questions of
runway) rather than an empty tank, because only Metaculus can refill this key and a
reminder that arrives when the balance hits $1 arrives too late to act on. The
floor is only checked against the donated key (the personal key reports no
`limit_remaining`). Per-run spend prefers the `limit_remaining` drop because the
donated key routes nearly all spend through BYOK provider integrations, which
leaves the plain `usage` field frozen while real money burns.

### The credit-alert suppression window (closed since 2026-09-03)

Credit alerting is ON. It was suppressed from 2026-07-26, when the donated key
drained and the operator started funding the season out of pocket, so an empty
donated key was the expected state rather than a defect. Metaculus granted $1,500
of credits on 2026-09-03, and `CREDIT_ALERT_RESUME_DATE` in `constants.py` was
moved up from 2026-09-10 to `2026-09-03` that day: a credit shortfall reddens CI
again. The machinery below is unchanged and re-armable — push
`CREDIT_ALERT_RESUME_DATE` forward in `constants.py`, or set
`OPENROUTER_CREDIT_ALERT_RESUME_DATE` in the workflow env, and the window reopens
with no other edit. Inside a window two paths are gated, because either one alone
would keep the check red:

1. The floor breach. `cli.py` skips the `sys.exit(1)` and logs an INFO line
   saying the breach was observed but alerting is suppressed until the resume
   date.
2. The credit-caused donated-to-personal key fallbacks. Each fallback counts
   toward `alertable` outside the window. `record_donated_key_fallback` tracks the
   suppressible subset in `_credit_key_fallback_count`, a subset of the
   all-causes `_generic_key_fallback_count`, and `cli.py` subtracts the subset
   back out while alerting is suppressed. Every event is counted exactly once:
   generic adds it, at most one subset subtracts it. That is why the whole
   accounting block in `record_donated_key_fallback` has to contain no `await`
   after the threaded probe — `+=` on a module global is interruptible between
   bytecodes, so an await there would let N forecasters failing on one dry key
   race the increment, undercount the generic total, and take a degraded run
   green.

Non-credit fallback causes alert in full whatever the window says, since each
means real breakage rather than an empty wallet: 401 invalid or disabled key, 404
"no allowed providers", 429 rate limit, and the guardrail / data-policy block.
Bot-side degradation is untouched by a suppression too: every counter in the
`Degradation counters:` summary always alerts in full (they are enumerated under
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
| 200, cap > 0, nothing remaining | `drained` | suppressible — the expected empty wallet |
| 200, cap == 0 | `zeroed` | **red** — Metaculus cut us off, never an "empty wallet" |
| 401 / 404 | `revoked` | **red** — key is gone, not empty |
| 200, money remaining | `funded` | **red** — the failure was not about credit |
| probe failed, or no donated key configured | `unknown` | **red** — fail safe |

Only `drained` is ever subtracted from `alertable`, and only inside a suppression
window (none is open since 2026-09-03). A probe that errors or times out classifies
as `unknown` and stays red, so a broken probe can never silently turn a red run
green.

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

### Checking balances

The donated Metaculus OpenRouter key (`OAI_ANTH_OPENROUTER_KEY`) is shared and
rate-limited, so its burn rate is worth checking periodically rather than only
when a run complains. `make check_credits` prints `limit` / `limit_remaining` /
`usage` for both `OAI_ANTH_OPENROUTER_KEY` (donated) and `OPENROUTER_API_KEY`
(personal); pass `ARGS="--key donated"` to check just one.

```bash
make check_credits                    # both keys
make check_credits ARGS="--key donated"
```

Raw curl backup, which avoids putting the key on disk by pulling it from `.env`:

```bash
curl -s -H "Authorization: Bearer $OAI_ANTH_OPENROUTER_KEY" \
  https://openrouter.ai/api/v1/auth/key | jq
```

Never paste a full key into chat and never commit one; `.env` is gitignored.

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

This section is the runbook — the commands, and what each one prints. The
methodology and the conventions that make a number trustworthy (era bucketing and
the merge-to-main rule, the exclusion cohorts, the PIT convention, the starved
outer tail, the supply probe, per-model recovery, the spot-peer rule,
`spot_peer_delta`, and the clip-threshold sweep) live in
`docs/performance_analysis.md`.

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

A resolution the platform reports as out of range (`above_upper_bound` /
`below_lower_bound`) carries no value, so its PIT is a SET rather than a number:
`[cdf[-1], 1]` above the ceiling, `[0, cdf[0]]` below the floor. Those readings
count toward every coverage column when the interval intersects the band, and are
excluded from PIT std and mean PIT, where no midpoint is imputed; the
`set-valued (pt n)` column states how many were excluded and what the
point-metric denominator therefore is. The convention lives in
`analysis.out_of_range_pit_reading` / `analysis.PitReading` and both PIT paths
read it. It matters because our own CDF decides the interval: q44842 deliberately
published 13% of its mass above the displayed ceiling, resolved
`above_upper_bound` and won spot peer +24.4, which the old PIT-1.0 convention
scored as a high-side band miss. A starved tail (`cdf[-1]` at the 0.999
open-bound floor) still misses the band, because that interval lies wholly above
0.90.

The same command prints a second, per-QUESTION section: the **starved outer tail**
scan, which lives in `performance_analysis/outer_tail.py` (the width monitor owns
only the CLI wiring). `docs/performance_analysis.md` defines the defect, why it is a
cliff at a fixed location that widening does not fix, and what the archived fire rate
means; this section covers running it and reading a row. q45218 published its winning
rig-count forecast with 27 such bins starting one rig above its declared p99, a flat
-219.5 zone sixteen rigs from the resolution, and the same shape is what made q44182
(-219.0) the worst record on the board. A side is flagged when its band's mean per-bin
mass is under `STARVED_OUTER_TAIL_FLOOR_MULTIPLE` (2.0) times the platform's per-bin
minimum step (`0.01/N`); each flagged row reports the declared anchor, how many member
curves set it and how many were dropped, the displayed bound, the band's mass and bin
count, the mass sitting beyond the bound, and the log score a resolution in the band's
thinnest bin would earn. The member census is there because the anchor is a median over the members
whose declared curve is usable, so dropping one (an anonymous positional
`Forecaster N` bucket, an unparseable curve, or one carrying fewer than two
distinct percentile labels) moves the boundary the verdict is measured against;
the section header states how many sides dropped a member, and every scanned side
carries `members_used` / `members_dropped` in the JSON dump.
`--output-starved-json <path>` writes every scanned side with its verdict,
flagged or not. This is a DETECTOR: any width response stays gated on the
standing `k_tail` hold, and there is no publish-time twin of it.

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
- `MEMBER_FORECAST: question=... model=... role=member|stacker qtype=... raw=... published=...`
  — one line per forecast VALUE that leaves a runner, for every ensemble member and
  for the stacker. `raw` is what the extraction ladder read off the rationale before
  any clamp, renormalise or sanitise; `published` is what the runner handed on.
  Both are whitespace-free JSON literals, so `json.loads` them whatever the type:
  binary a probability each (`raw=0.005 published=0.02`), multiple choice the
  option-probability vector in `question.options` order (`raw=[0.9,0.005,0.095]`),
  numeric the declared `[percentile, value]` pairs with the percentile as the
  block's decimal (`raw=[[0.025,9.2],...]`) and `published` the post-sanitise list.
  The numeric line precedes the unit-mismatch guard, so a withheld member still
  leaves its raw declaration (the drop is in `FORECASTER_DROPS`). Emitted by
  `forecaster_runners.py` (members), `stacking.py` (stacker binary / MC) and
  `aggregation_pipeline.py` (stacker numeric, where its percentiles are sanitised);
  formatter in `member_forecast.py`. Added 2026-09-02 because no marker carried a
  member's value on every question and the published comment, the only other
  writer, is middle-trimmed and carries the block only since 2026-05.
- `OPEN_BOUND_PILING: question=... model=... bound=... bin_mass=... ...` — a
  forecaster put enough mass on the terminal displayed bin of an open-bound
  numeric question, without declaring any percentile beyond the edge, to trip
  `OPEN_BOUND_PILING_THRESHOLD` (`numeric/config.py`). Emitted by
  `numeric/diagnostics.py`.
- `EXTREME_CALL: question=... model=... p=... side=low|high lone=... survivors=...`
  — one line per surviving ensemble member of a BINARY question whose probability
  sat at or past an edge of the extreme band (`EXTREME_CALL_LOW` /
  `EXTREME_CALL_HIGH` in `constants.py`, currently 0.05 / 0.95, inclusive).
  Emitted by `extreme_call.py` right after `FORECASTERS_SURVIVED`, which supplies
  the denominator: a member inside the band leaves no line, so a rate needs the
  survivor list too. `lone=true` means no other survivor was extreme on the same
  side, which is the measurement: the 2026-08-31 round found lone extremes right 4
  of 9 against 21 of 23 for accompanied ones (those counts used a looser
  either-side rule — `extreme_call.py` explains the difference before you pool old
  and new numbers). `survivors=1` marks a record where "lone" is vacuous because
  that member was the whole ensemble; drop those from a lone rate. This band
  membership check gates and clamps nothing; the single-survivor publish clamp
  below is a separate rule, keyed on the survivor count, that reuses the same two
  constants.
- `THIN_PUBLISH_FLOOR: question=... raw=... clamped=... survivors=1` — a WARN
  that a BINARY question published on exactly ONE surviving forecaster had its
  published probability clamped into `[THIN_PUBLISH_BINARY_FLOOR,
  THIN_PUBLISH_BINARY_CEIL]`, which `constants.py` defines by aliasing
  `EXTREME_CALL_LOW` / `EXTREME_CALL_HIGH` (currently 0.05 / 0.95), so the band is
  one definition rather than a second pair of literals. That range is narrower
  than the per-model `[BINARY_PROB_MIN, BINARY_PROB_MAX]` = [0.02, 0.98] clamp
  every member already passed: median-of-1 supplies no variance reduction, so the
  published value's admissible range is narrowed in exactly that state to price
  the missing aggregation. `raw` is what the survivor declared and is what the
  comment's per-model summary bullet still shows; `clamped` is what went to
  Metaculus. Emitted by `aggregation_pipeline.py` at the base-combine step, only
  when the value actually moved, so the line count is the floor's incidence: a
  lone survivor already inside the band leaves no line, and a multi-member median
  is never floored, however extreme (the receipt behind the rule priced that
  global variant at -52.02 spot peer). Expect it only alongside a
  `FORECASTERS_SURVIVED: ... survived=1/N` line for the same question.
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
- `AGENTIC_FETCH_THROTTLED: url=... method=... chars=... phrase=...` — a WARN, one per
  gap-fill v2 fetch whose HTTP 200 body was the host's rate-limit interstitial rather than
  the page (a body at or under `FETCH_THROTTLE_PAGE_MAX_CHARS` carrying one of
  `FETCH_THROTTLE_PHRASES`, both in `research/agentic/fetch_outcomes.py`). Such a fetch
  returns `status=throttled`, earns no verification tier, and is never cached, so the
  driver's retry is a real request. `phrase` names the rule that fired and `chars` the body
  it fired on: together they say whether a line is a true throttle or the rule over-reaching,
  which is what the phrase list and the cap get retuned on. Emitted by
  `_throttled_fetch_outcome` in `research/agentic/tools.py`; harvested as
  `agentic_fetch_throttled`. Receipt: q45191, where two throttled ogimet.com fetches reached
  the driver as successful ones and the driver's own retry was served the cached refusal.
- `AGENTIC_FETCH_LOCAL_DOC: url=... method=pdf_local|digest_local chars=... pages=...
  passages=...` — an INFO, one per document the gap-fill v2 ladder read without paying a
  Gemini `url_context` call for it. `passages=0` on a `digest_local` is the reading that
  matters — the document does not discuss what was asked, which in the block itself reads
  exactly like a successful read — and `pages` is `n/a` for a page with no page structure.
  The line fires only where text was actually served, so its absence measures nothing: a
  refused digest leaves no line and the paid read that followed shows up only in the spend.
  `docs/agentic_gap_fill.md` defines the two methods and the `chars` convention. Emitted by
  `log_local_document_read` in `research/agentic/local_document.py`; harvested as
  `agentic_fetch_local_doc`. This is how
  the local-first rung is measured at all: before it every PDF the driver met went to a paid
  reader, 191 calls over the 2026 summer season, and the only trace of one was the spend.
- `RESOLUTION_SOURCE_FETCH: question=... url=... status=... http=... embeds=... [reason=...]
  [route=...]` — one line per URL the resolution-source provider fetched, emitted by
  `_log_fetch_outcome_markers` in `research/resolution_source.py`. `status` is `ok`
  for a success and the verbatim `FetchStatus` otherwise (`blocked`, `js_wall`,
  `no_resolving_content`, `stale_data`, `ungrounded`, ...). Since the escalation ladder it may
  be a RUNG's verdict rather than the direct fetch's: the Wayback rung's `stale_data` where the
  direct fetch said `blocked` / `error` / `not_found`, the paid reader's `ungrounded` where it
  said `blocked` / `js_wall` / `error` / `no_resolving_content`. An era-bucketed `blocked` rate
  off this field alone shows a drop at that merge that is bookkeeping, not hosts refusing us
  less; the direct outcome is `from_status` on the sibling `RESOLUTION_SOURCE_ESCALATION` line,
  and `route` partitions the two populations. `http` is `n/a` when no response ever
  arrived; `embeds` names the routeless data-embed providers (Infogram / Flourish /
  Tableau) found in the page's raw HTML, which is what makes an unreadable-embed
  page queryable even when its prose made the fetch a legitimate `ok`. `reason` is
  appended only where the status alone is ambiguous — `no_resolving_content` is
  `embed_shell` when the page named such a provider, `thin_page` when the extraction was
  simply under the chrome floor (the population the floor gained on 2026-09-02 when it
  stopped being gated on a named provider), and `no_matching_passage` when a cited
  document read in full discusses nothing the question asks about;
  `unreadable_document` splits into `no_text_layer` / `encrypted` / `malformed`, and
  `unsupported_type` carries `budget_skipped` / `parse_contention` when it was a document
  we held and declined to parse. Its absence means no reason applies, on a fresh line as much as on an
  archived one. `route` names which rung of the escalation ladder produced the recorded
  outcome: `direct` for the plain fetch, and `meta_refresh`, `pdf_local`, `derived_api`,
  `rendered`, `wayback` or `url_context` for an escalated one (`impersonate` is reserved in
  the vocabulary for a rung that is not built). Without it
  a rescued page reads exactly like one the direct route managed on its own, so "what
  did the ladder actually buy" would not be a query. Three more optional keyed fields carry
  failure diagnostics on a non-success fetch, so the archive can separate an egress-reputation
  refusal from a host fault (the archived Akamai 403s reproduce only from the GitHub runner
  IP): `failure_class` is a small token vocabulary (`http_403`, `http_4xx`, `http_5xx` off the
  response, or `tls`, `dns`, `timeout`, `connection`, `decode`, `malformed_response` off the
  transport exception), `exc` is that exception's class name, and `server` is the `Server`
  response header lower-cased with internal spaces collapsed to `_` (the strongest tell of which
  CDN refused us). `malformed_response` is the one our own client raised rather than the host: a
  `ClientResponseError` on the fetch path, which aiohttp uses for a response it will not accept at
  all (a `Content-Encoding` it cannot decode, a header over the byte cap, a bad status line), and
  which used to fall into the catch-all `connection` bucket alongside genuine connect failures.
  All optional fields are keyed and sit at the end of the line in a fixed order
  (`reason`, `route`, `failure_class`, `exc`, `server`), so a line carrying a later field but
  not an earlier one parses correctly and every archived line still parses byte-identically.
  Tier-2 Datawrapper dataset hops ride the same line and are identifiable by their url
  (`static.dwcdn.net/data/<chart_id>.csv`). This replaced the older free-text
  `resolution_source fetched <netloc> (<status>)` lines rather than joining them, so
  each fetch appears exactly once; the remaining free-text lines are REASON lines (a
  decode score, an unread content-type, an SSRF rejection) carrying what the marker
  cannot.
- `RESOLUTION_SOURCE_ESCALATION: question=... url=... from_status=... rung=... outcome=...
  wall_s=...` — one line per escalated rung attempt, emitted by
  `research/resolution_source.py` when the direct fetch could not read a page and a
  heavier route was tried. `from_status` is the verbatim `FetchStatus` that triggered
  the escalation, and its domain is per rung rather than shared, because each rung's trigger set
  is: `js_wall` or `no_resolving_content` for `meta_refresh`, `derived_api` and `rendered`, the
  200s that carried nothing readable; `unsupported_type` for `pdf_local`, a body we held and had
  not parsed; `blocked`, `error` or `not_found` for `wayback`, whose whole point is a page our
  address never reached; and `blocked`, `error`, `js_wall` or `no_resolving_content` for
  `url_context`, the only rung that draws from both families (`not_found` is not in its set,
  because a 404 or 410 has no page for a third-party fetcher to read, and a
  `no_resolving_content` whose reason is `no_matching_passage` is excluded on the reason). A pair
  outside that table (a `blocked` render, say) is a defect rather than a rare case. `rung` is the route tried. `outcome`
  and `wall_s` are that RUNG's own, stamped as
  the dispatcher closes it: `outcome` is the status that stood once the rung was over (its
  rescue, its own verdict such as `stale_data` or `ungrounded`, or the direct status it left
  standing when it declined) and `wall_s` is what that rung alone cost. Two exclusions in `wall_s`
  are worth knowing before it is read as latency: the `url_context` line excludes the free robots
  pre-check that runs in front of the paid read, and the `pdf_local` line excludes time the
  document spent queued for a parse slot. A URL with several
  lines therefore reads as a sequence, and on a page where a dead feed GET was followed by a
  rescuing render the first line carries the direct status and the second carries `success`,
  with neither billed for the other's latency. One combination reads oddly until you know the
  order it comes from: after the Wayback rung withheld a capture as `stale_data`, a paid attempt
  that then declines records `outcome=<the direct fetch's status>` rather than the withhold, while
  the `RESOLUTION_SOURCE_FETCH` line for that URL keeps `route=wayback` and the `stale_data` verdict
  that replaced the direct result. The two lines disagree by design, so take the page's own outcome
  from the FETCH line or from `from_status`, never from the last escalation line. Live since the
  paid flag went on in every bot workflow. The `RESOLUTION_SOURCE_FETCH` line above records
  only the FINAL outcome per URL, so on its own it cannot say how many rungs were spent or
  which one rescued the page; this marker is where a rung that fires often and rescues
  nothing becomes distinguishable from one that never fires, and where the latency case
  for keeping a rung on a question under a close-derived time budget gets made. A rung that
  never RAN (no wall budget, no browser, the per-question snapshot cap, the robots pre-check)
  emits no line here by design and is counted in the provider's `details["counts"]` instead;
  `docs/research.md` lists every count key.
  Harvested as `resolution_source_escalation`.
- `RESOLUTION_SOURCE_URLCONTEXT_ROBOTS_SKIP: url=... host=...`, an INFO, one per paid
  `url_context` read the resolution-source ladder skipped because the host's robots.txt disallows
  `Google-Extended` (`research/resolution_source.py`; the same pre-check and per-host cache as
  `AGENTIC_URLCONTEXT_ROBOTS_SKIP` below, through `research/robots_policy.py`). A fire is a paid
  call NOT billed, so it is not a failure; the rate against the handful of hosts publishing the
  directive is what says whether the group parser is over-matching. No question id: the rung runs
  per cited URL inside its provider, so a join goes through the run id. Registered on 2026-09-04
  together with the flag flip that put `RESOLUTION_SOURCE_URL_CONTEXT_ENABLED` on in every bot
  workflow, so no archived run from before that merge carries one. Harvested as
  `resolution_source_urlcontext_robots_skip`.
- `RESOLUTION_SOURCE_URLCONTEXT_UNGROUNDED_SUPPRESSED: url=... statuses=...`, a WARN, one per
  paid `url_context` read on the ladder that came back with zero successful retrievals and was
  discarded as `ungrounded` rather than rendered under the primary-grading-evidence caption: the
  same floor `GEMINI_UNGROUNDED_SUPPRESSED` and `AGENTIC_DOCUMENT_UNGROUNDED_SUPPRESSED` apply, so
  the three suppression rates read as one family. The read WAS billed, so each record is money
  spent on nothing served. `statuses` is the comma-joined list of `url_retrieval_status` values
  the SDK reported, or `none` (harvested as null) when it attached no entry at all, which splits a
  retrieval that failed for a nameable reason from one that never happened. Same registration
  date and no question id, as above. Harvested as
  `resolution_source_urlcontext_ungrounded_suppressed`.
- `RESOLUTION_SOURCE_URLCONTEXT_NOT_ADDRESSED: url=... host=...`, a WARN, one per paid
  `url_context` read that retrieved the page but answered with the prompt's `NOT_ADDRESSED`
  sentinel, the model's designed reply when the page does not discuss the ask, so the read was
  withheld as `no_resolving_content` / `not_addressed` instead of rendered as prose standing in
  for an absent section. Distinct from the ungrounded line: the page WAS retrieved, so Gemini
  reaches the host, and the money bought a true negative. `host` because the rollout question is
  which hosts Gemini reaches but finds nothing on. Same registration date and no question id.
  Harvested as `resolution_source_urlcontext_not_addressed`.
- `url_context not_addressed reply for <host>: <first 300 chars>` and its `ungrounded` twin — an
  INFO line each, emitted immediately after the two markers above and deliberately NOT registered,
  so nothing archives them and the markers' own line shapes stay the contract. They carry the head
  of the reply the withhold discarded, whitespace-collapsed and capped at
  `RESOLUTION_SOURCE_WITHHELD_REPLY_LOG_CHARS`. Read them when a withhold needs explaining: a
  `not_addressed` verdict meaning the page truly does not discuss the ask and one meaning the model
  summarized the bot-challenge page the host served it instead are otherwise indistinguishable, and
  the reply is what tells them apart. The `ungrounded` line appears only when that read said
  something, since the same branch also fires on an empty reply.
- `AGENTIC_DOCUMENT_UNGROUNDED_SUPPRESSED: url=... [statuses=...]` — a WARN, one per
  gap-fill v2 `read_document` call whose `url_context` retrieval brought back nothing,
  so the answer would have been unsourced recall and the `fetched` verification tier is
  withheld (`research/agentic/tools.py`). Worth watching because a `fetched` document
  discrepancy is the only kind that enters the findings artifact's SUPERSEDE block, the
  one that tells every forecaster to override the briefing. `statuses` is the
  comma-joined list of url_context retrieval statuses the SDK reported for that call, or
  `none` when it reported none at all, which splits a retrieval that was attempted and
  failed for a nameable reason from one that never happened. Both `none` and an absent
  field harvest as null, so an archived pre-field line reads the same way. Harvested as
  `agentic_document_ungrounded_suppressed`.
- `AGENTIC_URLCONTEXT_ROBOTS_SKIP: url=... host=...` — an INFO, one per paid `url_context`
  read skipped because the host's robots.txt disallows `Google-Extended`, the product token
  Gemini's retrieval obeys, so the read would have been spend with a known-zero return
  (`research/agentic/tools.py`; the group parser is `research/robots_policy.py`, moved out of
  `research/agentic/` on 2026-09-03 when the resolution-source ladder became its second caller
  and now sharing one per-host cache between them).
  Non-alertable: a fire is a paid call NOT billed, and the free fetch rungs are unaffected by
  the check. Harvested as `agentic_urlcontext_robots_skip`. `docs/agentic_gap_fill.md` covers
  the group parser and what a high rate would mean.
- `FINANCIAL_NOISE_FLAG: surface=financial_data|ts_anchor symbol=... vr_lag=... vr=...
  floor=... short_vol=... long_vol=... robust_vol=...` — the series behind a rendered
  volatility is noise-dominated: its variance ratio sits below
  `FINANCIAL_VARIANCE_RATIO_FLOOR`, meaning most of each day's move is reversed the
  next, which inflates any volatility computed from one-day returns. The flagged
  block leads with `robust_vol`, measured on overlapping `vr_lag`-step returns, and
  labels the short-window figure noise-suspect. Two surfaces log it, sharing the
  screen and the line itself (`research/noise_flag.py`): `financial_data.py`'s
  `_volatility_lines` and `ts_render.py`'s `_realized_vol_lines`. Only the
  financial-data surface computes a long-horizon window, so a `ts_anchor` record
  reads `long_vol` as null rather than zero — `surface` is what tells that apart
  from a yfinance series too short to hold one. Per-identifier, so
  one question can fire several and the line carries no question id — `symbol` (the
  ticker or FRED series id, same field position as `FINANCIAL_STALE_LATEST`) is what
  tells two flagged identifiers in one run apart and joins a noise-flag record to the
  stale-latest record for the same series. Informational and NOT alertable — it
  describes the vendor's data, not a bot defect.
- `GEMINI_USAGE: role=grounded_search|read_document|resolution_source model=... prompt_tokens=...
  tool_use_prompt_tokens=... candidates_tokens=... thoughts_tokens=... total_tokens=...
  search_queries=... [question=...]` — one line per response from the paths that call
  Google natively rather than through OpenRouter, so their spend on the operator's personal
  AI Studio key is readable from a run log. Emitted by `log_gemini_usage`
  (`research/gemini_usage.py`), called from `gemini_search.py` (`grounded_search`, before
  any formatting branch, so an ungrounded-and-suppressed response still records what it
  cost), `research/agentic/tool_backends.py` (`read_document`, which carries no question
  id, hence the trailing field's absence there), and the resolution-source ladder's paid
  url_context rung (`resolution_source`, which likewise carries no question id and appears
  only from runs with `RESOLUTION_SOURCE_URL_CONTEXT_ENABLED` on: every bot workflow since
  2026-09-04, so an archived run from before that merge has no rows in this role). No surface
  here bills through OpenRouter,
  so none shows up in `CREDIT_ROLE_SPEND`, and before this marker the whole Google AI
  Studio side of a run's spend was invisible to the archive. That side is metered against a
  monthly grounded-prompt allowance per project and billed per QUERY on overage, which makes
  `search_queries` the billable unit and any feature that multiplies grounded calls a re-run
  of the spring-2026 billing arc. `model` is the response's own `model_version` where it
  reported one and the configured id otherwise. Any count Google
  did not report reads `n/a` rather than 0, since `thoughts_tokens=0` is a real reading;
  `search_queries` is the exception and reads a genuine 0 when the search tool issued none
  (an absent `web_search_queries` list IS a count of none), `n/a` only when the grounding
  metadata could not be walked — so separate the two surfaces on `role`, never on this
  field. **The ledger covers COMPLETED responses only.** `log_gemini_usage` runs after the
  SDK returns, so a call that timed out or raised billed unknown tokens and emitted no row —
  14 of 154 archived `read_document` calls (9.1%) hit that handler. A spend total from these
  rows is a LOWER bound, biased toward undercounting the largest calls; the denominator is
  `provider_results['gemini_search'].status` per question plus `research_provider_failures`,
  never this marker's row count. `thoughts_tokens` is the field worth watching — 71% of
  grounded-search output tokens were thinking before the
  explicit levels (`GEMINI_SEARCH_THINKING_LEVEL`, `GAP_FILL_V2_READER_THINKING_LEVEL`) were
  set. Nothing about this is alertable; it is spend accounting, not degradation. Harvested as
  `gemini_usage`.
- `CREDIT_BALANCE` / `CREDIT_SPEND` / `CREDIT_ROLE_SPEND` / `CREDIT_FLOOR_BREACH`
  — credit telemetry, described above. `CREDIT_FLOOR_BREACH` fires whatever the
  credit-alert window says, so a breach on a GREEN run means a suppression window
  is open (none is, since 2026-09-03); the adjacent INFO line names the resume
  date.
  `CREDIT_ROLE_SPEND` is the per-(role, key) decomposition of the run's
  OpenRouter spend; a run with no completions logs a single no-completions line
  under the same token instead of rows.
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

A third market line, `MARKET_TIER_CAPPED: question=... rows=... capped=venue@rank`,
fires only when the deterministic staleness pass refuses a row the top relation tier
— the ranker graded a market that stopped trading more than
`MARKET_STALENESS_TIER_CAP_DAYS` (60, in `market_retrieval/ranking.py`) before the
question opened as `same_quantity_same_date`. The row keeps its rank, price and
liquidity cells; what it gains is a note in the `why` cell stating the demotion and
its arithmetic (`demoted from same-date: closed 162d before the question opened`).
Silence is the normal case, and the cap fires on nothing in the 102 archived
snapshots, so a first line in a run log IS the finding. The demotion also rides the
archived snapshot as `MarketMatch.tier_cap_note`, so its incidence is answerable
offline; this line is the prod-log half and the one that survives a run whose
snapshot the research archive never captured.

A run can also exit non-zero for degradation alerts — the counters above,
personal-key fallbacks, or the model-deprecation tripwire — even when every
question that met the minimum-forecaster threshold was published. The non-zero
exit is the CI red-check signal to investigate; it does not mean publishing
failed. Credit-caused shortfalls alert again as of 2026-09-03, and are exempt only
inside a suppression window (see that section above); every other cause always alerts. See the alert block near the end
of `cli.py` for the exact conditions.
