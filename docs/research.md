# Research subsystem

This is the reference for how the bot gathers evidence before any forecaster
LLM sees a question. If you want the forecasting/aggregation side, see the other
docs; this file covers only the research phase.

For a quick map of the code: orchestration lives in
`metaculus_bot/research/orchestrator.py`, provider selection in
`metaculus_bot/research/providers.py`, and each add-on provider is its own module
under `metaculus_bot/research/`.

## The shape of a research run

Every question goes through `ResearchOrchestrator.run_research`
(`research/orchestrator.py:114`). The flow is:

1. **Cache check.** In benchmarking runs the orchestrator caches research per
   question id so a replayed backtest doesn't re-pay for the same pull. Live runs
   don't cache (`_lookup_research_cache`, `research/orchestrator.py:292`).
2. **Provider selection.** `_select_research_providers`
   (`research/orchestrator.py:317`) picks exactly one **primary** provider, then
   appends every enabled **add-on** provider. Each add-on is behind its own env
   flag, so the set that actually runs depends on configuration.
3. **Parallel fan-out.** `_run_providers_parallel`
   (`research/orchestrator.py:383`) runs all selected providers concurrently via
   `asyncio.gather`. One provider failing never kills the phase; the failure is
   recorded as a per-provider result and the rest proceed. Global concurrency is
   bounded by a semaphore (`DEFAULT_MAX_CONCURRENT_RESEARCH = 6`).
4. **Assembly.** Each provider's output is prefixed with a fixed `##` section
   header (`_provider_header`, `research/orchestrator.py:476`) and the sections
   are joined with `---` rules. Any stray `#`/`##` heading inside a provider's
   body is demoted two levels so it never competes with the section headers.
5. **Gap-fill.** Two second-pass gap-fill passes (v1 and v2) run concurrently on
   the assembled bundle and append their own sections. See "Gap-fill" below.
6. **Diagnostics + persistence.** A provider-diagnostics block (which provider
   succeeded, char counts, latency) is logged and stashed for the published
   Metaculus comment, but is deliberately kept OUT of the text forecasters read.
   If a research sink is wired, the full record is written for backtest replay.

The single string returned from `run_research` is what every forecaster in the
ensemble reads verbatim.

## Primary provider: a priority ladder

There is always exactly one primary provider, chosen by
`choose_provider_with_name` (`research/providers.py:448`). It walks a fixed
priority order and returns the first provider whose credentials are present:

1. **AskNews** if `ASKNEWS_CLIENT_ID` and `ASKNEWS_SECRET` are set. This is the
   production case.
2. **Exa.ai** (`SmartSearcher`) if `EXA_API_KEY` is set.
3. **Perplexity direct** (`perplexity/sonar-pro`) if `PERPLEXITY_API_KEY` is set.
4. **Perplexity via OpenRouter** (`openrouter/perplexity/sonar-reasoning-pro`) if
   `OPENROUTER_API_KEY` is set.
5. **Empty stub** if none of the above — research is just the add-on providers.

In production the AskNews credentials are present, so Exa and the two Perplexity
routes never run as the primary. They are fallbacks, not peers. To force a
specific primary regardless of credentials, set `RESEARCH_PROVIDER=<name>`
(`asknews` / `exa` / `perplexity` / `openrouter`); any other value behaves as
auto. Forcing `asknews` without the AskNews creds fails loudly rather than
silently picking a different provider.

### AskNews fallback (primary-only)

AskNews is the only primary that gets a runtime fallback. If the AskNews fetch
raises, `_fetch_research_with_fallback` (`research/orchestrator.py:493`) tries a
prose provider instead, in a **different** order than the primary ladder:
OpenRouter-Perplexity first (cheapest, prose-returning), then direct Perplexity,
then Exa last (its `SmartSearcher` spins up its own multi-search loop, the most
expensive path). The primary ladder orders by index quality; this fallback list
orders by cost, because it only ever fires after AskNews has already failed.

One AskNews error is treated specially: a `403011` "subscription is not currently
active" signature (`is_asknews_subscription_error`, `research/providers.py:92`) is
logged as `inactive` (an expected off-season state), not `errored`, so it doesn't
inflate the timeout counter or look like a real failure in diagnostics.

### AskNews dual-phase search

AskNews is the one provider that returns raw article text rather than
LLM-written prose, so it has two distinct stages: fetch, then summarize.

**Fetch** (`_asknews_provider`, `research/providers.py:106`) runs two phases
against the AskNews SDK:

- **Phase 1 — HOT:** `strategy="latest news"`, 6 articles.
- **Phase 2 — HISTORICAL:** `strategy="news knowledge"`, 10 articles.

Both phases share a retry budget (`ASKNEWS_MAX_TRIES = 3`) that only retries on
known-transient rate/concurrency errors (429, "rate limit", "concurrency
limit"); anything else raises immediately. A process-wide semaphore
(`ASKNEWS_MAX_CONCURRENCY = 1`) and an RPS gate (`ASKNEWS_MAX_RPS = 0.8`)
throttle calls, plus fixed ~10s waits before each phase, because the API
rate-limits aggressively even when we stay under our own limits. A hard
wall-clock timeout (`ASKNEWS_WALL_TIMEOUT = 300s`) backstops a network hang so a
stuck AskNews call can't hold the whole phase hostage.

The two article lists are formatted into two labeled sections — "Historical
Context & Background" and "Recent Developments & Current News" — with within-list
and cross-list URL deduplication (`_format_asknews_dual_sections`,
`research/providers.py:241`). Dedup normalizes URLs first (drops tracking params,
`m.` mobile subdomains, `/amp` suffixes, fragments) so the same story from two
feeds collapses to one entry.

The raw pre-summarization article markdown is captured separately and archived
(the `asknews_raw` field) so a later audit can replay the summarizer or attribute
a bad briefing to fetch-vs-summarize without paying for a fresh AskNews pull.

### AskNews summarizer (the analyst briefing)

Raw AskNews articles are compressed into an analyst briefing by an LLM before any
forecaster sees them (`_summarize_asknews`, `research/orchestrator.py:251`). The
summarizer model is a low-effort utility slot defined in `llm_configs.py`
(`SUMMARIZER_LLM`); it runs at `allowed_tries=1` and is wrapped in a 30s-gated
broad retry with a wall cap (`SUMMARIZER_WALL_TIMEOUT = 300s`). If the summarizer
hits a transient LLM error or returns blank, the orchestrator soft-fails to the
**raw** articles rather than dropping the news entirely. A non-transient error
(a prompt-construction bug, a refactor's `AttributeError`) is allowed to
propagate, because that's a real bug, not a degradation to tolerate.

The prompt (`asknews_summarizer_prompt`, `prompts.py:251`) tells the model to
produce a comprehensive briefing that extracts every decision-relevant fact,
number, quote, and expert opinion, dates each one, and separates facts from
opinion. It also shares the source-provenance / trust-ladder vocabulary with the
web-research prompt. Several rules are load-bearing for calibration:

- **Never paraphrase numbers.** Percentages, probabilities, dates, and counts are
  copied exactly.
- **Pre-window flagging.** Any event that happened before the question opened —
  and so can't itself satisfy the resolution criteria — is tagged `[PRE-WINDOW]`
  but kept as base-rate context.
- **Single-source labeling.** A claim resting on one outlet is tagged
  `[SINGLE-SOURCE]` with its original hedges preserved; it's never promoted to a
  confirmed fact.

The 2026-07-18 revision to this prompt added three things worth calling out:

- **Evidence-age lead.** The briefing must open by stating the date of the newest
  article that *directly* bears on resolution ("Newest directly-relevant article:
  2026-07-14"), or say explicitly when nothing directly reports on the resolution
  quantity and the section is only background.
- **Supersession.** When a newer article supersedes an older one on the same fact
  (a withdrawal, an updated count, a final decision), the briefing states which
  version governs today and compresses the stale version to one line, instead of
  giving obsolete detail equal weight. Deadline/window questions must quote the
  underlying dates and rules rather than assert a conclusion.
- **Relevance screen + proportionality.** Each article is screened for direct
  bearing on the resolution criteria; anything off-topic is dropped and listed on
  a single "Screened out as not decision-relevant" line. Briefing length must
  track surviving decision-relevant content — comprehensive when there's real
  material, short when few articles survive, never padded to look thorough.

## Add-on providers (parallel, each independently gated)

On top of the single primary, every enabled add-on provider runs in parallel.
Each is behind its own env flag and produces its own `##` section. In production
all of these are on.

### OpenAI native search — `NATIVE_SEARCH_ENABLED`

OpenAI web search via OpenRouter's native web plugin
(`_native_search_provider` / `build_native_search_llm`,
`research/providers.py:327`). Default model `openai/gpt-5.6-terra`
(`NATIVE_SEARCH_DEFAULT_MODEL`) at `reasoning={"effort":"low"}` and
`verbosity="low"`, with a 360s per-request timeout (`NATIVE_SEARCH_TIMEOUT`) and a
420s hard wall-clock cap (`NATIVE_SEARCH_WALL_TIMEOUT`). Model and effort are
overridable via `NATIVE_SEARCH_MODEL` / `NATIVE_SEARCH_REASONING_EFFORT` /
`NATIVE_SEARCH_VERBOSITY`.

The model is built at `allowed_tries=1` on purpose: an earlier incident
(2026-05-20) had OpenRouter drip whitespace keep-alive bytes for over eight
minutes before returning malformed JSON, and retrying that call just multiplies
the wait. The wall-clock cap plus a single try bounds the worst case at ~7
minutes. It routes through `build_llm_with_openrouter_fallback`, so it bills the
Metaculus-donated `OAI_ANTH_OPENROUTER_KEY` first and falls back to the personal
`OPENROUTER_API_KEY` on credential/credit errors. The prompt is the shared
`web_research_prompt` with markdown citations.

### Gemini grounded search — `GEMINI_SEARCH_ENABLED` + `GOOGLE_API_KEY`

Real first-party Google Search grounding via the `google-genai` SDK
(`research/gemini_search.py`), NOT via OpenRouter. This adds a genuinely distinct
search index to the ensemble. Default model `gemini-3-flash-preview`
(`GEMINI_SEARCH_DEFAULT_MODEL`), 360s timeout (`GEMINI_SEARCH_TIMEOUT`). It
enables both the `google_search` tool and the `url_context` tool, so the model
can read specific URLs named in a question's fine print directly.

Output is stitched together with inline citation markers spliced in from the
response's grounding metadata, plus a `### Sources` list
(`_format_grounded_response`). url_context fetches are logged and only
*successful* fetches are surfaced to forecasters (a "fired but fetched nothing"
run collapses to a terse `_url_context: none_` marker rather than pushing dead
URLs at the model).

This provider uses the operator's personal `GOOGLE_API_KEY` (a paid-tier Google
AI Studio key). There is no Metaculus-donated key on the google-genai side — the
donated path only exists for OpenRouter-routed Gemini. If grounded search starts
soft-failing across a run, check the AI Studio prepaid-credit balance first
(exhaustion shows up as 429s, not surprise charges).

### Financial data — `FINANCIAL_DATA_ENABLED` (+ `FRED_API_KEY` for live FRED)

For questions about trackable financial/economic metrics
(`research/financial_data.py`). A cheap LLM classifier
(`FINANCIAL_CLASSIFIER_MODEL`, low effort) decides whether the question is
financial and which tickers / FRED series apply, and — critically — resolving
identifiers are *also* extracted deterministically from URLs in the resolution
criteria (`extract_financial_identifiers_from_criteria`). That extraction is the
load-bearing guarantee: even if the classifier misreads the question, the series
the question actually resolves on still fires. The two sets are merged
(extraction is additive), then fetched in parallel:

- **yfinance** for tickers: current price, period returns, 30-day annualized
  volatility, 52-week range, recent closes, and (live only) fundamentals.
- **FRED** for economic series: latest/previous value, MoM and YoY change, recent
  observations.

Under benchmarking every fetch is ceilinged to the question's `open_time`:
yfinance uses a bounded history window and skips the leaky live `.info` call, and
FRED routes through a keyless point-in-time path (`ts_fetch`, ALFRED vintages) so
revised macro series return the vintage known at forecast time, not today's
revisions. A forecaster-invisible HTML-comment routing marker records which
identifiers fired, which came from extraction vs. the classifier, and any
unrecognized (fetched-anyway-but-flagged) IDs.

### Prediction-market snapshot — `PREDICTION_MARKETS_ENABLED`

A crowd-forecast cross-check (`research/prediction_market.py`). Fans out to four
venues concurrently:

- **Polymarket** (Gamma public-search, bounded retry on 403 IP rate limits).
- **Kalshi** (no keyword endpoint — prefetch ~3k open events once per session,
  cached ~6h, fuzzy-match client-side with rapidfuzz).
- **Manifold** (search endpoint, plus an extra natural-language query since its
  search prefers that framing).
- **PredictIt** (prefetch the full market dump, fuzzy-match locally, and pick the
  *contract* whose name best matches the query so a "Trump 2028?" query surfaces
  the Trump contract's price, not whichever contract is listed first).

Keyword extraction defaults to `s4_s5_union`: two LLM prompts (noun phrases +
entity/event/deadline) run in parallel via a small model, unioned and deduped
(`PREDICTION_MARKET_KEYWORD_STRATEGY`, valid: `s4_s5_union` / `s5_only` /
`simple`). The result is rendered as a markdown table: implied probability, total
volume, open interest, a liquidity/participation `signal` label (thin / decent /
deep for real-money venues, thin / decent / high by bettor count for Manifold,
`no-liquidity-data` for PredictIt), close date, and match confidence, followed by
each market's resolution rules. The prompt tells forecasters to verify each
market's criteria and date against the question — a matched market is strong
evidence, a mismatched one is discounted proportionally — and to weight by the
liquidity label.

This provider is **hard-disabled under benchmarking** (`is_benchmarking=True`
returns `""`), regardless of the env flag. The `as_of` filter only drops markets
that closed *before* `as_of`; still-open markets would leak post-`as_of`
information into a backtest, so the benchmarking guard is the only safe defense.
That guard is why its forecasting value can't be measured by the standard
backtest gate — it was validated with live `test_bot.yaml` runs and opt-in live
integration tests.

### Resolution-source fetcher — `RESOLUTION_SOURCE_ENABLED`

Fetches the exact URL(s) a question cites as its grading source
(`research/resolution_source.py`), so forecasters read the ground truth the
question resolves against. This is Tier-1 only: plain HTTP with browser-like
headers, no LLM calls, no retries.

It deterministically extracts URLs from resolution criteria + fine print (markdown
links and bare URLs, order-preserving dedup, Metaculus markdown-escapes undone),
skip-filters URLs that add nothing or belong to another provider (Metaculus
self-refs, FRED series owned by financial-data, Yahoo `/quote/` pages owned by
yfinance), and caps at `RESOLUTION_SOURCE_MAX_URLS = 5` *after* the skip filter so
a run of leading self-refs doesn't starve the real sources. Fetches run in
parallel with one-request-per-host politeness (a `Semaphore(1)` per netloc, keyed
per redirect hop). Content is extracted with trafilatura (HTML), or read raw
(JSON / text / CSV); anything else (PDF, binary, missing Content-Type) is left
unread.

It is **SSRF-hardened** because these URLs are user-authored and fetches run from
CI on AWS: a preflight `is_public_http_url` check rejects private / loopback /
link-local / non-global IPs, userinfo tricks, and non-HTTP schemes, and a
connect-time `FilteringResolver` (the actual DNS-rebinding boundary, not the
preflight) re-checks every resolved IP. Redirects are followed manually with a
hop cap, re-guarding each `Location`. Per-URL truncation appends a
`[truncated at N chars — full source at URL]` marker, and the section formatter
appends `[N additional source(s) omitted — section budget]` when later sections
are dropped for length. Unfetchable pages (`blocked` / `js_wall`) are retained in
the per-URL `FetchStatus` as the seam for a future Tier-2 LLM-driven fetch pass.

Like prediction markets, it is **hard-disabled under benchmarking** (current page
content post-dates any backtest window).

### Time-series anchor — `TS_ANCHOR_ENABLED` (chart side-channel `TS_ANCHOR_CHART_ENABLED`, off)

A deterministic empirical anchor for numeric questions whose resolution series is
a fetchable FRED/yfinance series (`research/timeseries_anchor.py`). No LLM, no
model selection — it renders the latest value, a multi-resolution history, a
52-week range, and a horizon-matched empirical band built only from the series'
own past. The Phase-A offline replay found CV-gated model picks beat naive
out-of-sample only 43% of the time, while the naive empirical band was sharper and
better tail-calibrated, so this ships the naive band on purpose.

It is the *first* backtest-safe research provider: instead of hard-disabling under
benchmarking, it pins `as_of` to the question's `open_time` and fetches
point-in-time up to that date (ALFRED vintages for revising macro series), so the
data known at forecast time is the answer without leaking the resolution. The
text anchor is on in production; the chart-image side-channel
(`TS_ANCHOR_CHART_ENABLED`) is a separate flag and is off.

## Gap-fill (two passes, both concurrent, both on in prod)

After the primary + add-on bundle is assembled, two independent gap-fill passes
run **concurrently** in one `asyncio.gather` (`research/orchestrator.py:144`), so
the research-phase wall-clock is `max(v1, v2)`, not the sum. Each runs inside its
own try/except so a defect in one can never zero the other's output. Both consume
the pre-gap-fill bundle, which means the v2 driver's brief does not see v1's
addendum; v2's section appends after v1's.

### v1 — targeted gap-fill (`research/targeted.py` `run_gap_fill_pass`)

Two stages, gated by `GAP_FILL_ENABLED` (and skipped when the first-pass bundle is
shorter than `GAP_FILL_MIN_RESEARCH_CHARS = 200`):

1. A non-grounded analyzer LLM (`GAP_FILL_ANALYZER_MODEL`, low effort) reads the
   first-pass research and emits a JSON list of up to `GAP_FILL_MAX_GAPS` factual
   gaps.
2. Each gap is resolved by a parallel OpenAI native web search
   (`GAP_FILL_RESOLVER_MODEL`, low effort, via OpenRouter on the donated key).
   Because the searches run in parallel, latency is the slowest call, not the sum.

The resolver migrated off direct-Google grounding on 2026-06-25, which is why
`GOOGLE_API_KEY` is no longer required for gap-fill. The whole pass never raises —
it returns `""` on any error — and appends its results under
`## Targeted Gap-Fill (second pass)`.

### v2 — agentic gap-fill (`research/agentic_gap_fill.py` `run_gap_fill_v2`)

A bounded agentic tool loop run by a driver LLM
(`GAP_FILL_V2_DRIVER_MODEL = openai/gpt-5.6-terra`, `effort=low`), gated by
`GAP_FILL_V2_ENABLED`. The driver is briefed with the forecaster prompt, privately
dry-runs a forecast to find things worth filling or verifying, then iterates over
four tools (news search, web search, fetch, document read) under a wall deadline
(`GAP_FILL_V2_WALL_DEADLINE = 540s`) and a tool-call budget. It appends a
citation-only findings artifact under `## Agentic Research Findings`, leading with
a corrections-to-the-briefing block. Like the other leakage-sensitive providers,
it is benchmarking-guarded off. See `docs/agentic_gap_fill.md` for the full
tool loop, escalation ladder, telemetry, and design rationale.

## Diagnostics and persistence

`run_research` returns forecaster-clean text. The **provider-diagnostics block**
(which provider succeeded, char counts, latency per provider) is computed
separately (`format_provider_diagnostics_block`) and deliberately kept out of the
returned research — forecasters and the v2 driver must never see it. It reaches
three places instead: an INFO log line, the research archive (as its own field),
and the published Metaculus comment (stashed per question id, popped by the
forecaster at comment-build time via `pop_provider_diagnostics`).

When a research sink is wired, each question's research is written for backtest
replay by `ResearchPersistenceWriter` (`research/persistence.py`, schema version
2). The record carries the assembled `research_text`, the per-provider
`provider_results` (the authoritative outcome list), derived
`providers_attempted` / `providers_succeeded`, the `gap_fill_used` flag, and —
when they exist — the v2 agentic trace (`gap_fill_v2`), the diagnostics block, and
the raw pre-summarization AskNews articles (`asknews_raw`). Records flush to a
timestamped JSONL file. `providers_used` is retained but legacy/ambiguous; prefer
`provider_results` for any analysis.

## Production configuration

All four workflows (`.github/workflows/run_bot_on_{tournament,metaculus_cup,minibench}.yaml`
and `test_bot.yaml`) enable the full research stack:

| Flag | Provider |
|---|---|
| `NATIVE_SEARCH_ENABLED` | OpenAI native search |
| `GEMINI_SEARCH_ENABLED` | Gemini grounded search |
| `FINANCIAL_DATA_ENABLED` | yfinance + FRED |
| `PREDICTION_MARKETS_ENABLED` | prediction-market snapshot |
| `RESOLUTION_SOURCE_ENABLED` | resolution-source fetcher |
| `TS_ANCHOR_ENABLED` | time-series anchor (text; chart off) |
| `GAP_FILL_ENABLED` | v1 targeted gap-fill |
| `GAP_FILL_V2_ENABLED` | v2 agentic gap-fill |

So in production the active research stack is: AskNews (primary, summarized) +
OpenAI native search + Gemini grounded search + financial data (when classified
financial) + prediction-market snapshot + resolution-source fetcher + time-series
anchor + both gap-fill passes. Env flags, models, and timeouts live in
`metaculus_bot/constants.py`; provider models route through the shared
donated-then-personal OpenRouter fallback (`fallback_openrouter.py`), except
Gemini grounded search, which uses the personal Google key directly.

## Cost note

The research providers hit live, paid APIs (AskNews, Exa, Perplexity, OpenRouter
credits, Google grounding, FRED). Running the bot or a backtest spends real money
and, in live modes, publishes to Metaculus. Do not launch a paid run without the
operator's approval — see `AGENTS.md` "Cost discipline". The unit/integration test
suite is self-contained and hits no paid APIs.
