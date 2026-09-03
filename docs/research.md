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
(`research/orchestrator.py`). The flow is:

1. **Cache check.** In benchmarking runs the orchestrator caches research per
   question id so a replayed backtest doesn't re-pay for the same pull. Live runs
   don't cache (`_lookup_research_cache`, `research/orchestrator.py`).
2. **Provider selection.** `_select_research_providers`
   (`research/orchestrator.py`) picks exactly one **primary** provider, then
   appends every enabled **add-on** provider. Each add-on is behind its own env
   flag, so the set that actually runs depends on configuration.
3. **Parallel fan-out.** `_run_providers_parallel`
   (`research/orchestrator.py`) runs all selected providers concurrently via
   `asyncio.gather`. One provider failing never kills the phase; the failure is
   recorded as a per-provider result and the rest proceed. Global concurrency is
   bounded by a semaphore sized by `DEFAULT_MAX_CONCURRENT_RESEARCH`.
4. **Assembly.** Each provider's output is prefixed with a fixed `##` section
   header (`provider_header`, `research/section_format.py`) and the sections
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
`choose_provider_with_name` (`research/providers.py`). It walks a fixed
priority order and returns the first provider whose credentials are present:

1. **AskNews** if `ASKNEWS_CLIENT_ID` and `ASKNEWS_SECRET` are set. This is the
   production case.
2. **Exa.ai** (`SmartSearcher`) if `EXA_API_KEY` is set.
3. **Perplexity direct** if `PERPLEXITY_API_KEY` is set. Model:
   `PERPLEXITY_RESEARCH_MODEL` (`constants.py`).
4. **Perplexity via OpenRouter** if `OPENROUTER_API_KEY` is set. Same model,
   prefixed for the OpenRouter route:
   `PERPLEXITY_RESEARCH_MODEL_VIA_OPENROUTER`.
5. **Empty stub** if none of the above — research is just the add-on providers.

In production the AskNews credentials are present, so Exa and the two Perplexity
routes never run as the primary. They are fallbacks, not peers. To force a
specific primary regardless of credentials, set `RESEARCH_PROVIDER=<name>`
(`asknews` / `exa` / `perplexity` / `openrouter`); any other value behaves as
auto. Forcing `asknews` without the AskNews creds fails loudly rather than
silently picking a different provider.

### AskNews fallback (primary-only)

AskNews is the only primary that gets a runtime fallback. If the AskNews fetch
raises, `_fetch_research_with_fallback` (`research/orchestrator.py`) tries a
prose provider instead, in a **different** order than the primary ladder:
OpenRouter-Perplexity first (cheapest, prose-returning), then direct Perplexity,
then Exa last (its `SmartSearcher` spins up its own multi-search loop, the most
expensive path). The primary ladder orders by index quality; this fallback list
orders by cost, because it only ever fires after AskNews has already failed.

One AskNews error is treated specially: a `403011` "subscription is not currently
active" signature (`is_asknews_subscription_error`, `research/providers.py`) is
logged as `inactive` (an expected off-season state), not `errored`, so it doesn't
inflate the timeout counter or look like a real failure in diagnostics.

### AskNews dual-phase search

AskNews is the one provider that returns raw article text rather than
LLM-written prose, so it has two distinct stages: fetch, then summarize.

**Fetch** (`_asknews_provider`, `research/providers.py`) runs two phases
against the AskNews SDK, asking HISTORICAL for a larger article budget than HOT:

- **Phase 1 — HOT:** `strategy="latest news"`.
- **Phase 2 — HISTORICAL:** `strategy="news knowledge"`.

Both phases share a retry budget (`ASKNEWS_MAX_TRIES`) that only retries on
known-transient rate/concurrency errors (429, "rate limit", "concurrency
limit"); anything else raises immediately. A process-wide semaphore
(`ASKNEWS_MAX_CONCURRENCY`) and an RPS gate (`ASKNEWS_MAX_RPS`) throttle calls,
plus a fixed wait before each phase (`WAIT_FOR_HOT_SEC` /
`WAIT_FOR_HISTORICAL_SEC`, function-local in `_asknews_provider`), because the
API rate-limits aggressively even when we stay under our own limits. All three
throttles take env overrides, and the shipped `.env.template` deliberately sets a
*lower* RPS than the `constants.py` default — the constant is the ceiling, not the
operating point, so the two disagreeing is expected rather than a drift bug. A
hard wall-clock timeout (`ASKNEWS_WALL_TIMEOUT`) backstops a network hang so a
stuck AskNews call can't hold the whole phase hostage.

The two article lists are formatted into two labeled sections — "Historical
Context & Background" and "Recent Developments & Current News" — with within-list
and cross-list URL deduplication (`_format_asknews_dual_sections`,
`research/providers.py`). Dedup normalizes URLs first (drops tracking params,
`m.` mobile subdomains, `/amp` suffixes, fragments) so the same story from two
feeds collapses to one entry.

**Both phases empty returns `""`, not a "No articles were found" sentence.** That
sentence defeated every downstream empty guard at once: the orchestrator's
`has_output` check saw chars>0 and reported `ok`, the summarizer (whose prompt has
no no-data escape) was handed the sentence as its article set, and the resulting
briefing rendered under the AskNews header as though it were research. The
formatter now logs `ASKNEWS_NO_ARTICLES`, records an `articles: empty(no_articles)`
source loss so the diagnostics line reads `empty | 0 chars | lost=articles:...`
rather than a bare `empty`, and the orchestrator skips the summarizer call
entirely. Gemini's grounded-chunk floor is the same pattern one provider over.

The raw pre-summarization article markdown is captured separately and archived
(the `asknews_raw` field) so a later audit can replay the summarizer or attribute
a bad briefing to fetch-vs-summarize without paying for a fresh AskNews pull.

### AskNews summarizer (the analyst briefing)

Raw AskNews articles are compressed into an analyst briefing by an LLM before any
forecaster sees them (`_summarize_asknews`, `research/orchestrator.py`). The
summarizer model is a low-effort utility slot defined in `llm_configs.py`
(`SUMMARIZER_LLM`); it runs at `allowed_tries=1` and is wrapped in an
elapsed-gated broad retry (`invoke_with_broad_retry`, whose gate is
`TRANSIENT_RETRY_MAX_ELAPSED_S` in `llm_retry.py`) with a wall cap
(`SUMMARIZER_WALL_TIMEOUT`). If the summarizer hits a transient LLM error or returns
blank, the orchestrator soft-fails to the **raw** articles rather than dropping the
news entirely. A non-transient error
(a prompt-construction bug, a refactor's `AttributeError`) is allowed to
propagate, because that's a real bug, not a degradation to tolerate.

The prompt (`asknews_summarizer_prompt`, `prompts.py`) tells the model to
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
`research/providers.py`). The model is `NATIVE_SEARCH_DEFAULT_MODEL`, run at the
reasoning effort and verbosity in `NATIVE_SEARCH_REASONING_EFFORT_DEFAULT` /
`NATIVE_SEARCH_VERBOSITY_DEFAULT`, under a per-request timeout
(`NATIVE_SEARCH_TIMEOUT`) and a hard wall-clock cap
(`NATIVE_SEARCH_WALL_TIMEOUT`) set just above it. Model, effort, and verbosity are
overridable via `NATIVE_SEARCH_MODEL` / `NATIVE_SEARCH_REASONING_EFFORT` /
`NATIVE_SEARCH_VERBOSITY`.

The model is built at `allowed_tries=1` on purpose: an earlier incident
(2026-05-20) had OpenRouter drip whitespace keep-alive bytes for over eight
minutes before returning malformed JSON, and retrying that call just multiplies
the wait. `NATIVE_SEARCH_WALL_TIMEOUT` plus a single try is what bounds the worst case.
It routes through `build_llm_with_openrouter_fallback`, so it bills the
Metaculus-donated `OAI_ANTH_OPENROUTER_KEY` first and falls back to the personal
`OPENROUTER_API_KEY` on credential/credit errors. The prompt is the shared
`web_research_prompt` with markdown citations.

### Gemini grounded search — `GEMINI_SEARCH_ENABLED` + `GOOGLE_API_KEY`

Real first-party Google Search grounding via the `google-genai` SDK
(`research/gemini_search.py`), NOT via OpenRouter. This adds a genuinely distinct
search index to the ensemble. Model and request timeout come from
`GEMINI_SEARCH_DEFAULT_MODEL` / `GEMINI_SEARCH_TIMEOUT` (`constants.py`). It
enables both the `google_search` tool and the `url_context` tool, so the model
can read specific URLs named in a question's fine print directly.

Output is stitched together with inline citation markers spliced in from the
response's grounding metadata, plus a `### Sources` list
(`_format_grounded_response`). url_context fetches are logged and only
*successful* fetches are surfaced to forecasters (a "fired but fetched nothing"
run collapses to a terse `_url_context: none_` marker rather than pushing dead
URLs at the model).

**Two citation systems, one of them ours.** Gemini also writes its own
hierarchical `[2.4.1]` / `[1.1.1, 1.1.2]` / `[A: NASA, 1.1.2]` indices, pointing
at a source list nobody outside the model holds. 173 of 323 archived sections
carried them and 163 carried both families at once, so half the corpus handed a
forecaster a bracket field where some brackets resolve against the rendered
`### Sources` list and some are decoration, with nothing to tell them apart.
`_strip_model_citation_indices` removes them, running after the citation splice
because that splice indexes the original response text by byte offset. It only
removes a dotted run that is delimited the way a citation is and whose every
component is at most two digits, so bracketed quantities, currency, versions,
years and IP-like tokens survive; the `### Sources` block is appended afterwards
and never passes through the strip. Validated over all 323 archived sections at
zero false positives (`scratch/next_season_bundle_2026-09/item3_citation_strip/`).
The Gemini-only prompt clause also asks the model not to write the indices in the
first place, and it carves the source-tier tags back out by name, because the
same prompt orders bracketed `[A: official]` tags 26 lines further down: a
literal reader that over-complies stops tagging, which costs the forecaster
prompts the tier signal they weight on and leaves the attribution check below
nothing to check.

**Attributions the response's own grounding record cannot back.** Gemini also
writes self-invented source-tier tags — `[A: NASA]`, `[B: Reuters]`,
`[C: Time and Date]` — and across the 323 archived sections, 478 of the 681
outlet-named tier attributions (70%) name an outlet absent from that same
response's grounded-domain list. q44953 claimed `[A: NASA]` for the eclipse path
over a source list of perlan.is / guidetoiceland.is / timeanddate.com; q45401
named 19 institutions (Bloomberg, FactSet, Goldman Sachs, Kalshi, AP, …) over a
single grounded domain. The zero-chunk floor cannot see any of this, because it
fires only when nothing grounded at all, and the forecaster prompts instruct
weighting by source tier — so an unbacked tier tag is an authority claim we
manufactured. `gemini_attribution.rewrite_unsupported_attributions` replaces each
one with `[unverified attribution]` at format time, after the citation-index
strip, keeping any outlet in the same bracket that the record does name and
dropping the tier grade along with the outlet it was read off. It never touches a
word outside a bracket: the FACT is not what is being disputed (an aggregator
domain can carry another outlet's copy), only the provenance claim. Matching is
biased hard toward keeping, on six rules — concatenation, all-tokens,
token-overlap, parent-domain-inside-name (`Chosunbiz` / chosun.com),
name-abbreviates-outlet (`WaPo` / washingtonpost.com), domain-abbreviates-name
(`Times of Central Asia` / timesca.com) — because a false strip discards real
provenance while a false keep leaves one tag standing. A response whose chunks
carry no renderable label is skipped rather than blanket-marked: with no evidence
base, a rewrite would dress our own render failure as the model's embellishment.
The token is defined where the forecaster reads it: `prompts._SOURCE_PROVENANCE_LADDER`
carries one bullet saying the pipeline could not match the named outlet against its
own retrieval record, that the claim itself may still be correct, and that the
evidence is untiered rather than low-tier. Without that, the ladder tells the model
to weight by tier while a token it has never seen stands where the tier was.
Per-response counts ride `GEMINI_UNSUPPORTED_ATTRIBUTION` (only when non-zero)
and the provider-diagnostics `unsupported_attributions` count (always, so a zero
is a measurement); nothing keys on either. The diagnostics line carries its
denominator, `tier_tags`, next to it, because the marker is gated on
`unsupported`: without the denominator a response that carried no outlet-named
tier tag at all and one whose every tag was backed both archive as
`unsupported_attributions=0`, so a model that quietly stopped tagging would read
as a model that tagged accurately. `tier_tags` counts outlet-named items only
(the generic tier words are excluded before matching), so a zero there means "no
outlet-named tags"; the definitive check for whether any tag was written is a
grep for `[A: ` over the archived section. Validation over all 323 sections,
including the hand review of every section where every attribution was marked:
`scratch/next_season_bundle_2026-09/item4_attribution_check/`.

**Grounding density, as telemetry only.** Every response that passes the floor
below logs `GEMINI_GROUNDING_DENSITY: question=... chunks=... supports=...
chars=...`, where `chars` is the raw model text. Post-floor the median response
carries one grounding support per ~872 chars and 41% of passers carry three or
fewer, which is the surface the floor cannot see. Nothing keys on these values
and there is deliberately no density gate: a decisive, true, later-verified
figure once came out of a one-support response, so a gate would have suppressed
it. The marker exists so "did embellishment move" is a query over the telemetry
archive rather than a hand audit.

**Grounded-chunk floor.** A response with no grounding evidence at all — zero
`google_search` chunks AND no successful `url_context` read — is suppressed
(returns `""`, logs `GEMINI_UNGROUNDED_SUPPRESSED`, records a
`grounding: error(ungrounded_suppressed)` loss token) rather than passed through:
ungrounded Gemini text is a demonstrated fabrication vector (Q38195, 2026-07-19 —
30 search queries, 0 grounding chunks, a confident fabricated contract table with
fake `[primary]` tags reached forecasters). "No grounding evidence" includes a
response carrying no candidates at all; that case used to return its text via an
early exit that walked straight past this floor. There is now no path around it.

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

- **yfinance** for tickers: the latest price, dated ("Latest price: X (as of
  DATE)"), with " — today's bar, in progress" appended (live only) when the
  newest bar is today's, and a stale-latest warning when the newest bar is older
  than its own cadence explains (also logged as the `FINANCIAL_STALE_LATEST`
  telemetry marker); period returns via date-based at-or-before lookups, where a
  label whose match slips past the basis's grace discloses the actual span
  ("1d (actual 2d)") and a label with no observation at or before its target is
  omitted; an annualized volatility over the recent window
  (`FINANCIAL_YFINANCE_RECENT_DAYS` trailing observations); a 52-week high/low
  range (a row-count slice of roughly one year of bars on the series' own daily
  basis); recent closes; and (live only) fundamentals.
- **FRED** for economic series: latest value (dated), previous value, change from
  the previous observation (a row step, whatever the series' cadence), a
  date-based year-over-year change, recent observations.

Both yfinance paths — live and backtest — fetch by explicit calendar start date,
`as_of − FINANCIAL_YFINANCE_LOOKBACK_DAYS` (390 days; `as_of` defaults to now). A
bare `period="Nd"` is deliberately avoided: Yahoo's chart API reads that custom
range as N trading BARS for listed assets but ~N calendar DATES for 24/7 ones —
one integer under two unit systems. Under benchmarking every fetch is
additionally ceilinged to the question's `open_time`: yfinance sets an explicit
`end` at `as_of` and skips the leaky live `.info` call, and FRED routes through a
keyless point-in-time path (`ts_fetch`, ALFRED vintages) so revised macro series
return the vintage known at forecast time, not today's revisions. A
forecaster-invisible HTML-comment routing marker records which identifiers fired,
which came from extraction vs. the classifier, and any unrecognized
(fetched-anyway-but-flagged) IDs.

### Prediction-market snapshot — `PREDICTION_MARKETS_ENABLED`

A crowd-forecast cross-check (`research/prediction_market.py`, with the retrieval
machinery in `research/market_retrieval/`). Generation is deliberately
recall-maximal and ALL the judgment sits in one LLM ranking call, because the
bake-off measured that selection, not generation, is the binding constraint: a
perfect ranker over the pool that already exists reaches 14/16 questions while the
same pool's deterministic top-4 reaches 5/16.

Four stages per question:

1. **Catalogue prefetch**, concurrent with a **query author**. Kalshi's complete
   open-events catalogue is paginated (`KALSHI_CATALOGUE_WALL_TIMEOUT`,
   `KALSHI_PREFETCH_MAX_PAGES`, `KALSHI_PAGE_SLEEP_S` between pages) and projected
   down as each page streams in, tiered so that only the fields read across an
   event's nested markets are kept past the first one. It is cached for
   `KALSHI_CACHE_TTL_S` only if it **completed**: a pull cut short by a 429, the wall
   or a runaway bound still serves the question that paid for it, but pinning that
   partial list would let one blip on the first question starve the whole run.
   PredictIt's whole ~197-market dump is one GET. Neither needs a query, which is
   what makes the concurrency free. The query author is one LLM call
   (`MARKET_QUERY_AUTHOR_LLM_CONFIG`) emitting domain vocabulary the question's own
   tokens cannot reach; its output is ADDITIVE to a deterministic query set, so its
   failure costs no recall.
2. **Venue-native search** for Manifold and Polymarket — the two venues whose own
   index is the only way in. Every deduped query is issued in parallel, with
   per-query failure isolation, and every query is stripped of digit-bearing tokens
   first because Manifold's `term` is a strict conjunction that one date token
   zeroes. The enumerable venues score against the UN-stripped set, where a year is
   real signal against a catalogue of dated market titles. Manifold is asked for
   `contractType=ALL` rather than `BINARY`: multi-outcome markets are ~30% of its
   catalogue and were structurally unreachable before, and their price arrives from
   the stage-3 detail fan-out rather than the search listing, which carries no
   per-answer data at all.

   At PARSE time the author's own synonyms are filtered by a narrower rule than the
   blanket digit strip: a synonym is **dropped whole** (never trimmed to a remnant,
   which would reach the floorless fuzzy channel and score ~100 against every event
   whose rules mention one generic word) only when it carries a DATE-like token — a
   four-digit group in 1900-2099, a bare number in a synonym that names nothing else,
   or a 1-2 digit day beside a month word. Digits belonging to a name survive
   verbatim (`U-3`, `S&P 500`, `10-K`, `50bp`), because series-code vocabulary is
   most of what the author exists to contribute and the conjunction cliff is a
   property of the enumerable venues' call site, which still strips. The rule
   knowingly mistakes `Russell 2000` for a year: a bare in-range four-digit token is
   the measured hazard, and the question's own words reach the venues regardless.
3. **Pool assembly**, three channels unioned, with channel order as the ranking: a
   settlement-source join (Kalshi events settling on a publisher the question's own
   resolution text names — the recall channel a word-overlap scorer structurally
   cannot see), then the venue-index hits, then the enumerable universes ranked by
   a fuzzy scorer with NO floor. A bounded Manifold detail fan-out then fills in the
   rules text the search listing omits, and on a multi-outcome row its leading
   answers — the only price such a row has, since the search reports none. A row
   whose detail GET failed stays title-only rather than costing the snapshot.
4. **Ranking**: one call (`MARKET_RANKER_LLM_CONFIG`) over the whole ~380-440
   candidate pool, returning up to 8 rows in ranked order with a relation tier and a
   one-phrase reason. Width is the model's choice in 0..8 — an empty array is a
   VALID answer, not a failure — and nothing downstream re-orders, re-scores or
   caps per venue. Exactly one deterministic pass runs after it, and it changes no row's
   POSITION: `cap_stale_top_tier` refuses `same_quantity_same_date` on a row whose close
   date precedes the question's own `open_time` by more than
   `MARKET_STALENESS_TIER_CAP_DAYS`, capping the grade one rung to
   `same_quantity_other_cut` and writing a `tier_cap_note` that states the demotion and
   its arithmetic (`demoted from same-date: closed 162d before the question opened`, a
   shape the rendered legend defines; it does not restate the withdrawn grade, which the
   note's presence already implies because only the top tier is ever capped). It shares the
   `why` cell's per-row character budget with the ranker's phrase rather than riding on top
   of it, so a capped row costs the section nothing. It is disclosure rather than a drop (the row keeps its
   rank, its price and its rules bullet), because a wrongly excluded market is evidence
   the forecaster never sees. It fires on nothing in the archived corpus, so read it as a
   guard on a claim a long-closed market cannot make rather than as a measured fix: only 9
   archived rows are graded `same_quantity_same_date` at all, and q45163's own offender was
   graded one tier below that. Ordering WITHIN a tier is the ranker prompt's job for the
   same reason (its `closes` recency signal), since the render is its order verbatim.
   Everything else falls open to the pool-order top rows, marked as
   such: unreadable output, and equally a transient LLM error on the call itself
   (the retry wrapper catches the `openai.APIError` family, which is what every
   litellm transport exception subclasses, and returns an empty completion the
   parser then reports unusable). Both land on the fail-open slate rather than
   costing the whole snapshot.

The render is that order verbatim: implied probability, total volume and open
interest (approximate USD on the real-money venues, play-money mana on Manifold —
the legend says which, since the two are not comparable), a liquidity/participation
`signal` label (thin / decent / deep for real-money venues, thin / decent / high by
bettor count for Manifold, `no-liquidity-data` for PredictIt), close date,
`open`/`RESOLVED` status, and the ranker's `relation` + `why`, followed by each
market's resolution rules.

A close date already in the past when the forecast was made carries a `(Nd ago)` suffix,
dated against the snapshot's own `forecast_time` so a later replay of the archived payload
reproduces what the forecaster saw rather than re-aging every row against the replay's clock.
The suffix claims only that the DATE has passed, not that trading stopped, because Manifold's
close dates are soft and its rows can read `status=open` past them. That is exactly how a
five-month-dead market reached rank 0 on q45163 with nothing in the table saying so, and the
disclosure fires on 62 of the 711 archived rendered rows (7 of them still labelled
`status=open`) at a measured cost of 1,017 characters across 102 archived snapshots. The
legend carries the reading, plus one caveat: the column is the venue's TRADING close rather
than its settlement date, and on the Kalshi rows this bot has rendered the two sit a median
+317 days apart, so a forecaster told to verify each market's resolution date was checking
against a different number.

Two row shapes have **no single probability** and render `-` rather than a number:
a Kalshi event that is a threshold FAMILY (86.5% of that catalogue), where one
strike's price under the event's own title would answer a question the row never
asked, and a Manifold multi-outcome market, whose leading answers ride inside the
rules bullet instead. Both keep their liquidity figures, which is what keeps the row
worth its width — and on a Kalshi family those figures are the SUM over its live
strikes, each converted at its own price, rather than the first strike's alone. The
forecaster prompts tell models to weight by both axes — the liquidity label and the
relation tier — and to read a RESOLVED price as a realized outcome rather than a
forecast.

This provider is **hard-disabled under benchmarking** (`is_benchmarking=True`
returns `""`), regardless of the env flag, and that guard is the ONLY leakage
defence: markets retain their last-trade price after resolution, so a market that
closes between an `as_of` instant and now leaks either way. The provider path
therefore passes `as_of=None`. The filter itself survives for explicit callers
(backtests, replay tooling) and runs INSIDE pool assembly, ahead of the per-venue
width slice, so a leaked market never reaches the ranker and an ineligible row
frees its slot for an eligible one instead of consuming it. As a post-hoc filter
over the already-truncated pool it deleted rows the width had spent its slots on
and zeroed the per-venue counts provider health reads. Prod ran with a derived
`as_of` until 2026-08-04, and it cost real recall: it dropped every market closing
before the question resolved — the "same quantity, adjacent month" class that
carries most of the evidential value —
and prod telemetry recorded 20 of 47 archived runs where Polymarket fetched
candidates and rendered nothing because of it.

The benchmarking guard is also why this provider's forecasting value can't be
measured by the standard backtest gate — it was validated with live
`test_bot.yaml` runs and opt-in live integration tests.

### Resolution-source fetcher — `RESOLUTION_SOURCE_ENABLED`

Fetches the exact URL(s) a question cites as its grading source
(`research/resolution_source.py`), so forecasters read the ground truth the
question resolves against. This is Tier-1 only: plain HTTP with browser-like
headers, no LLM calls, no retries.

It deterministically extracts URLs from resolution criteria + fine print (markdown
links and bare URLs, order-preserving dedup, Metaculus markdown-escapes undone),
skip-filters URLs that add nothing or belong to another provider (Metaculus
self-refs, FRED series owned by financial-data, Yahoo `/quote/` pages owned by
yfinance), and caps at `RESOLUTION_SOURCE_MAX_URLS` *after* the skip filter so
a run of leading self-refs doesn't starve the real sources. Fetches run in
parallel with one-request-per-host politeness (a `Semaphore(1)` per netloc, keyed
per redirect hop, and shared process-wide since 2026-09-03 — with a map per provider
call, six questions citing one host each held their own semaphore and hit it six times
at once). Content is extracted with trafilatura (HTML), or read raw (JSON / text /
CSV). A PDF is read locally with pypdf and rendered as a passage digest (below);
anything else is left unread as `unsupported_type`.

Two free rungs sit under the HTML path, both reached only when the page carried nothing
readable. A **meta-refresh hop** follows the redirect no HTTP status announces:
cdc.gov's surveillance URLs answer 200 with a ~300-byte stub whose only content is
`<meta http-equiv="refresh" content="0; url=...">`, which the manual redirect loop
cannot see (no 3xx, no `Location`), so the stub used to be classified a JS wall and the
resolving page never fetched. The target is returned as the next hop, so it re-enters
the same classification path and consumes one of the same `MAX_REDIRECTS` slots, and it
passes exactly the checks a `Location` header does. An **ARIA-table rewrite** runs
before every extraction: cdc.gov builds its outbreak stat blocks out of
`<div role="table">` / `role="row"` / `role="cell"`, which is valid accessible markup
and invisible to trafilatura's table handling — the cyclosporiasis block rendered as a
bare "17,180 / 2" with no labels and no hospitalization count at all, because 922 sat in
an unwrapped cell. Rewritten to real table tags, the same page extracts
`| Hospitalizations | 922 |`. A page with no ARIA role is handed to trafilatura as the
original bytes, so its extraction is unchanged.

A **cited PDF is now read** rather than dropped: `research/document_text.py` extracts the
text with pypdf and selects the passages most relevant to the question's title plus its
resolution criteria (BM25, deterministic, no model call), rendering a digest that states
how many pages were read and labels each passage with its page. Measured 2026-09-03,
that path pulled 833,450 chars out of a 6.7 MB 220-page document in 5.3 s with the wanted
passage in it, while the paid alternative returned nothing for the same file. A body the
server did not declare as a PDF is still sniffed by its `%PDF-` magic, since several
government hosts serve documents as `application/octet-stream` — a declared document gets
the larger `DOCUMENT_TEXT_PDF_MAX_BYTES` cap (the receipt file is over the 5 MiB response
cap), an undeclared one keeps the smaller one. Bytes we read and could not turn into text
get their own status, `unreadable_document`, with `status_reason` naming which of
`no_text_layer` / `encrypted` / `malformed` applies; only the first could ever be rescued
by a paid document read, which is why it is not folded into `unsupported_type`.

Every rung is self-bounding on the Datawrapper hop's pattern — wall minus elapsed minus a
margin, skipped below a floor — because the provider's outer `asyncio.wait_for` discards
every page that already fetched when it fires, so an overrunning rung costs the whole
question's resolution evidence rather than just its own attempt.

It is **SSRF-hardened** because these URLs are user-authored and fetches run from
CI: a preflight `is_public_http_url` check rejects private / loopback /
link-local / non-global IPs, userinfo tricks, and non-HTTP schemes, and a
connect-time `FilteringResolver` (the actual DNS-rebinding boundary, not the
preflight) re-checks every resolved IP. Redirects are followed manually under the
`MAX_REDIRECTS` hop cap (`research/http_fetch.py`, shared with the v2 agentic
tools), re-guarding each `Location`. Per-URL truncation appends a
`[truncated at N chars — full source at URL]` marker, the aggregate section-budget
trim routes through the same marker-emitting truncator (a bare slice could cut
mid-sentence and eat that marker, so an already-truncated page rendered as
complete), and the formatter appends `[N additional source(s) omitted — section
budget]` when later sections are dropped for length.

The per-URL `FetchStatus` distinguishes two kinds of non-success, and only one is a
seam. `blocked` / `js_wall` / `no_resolving_content` are pages we could not READ, and
they remain the target of a future Tier-2 LLM-driven fetch pass, as is the `no_text_layer`
half of `unreadable_document` (a scan, where a model really is the only route). `empty_body`
(a 200 whose body is empty or whitespace-only) and `unsupported_type` (including a body whose
declared charset decodes to mojibake) are bodies that carried no information — refusals
rather than seams, because there is nothing on the other side to fetch harder. Both
exist because `status="success"` has to mean CONTENT: as `success`, an empty body
rendered an empty section under the "primary grading evidence" caveat, suppressed the
all-failed "yielded no usable content" notice for every sibling URL, and reported `ok`
to provider diagnostics. That notice says "yielded no usable content" rather than "was
unreachable" because two of the statuses it covers — `no_resolving_content` and
`empty_body` — are pages that answered HTTP 200 and carried nothing, and "the tracker
was down" is different evidence from "the tracker has no reading"; the per-domain status
token beside it says which happened.

`no_resolving_content` is the newest of those seams and covers the page that answers 200
with nothing but chrome. The floor is what decides it: below
`RESOLUTION_SOURCE_EMBED_SHELL_MAX_CHARS` of extracted text the page is withheld under
this status, which costs nothing because everything archived below that floor is site
chrome and the shortest archived extraction that carries the resolving content is 401
chars. Above the floor the page kept real prose, so the prose is rendered, and where a
third-party data embed hid figures from it one bracketed line says plainly that those
figures are not in the text.

`status_reason` records which shape of chrome it was. `embed_shell` means the RAW HTML
named an embed whose numbers are real but locked inside it — Infogram, Flourish or
Tableau, detected by `unreadable_data_embed_providers` because trafilatura emits no
iframe or embed-script URLs at any setting; Datawrapper is deliberately excluded from
that scan since the Tier-2 hop reaches it. `thin_page` means no such provider was named.
That distinction used to be a GATE rather than a label, and the gate was wrong: the
2026-09-01 residual round found five content-free `success` renders and not one of them
named a provider, among them q45088's 127-char single-page-app tab list and q45215's 385
chars of Kazakh region names, both published under the primary-grading-evidence caveat.
The embed half of the story still comes from qids 44554/44556, where a Senate-forecast
tracker returned HTTP 200, extracted 2.9k chars of background, and published with zero
polling numbers in it — byte-identical across three questions, with the resolving average
sitting in two Infogram iframes and nothing anywhere saying so. `js_wall` keeps its own,
much lower floor and is checked between the two verdicts, so generalising the chrome
floor did not absorb the JS-walled population. A page can also draw both verdicts at once
and should: Tier-1 `no_resolving_content` on the page next to a Tier-2 `success` on its
Datawrapper dataset is the correct reading of a tracker whose prose we cannot use and
whose series we can.

One more rung reads data out of the page we already hold, with no second request and no
LLM call. `resolution_chart_data.render_inline_chart_data` scans the raw HTML for a
Highcharts config — a `data-chart="{…}"` attribute, or a `Highcharts.chart(…)` call whose
argument is strict JSON — and renders each series' most recent points as a compact
labelled block that leads the page text. Nothing is summed, interpolated or
unit-converted: the block states the values the page's own chart holds, and a config that
does not parse is skipped at DEBUG. It runs on every fetched HTML page rather than only
thin ones, because the record it exists for is q43949, whose resolving IOM page extracted
roughly 80k chars of incident rows and prose carrying none of the resolving figures while
its annual series sat in the attribute — reading 1,240 for 2026 in a snapshot 25 days
before a forecast that landed about 340 too high. Because chart data counts as content,
it also rescues a page the chrome floor would otherwise withhold.

Every fetched URL emits one harvested `RESOLUTION_SOURCE_FETCH` line (status, HTTP
code, any routeless embed providers, `reason` where the status alone is ambiguous, and
`route` naming which rung of the escalation ladder produced the outcome), so per-domain
fetch health is a query against the telemetry archive instead of a re-scrape of run logs
that expire from GHA at 90 days. Because that line carries only the FINAL outcome per
URL, each escalated rung additionally emits `RESOLUTION_SOURCE_ESCALATION` with the
status that triggered it, the rung tried, what came back, and the wall-clock the rung
cost — which is what makes "does this rung rescue anything, and is it worth its latency"
answerable. See "Reading run logs" in `docs/operations.md` for the field meanings.

Like prediction markets, it is **hard-disabled under benchmarking** (current page
content post-dates any backtest window).

### Time-series anchor — `TS_ANCHOR_ENABLED` (chart side-channel `TS_ANCHOR_CHART_ENABLED`, off)

A deterministic empirical anchor for numeric questions whose resolution series is
a fetchable FRED/yfinance series (`research/timeseries_anchor.py`). No LLM, no
model selection — it renders the latest value, dated ("as of DATE", with an
in-progress marker when today's bar is still forming and a stale-latest warning —
the same `FINANCIAL_STALE_LATEST` marker as the financial-data provider — when
the newest observation is older than its cadence explains), a multi-resolution
history, a 52-week range, and a horizon-matched empirical band built only from
the series' own past. The Phase-A offline replay found CV-gated model picks beat naive
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
run **concurrently** in one `asyncio.gather` inside `run_research`
(`research/orchestrator.py`), so the research-phase wall-clock is `max(v1, v2)`,
not the sum. Each runs inside its
own try/except so a defect in one can never zero the other's output. Both consume
the pre-gap-fill bundle, which means the v2 driver's brief does not see v1's
addendum; v2's section appends after v1's.

### v1 — targeted gap-fill (`research/targeted.py` `run_gap_fill_pass`)

Two stages, gated by `GAP_FILL_ENABLED` (and skipped when the first-pass bundle is
shorter than `GAP_FILL_MIN_RESEARCH_CHARS`):

1. A non-grounded analyzer LLM (`GAP_FILL_ANALYZER_MODEL`, low effort) reads the
   first-pass research and emits a JSON list of up to `GAP_FILL_MAX_GAPS` factual
   gaps.
2. Each gap is resolved by a parallel OpenAI native web search
   (`GAP_FILL_RESOLVER_MODEL` at `GAP_FILL_RESOLVER_REASONING_EFFORT`, via
   OpenRouter on the donated key).
   Because the searches run in parallel, latency is the slowest call, not the sum.

The resolver migrated off direct-Google grounding on 2026-06-25, which is why
`GOOGLE_API_KEY` is no longer required for gap-fill. The whole pass never raises —
it returns `""` on any error — and appends its results under
`## Targeted Gap-Fill (second pass)`.

### v2 — agentic gap-fill (`research/agentic_gap_fill.py` `run_gap_fill_v2`)

A bounded agentic tool loop run by a driver LLM (`GAP_FILL_V2_DRIVER_MODEL` at
`GAP_FILL_V2_DRIVER_EFFORT`), gated by
`GAP_FILL_V2_ENABLED`. The driver is briefed with the forecaster prompt, privately
dry-runs a forecast to find things worth filling or verifying, then iterates over
four tools (news search, web search, fetch, document read) under a wall deadline
(`GAP_FILL_V2_WALL_DEADLINE`) and a tool-call budget
(`GAP_FILL_V2_MAX_TOOL_CALLS`). It appends a
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
replay by `ResearchPersistenceWriter` (`research/persistence.py`, at
`RESEARCH_SCHEMA_VERSION`). The record carries the assembled `research_text`, the
per-provider
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
