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
2. **Exa.ai** (`SmartSearcher`) if `EXA_API_KEY` is set — a generic rundown
   (`_exa_provider`, `research/providers.py`).
3. **Perplexity direct** if `PERPLEXITY_API_KEY` is set. Model:
   `PERPLEXITY_RESEARCH_MODEL` (`constants.py`); the function is
   `_perplexity_provider` (`research/providers.py`), and its prompt explicitly
   asks for prediction-market consideration unless the run is benchmarking.
4. **Perplexity via OpenRouter** if `OPENROUTER_API_KEY` is set. Same function
   called with `use_open_router=True`, same model, prefixed for the OpenRouter
   route: `PERPLEXITY_RESEARCH_MODEL_VIA_OPENROUTER`.
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
`NATIVE_SEARCH_VERBOSITY`. On the wire the effort goes out as
`reasoning={"effort": ...}` and the verbosity as `extra_body={"verbosity": ...}`.

The model migrated on 2026-07-09 to `gpt-5.6-sol`, then on 2026-07-17 to
`gpt-5.6-terra` per the blind research-role audit
(`scratch/research_role_audit_2026-07-17/` — terra 1st, sol 2nd, luna 3rd; verdict
"MARGINAL EDGE", terra at −42% cost). Effort has been low since 2026-05-20 for
latency reasons; see `constants.py`.

The model is built at `allowed_tries=1` on purpose: an earlier incident
(2026-05-20) had OpenRouter drip whitespace keep-alive bytes for over eight
minutes before returning malformed JSON, and retrying that call just multiplies
the wait. `NATIVE_SEARCH_WALL_TIMEOUT` plus a single try is what bounds the worst case.
It routes through `build_llm_with_openrouter_fallback`, so it bills the
Metaculus-donated `OAI_ANTH_OPENROUTER_KEY` first and falls back to the personal
`OPENROUTER_API_KEY` on credential/credit errors — that fallback is
`FallbackOpenRouterLlm` (`metaculus_bot/fallback_openrouter.py`). The donated key
used to be blocked here by a data-policy restriction; that block has been
RESOLVED, verified 2026-06-25 by a live call returning 200 with grounding, so
native search now routes through and bills the donated key. The prompt is the
shared `web_research_prompt` with markdown citations.

### Gemini grounded search — `GEMINI_SEARCH_ENABLED` + `GOOGLE_API_KEY`

Real first-party Google Search grounding via the `google-genai` SDK
(`research/gemini_search.py`), NOT via OpenRouter. This adds a genuinely distinct
search index to the ensemble. Model and request timeout come from
`GEMINI_SEARCH_DEFAULT_MODEL` / `GEMINI_SEARCH_TIMEOUT` (`constants.py`). It
enables both the `google_search` tool and the `url_context` tool, so the model
can read specific URLs named in a question's fine print directly.

Output is stitched together with real `[N]` citation markers spliced in from the
response's grounding metadata, plus a matching `### Sources` domain list
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
`_strip_model_citation_indices` (shipped 2026-09-01) removes them. It MUST run
after `_splice_inline_citations`, because that splice indexes the ORIGINAL
response text by grounding-support byte offsets — rewrite the text first and our
real markers land mid-word, a bug class this repo has already shipped and fixed.
It only removes a dotted run that is delimited the way a citation is AND whose
every dot-separated component is at most two digits, so bracketed quantities,
currency, versions, years and IP-like tokens survive. Both bounds were measured on
the archive: across 2,609 dotted bracket groups the largest component anywhere is
39. The plan's alternative — require at least three components — was NOT adopted,
because all 165 two-component groups read in context are genuine indices, so the
stricter rule would have left 318 fake markers standing for no safety gain. The
strip runs on both forecaster-facing branches (the grounded path and the
url_context-only escape) and never on the rendered `### Sources` block, which is
appended afterwards and whose labels are page titles that legitimately carry
version numbers. Validated over all 323 archived sections at zero false positives
(`scratch/next_season_bundle_2026-09/item3_citation_strip/`).
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
manufactured. `_check_attributions` → `rewrite_unsupported_attributions`
(`research/gemini_attribution.py`, shipped 2026-09-01)
replaces each one with `[unverified attribution]` at format time. It runs on the
grounded path ONLY — the url_context-only escape gets the citation strip and
returns — after that strip and before the `### Sources` block is appended, with
`_grounded_source_labels` the single derivation of both the check's evidence base
and the rendered block, so the two can never disagree about what our record says.
A supported outlet in the same bracket survives verbatim with its own separator:
`[A: FDA, B: Food Safety Magazine]` on a record holding fda.gov renders
`[A: FDA, unverified attribution]`. Several unsupported names in one bracket
collapse to a single marker. The tier grade goes with the outlet it was read off,
because the grade IS the claim. It never touches a word outside a bracket: the
FACT is not what is being disputed (an aggregator domain can carry another
outlet's copy), only the provenance claim — which is why the marker says
*unverified* and not *false*.

Generic tier words that name a CLASS rather than an outlet are skipped before
matching (`official` alone is 243 of the corpus's 790 tier items, and the skip
list is carried over from the audit's own so the shipped rate stays comparable to
its measurement). Matching is then biased hard toward KEEPING, because a false
strip discards real provenance while a false keep merely leaves one tag standing.
Any one of six rules credits a name: it concatenates into the domain
(`Golf Channel` / golfchannel.com); all of its identity tokens appear in the
domain (`The Guardian` / guardian.co.uk); the token sets intersect (`LSE Blogs` /
lse.ac.uk); the domain's registrable core sits inside the name, the sub-brand
shape (`Chosunbiz` / chosun.com); a single-token name is a subsequence of the
label (`WaPo` / washingtonpost.com — single-token only, since a subsequence test
over a multiword name credits almost anything); or the domain core abbreviates the
name (`Times of Central Asia` / timesca.com). A response whose chunks carry no
renderable label is skipped rather than blanket-marked (q44802): with no evidence
base, a rewrite would dress our own render failure as the model's embellishment.
That skip is what makes the count's ABSENCE meaningful — on a schema-v2 record an
absent `unsupported_attributions` means the check had no evidence base or the
record predates the change, while a recorded 0 means it ran and found nothing.
The token is defined where the forecaster reads it: `prompts._SOURCE_PROVENANCE_LADDER`
carries one bullet saying the pipeline could not match the named outlet against its
own retrieval record, that the claim itself may still be correct, and that the
evidence is untiered rather than low-tier. Without that, the ladder tells the model
to weight by tier while a token it has never seen stands where the tier was.
Per-response counts ride
`GEMINI_UNSUPPORTED_ATTRIBUTION: question=... tagged=N unsupported=N groups=N
labels=N` (INFO, emitted only when `unsupported` > 0, harvested as
`gemini_unsupported_attribution`, and deliberately NOT alertable — the habit is the
model's, not a bot defect) and the provider-diagnostics
`unsupported_attributions` count (always, so a zero is a measurement); nothing
keys on either. `labels` rides the line because the same `unsupported` count reads
completely differently against it — q38195 named 21 outlets over ONE grounded
domain, aft.org. `groups` is the render footprint, which sits below `unsupported`
because of the collapse. There is no `rewritten` or `stripped` field, because under
this design `rewritten` always equals `unsupported` and the check never removes a
bracket outright. The diagnostics line carries its
denominator, `tier_tags`, next to it, because the marker is gated on
`unsupported`: without the denominator a response that carried no outlet-named
tier tag at all and one whose every tag was backed both archive as
`unsupported_attributions=0`, so a model that quietly stopped tagging would read
as a model that tagged accurately. `tier_tags` counts outlet-named items only
(the generic tier words are excluded before matching), so a zero there means "no
outlet-named tags"; the definitive check for whether any tag was written is a
grep for `[A: ` over the archived section.

Measured over all 323 sections: 48 sections rewritten, 203 attributions kept, 478
marked, 0 idempotency failures, and 0 sections where any text outside a bracket
changed. The 70% headline reconciles with the audit's published 87% (276/318) via
86% (590/685), which is what the audit's own matching rule gives through this
harness's extraction — the residual gap is occurrence- versus distinct-name
counting and the three-source union versus artifact-only — and the six keep rules
then move 86% → 70%. All 11 fully-unsupported sections were read in context and
all 11 are true positives; one residual arguable case is test-pinned
(`NewsRadio WFLA` against a grounded iheart.com, 2 of 681 — Google reported only
the parent domain, and no general rule recovers a subdomain the SDK never sent);
and the deliberate false-KEEP exposure is enumerated at 20 occurrences across 10
distinct names (2.9%), all short acronyms or shared tokens. Rules, counts, both
review sets and the similarity screen behind the false-strip review:
`scratch/next_season_bundle_2026-09/item4_attribution_check/VALIDATION.md`; the 87%
receipt is `scratch/residual_2026-08-31/gemini_search_audit/cutB_pattern.md` §3.2.

**Grounding density, as telemetry only.** Every response that passes the floor
below logs `GEMINI_GROUNDING_DENSITY: question=... chunks=... supports=...
chars=...`, where `chars` is the raw model text (which is the density the audit
measured); it is harvested as `gemini_grounding_density`. Post-floor the median
response carries one grounding support per ~872 chars and 41% of passers carry
three or fewer, which is the surface the floor cannot see. Nothing keys on these
values and there is deliberately no density gate: q44944's decisive, later-verified
ICE figure came out of a one-support response, so a gate would have suppressed the
round's best find. The marker exists so "did embellishment move" is a query over the telemetry
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

**Where the code lives.** `financial_data.py` itself keeps the classifier, the
identifier extraction and capping, the job fan-out and the yfinance block. The
2026-09-01 additions below pushed it past the file-size ceiling, so two feature
areas moved to siblings: `research/currency_pegs.py` (the `HARD_PEG_ANCHORS` table
plus `peg_for_ticker` / `peg_disclosure_lines`, stdlib-only so it can never cycle)
and `research/fred_rendering.py` (value/change formatting, `_render_fred_series`,
the first-release table, and BOTH FRED fetchers). The fetchers ride with the
renderers deliberately: `_fetch_fred_first_releases` reads
`Fred.earliest_realtime_start` / `latest_realtime_end` off the class, and its test
proves that by patching `Fred` where the client is constructed, so client
construction and the class-attribute reads have to stay in one patchable
namespace. **Every `Fred` / `fetch_series` patch target is therefore
`metaculus_bot.research.fred_rendering`, not `financial_data`** — fredapi's real
class carries the identical literals, so a patch at the wrong module stays green
while proving nothing. Tests split the same way: `tests/test_currency_pegs.py`,
`tests/test_fred_rendering.py`, with the shared yfinance mock and synthetic series
in `tests/financial_fakes.py`.

The rendered block is no longer just derived stats. Five additions landed
2026-09-01 out of the 2026-08-31 residual round (q44797, q44944):

- **Hard pegs arrive labeled.** `HARD_PEG_ANCHORS` is a static table of eleven
  currencies whose dollar cross is a fixed quote rather than a traded market: the
  Common Monetary Area trio against the rand, Denmark and the two CFA francs
  against the euro, Brunei against the Singapore dollar, and Hong Kong / UAE /
  Saudi / Qatar against the dollar itself (every rate and date verified against
  the issuing authority 2026-09-01). A pegged ticker's block carries a warning
  saying what is fixed and that day-to-day movement is mostly quote noise, then
  appends the liquid anchor cross's whole block below, labeled. Nothing is
  SUBSTITUTED — the question still resolves on the pegged pair, so its own quote
  stays on the page — and a dollar-pegged currency has no substitute cross, so the
  block says so instead of inventing one. Deliberately a static table, not a
  correlation detector: hard pegs are published policy and do not need inferring.
- **A variance-ratio noise flag, with the robust figure as the headline.**
  `variance_ratio` (`research/ts_estimators.py`) is an overlapping Lo-MacKinlay
  ratio on log returns over the provider's full held history (~265 daily bars, NOT
  the 30-row volatility window — the statistic is uninformative at n=30). Below
  `FINANCIAL_VARIANCE_RATIO_FLOOR` the block prints the flag, leads with
  `multi_period_annualized_vol_pct` measured on overlapping
  `FINANCIAL_VARIANCE_RATIO_LAG`-step returns, and labels the 30-row figure
  noise-suspect. **Promoting the long window alone does not reach the case:** both
  volatilities are computed from ONE-day returns and independent quote noise
  inflates both equally, so on q44797's series the long window moves 17.85% to
  15.2% where an honest estimate was 11-14%; the robust figure equals the one-day
  figure times √VR by construction, which is the same statistic's own remedy. Both
  estimators return None on a sample with no measurable return variation, since the
  ratio there is a quotient of floating-point rounding noise (it read 0.369 on an
  exact ramp — a confident noise flag manufactured out of mantissa bits). The
  screen, its `FINANCIAL_VARIANCE_RATIO_MIN_RETURNS` sample floor and the marker's
  one format string live in `research/noise_flag.py` (`screen_for_quote_noise` /
  `noise_flag_line`), shared by both surfaces, because the two copies of the vol
  estimator had already drifted once — the q44882 `sqrt(252)`-on-a-24/7-series
  defect was fixed in one copy weeks before the other. Only the forecaster-facing
  prose is local to each renderer, since the two say different things about what
  else in their section the noise affects.
- **The long-horizon volatility now prints beside the 30-row one on every block**,
  flag or no flag, labeled with its actual row count and step unit.
- **FRED levels render at full precision.** Five sites in `_render_fred_series`
  used `:.4g`, which turned a Case-Shiller print of 331.893 into "331.9" on a
  question whose displayed range was four index points wide, and the Fed balance
  sheet into "6.7e+06". They go through `format_decimal_value` /
  `format_decimal_change` now, which live in `research/number_format.py`
  (stdlib-only, so the FRED block and the inline-chart rung in
  `research/resolution_chart_data.py` can share one rule without dragging pandas or
  fredapi into the latter): fixed-point, up to six decimals, trailing zeros
  stripped, never scientific notation — which also cleans up float-subtraction
  noise, so 0.8729999999999905 renders "0.873". The time-series anchor's own
  formatter, `ts_render._fmt`, was swept the same way at the same time:
  fixed-point up to THREE decimals above 100 (three, not six, because it also
  renders the empirical P10/P50/P90 band, where six decimals on an estimate would
  be fabricated precision), `:.4g` below. Both providers append unconditionally, so
  before that sweep one bundle could state two different values for one
  observation in two adjacent sections.
- **First release vs current vintage.** For a revising FRED series the question
  actually resolves on (`is_resolving_source`, URL-extracted from the resolution
  criteria rather than merely named by the classifier, and not on
  `FRED_NON_REVISING_SERIES`), one extra free ALFRED call renders a table of the
  recent prints' initial releases, current values, revisions, and the observed
  revision direction. It carries the q44944 dossier's mandatory caveat: a
  revision-direction adjustment and a same-source leading indicator measure the
  same underlying data, so apply one, not both.

**A hallucinated FRED series no longer erases the block, and no longer hides as an
`empty`.** The reference tables carry no exchange-rate FRED series at all and only
three currency crosses, so on a question about any other currency the classifier
had nothing to route to and invented an id: q45363 (the Boliviano-USD rate) got
`DEXBOUS`, which does not exist on FRED, with no Yahoo cross named beside it, so
the forecasters got no level and no realized volatility on a currency question —
the verification pass measured that a member sized off the resolving series' own
30-print volatility would have scored +55.35 spot peer alone, better than every
member that ran. Three changes. (1) The classifier prompt now routes every
exchange rate to a Yahoo cross (`USD<ISO>=X` / `<ISO>USD=X`, the spelling matching
how the question quotes the rate) and forbids inventing a FRED id. That is the fix
at the cause, because the currency's ISO code is not recoverable downstream —
FRED's country codes are not ISO currency codes and the `BO` in `DEXBOUS` is a
country, so the classifier is the only step that can name the pair. (2) A series
FRED reports as nonexistent raises `UnknownFredSeries`
(`research/fred_rendering.py`, keyed on FRED's own
`400 "The series does not exist"` body, which fredapi surfaces as a `ValueError`),
so it reaches diagnostics
as `unknown_series` rather than the ambiguous `empty` that was q45363's only trace,
with one `FRED_UNKNOWN_SERIES: series_id=... proposed_by=classifier|resolution_url`
WARN harvested as `fred_unknown_series` — non-alertable, since an invented id is
the classifier's habit rather than a bot crash, and `proposed_by` separates that
from a question whose own resolution criteria link a dead FRED page. (3) When a
question's exchange-rate identifiers carry nothing, the section is ABSENT — no "we
looked and found nothing" line, for the same reason AskNews returns `""` rather
than its old `No articles were found` sentence: any non-empty return flips the
orchestrator's status from `empty` to `ok`, counts the provider in
`providers_succeeded`, and defeats every downstream empty guard at once, so prose
can never stand in for an absent section. What carries the signal instead is
`counts.fx_identifiers_empty`, the number of attempted identifiers whose name has
the shape of an exchange rate on either vendor (`DEX????` on FRED, `???=X` /
`??????=X` on Yahoo — shape predicates `is_fred_fx_series` / `is_yahoo_fx_ticker` /
`is_fx_identifier` in `research/fx_identifiers.py`) and whose `details["sources"]`
token is a loss under the canonical `is_lost_source`, so `empty`,
`unknown_series`, `error` and `skipped(no_fred_api_key)` all count. It is recorded
on every path, so a 0 means the check ran rather than never having run, and it is
independent of whether the section rendered: an FX identifier lost beside a ticker
that rendered fine is the same partial gap and reads the same way, beside the
`sources=<ok>/<total>` the diagnostics line already carries. Separately, a FRED
series skipped for a missing `FRED_API_KEY` now records `skipped(no_fred_api_key)`,
where it used to leave no source token at all and N unfetched series read as a
fully healthy line. The keyless benchmarking fetcher stays silent on all of this:
fredgraph cannot tell a bad id from a vintage predating the series.

Both volatility surfaces emit
`FINANCIAL_NOISE_FLAG: surface=financial_data|ts_anchor symbol=... vr_lag=... vr=...
floor=... short_vol=... long_vol=... robust_vol=...` at INFO, harvested as
`financial_noise_flag` and non-alertable — it describes the vendor's data, not a
bot defect. The sibling flag on the time-series-anchor surface
(`ts_render._realized_vol_lines`) runs the same screen with the same constants,
because the anchor routes to any Yahoo ticker a resolution URL cites and would
otherwise render an equally inflated figure with no disclosure; its prose also
states that the anchor's change BANDS are unaffected, since those are empirical
multi-observation quantiles over which the noise cancels. `long_vol` reads `None`
on the anchor surface, which computes no long-horizon window at all — `surface=` is
what tells that apart from a yfinance series too short to hold one.

### Prediction-market snapshot — `PREDICTION_MARKETS_ENABLED`

A crowd-forecast cross-check. `research/prediction_market.py` is the seam module;
the retrieval pipeline lives in `research/market_retrieval/`. This is **ranked
retrieval**, live since 2026-08-04 (`e75e708`), and it replaced a keyword/fuzzy
design that the 2026-08-03 bake-off measured at 0/17 on near-identical markets.
Generation is deliberately recall-maximal and ALL the judgment sits in one LLM
ranking call, because the bake-off measured that selection, not generation, is the
binding constraint: a perfect ranker over the pool that already exists reaches
14/16 questions while the same pool's deterministic top-4 reaches 5/16.

Four stages per question:

1. **Catalogue prefetch**, concurrent with a **query author**. Kalshi's complete
   open-events catalogue (~10k open events, streamed from `/events`) is paginated
   (`KALSHI_CATALOGUE_WALL_TIMEOUT`,
   `KALSHI_PREFETCH_MAX_PAGES`, `KALSHI_PAGE_SLEEP_S` between pages) and projected
   down as each page streams in, tiered so that only the fields read across an
   event's nested markets are kept past the first one. It is cached for
   `KALSHI_CACHE_TTL_S` only if it **completed**: a pull cut short by a 429, the wall
   or a runaway bound still serves the question that paid for it, but pinning that
   partial list would let one blip on the first question starve the whole run.
   PredictIt's whole ~197-market dump is one GET, and all ~197 go into the pool
   UNFILTERED: its old fuzzy pre-filter ranked "Will the Pope visit Cuba" above the
   on-topic market. Neither venue needs a query, which is
   what makes the concurrency free. The query author is one LLM call
   (`MARKET_QUERY_AUTHOR_LLM_CONFIG`) emitting domain vocabulary the question's own
   tokens cannot reach; its output is ADDITIVE to a deterministic query set, so its
   failure costs no recall.
2. **Venue-native search** for Manifold and Polymarket — the two venues whose own
   index is the only way in, at width 60 each. Every deterministic query plus every
   query-author addition is issued unconditionally, in parallel after dedup, with
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
   settlement-source join (Kalshi events whose `settlement_sources` domains match a
   publisher the question's own resolution URLs name, matched through a vendored
   public-suffix list — the recall channel a word-overlap scorer structurally
   cannot see), then the venue-index hits, then the enumerable universes ranked by
   a fuzzy scorer with NO floor (Kalshi to width 100). A bounded Manifold detail
   fan-out then fills in the
   `textDescription` rules text the search listing omits, via per-market detail
   GETs, and on a multi-outcome row its leading
   answers — the only price such a row has, since the search reports none. A row
   whose detail GET failed stays title-only rather than costing the snapshot.
4. **Ranking**: one call (`MARKET_RANKER_LLM_CONFIG` — luna at low effort, a
   ~36k-token prompt) over the whole ~380-440
   candidate pool, returning up to 8 rows in ranked order, each stamped with a
   relation tier and a one-phrase `why`. The tier vocabulary is exactly four words,
   in this order of strength: `same_quantity_same_date` >
   `same_quantity_other_cut` > `driver_or_consequence` > `weak`.
   Width is the model's choice in 0..8 — an empty array is a
   VALID answer, not a failure — and nothing downstream re-orders, re-scores or
   caps per venue. Exactly one deterministic pass runs after it, and it changes no row's
   POSITION: `cap_stale_top_tier` (`market_retrieval/ranking.py`) refuses
   `same_quantity_same_date` on a row whose close
   date precedes the QUESTION's own `open_time` by more than
   `MARKET_STALENESS_TIER_CAP_DAYS` (60, local to that module), capping the grade
   one rung to
   `same_quantity_other_cut` and writing a `MarketMatch.tier_cap_note` that states
   the demotion and
   its arithmetic (`demoted from same-date: closed 162d before the question opened`, a
   shape the rendered legend defines; it does not restate the withdrawn grade, which the
   note's presence already implies because only the top tier is ever capped — the old
   wording closed the cell with `(ranker said same_quantity_same_date)` inside a table
   whose preamble tells forecasters to anchor on a same-date market's price). It shares the
   `why` cell's `WHY_CHARS` budget with the ranker's phrase (note first, phrase
   truncated to the remainder) rather than riding on top
   of it, so a capped row costs the section zero characters; while it was exempt,
   three capped rows on a maxed slate crossed the section budget and no fixture set a
   note, so nothing could see it. The note is its own field on purpose:
   `relation_tier` must stay one of the four vocabulary words (`STRONG_TIERS`
   membership picks which preamble renders, and every tier-conditioned residual cut
   tests it by equality) and `relevance_label` must stay the ranker's verbatim
   phrase, or "what the model said about a row our arithmetic overruled" becomes
   unrecoverable from the archive. It keys on question OPEN time, not forecast time,
   so the same market grades the same however late in the window the bot runs. It is
   disclosure rather than a drop (the row keeps its
   rank, its price, its liquidity cells and its rules bullet), because a wrongly
   excluded market is evidence
   the forecaster never sees. It fires on nothing in the 102 archived snapshots, so
   read it as a
   guard on a claim a long-closed market cannot make rather than as a measured fix: only 9
   archived rows are graded `same_quantity_same_date` at all, and q45163's own offender was
   graded one tier below that — whether to demote that tier too (the dossier's own
   recommendation, `driver_or_consequence`) is an open operator decision in FUTURE.md,
   because that tier changes how the forecaster prompt weights the row. An actual
   demotion logs `MARKET_TIER_CAPPED: question=... rows=... capped=venue@rank` at
   INFO — only on a real cap, harvested as `market_tier_capped`.
   Ordering WITHIN a tier is the ranker prompt's job for the
   same reason: its signals block carries a fourth bullet making `closes` a RECENCY
   tiebreaker within a tier, and that has to live in the prompt because the renderer
   shows the ranker's order verbatim and re-sorting downstream is a measured
   non-option (a previous re-ordering pass lost 43 of 58 wanted rows).
   Everything else falls open to the deterministic retrieval-order top 8, marked
   `[ranking unavailable — showing retrieval order]`: unreadable output; a non-empty
   ranking array from which NO usable row can be read (a renamed index key, every
   index out of range), which raises `RankingShapeRegression`; and equally a
   transient LLM error on the call itself
   (the retry wrapper catches the `openai.APIError` family, which is what every
   litellm transport exception subclasses, and returns an empty completion the
   parser then reports unusable). Both land on the fail-open slate rather than
   costing the whole snapshot.

   That fail-open is kept strictly distinct from a DELIBERATE zero-row ranking over
   a non-empty pool, which renders one sentence saying so, because every genuine
   failure path renders nothing and that is what keeps an outage distinguishable
   from a considered empty answer. `RankingShapeRegression` exists for exactly that
   boundary: the renamed-index case used to arrive as `ok(0)` and render the
   deliberate-empty sentence — a forecaster-facing affirmative claim ("prediction
   markets were retrieved and reviewed… none was judged to bear on it") on a path
   where our own prompt/parser contract had broken.

The render is that order verbatim: implied probability, total volume and open
interest (approximate USD on the real-money venues, play-money mana on Manifold —
the legend says which, since the two are not comparable), a liquidity/participation
`signal` label (thin / decent / deep for real-money venues, thin / decent / high by
bettor count for Manifold, `no-liquidity-data` for PredictIt), close date,
`open`/`RESOLVED` status, and the ranker's `relation` + `why`, followed by each
market's resolution rules.

A close date already in the past when the forecast was made carries a `(Nd ago)` suffix
(`_close_cell`, `rendering.py`), on parent rows and `↳` sub-rows alike,
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

Two row shapes have **no single probability** and render `-` rather than a number
on the parent row: a Kalshi event that is a threshold FAMILY (86.5% of that
catalogue), where one strike's price under the event's own title would answer a
question the row never asked, and a Manifold multi-outcome market. Both keep their
liquidity figures, which is what keeps the row
worth its width — and on a Kalshi family those figures are the SUM over its live
strikes, each converted at its own price, rather than the first strike's alone.

**Since 2026-08-25 (`58175a7`) a multi-outcome family renders WHOLE**, and that is
`rendering.py`'s most load-bearing contract. A family is a distribution over its own
outcome space, its forecast content is the SHAPE of that distribution, and no
subset carries a shape, so truncating from the end was answering the wrong
question. Measured before the fix: 108 of 162 archived families (67%) were
truncated, and truncation correlated WITH relevance (81% of
`same_quantity_other_cut` families versus 50% of `weak` ones); on q45189 all three
forecasters read the one surviving bracket of a ten-bracket margin ladder as an
equality constraint and cut the resolving bucket below their own prior. Now the
leading outcomes get full `↳` sub-rows (`MAX_CHILD_ROWS_PER_SNAPSHOT`, cut 24 → 14
because the bound now sizes only the FULL sub-rows), and every remaining outcome is
NAMED with its own price in one `↳ [remaining N]` ladder row instead of being
dropped. Under character pressure that ladder collapses groups in increasing order
of forecast content — unquoted first (no price, nothing to say), then settled, then
open outcomes by an escalating floor (`LADDER_PRICE_FLOORS`, walked one stage at a
time up to `LADDER_MAX_STAGE`) — and every collapsed group states its count and its
summed price, a counted set rather than a silent cut. Settled is deliberately not
collapsed before unquoted: a Manifold threshold ladder settles its crossed rungs to
exactly 1.0 while the market stays open (10 of 17 on that module's committed
fixture), so those titles are the floor the series has already passed, which is why
the group names its LAST member.

Which open outcomes count as "least informative" depends on the family's SHAPE, and
`LADDER_CUMULATIVE_PRICE_SUM` (1.2) is what tells the two apart. A mutually
exclusive partition's prices sum to ~1 by construction (q45189's ten margin
brackets sum to 0.965; a real one lands in roughly 0.95-1.05), while a threshold
ladder's nested prices are SURVIVAL probabilities and sum to roughly the rung count
times the average survival — a median 1.46 across the archived Kalshi families, and
25.4 on the 50-rung gold ladder. On a partition the informative outcomes are the
highest-priced ones, so the collapse is cheapest-first. On a cumulative ladder they
are the ones nearest the CROSSING, so the collapse ranks by distance from
certainty, `min(p, 1-p)` — a 0.99 "above $3251" rung on a gold ladder trading near
$4400 is a near-certainty carrying no forecast content at all, and price-ranking is
close to worst-possible there. Half the archived Kalshi families are cumulative
threshold ladders. When every stage is spent and the section is still over, a
per-family hard bound keeps the highest-priced terms that fit
(`LADDER_HARD_BOUND_STAGE`, the sentinel `99` so the marker's `max_stage=` reads
"fell off the end of the ladder" rather than as one more ordinary stage) and closes
with a counted, summed remainder.

Every summed price names the count it covers. `_open_price_total` sums only OPEN,
PRICED members — a settled rung's price is a realized outcome and an unquoted one
has none — so the per-family hard bound renders `+N more (K priced, X summed)`. A
bare `+160 more (78.50 summed)` read as 78.50 across 160 outcomes when it was 78.50
across 157, and on a settled ladder the same shape hid rungs realized at 1.00.

Two supporting rules make that render trustworthy. The venue parsers no longer sort
children: all three price-bearing venues return the venue's own catalogue order and
the renderer owns presentation, because a parser sorting its children was deciding
what survived a budget it could not see, and price-descending scrambles threshold
ladders. And each venue BLANKS its own manufactured ~0.50 default at parse time, so
a fabricated price reaches neither the ranker nor the render nor a disclosure figure
(192 of 1,839 archived ranked-era child outcomes were in that class). Kalshi blanks
on a book at least `KALSHI_NO_PRICE_SPREAD` wide — an empty book is
`0.0000`/`1.0000`, whose midpoint is a synthetic $0.50 nobody quoted, and the cell
then renders the raw range `0.00-1.00`, which cannot be read as a point
probability. Polymarket blanks on Gamma's `["0.5","0.5"]` placeholder when the leg
carries no volume and no open interest. Manifold blanks (`_priced_or_none`) an
answer sitting at its untouched 0.5 prior with zero volume — in the ranker's
candidate segment as well as in the children, where a defaulted price had been
distorting selection upstream of the render.

Two liquidity-label corrections came out of the same work, both in
`market_retrieval/types.py` so a sub-row is labelled by the SAME rule as its
parent. A Manifold child whose OWN volume is present and zero now reads `thin`
whatever its parent market's bettor pool: Manifold publishes no per-answer bettor
count, so a market with 150 bettors labelled every one of its untouched answers
`high`, including the ones `_priced_or_none` had just refused a price for (62 of 399
archived Manifold children, 15.5%, rendered decent/high on zero own volume). Absent
volume is not evidence of no trading, so the same `is not None` gate applies. And a
Kalshi contract count with no price to convert by (no book AND no last trade) leaves
`total_vol` unknown rather than stating `$0`, which read as a market nobody traded.

The forecaster prompts tell models to weight by both axes — the liquidity label and
the relation tier, whose shared constants live in `prompts.py` — to read a RESOLVED
price as a realized outcome rather than a forecast, to read a family of `↳` rows as
a DISTRIBUTION rather than an equality constraint on a tail, and to resolve a
relation-vs-liquidity conflict in favour of liquidity: a thin market's price is
noisy even when its relation is tight, so widen around it rather than transplant it.

The staleness disclosure is measured against the section's own character budget.
It cost 1,017 characters across 102 archived snapshots (median 0), but the
adversarial worst case is tighter: the maxed budget in `tests/` moved 10,600 →
11,050 for the disclosure and 11,050 → 11,150 for the demotion note's legend
sentence, against a `RESEARCH_SECTION_CHAR_LIMIT / 4` ceiling of 11,249. That
leaves 99 characters of structural headroom (164 from the measured worst case of
11,085), so the next change that widens this section has to cut prose rather than
spend slack.

`MarketSnapshot.forecast_time` is set in `_fetch_market_snapshot_impl` to
`as_of or datetime.now(UTC)`, and the staleness suffix reads it rather than the
renderer taking the clock: the render has to be reproducible from an archived
snapshot alone, and a replay months later would otherwise stamp staleness on rows
the forecaster never saw it on. Archived snapshots predating the field carry None
and render exactly as before, and a backtest with `as_of` supplied can fire no
disclosure at all, because pool assembly already dropped everything closing at or
before it.

**Per-question telemetry**, all three harvested into the telemetry archive
(`market_ranking` / `market_child_render` / `market_ranking_degraded` in
`scripts/telemetry/markers.py`), so pool indices, prompt sizes, child-render counts
and degradation causes survive the 90-day GHA log expiry:

- `MARKET_RANKING: question=... pool=N outcome=ranked|failopen|empty rows=K
  prompt_chars=M rendered=...`, where `rendered` is the per-row
  `venue:pool_index@rank` list.
- `MARKET_CHILD_RENDER: question=... families=... full_rows=... ladder_rows=...
  outcomes=... named=... collapsed=... withheld=... max_stage=...
  ladder_chars=...`. `named + collapsed == outcomes` is the completeness invariant,
  so a line where they disagree is a render bug rather than a tuning signal;
  `withheld=` turns the blanking rules' prod incidence into a query rather than a
  guess (the Kalshi no-price spread threshold is calibrated on eleven fixture
  strikes); and `max_stage` / `ladder_chars` say whether `LADDER_SECTION_MAX_CHARS`
  binds on real slates.
- `MARKET_RANKING_DEGRADED: question=... pool=N reason=shape_regression|unreadable
  detail=...`, which says WHICH failure produced a fail-open —
  `outcome=failopen` alone cannot, and `reason=shape_regression` is the one that
  means OUR contract broke.

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

The research section header is `## Prediction Market Snapshot`, and the prompts
import it from `prompts.py` as `MARKET_SNAPSHOT_SECTION_HEADER` to decide whether to
render their market-reading rules at all.

### Resolution-source fetcher — `RESOLUTION_SOURCE_ENABLED`

Fetches the exact URL(s) a question cites as its grading source
(`research/resolution_source.py`), so forecasters read the ground truth the
question resolves against. The page fetch is Tier-1: plain HTTP with browser-like
headers, no LLM calls, no retries. One narrow Tier-2 hop sits beside it (the
embedded Datawrapper dataset, below). When the direct fetch cannot read a page an
escalation ladder runs (`_escalate_unresolved`), each rung self-bounded against the same
provider wall and each returning a result that went through the same classification path,
so a rescued page is indistinguishable downstream from a directly-fetched one; the `route`
on every result says which rung produced it. The four rungs added on 2026-09-03, and the
one paid rung among them, are described under "The escalation ladder" below.

It deterministically extracts URLs from resolution criteria + fine print (markdown
links and bare URLs, order-preserving dedup, Metaculus markdown-escapes undone),
skip-filters URLs that add nothing or belong to another provider (Metaculus
self-refs, FRED series owned by financial-data, Yahoo `/quote/` pages owned by
yfinance), and caps at `RESOLUTION_SOURCE_MAX_URLS` *after* the skip filter so
a run of leading self-refs doesn't starve the real sources. Fetches run in
parallel with one-request-per-host politeness (a `Semaphore(1)` per netloc, keyed
per redirect hop, and shared process-wide since 2026-09-03 in the loop-scoped
`http_fetch.host_semaphores` map, handed out by `http_fetch.semaphore_for_host` —
with a map per provider
call, six questions citing one host each held their own semaphore and hit it six times
at once). Content is extracted with trafilatura (HTML), or read raw (JSON / text /
CSV). A PDF is read locally with pypdf and rendered as a passage digest (below);
anything else is left unread as `unsupported_type`.

**Which extraction publishes** is a policy, not a flag (`_extract_page_text`,
calibrated 2026-09-03 on 118 re-fetched bodies with five extractor variants run on identical
bytes; receipt `scratch/fetch_ladder_2026-09-03/chrome_calibration.md`). Trafilatura's default
recall is the primary extraction. Its text is scored by line shape: `content_share` is the
share of extracted characters that sit in table rows (lines starting with `|`) or in lines of
at least `RESOLUTION_SOURCE_CONTENT_LINE_MIN_CHARS` (60), and a text at or above
`RESOLUTION_SOURCE_CONTENT_SHARE_MIN` (0.38) that also clears the 400-char chrome floor
publishes. A text that clears the floor on short lines alone is chrome, and the same input is
re-extracted with `favor_precision=True`, which is the one trafilatura setting that prunes
navigation out of the backup tree its readability fallback swaps in; that text publishes only
if it clears the floor and the same metric. Otherwise the page is withheld as
`no_resolving_content` with reason `thin_page`, so the rendered rung still fires. The two
cases that fixed the policy: congress.gov, where the default extraction replaces the 2,411-char
bill-status card ("Latest Action", "Passed House") with 54,393 chars of a member-name
dropdown and precision restores the card; and uk.finance.yahoo.com, whose 1,191-char direct
body is a menu plus one quote line while its render is the full 23,991-char price table that
a floor-only check never reached because the menu counted as success. On the labelled
corpus the policy publishes every content text (46 of 46), publishes 2 chrome texts against
11 for default-only and 4 for precision-only, and withholds no content. The margin is about
0.05 on each side: navigation-tree chrome tops out at 0.329 (kasa.go.kr's homepage, a menu
with a news ticker) and the thinnest labelled content is 0.431 (a wastewaterscan dashboard
of 79-char readings). What it gives up, deliberately: prose-shaped boilerplate (AP's
cookie-consent wall, clinicaltrials.gov's glossary) is sentences and passes any line-shape
metric, and kasa's ticker line is withheld with its menu. Precision alone shipped until
2026-09-03 and withheld readable pages (kasa.go.kr pruned to 78 chars, two tracxn funding
tables, manifold's market body); default alone shipped for one day on a character-count
measurement, which under a head-preserving 6,000-char cap is the wrong metric, and the
earlier claim that its biggest gainers had been read by hand and were all content was
wrong. Both decisions ride `details["counts"]` as `chrome_metric_withholds` and
`precision_fallback_rescues`; no status or reason token changed.

Two free rungs sit under the HTML path, both reached only when the page carried nothing
readable. A **meta-refresh hop** follows the redirect no HTTP status announces:
cdc.gov's surveillance URLs answer 200 with a ~300-byte stub whose only content is
`<meta http-equiv="refresh" content="0; url=...">`, which the manual redirect loop
cannot see (no 3xx, no `Location`), so the stub used to be classified a JS wall and the
resolving page never fetched. The target is returned as the next hop, so it re-enters
the same classification path and consumes one of the same `MAX_REDIRECTS` slots, and it
passes exactly the same three checks a `Location` header does, via the shared
`_vetted_hop_target` helper. An **ARIA-table rewrite** (`rewrite_aria_tables`,
`research/http_fetch.py`) runs
before every extraction: cdc.gov builds its outbreak stat blocks out of
`<div role="table">` / `role="row"` / `role="cell"`, which is valid accessible markup
and invisible to trafilatura's table handling — the cyclosporiasis block rendered as a
bare "17,180 / 2" with no labels and no hospitalization count at all, because 922 sat in
an unwrapped cell. Rewritten to real table tags, the same page extracts
`| Hospitalizations | 922 |`. A page with no ARIA role is handed to trafilatura as the
original bytes, so its extraction is unchanged.

A **cited PDF is now read** rather than dropped (`_resolution_pdf_outcome` is the
branch): `research/document_text.py` extracts the
text with pypdf and selects the passages most relevant to the question's title plus its
resolution criteria (BM25, deterministic, no model call), rendering a digest that states
how many pages were read and labels each passage with its page as `[p.N]`. The ranking
query is title plus resolution criteria and deliberately not fine print, which is mostly
procedural boilerplate about ambiguity and annulment and would dilute the term set.
Measured 2026-09-03,
that path pulled 833,450 chars out of a 6.7 MB 220-page document in 5.3 s with the wanted
passage in it, while the paid alternative returned nothing for the same file. A body the
server did not declare as a PDF is still sniffed by its `%PDF-` magic, since several
government hosts serve documents as `application/octet-stream` — a declared document gets
the larger `DOCUMENT_TEXT_PDF_MAX_BYTES` cap (the receipt file is over the 5 MiB response
cap), an undeclared one keeps the smaller one. Bytes we read and could not turn into text
get their own status, `unreadable_document`, with `status_reason` naming which of
`no_text_layer` / `encrypted` / `malformed` applies; only the first could ever be rescued
by a paid document read, which is why it is not folded into `unsupported_type`. A document
we read in full whose passage selection matches no query term is WITHHELD as
`no_resolving_content` / `no_matching_passage` rather than published: its block is the
header, the outline and one sentence saying nothing matched, and published as `success`
that was prose standing in for an absent section, indistinguishable in the run log from a
document that handed the forecasters the resolving paragraph. It is also the one
`no_resolving_content` the paid url_context rung is not allowed to re-read, since we
already hold the document's text.

**Tier 2: the embedded Datawrapper dataset.** The first (and so far only) Tier-2 hop
shipped 2026-08-25 (`5f27c46`, receipt qids 44858/44841) and is narrow by design.
When a fetched page's RAW HTML embeds a Datawrapper chart, the chart also serves its
live "Get the data" CSV, and poll trackers lock their resolving daily series inside
exactly those iframes — which trafilatura drops at every setting. The hop uses ONLY
the version-free `static.dwcdn.net/data/<chart_id>.csv` route, because the page HTML
pins a stale chart version whose `datawrapper.dwcdn.net/<id>/<version>/dataset.csv`
form keeps serving 5-14-month-old snapshots as HTTP 200 (the naive fix the 2026-08-24
verifications refuted). A `Last-Modified` freshness guard then withholds anything
outside the window under the `stale_data` status rather than serving stale data as live
(the Wayback rung below uses the same status for an over-age archived capture of a cited
page; the two are told apart by `chart_id`): older than
`RESOLUTION_SOURCE_DATAWRAPPER_MAX_AGE_DAYS`, undatable, or
implausibly far in the FUTURE — a future date past a six-hour clock-skew tolerance
means a broken clock, not maximal freshness. The hop is bounded by
`RESOLUTION_SOURCE_DATAWRAPPER_MAX_CHARTS`,
`RESOLUTION_SOURCE_DATAWRAPPER_PER_DATASET_MAX_CHARS`,
`RESOLUTION_SOURCE_DATAWRAPPER_HOP_WALL_MARGIN_S` and
`RESOLUTION_SOURCE_DATAWRAPPER_MIN_HOP_BUDGET_S`. A served dataset leads with a
`Dataset published <ts>` liveness stamp.

**The escalation ladder.** Four more rungs shipped on 2026-09-03, tried cheapest first behind
the free hops above. Each declines by returning nothing, in which case the direct route's own
status stands.

`route=derived_api` (`research/derived_api.py`) serves the JSON feed a page loads its own
figures from. A JavaScript dashboard's numbers arrive over XHR after the DOM is ready and sit
in the served HTML at no wait condition, so the endpoint is found by the browser rung recording
the page's own requests, then remembered per HOST for the rest of the run: a second cited URL on
that host costs one GET (floor `RESOLUTION_SOURCE_DERIVED_API_MIN_BUDGET_S`, the one-request
floor) instead of a second browser launch. Because a host's feed is usually parameterised, the
lead on each served block names the endpoint and says whether it was discovered on that page or
on another page of the same host, so a forecaster can check that the feed covers the quantity
asked about.

`route=rendered` re-reads the page out of a headless Chromium render
(`research/rendered_fetch.py`, the transport shared with gap-fill v2's fetch ladder and its
process-global two-launch cap). It triggers on `js_wall` and on the `thin_page` shape of
`no_resolving_content`, both pages that answered 200 with nothing readable, and deliberately not
on `embed_shell`, since `page.content()` returns the main frame's HTML and an Infogram or
Flourish iframe comes back as a bare tag. Its floor,
`RESOLUTION_SOURCE_RENDER_MIN_BUDGET_S`, is far above the one-request rungs' because a launch
plus a DOM-ready navigation costs several seconds even on a page that renders cleanly, and the
launch slot is contended process-wide, so a question with no budget left would take a slot a
sibling question could still land a page with. The rendered DOM re-enters `_classify_html_body`,
so a rescued page gets the same chart read, ARIA rewrite, floors and disclosure leads as a
directly-fetched one, and can still be withheld. A transport that declines (Playwright missing
or broken, a host that will not pin to a public IP, or a browser error) is recorded as a SKIP
with the reason `renderer_unavailable` rather than as a fired rung, because nothing was rendered
and so nothing about the page changed. Two nearby cases are kept OUT of that reason so a memo
hit or a queue timeout cannot read as the Chromium install having failed: a URL an earlier
question already rendered to nothing this run is `rendered_no_text` (the memo doing its job),
and a render that ran out of budget queued behind the launch gates, which the transport signals
with `RenderBudgetExpired`, is `wall_budget` (the same reason the pre-gate floor check records).
Two more bounds hold the rung inside the wall, and a render cut off by either is recorded as a
SKIP with its own reason, `render_timeout`, rather than as the renderer being unavailable. Inside the
transport, `page.content()` is capped at `RENDER_DOM_READ_TIMEOUT_MS`: on a settled DOM it is a
sub-second round trip, and it runs long only when the page keeps navigating after the settle
(measured 2026-09-03 on ogimet.com, where the goto timed out at 33 s as designed and the
unbounded read then blocked for a further 40 s, so the render ran 76 s against the 45 s wall and
every page the question had already fetched was discarded). Around the transport, the rung holds
the whole `render_page` call to the remaining budget with `asyncio.wait_for`, so no Playwright
call can overrun the wall from inside it. A cut-off render says nothing about whether Chromium
works, so it does not trip the once-per-run "rung unavailable" warning, and the direct result is
what stands. The URL is memoised for the run (so a second question citing the same hostile page
does not pay for it again) only when the transport's own DOM-read bound fires before the rung's
outer cut. In the salvage shape (the goto ran its budget out) the outer cut lands first by the
launch time, because the launch runs after the transport recomputes its navigation budget and is
not reserved in it, so that URL is not memoised (FUTURE.md item 5 prices the residual).

`route=wayback` (`research/wayback.py`) serves an archived capture. The archive earns a rung
because it is the one free route whose EGRESS IS NOT OURS: measured 2026-09-03, the same client
with the same headers gets 403 from a GitHub Actions runner and 200 from a residential address
on bls.gov, cdc.gov and fsis.usda.gov alike. It triggers on `blocked` / `error` / `not_found`
and never on `js_wall`, because the archive stores the unrendered shell. Three bounds: the
`RESOLUTION_SOURCE_WAYBACK_MIN_BUDGET_S` floor, at most
`RESOLUTION_SOURCE_WAYBACK_MAX_ATTEMPTS` snapshots per QUESTION (every snapshot shares the one
`web.archive.org` host gate, so N cited URLs would otherwise queue into N sequential archive
fetches inside the provider wall), and the age bound
`RESOLUTION_SOURCE_WAYBACK_MAX_AGE_DAYS`, which matches the Datawrapper freshness bound and is
the same judgment rather than a measurement. A capture is admissible as primary grading evidence
only with its age stated, which is what the lead renders. Three outcomes, in this order, and the
order is the design. The wrapped inner URL is unwrapped and re-checked first, because
`is_metaculus_self_ref` keys on hostname and a `web.archive.org/web/.../metaculus.com/...` URL
sails past every self-reference filter in the pipeline, and a failed SSRF or self-ref re-check
refuses the rung outright. Then a capture the archive never served DECLINES, leaving the direct
status standing, because "no archived copy exists" is a different fact from a stale one and the
direct status says more about the source. Only a capture we did read and cannot date, or can date
and it is too old, is withheld as `stale_data`. A withhold does not end the ladder: the paid rung
below is still asked about the DIRECT outcome (a stale archive is still a page we could not read
fresh), and the withhold is what stands when that rung is off or declines. This rung is NOT
flag-gated, so from its merge a cited page's `status` can be a rung's verdict where it used to
be the direct outcome: `stale_data` where the direct fetch said `blocked` / `error` /
`not_found`, and (flag on) `ungrounded`, or `no_resolving_content` with reason `not_addressed`,
where it said `blocked` / `js_wall` / `error` / `no_resolving_content`. An era-bucketed
`blocked` or `error` rate read off `status` alone will
show a drop at that merge that is a bookkeeping change, not hosts refusing us less; take the
direct outcome from `from_status` on the `RESOLUTION_SOURCE_ESCALATION` line, or partition
`status` by `route`, where `direct` rows are unchanged.

`route=url_context` is the LAST rung and the only paid one: Gemini reads the page for us
(`research/url_context_reader.py`, the reader shared with gap-fill v2's `read_document`),
reaching hosts our own client cannot because Gemini dials from Google's address. It is OFF by
default behind `RESOLUTION_SOURCE_URL_CONTEXT_ENABLED` and set in no workflow yaml, so it fires
nowhere in production today; `docs/operations.md` covers what turning it on costs and on whose
key. Every gate is checked in increasing cost order before a cent is spent: the trigger statuses
(`blocked`, `js_wall`, `error`, `no_resolving_content`, tested against the DIRECT outcome, so a
withheld Wayback capture on the way down does not close the rung), the flag, the API key, the
`RESOLUTION_SOURCE_URL_CONTEXT_MIN_BUDGET_S` floor, then the per-host `Google-Extended` robots
pre-check (`research/robots_policy.py`, whose cache and `ROBOTS_FETCH_TIMEOUT_S` bound are shared
with v2's reader; worth a request of its own because a host disallowing that token refuses
Gemini's fetch server-side), and the budget floor a SECOND time. The pre-check is the one gate
that costs a request, and the paid read runs in a thread that `asyncio.wait_for` cannot cancel, so
the client-side ceiling sized off the remaining budget is the only bound that can stop it; re-reading
the budget after the pre-check is what keeps that ceiling honest, and a pre-check that ate the room
records a `wall_budget` skip rather than a paid call nothing reads. It has its
own retry count, `RESOLUTION_SOURCE_URL_CONTEXT_ATTEMPTS`, deliberately lower than the v2
reader's, because a retry inside a wall shared with every other cited URL spends the budget the
pages already fetched need in order to render. A per-QUESTION paid-read cap,
`RESOLUTION_SOURCE_URL_CONTEXT_MAX_ATTEMPTS`, bounds how many reads a single question can pay
for across its cited URLs — the analogue of the Wayback per-question cap and a distinct quantity
from the SDK retry count — claimed last, only for a read that has cleared every cheaper gate, and
a read the cap declines records a `url_context_cap` skip. Zero successful retrievals
DISCARDS the text under the new terminal status `ungrounded`, the same floor `gemini_search` and
v2's `read_document` apply: Gemini answers fluently out of parametric memory when every
retrieval failed, and a fluent unsourced answer under the primary-grading-evidence caption is
the Q38195 failure with a forecaster-facing blast radius. A read whose answer opens with the
prompt's `NOT_ADDRESSED` sentinel (`research/url_context_reader.py`, the model's designed reply
when the retrieved page does not discuss the ask) is withheld as `no_resolving_content` /
`not_addressed` rather than rendered: the page WAS retrieved, so it is not `ungrounded`, but under
the url_context lead the non-answer was prose standing in for an absent section, the shape the
PDF digest closes with `no_matching_passage`. The read stays on the record as the rung's own
verdict, since it was paid for. A read that lands leads with a
mandatory disclosure saying why the route was taken and that the text is a model's reading
rather than a copy of the page. Its spend is visible as a third `GEMINI_USAGE` role,
`resolution_source`.

Every non-direct route present in the sections that will RENDER also contributes one
forecaster-facing caveat sentence, from `ROUTE_CAVEATS` in
`research/resolution_fetch_result.py`: where the bytes came from, and what the reader must not
conclude from having them. The mapping is keyed by route and iterated, so it is both the
vocabulary check and the render order, cheapest and most transparent first and model-mediated
last. `direct` is deliberately ABSENT from it rather than mapped to an empty string, which is
what keeps an all-direct question's section byte-identical to what it rendered before the ladder
existed (the overwhelming majority of questions, pinned by a test). One sentence belongs to a
rung that has not shipped: `impersonate` is in the route vocabulary and carries a caveat, but
nothing produces that route, and a completeness test asserts every non-direct token has a
sentence so a future rung cannot render rescued content with no disclosure at all.

Every rung is self-bounding on the Datawrapper hop's pattern — wall minus elapsed
minus `RESOLUTION_SOURCE_RUNG_WALL_MARGIN_S`, skipped below its own floor — because
the provider's outer `asyncio.wait_for` discards
every page that already fetched when it fires, so an overrunning rung costs the whole
question's resolution evidence rather than just its own attempt. A rung that FIRED
records itself on the result (`route=` on the fetch marker, plus one
`RESOLUTION_SOURCE_ESCALATION` line); a rung that was SKIPPED is counted under
`details["counts"]` instead of logged, alongside the fired counts. A zero
renders nothing while still surviving into the archive, which is what makes "the rung existed
and never fired" distinguishable from "this record predates the rung". Six of the keys count
rungs that FIRED: `meta_refresh_hops`, `pdf_documents_read`, `rendered_attempts`,
`derived_api_reads`, `wayback_attempts` and `url_context_reads`. Two count the extractor
policy's decisions rather than rungs, per final result: `chrome_metric_withholds` is an
extraction the line-shape metric withheld because it cleared the chrome floor on navigation
alone, including a chart-rescued page whose chart block still published without that text (on
a page with no chart block its `reason` is the same `thin_page` an under-floor page carries, so
this count is what separates the two), and `precision_fallback_rescues` is a page published from the
`favor_precision` re-extraction after the default one failed that metric. The rest count rungs that
were SKIPPED, one key per skip reason rather than everything folded into `rung_budget_skips`,
because each names a different binding constraint. `rung_budget_skips` is the question that ran
out of wall, summed over every rung; the same skips are broken out per rung as
`meta_refresh_budget_skips`, `pdf_local_budget_skips`, `derived_api_budget_skips`,
`rendered_budget_skips`, `wayback_budget_skips` and `url_context_budget_skips`, because the
aggregate cannot say WHICH rung the wall is binding on and "how often is the paid rung starved
by the pages before it" is the question the flag's rollout asks.
`pdf_contention_skips` is a document left unread while two others were parsing, so the two-slot
parse gate is what binds. `renderer_unavailable_skips` is a browser rung that never rendered,
most often because Chromium is missing on the runner (the install step is `continue-on-error` in
every workflow, so its absence is by design), and it is invisible in `rendered_attempts`; it no
longer includes a URL an earlier question rendered to nothing, which is its own
`rendered_no_text_skips` so a memo hit cannot inflate the install-failed signal.
`render_timeout_skips` is a browser rung that launched and was cut off, by the transport's
DOM-read cap or by the question's remaining wall budget: a page that keeps navigating, which is a
fact about the page rather than about the runner or the question's clock, and also invisible in
`rendered_attempts`.
`wayback_cap_skips` is a question that spent its snapshot attempts on earlier cited URLs, so the
per-question cap is what binds; `url_context_cap_skips` is the paid rung's analogue, a question
that spent its per-question paid-read budget (`RESOLUTION_SOURCE_URL_CONTEXT_MAX_ATTEMPTS`) on
earlier cited URLs, so the spend cap binds rather than the wall or the flag. `fast_path_skips`
is an expensive rung (the render, the paid read)
declined because the QUESTION's close-derived budget put it on the time-budget fast path, a fact
about the question's window rather than about the provider's own 45 s wall, which is what
`rung_budget_skips` counts. `url_context_robots_skips` is the free `Google-Extended`
pre-check earning its request: the host would have refused the read server-side, so that is
spend avoided rather than a page lost and it must not read as a failure. `url_context_no_api_key_skips`
is the paid rung enabled with no `GOOGLE_API_KEY` set: a misconfiguration rather than a tuning
signal, and counted precisely because without it "flag on, key missing" is byte-identical in the
archive to "flag off".

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
they are what the escalation rungs above trigger on, as is the `no_text_layer`
half of `unreadable_document` (a scan, where a model really is the only route). The one
exception inside that family is `no_resolving_content`'s `no_matching_passage`: we read the
whole document, so there is no harder fetch to try and the paid rung skips it. `empty_body`
(a 200 whose body is empty or whitespace-only) and `unsupported_type` (including a body whose
declared charset decodes to mojibake) are bodies that carried no information — refusals
rather than seams, because there is nothing on the other side to fetch harder. Both
exist because `status="success"` has to mean CONTENT: as `success`, an empty body
rendered an empty section under the "primary grading evidence" caveat, suppressed the
all-failed "yielded no usable content" notice for every sibling URL, and reported `ok`
to provider diagnostics.

`ungrounded` is the twelfth and newest status, and the only one that cost money: the paid
url_context rung answered with zero successful retrievals, so what came back is recall rather
than a read of the page and it is discarded rather than rendered. It has its own token because
it says something no other status does, that the host answered a third-party fetcher's request
with nothing while refusing ours. On the reason side there are now two vocabularies rather than
one, split by what each qualifies. `FetchStatusReason` qualifies a result's STATUS —
`embed_shell` / `thin_page` / `no_matching_passage` / `not_addressed` under
`no_resolving_content`, `no_text_layer`
/ `encrypted` / `malformed` under `unreadable_document`, and `budget_skipped` / `parse_contention`
under the `unsupported_type` a held-but-unparsed document earns. `RungSkipReason` qualifies a
rung ATTEMPT that never ran (`RungAttempt.skipped_reason`), which produced no result and so has
nowhere else to record its reason: `wall_budget`, `wayback_cap`, `fast_path`, `no_api_key`,
`robots_disallowed`, `rendered_no_text`, `renderer_unavailable`, `render_timeout`, and
`parse_contention` again (a held document declined for want of a parse slot records the skip AND
stamps the withheld result's `status_reason`, the one token shared by both Literals). Both are a
closed `Literal`, so a misspelt reason is a type error rather than a permanently-zero count.
`renderer_unavailable` is the browser declining before it rendered anything (missing, broken,
unpinnable host); `render_timeout` is a render that launched and was cut off because the page
kept navigating, and the two are kept apart because only the first says anything about Chromium.
`no_resolving_content` has four reasons (`embed_shell`, `thin_page`, `no_matching_passage` and
`not_addressed`): the third is the only one that is a document rather than a page, and the fourth
is the paid reader's, a page Gemini retrieved whose answer said it does not discuss the ask. All
of them live in `research/resolution_fetch_result.py` with the rest of the vocabulary.

`vacuous_body_status` (`research/resolution_fetch_result.py`) is the one place that
decision is made, on every raw-body branch — Tier-1 JSON/text/CSV and the Tier-2
dataset alike. Three ways a 200 carries nothing. It could not be DECODED: the body is
decoded BOM-first, then by its declared charset, and an undecodable-character ratio
above `MAX_UNDECODABLE_CHAR_RATIO` is refused as `unsupported_type`, because mojibake
like `0�.�4�2�` type-checks as text and rendered as grading evidence. It is empty or
whitespace-only, which is `empty_body`. Or — datasets only — it is not row-shaped, so
nothing may claim it is the chart's live series. That third check is deliberately
ordered BEFORE the freshness verdict, so an empty CDN body cannot borrow
`stale_data`'s benign diagnostics token (a DATASET's `stale_data`, the one carrying a
`chart_id`, reports to diagnostics as the benign "guard working as designed", which would
hide a broken hop; a cited page's `stale_data` from the Wayback rung is a lost source and
keeps its loss token). Row shape is also
decided on the PRE-strip text, because `looks_like_csv_rows` rejects markup by its
leading `<` and stripping first would remove exactly the allow-listed fragment tags
(`<p>`, `<div>`) a CDN soft-404 opens with, letting an error page carry the
authoritative `Dataset published` lead if its prose holds a comma. Those allow-listed
HTML tags ARE stripped from raw CSV/text bodies before truncation, which is worth 58
rows versus 13 at the same character budget on a live-shaped poll table.

That notice says "yielded no usable content" rather than "was
unreachable" because two of the statuses it covers — `no_resolving_content` and
`empty_body` — are pages that answered HTTP 200 and carried nothing, and "the tracker
was down" is different evidence from "the tracker has no reading"; the per-domain status
token beside it says which happened.

`no_resolving_content` (2026-09-01) is the newest of those seams and covers the page
that answers 200
with nothing but chrome. The floor is what decides it: below
`RESOLUTION_SOURCE_EMBED_SHELL_MAX_CHARS` — 400 characters — of extracted text the
page is withheld under
this status, which costs nothing because everything archived below that floor is site
chrome and the shortest archived extraction that carries the resolving content is 401
chars. That calibration was re-checked against the same census when the gate was
generalised: of 68 cited successes, all 8 below 400 chars are chrome, and the
per-URL list is in the constant's own comment. Above the floor the text still has to be
content-shaped (the extractor policy above: table rows and long lines, not a menu tree),
and a page that is chrome at both extractor settings takes the same `thin_page` withhold.
A page that passes is rendered as is, and where a
third-party data embed hid figures from it one bracketed line says plainly that those
figures are not in the text.

`FetchResult.status_reason` records which shape of chrome it was. `embed_shell` means the RAW HTML
named an embed whose numbers are real but locked inside it — Infogram, Flourish or
Tableau, detected by `unreadable_data_embed_providers` because trafilatura emits no
iframe or embed-script URLs at any setting; Datawrapper is deliberately excluded from
that scan since the Tier-2 hop reaches it. `thin_page` means no such provider was named.
The status's third reason is not a shape of chrome at all: `no_matching_passage` is a cited
document we read in full that discusses nothing the question asks about, withheld under the
same status because the outcome for a forecaster is the same, a section with nothing in it to
grade against. Its fourth, `not_addressed`, is the paid url_context rung's equivalent: a page
Gemini retrieved whose answer opened with the prompt's `NOT_ADDRESSED` sentinel, withheld for
the same reason.
That distinction used to be a GATE rather than a label, and the gate was removed on
2026-09-02 because it was wrong: the
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

One more rung (2026-09-02) reads data out of the page we already hold, with no second
request and no
LLM call. `resolution_chart_data.render_inline_chart_data` scans the raw HTML for a
Highcharts config — a `data-chart="{…}"` attribute, or a `Highcharts.chart(…)` call whose
argument is strict JSON — and, after an HTML-entity unescape plus a plain `json.loads`,
renders each series' most recent points as a compact
labelled block that leads the page text. Nothing is summed, interpolated,
unit-converted or re-derived: the block states the values the page's own chart holds; a
declared `datetime` x axis renders as UTC dates rather than epoch millis; and a config
that
does not parse is skipped at DEBUG. It runs on every fetched HTML page rather than only
thin ones, because the record it exists for is q43949, whose resolving IOM page extracted
roughly 80k chars of incident rows and prose carrying none of the resolving figures while
its annual series sat in the attribute — reading 1,240 for 2026 in a Wayback snapshot 25
days
before a forecast that landed about 340 too high. A thin-only gate would have missed the
record the rung exists for. Because chart data counts as content,
it also rescues a page the chrome floor would otherwise withhold.

Every fetched URL — Tier-1 page and Tier-2 dataset hop alike — emits one
`RESOLUTION_SOURCE_FETCH: question=... url=... status=... http=... embeds=...
[reason=...] [route=...]` line, harvested as `resolution_source_fetch`. `reason` is
appended only where the status alone is ambiguous, so archived lines stay
byte-identical and its absence keeps meaning "no reason applies"; `route` names which
rung of the escalation ladder produced the outcome. It REPLACED the older free-text
`resolution_source fetched <netloc> (<status>)` lines rather than joining them, so each
fetch is logged exactly once, and it is what turns a cut like "cdc.gov is 0 successes
in 1,069 fetch records" into a query rather than a re-scrape of run logs
that expire from GHA at 90 days. Because that line carries only the FINAL outcome per
URL, each escalated rung additionally emits `RESOLUTION_SOURCE_ESCALATION` with the
status that triggered it, the rung tried, what came back, and the wall-clock the rung
cost — which is what makes "does this rung rescue anything, and is it worth its latency"
answerable. The two lines spell one state two ways: a fetch that worked is
`status=ok` on the fetch line (the shared `fetch_outcome_token`, whose `ok` is what the
diagnostics formatter reads as "this source contributed") and `outcome=success` on the
escalation line (the verbatim `FetchStatus`), so a query joining the two has to treat
`ok` and `success` as the same outcome. Both are data contracts, so the difference is
documented here rather than re-spelled on either side. See "Reading run logs" in
`docs/operations.md` for the field meanings.

The paid rung adds three greppable log lines that are deliberately NOT registered as marker
specs: `RESOLUTION_SOURCE_URLCONTEXT_ROBOTS_SKIP: url=... host=...` (an INFO, the free pre-check
avoiding a known-zero paid read), `RESOLUTION_SOURCE_URLCONTEXT_UNGROUNDED_SUPPRESSED: url=...
statuses=...` (a WARN, a paid read discarded for retrieving nothing; `statuses` is every reported
`url_retrieval_status`, `none` when the SDK attached no entry) and
`RESOLUTION_SOURCE_URLCONTEXT_NOT_ADDRESSED: url=... host=...` (a WARN, a paid read withheld
because its answer opened with the `NOT_ADDRESSED` sentinel, so the page was retrieved and has
nothing on the ask). With the flag off in every workflow none of them can fire in production, so
a spec would only add an always-empty archive column; all three are FUTURE marker-spec
candidates, to be registered if the flag is ever turned on, which is when their rates start
meaning something. Their spellings are pinned by tests so the eventual spec matches the lines,
and the first two are parallel to their `AGENTIC_URLCONTEXT_ROBOTS_SKIP` /
`AGENTIC_DOCUMENT_UNGROUNDED_SUPPRESSED` twins on the gap-fill v2 reader, which already carry
specs and are the pattern to follow.

Like prediction markets, it is **hard-disabled under benchmarking** (current page
content post-dates any backtest window), on the same leakage rationale. The section
header is `## Resolution Source Snapshot`, and `RESOLUTION_SOURCE_ENABLED` was flipped
on in the three prod yamls on 2026-07-10 after a live-output eyeball.

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
shorter than `GAP_FILL_MIN_RESEARCH_CHARS`, or when the question's close-derived time
budget drops it — the fast path, or a research phase that ran out of budget):

1. A non-grounded OpenRouter analyzer LLM (`GAP_FILL_ANALYZER_MODEL`, low effort)
   reads the
   first-pass research and emits a JSON list of up to `GAP_FILL_MAX_GAPS` factual
   gaps, ranked by decision-relevance, so the trailing slot holds the least
   forecast-moving gap.
2. Each gap is resolved by a parallel OpenAI native web search
   (`GAP_FILL_RESOLVER_MODEL` at `GAP_FILL_RESOLVER_REASONING_EFFORT`, via
   OpenRouter on the donated key).
   Because the searches run in parallel, latency is the slowest call, not the sum.

The resolver migrated off direct-Google grounding on 2026-06-25, which is why
`GOOGLE_API_KEY` is no longer required for gap-fill, and its model went sol → terra on
2026-07-20: terra was preferred-or-within-noise in all three 2026-07 blind role audits
at ~40-50% lower cost, which matters here because these searches are the single biggest
research line item at ~44% of spend. The whole pass never raises —
it returns `""` on any error — and appends its results under
`## Targeted Gap-Fill (second pass)`.

### v2 — agentic gap-fill (`research/agentic_gap_fill.py` `run_gap_fill_v2`)

A bounded agentic tool loop, living in `metaculus_bot/research/agentic/` behind the
seam `research/agentic_gap_fill.py` `run_gap_fill_v2`, run by a driver LLM
(`GAP_FILL_V2_DRIVER_MODEL` at
`GAP_FILL_V2_DRIVER_EFFORT`, both picked by the 2026-07-17 blind driver eval —
`scratch/driver_replay_2026-07-17/blind_judge_report.md`), gated by
`GAP_FILL_V2_ENABLED`. It has been on in every bot workflow since 2026-07-21T17:07Z
(`b4e9df0`), with v1 left on alongside for an overlap window.

The driver is briefed with the forecaster prompt template, privately
dry-runs a forecast to find fill/verify targets, then iterates over
four tools (`research/agentic/tools.py`): `search_news` (AskNews, through the same
rate gate as the primary provider), `search_web` (Exa direct), `fetch` (an
auto-escalating ladder: plain → local PDF extraction → headless Chromium →
`read_document`), and `read_document` (acquisition-first — the free rungs, then
`GAP_FILL_V2_READER_MODEL` via Gemini url_context). Two of those rungs are transports shared
with the Tier-1 resolution-source ladder rather than copies of it: the Chromium render
(`research/rendered_fetch.py`, over which `tools._try_rendered_fetch` is now a thin mapping onto
this ladder's own result type) and the url_context read
(`research/url_context_reader.py`, with the `Google-Extended` pre-check in
`research/robots_policy.py`, moved out of `research/agentic/` when the second caller arrived).
It runs under a wall deadline
(`GAP_FILL_V2_WALL_DEADLINE`) and a tool-call budget
(`GAP_FILL_V2_MAX_TOOL_CALLS`), producing anytime output, and soft-fails to `""` at
every boundary. It appends a detached
citation-only findings artifact under `## Agentic Research Findings`, leading with a
`### ⚠ Corrections to the briefing` priority block; a ghost forecast is logged for
telemetry only (the `GHOST_FORECAST` marker) and never published. Like the other
leakage-sensitive providers,
it is benchmarking-guarded off (`is_benchmarking=True` returns `""`). See
`docs/agentic_gap_fill.md` for the full
tool loop, escalation ladder, telemetry, and design rationale.

## Diagnostics and persistence

`run_research` returns forecaster-clean text. The **provider-diagnostics block**
(which provider succeeded, char counts, latency per provider) is computed
separately (`format_provider_diagnostics_block`) and deliberately kept out of the
returned research — forecasters and the v2 driver must never see it. It reaches
three places instead: an INFO log line, the research archive (as its own field),
and the published Metaculus comment (stashed per question id, popped by the
forecaster at comment-build time via `pop_provider_diagnostics`).

A provider's `details` dict carries two conventions, and they answer different
questions. `details["sources"]` is the per-source outcome map, rendered into the
`lost=` suffix. `details["counts"]` (`provider_diagnostics._counts_suffix`) is the
second: an ordered `{name: number}` map of provider-INTERNAL quantities that are
neither a source outcome nor a failure — Gemini's `tier_tags` /
`unsupported_attributions`, financial-data's `fx_identifiers_empty`, and the
resolution-source rung counts. **A zero renders nothing**, so every healthy provider's
`## Provider Diagnostics` line stays byte-identical to what it was before the map
existed, while `asdict` keeps the zero in the schema-v2 archive — which is exactly what
makes "the check ran and found none" distinguishable from "the check never ran".

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

All five bot workflows
(`.github/workflows/run_bot_on_{tournament,metaculus_cup,minibench}.yaml`,
`test_bot.yaml` and `test_bot_basic.yaml`) enable the full research stack:

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
financial) + prediction-market snapshot + Tier-1 resolution-source fetcher +
time-series
anchor + both gap-fill passes. Env flags, models, and timeouts live in
`metaculus_bot/constants.py`; provider models route through the shared
donated-then-personal OpenRouter fallback (`fallback_openrouter.py`), except
Gemini grounded search, which uses the personal Google key directly.

All of that is subject to the question's close-derived time budget: a question on the
fast path runs the primary plus the cheap hard-capped providers only, with the slow
optional search providers dropped and BOTH gap-fill passes skipped. The resolution-source
fetcher stays in but learns about the fast path too (`resolution_source_provider(...,
fast_path=)`), and its two EXPENSIVE escalation rungs, the Chromium render and the paid
`url_context` read, decline on it before any side effect, each recording a `fast_path` skip
(`counts["fast_path_skips"]`); the cheap rungs run as they do off it. See the pipeline's
time-budget step for how the budget is granted and what it cuts.

## Cost note

The research providers hit live, paid APIs (AskNews, Exa, Perplexity, OpenRouter
credits, Google grounding, FRED). Running the bot or a backtest spends real money
and, in live modes, publishes to Metaculus. Do not launch a paid run without the
operator's approval — see `AGENTS.md` "Cost discipline". The unit/integration test
suite is self-contained and hits no paid APIs.
