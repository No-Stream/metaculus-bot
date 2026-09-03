# Resolution-source fetch ladder: implementation plan

Written 2026-09-03 for the operator's sign-off. Supersedes the routing half of
`fetch_escalation_ladder_design.md` (2026-09-01, design only, nothing implemented); that
document's SSRF posture, budget pattern, telemetry plan and open questions still bind and are
cited rather than restated. Nothing here is implemented yet.

A reader who has never seen this repository should be able to follow this end to end. Every
number comes from a file or a command named beside it. Where a sentence rests on inference
rather than something read or run, it says so.

## What this is about

The bot forecasts Metaculus questions. When a question names a URL as its resolution source
("resolves according to the CDC's cyclosporiasis surveillance page"), the bot fetches that page
and puts its text in front of every forecaster model under a caveat calling it primary grading
evidence (`metaculus_bot/research/resolution_source.py`). A second, separate path — the
gap-fill v2 agentic research loop — fetches pages a driver model chooses
(`metaculus_bot/research/agentic/tools.py`).

Between 2026-07-26 and 2026-08-28 the first path failed on 40 of 96 cited URLs, and on 23 of
the 71 questions citing any URL, every cited URL failed
(`scratch/next_season_bundle_2026-09/item13/archive_fetch_status_tally.txt`).

## Summary, and the one finding that reshapes the design

**The dominant cause of those failures is not fetch technique. It is where the request comes
from.** Measured 2026-09-03 with the bot's own aiohttp session, its own `BROWSER_HEADERS`, and
its own resolver:

| environment | bls.gov/wsp/ | cdc.gov surveillance | fsis.usda.gov |
|---|---|---|---|
| GitHub Actions runner (archived runs, 2026-07/08) | 403 x4 | 403 x3 | 403 x3 |
| operator laptop, residential | 200 | 200 | 200 |
| operator EC2 box (Amazon datacenter, 205.251.233.179) | 200 | 200 (234-byte stub) | 200 |

Identical client, identical headers, opposite outcome. So the discriminator is the GitHub
Actions egress range specifically, not TLS fingerprinting and not "datacenter IP" in general.
A free replay of 47 archived failure URLs from the laptop found the same thing at scale: 25 of
47 do not reproduce at all, including 16 of the 24 recorded as `blocked`
(`scratch/fetch_ladder_2026-09-03/replay_report.md`).

Two consequences drive this plan.

1. **The TLS-impersonation rung is UNPROVEN and must not be built on today's evidence.** Every
   environment where impersonation succeeded also succeeds without it. We have no measurement
   from the only environment that fails. A free diagnostic (step 0) settles it before any code.
2. **Rungs whose egress is not ours are structurally more valuable than they looked.** The
   Wayback Machine and Gemini's url_context fetcher both reach a host from an address the host
   does not associate with GitHub. That is why url_context read bls.gov and sagaftra.org PDFs in
   production on 2026-09-03 while our own fetch got 403 on both.

Everything else in the plan is a set of free, deterministic wins that stand on their own
evidence and are independent of the egress question.

## Evidence table

Verified 2026-09-03 unless noted. "Replay" = `scratch/fetch_ladder_2026-09-03/replay_report.md`,
47 URLs probed from the laptop.

| claim | evidence |
|---|---|
| GHA-origin 403s do not reproduce from laptop or EC2 with the bot's own client | table above; replay 25/47 |
| TLS impersonation adds nothing measurable where we can measure | replay rung A vs rung B; EC2 plain 200 and impersonate 200 on all three hosts |
| sagaftra.org refuses every free route | DataDome 403 on plain, impersonated, and headless Chromium; `web.archive.org` returns 403 for it (Wayback exclusion), so no archived copy |
| trueup.io refuses every free route | Cloudflare managed challenge; Chromium never clears it; only Wayback snapshot is 517 days old and 363 chars |
| url_context reaches hosts we cannot | prod run 33775800806 read bls.gov + sagaftra.org PDFs that our fetch 403'd |
| url_context is refused by robots.txt `Google-Extended` | both internationalaisafetyreport.org URLs returned zero retrievals; that site's Cloudflare-managed robots.txt disallows `Google-Extended` and sets `Content-Signal: ai-train=no,use=reference`. Google's docs say the token governs Gemini training and grounding; they do not name url_context, so this is strongly consistent, NOT documented |
| Manifold refused by url_context for a different reason | its robots.txt is `Allow: /`; plain fetch returns a 2k-char JS shell |
| local PDF text extraction works, no OCR case found | pypdf: 833,450 chars from the 6.7 MB 220-page IAISR report in 5.3 s, exact wanted passage present; 4,689 chars from the BLS release; every PDF in the replay had a real text layer |
| cdc.gov meta-refresh stub is real and reachable off-GHA | 234 bytes plain / 340 impersonated from EC2 |
| Chromium's `networkidle` wait loses content | 4 of the replay's 10 render rescues came from pages where `page.goto` raised TimeoutError and the DOM was complete anyway (both ballotpedia questions, both fts.unocha.org summaries) |
| Wayback needs https and no availability API | `http://web.archive.org` refuses port 80 (reads like throttling); the availability API 429s under any concurrency; `https://web.archive.org/web/2026id_/<url>` works and carries the timestamp in the final URL |
| Wayback rescues zero JS walls | the archive faithfully stores the unrendered shell: bsky.jazco.dev (2.4-day snapshot) and dcas.dmdc.osd.mil (2.9-day) both extract 0 chars |
| derived APIs are findable only by rendering | grepping served HTML across six JS apps found one candidate (a Maps key); XHR capture in Chromium found a working unauthenticated endpoint for all six, and 3 of 4 hand-guessed endpoints were wrong |
| our own URL extractor emits malformed URLs | `_BARE_URL_RE` and `_MARKDOWN_LINK_URL_RE` (`resolution_url_scan.py:26,29`) both stop at `)`, so a markdown-escaped `Nuri_\(rocket\)` truncates to `Nuri_(rocket\`; the corrected URL returns 200 with 16,753 chars on the plain rung. A second instance on ballotpedia |
| the archived `js_wall` on dcas.dmdc.osd.mil was a DNS failure | SERVFAIL on 8.8.8.8 and 9.9.9.9, resolves on 1.1.1.1; 200 with 46 KB at 11:05 and DNS-failed by 11:09 in one session |

## Step 0 — the gate: one free GitHub Actions diagnostic

**This is an operator action and it blocks only the impersonation rung.** Everything in phase 1
below proceeds without it.

Why it is needed: the only environment that fails is the one we have never probed. Why it is
free: a `workflow_dispatch`-only job that does nothing but HTTP GETs to public pages. No API
keys, no LLM calls, no Metaculus publish, and GitHub Actions minutes are free on public repos
(`No-Stream/metaculus-bot` is public, verified). It is therefore outside CLAUDE.md's cost gate,
which is a gate on SPEND and on publishing.

What it does, per URL, from the runner: the bot's own aiohttp session as the baseline; the same
URL under `curl_cffi` `impersonate="chrome"`; and one `https://web.archive.org/web/2026id_/`
snapshot fetch. Ten URLs, the three Akamai hosts plus the DataDome and Cloudflare cases.
Prints a table. Nothing else.

The three outcomes and what each means:

- Plain 403, impersonated 200 → **impersonation is validated**; build it as rung 2.
- Both 403 → **it is pure IP reputation**; drop the impersonation rung entirely and the answer
  for that class is Wayback, url_context, or an egress change. This is the outcome I expect,
  stated as a prediction so it is falsifiable.
- Both 200 → the 403s were transient or have been lifted; re-measure the whole premise before
  building any escalation for the `blocked` class.

If the answer is "pure IP reputation", the cheapest real fix is an egress change rather than a
rung, and that is a new operator decision (see Decisions, D4).

## The ladder

Free and deterministic first, paid last. Steps 1b, 2 and 6 are **transport swaps**: they
produce an HTTP response that re-enters the same classification path (meta-refresh check,
inline chart read, chrome floor, JS-wall floor) rather than a terminal result. Only 4 and 7
produce a different artifact. Building it that way keeps the number of code paths linear and is
what makes the meta-refresh check correct on a body only a later rung could obtain.

| # | rung | cost | trigger | gate |
|---|---|---|---|---|
| 0 | direct fetch (unchanged) | free | always | — |
| 1a | URL-extractor paren/backslash fix | free, no request | always | none |
| 1b | meta-refresh hop | free, 1 request | near-empty 200 carrying the tag | none |
| 1c | ARIA-table rewrite before extraction | free, no request | every HTML page | none |
| 1d | inline Highcharts config read | already shipped 2026-09-02 | every HTML page | — |
| 2 | Chrome-TLS-impersonated retry | free, 1 request | `blocked` (403/406/429) and `error` with no HTTP status | **step 0** |
| 3 | local PDF text extraction | free, no new request | `application/pdf` by content type or `%PDF` magic | none |
| 4 | derived-API registry, endpoints harvested by XHR capture | free, <=3 requests | surviving `js_wall` / `no_resolving_content` | none |
| 5 | headless Chromium render | free, 1 render | surviving `js_wall` | none |
| 6 | Wayback snapshot, age-bounded and disclosed | free, 1-2 requests | anything unresolved, but SKIPPED for `js_wall` | none |
| 7 | Gemini url_context | **paid** | anything unresolved on a host robots.txt does not disallow | own flag |

Ordering notes that come from measurement, not taste:

- **Rung 6 is skipped for `js_wall`.** Wayback rescued 0 of 8 still-failing JS walls because it
  stores the unrendered shell. Rung 5 rescued 6 of 8 and rung 4 rescued 7 of 8.
- **Rung 5 must not treat a `goto` timeout as failure.** Use `domcontentloaded` plus a fixed
  settle, and always salvage `page.content()` after a timeout. Skipping this loses 4 of 10
  measured render rescues.
- **Rung 4 is rung 5 plus bookkeeping, not a static rung.** Harvest JSON XHR endpoints during a
  render and cache per domain so later questions on the same source skip to the API. A static
  page-source grep finds almost nothing.
- **Every rung switches off on the time budget's fast path**, per `design:427-432`.

Cumulative rescue over the 22 URLs that still fail from the laptop, in the order B → D → C → E → F
as the replay measured it: 5, then 6, then 9, then 17, then 18, leaving 4. Of those 4, only two
are genuine paid-rung candidates (sagaftra.org, trueup.io); one is a dead link no fetcher
recovers and one is our own extractor bug. **Read those counts as the residual after egress is
excluded, not as the ladder's total value** — the larger population is the 25 that need no rung
at all if the request comes from somewhere else.

## Both paths, and the seams

Anchors verified by a read of the tree at `f340e7d`.

Tier-1 (`resolution_source.py`): status table `resolution_fetch_result.py:86-92`
(403/406/429 → `blocked`, 404/410 → `not_found`, everything else non-200 → `error`; **401 is
absent and maps to `error`**). HTML branch `:659-763`, raw-body branch `:766-814`, content-type
router `:831-851` — the last is where a cited PDF dies today, returning **without reading the
body**, pinned by `tests/resolution_source/test_resolution_source_fetch.py:228`, so code and
test change together. Redirect dispatcher `:817-851` returns `FetchResult | str`, which is the
built-in seam for a meta-refresh hop; `_resolution_html_outcome` (`:659`) must widen its return
type. Per-URL budget `RESOLUTION_SOURCE_PER_URL_MAX_CHARS = 6000`, response cap
`RESOLUTION_SOURCE_MAX_RESPONSE_BYTES = 5 MiB` (`constants.py:546-547`) — note the IAISR PDF is
6.7 MB, so the PDF rung needs its own byte cap.

gap-fill v2 (`agentic/tools.py`): ladder body `:559-590`. The `blocked` early return at
`:565-566` is the hole rung 2 targets. PDFs become `document_needed` at `:247-255` **before any
bytes are read**, so a local-PDF rung requires moving the capped read above that check. Its 403
set is a second literal, `fetch_outcomes.py:42`, whose name (`_RETRYABLE_...`) is aspirational
— nothing retries it. A new `method` string grants **no** provenance tier unless added to
`_METHOD_TO_TIER` (`agentic/provenance.py:55-62`), and the conclude gate's fetch floor counts
`fetched`-tier reads (`agentic/gates.py:266`), so an untiered rung silently starves the gate.

Three corrections to what the repo's own docs claim, found while mapping:

- **The per-host politeness map is NOT shared between the two paths.** They share the helper,
  not the map: Tier-1 builds a fresh dict per provider call (`resolution_source.py:1210`), v2
  uses a module global (`agentic/tools.py:94`). So six concurrent questions can hit one host
  simultaneously from Tier-1 today. AGENTS.md and the v2 throttle notes describe this as
  shared, which is true of the function and false of the serialization. Fix this before adding
  any rung that adds requests, or we re-earn the blocks we are trying to route around.
- **`fetch_hardening.py` is not shared plumbing for either path** — it patches the Metaculus
  question-list GET.
- **`FUTURE.md:1661-1666`'s list of `agentic/` reaches into `resolution_source` privates is
  stale**: there are 11 call sites, not 6.

## SSRF invariants any new client must preserve

The governing rule is `design:444-446`: no derived URL may take a code path that bypasses
`_fetch_one_hop`'s session. A second HTTP client is exactly that bypass, so the equivalence has
to be argued explicitly rather than assumed.

1. `is_public_http_url` preflight on the initial URL and again on every hop.
2. **Connect-time IP vetting.** aiohttp gets this from `FilteringResolver`
   (`http_fetch.py:40-90`); libcurl does not participate. The equivalent is
   `resolve_vetted_public_ip` (`resolution_source.py:260-293`) then `CurlOpt.RESOLVE` host→IP
   pinning, the same technique `_resolve_pinned_host` uses for Chromium
   (`agentic/tools.py:368-404`). Must fail closed. **`uv.lock` pins curl_cffi 0.15.0** (a
   transitive dep of yfinance, `uv.lock:772`); I verified `CurlOpt.RESOLVE` exists at 0.16.2
   and 0.16.3, NOT at 0.15.0. Confirming it there, or bumping, is a hard gate on rung 2.
3. `allow_redirects=False` with a manual per-hop re-guard, reusing `MAX_REDIRECTS` and
   `REDIRECT_STATUSES` (`http_fetch.py:98-99`). Do not rely on `CurlOpt.RESOLVE` surviving a
   redirect.
4. **Body cap.** `read_body_capped` consumes `resp.content.iter_chunked` and is aiohttp-shaped;
   it is NOT reusable. A parallel capped reader needs its own tests — the abort-mid-stream pin
   at `tests/test_http_fetch.py:188` does not transfer.
5. Reusable as-is: `decode_text_body`, `vacuous_body_status`, `strip_html_tags`, both
   truncators, `_extract_main_text`.
6. **Wayback needs URL unwrapping before any guard.** `is_metaculus_self_ref` keys on hostname
   (`resolution_url_scan.py:102-113`), so `web.archive.org/web/.../metaculus.com/...` sails
   past both paths' self-reference refusals and would hand a forecaster an archived Metaculus
   page. Unwrap the inner URL and re-run every check on it.
7. **The test suite's egress guard does not cover libcurl.** `tests/conftest.py:45-46` patches
   `socket.socket.connect`; libcurl issues its own syscalls from C, the same hole Chromium
   already has. New rung tests must fake at the client-object level, and the guard should grow
   a curl-side check so a stray live call cannot pass CI silently.
8. Header contract: `tests/test_http_fetch.py:123` asserts `Accept-Encoding` excludes `br`
   because no brotli decoder is installed. A Chrome-fingerprint client advertising `br` breaks
   that premise; check whether curl_cffi decompresses brotli itself.

## Budgets

Tier-1 runs under `RESOLUTION_SOURCE_WALL_TIMEOUT = 45 s` with 20 s per request, up to 5 URLs,
6 questions concurrent. Archived timings: success p50 0.7 s / max 10.2 s; all-fail p50 0.3 s
but 3 of 23 ran the full 20 s (`design:395-399`).

- **Do not raise the wall yet.** `design:422-426` proposes 45 → 90 s when url_context is on.
  With the paid rung last and the free rungs each one request or single-digit-seconds of local
  CPU, build inside 45 s, measure, and revisit only if rung 7 is being starved by its own
  floor. This also respects the operator's standing rule that timing and fallback paths get
  only strictly-safer changes.
- **Per-rung self-bounding is a precondition, not a nicety.** The provider's outer
  `asyncio.wait_for` (`resolution_source.py:1352-1355`) returns `""` on timeout **before** the
  marker emission, the raw-archive write, the diagnostics record and the section render — so an
  overrun discards every page that already succeeded. The fix for that is `FUTURE.md:1229-1243`,
  **skipped by operator decision**. A longer ladder inside an unchanged wall therefore raises
  the chance of losing finished work unless every rung uses the Datawrapper pattern:
  wall-minus-elapsed-minus-margin, skip below a floor, degrade to what already succeeded
  (`resolution_source.py:1236-1268`).
- **Wayback needs its own cap.** Every snapshot shares netloc `web.archive.org`, so under
  Tier-1's per-host `Semaphore(1)` N URLs serialize into N requests behind one gate. Cap how
  many URLs get an attempt per question.
- **Chromium does not fit a 45 s wall unaided**: 35 s timeout plus launch, queued behind a
  process-global `Semaphore(2)` shared with v2. Needs a tighter render timeout and a
  wall-derived budget. Chromium install is `continue-on-error: true` in all five workflows
  after an apt hang forfeited two questions, so availability is best-effort by design.

## Models

Settled by the operator 2026-09-03:

- **`GAP_FILL_V2_READER_MODEL`: `gemini-3.5-flash` → `gemini-3.8-flash`.** Strictly better —
  half the input price ($0.75 vs $1.50 per 1M) and 42% of the output price, newer, stable. The
  reader's output enters the `fetched` provenance tier, the only tier that reaches the SUPERSEDE
  block telling every forecaster to override the briefing, so a misread has forecaster-facing
  blast radius nothing downstream can detect. Volume is a few calls per run once rungs 1-6 land,
  so the price gap to the lite tiers is cents.
- **Grounded search stays on `gemini-3-flash-preview`** for now. Moving it to 3.8 costs roughly
  a dollar a month more and needs one live call to verify the id on the native SDK, which is a
  paid action and the operator's to fire. Independent argument for eventually moving: it is a
  *preview* id in a production path and the pricing page now labels it legacy.
- **There is no free Google allocation to lean on.** A billing-enabled AI Studio project is
  paid tier for every API request, drawn from the prepaid balance in near real time; when that
  balance hits zero every key on the account stops at once. Only the 5,000 grounded *search
  requests* per month are free, and the tokens behind them are still billed. Prepaid credits
  expire after 12 months.
- **We cannot currently see Google-side spend.** Nothing reads `response.usage_metadata`, so
  grounded search and read_document are absent from the per-role ledger that covers OpenRouter.
  Add a `GEMINI_USAGE` INFO marker per native call carrying `prompt_token_count`,
  `tool_use_prompt_token_count` (the retrieved content), `candidates_token_count` and
  `thoughts_token_count`, harvested like `CREDIT_ROLE_SPEND`. This is the deterministic answer
  to "are we burning the Google balance".
- **Both clients are under-configured, differently.** Grounded search is
  `genai.Client(api_key=...)` with no `HttpOptions` at all (`gemini_search.py:69`): no
  client-side timeout, no retries, bounded only by a 360 s outer `wait_for`. The reader sets a
  timeout but leaves `retry_options` unset, and `retry_args(None)` yields
  `stop_after_attempt(1)` (`google/genai/_api_client.py:529-533`), which is why two calls died
  on `503 UNAVAILABLE` in production with no retry. Give both an explicit
  `HttpRetryOptions(attempts=2-3, http_status_codes=[429,500,502,503,504])`; do **not** take
  the SDK's 5-attempt default, whose worst case exceeds each path's own deadline.
- **Thinking is set nowhere today, so each model's default applies, and `gemini-3-flash-preview`
  defaults to `high`** (ai.google.dev/gemini-api/docs/thinking, read 2026-09-03), which is why 71% of
  grounded-search output tokens are thinking. Pin it explicitly in constants via
  `GenerateContentConfig(thinking_config=ThinkingConfig(thinking_level=...))` (installed SDK enum:
  MINIMAL / LOW / MEDIUM / HIGH; `minimal` is NOT offered on 3.8-flash, whose range is low/medium/high
  with default medium). Settled 2026-09-03: **reader `low`** (extraction, no planning); **grounded
  search `medium`** (the call is agentic and thinking is where query planning happens; high→medium
  roughly halves the dominant cost line), with `low` a MEASURED follow-up against
  `GEMINI_GROUNDING_DENSITY`, unsupported-attribution count and queries/call over a few dozen prod
  calls, since it shifts forecasts and belongs in the same merge as the bundle.
- **Size gate is the primary reader saving, model is secondary.** A document we already hold is never
  sent to url_context; local extraction plus BM25 passage selection serves it. The reader's residual is
  hosts our client cannot reach (two of 47 archived failures).

### Measured season spend on the Google key (prod runs only, 2026-05-29 to 08-28)

Reconstructed 2026-09-03 from the research archive (`backtests/research_archive/latest/`,
322 summer-futureeval-2026 artifact records) and the raw archive's `usage_metadata` on 113
grounded calls from 2026-07-20 on (`raw/*.jsonl`, provider `gemini_search`). Receipts and the
per-URL reader sizing: `scratch/fetch_ladder_2026-09-03/`. Backtests and ablations billed the
same key but are not in this archive and are EXCLUDED. The Google AI Studio billing page is the
only ground truth; this is the estimate to reconcile against it.

Grounded search (`gemini-3-flash-preview`, unchanged since 2026-05-01). Measured per call: 2,264
input tokens, 3,473 output tokens of which 71% are thinking, 12.7 search queries. Calls: one per
question (322) plus, before the 2026-06-25 resolver migration, one grounded call per gap-fill
gap (503 more, mean 3.99 gaps/question over 126 questions).

| month | questions | grounded calls | token $ | search queries | overage past 5,000 free |
|---|---|---|---|---|---|
| May 29-31 | 25 | 124 | 1.43 | ~1,600 | 0 |
| June | 119 | 523 | 6.04 | ~6,600 | ~23 (0 if resolver calls averaged 5 searches) |
| July | 111 | 111 | 1.28 | ~1,400 | 0 |
| August | 67 | 67 | 0.77 | ~850 | 0 |
| season | 322 | 825 | 9.53 | ~10,500 | 0-23 |

The June spike is the "started getting billed" episode CLAUDE.md records: the v1 resolver's
extra grounded calls pushed the month past the free search pool. Since the migration the bot
uses 850-1,400 searches a month, 17-28% of the pool, NOT the 3% CLAUDE.md estimates by counting
prompts rather than queries. **The free search pool is the binding constraint, not token price.**
One `backtest_large` (about 600 grounded prompts at 12.7 queries each) would consume more than a
month's pool in one run.

Reader (`read_document`, `gemini-3.5-flash`, v2 on in prod since 2026-07-21): 104 summer
questions with transcripts, 191 reader calls, 135 succeeded. Retrieved content on SUCCESSFUL
reads, sized by re-fetching every document and counting chars/4: 874k tokens (128 measured,
7 imputed at the median 1,484). Failed reads retrieve nothing and bill only the prompt. Output
tokens are unmeasured; the range below assumes 1,000-3,000 per call including thinking.

| model | season cost | per v2 question |
|---|---|---|
| gemini-3.5-flash (actual) | $3.20-6.64 | $0.031-0.064 |
| gemini-3.8-flash | $1.46-2.89 | $0.014-0.028 |
| gemini-3.1-flash-lite | $0.53-1.11 | $0.005-0.011 |

Nine documents over 100k tokens carried 67% of all sized reader tokens (SEC EDGAR filings, the
754-page XPT report, raw CSV time series). A size gate ahead of url_context, routing anything
over ~100k tokens to local extraction plus passage selection, removes most of the reader's
cost regardless of model.

**Google-side total, prod only: about $13-40 for the season, $0.04-0.12 per question.** The
OpenRouter side is the separately measured $0.38-0.41 per question (CLAUDE.md), so all-in
per-question spend excluding the AskNews and Exa subscriptions is roughly $0.45-0.55.

## Telemetry

Follow `design:378-383` rather than reinventing: grow the existing `RESOLUTION_SOURCE_FETCH`
marker with `route=`, `failure_class=` (fetcher / egress / host / unknown), `exc=` and
`server=`, and add `RESOLUTION_SOURCE_ESCALATION: question= url= from_status= rung= outcome= wall_s=`.
Mechanics: `MarkerSpec` named groups **are** the schema (`scripts/telemetry/markers.py:197-216`);
a new field must be optional and at the tail, because an optional group between same-shaped
`\S+` fields silently records `None` (documented at `markers.py:645-648`). The pattern to copy
is the existing optional `reason=` at `markers.py:726`.

`agentic_document_ungrounded_suppressed` already exists as a spec (`markers.py:1092-1102`), so
adding the SDK's per-URL `url_retrieval_status` to that WARN is an optional trailing group plus
a with/without parse test. That logging change is two lines and free:
`extract_url_context_telemetry` already returns the statuses and `tool_backends.py:254`
discards them.

Per-rung attempt counts belong in `details["counts"]` (`provider_diagnostics.py:169-184`), where
a zero renders nothing but survives into the archive.

Hand-maintained surfaces a new marker must touch: the spec plus the emitter list in
`markers.py`, a class in `tests/test_telemetry_markers.py` (fixture line copied verbatim from
the emitter), `tests/test_id_mapping.py:235-246`, `docs/operations.md`, `docs/research.md:563-567`,
and AGENTS.md's `make sync_all` roster at `:356`. Nothing enforces the docs.

## Testing and QA

Layered, cheapest first. The operator's direction is to test extensively and have a subagent
manually QA, because every rung talks to a third party that changes under us.

1. **Offline unit and contract tests on recorded fixtures**: an Akamai 403 body, a DataDome
   challenge page, small text-layer and scanned PDFs, the cdc.gov meta-refresh stub (a real
   234-byte capture), an ARIA-table stat block (fixture already on disk at
   `scratch/next_season_bundle_2026-09/item13/`), Wayback snapshot HTML, a Wayback-wrapped
   Metaculus URL for the unwrap guard. These pin ladder order, status tokens, disclosure
   wording, budgets and the SSRF invariants. CI's egress block forces this shape.
2. **Extend the existing pins**, which are extensive and enumerated in the seam map: PDF
   body-never-read (`test_resolution_source_fetch.py:228`), the 403-set parametrization
   (`test_agentic_tools.py:1068`), the tier-unmapped precedents
   (`test_agentic_loop.py:1111`), the Datawrapper budget-skip (`test_..._datawrapper.py:585`),
   `_reset_tool_state` (`test_agentic_tools.py:56`) for any new module global.
3. **A checked-in live replay corpus** of URL plus expected rung, the 47 URLs already probed,
   run under the existing `live` marker so CI never touches it. Free, no model calls.
   Monitoring rather than a gate, because sites drift.
4. **Independent manual QA by a fresh-context subagent** after implementation: run the live
   corpus, drive the resolution-source provider alone over ~10 archived questions (network
   only, no forecasters), diff rendered sections against the archived ones, and report per URL.
   Separate from `/forge`.
5. **One paid smoke test at the end**, operator-fired, as the final pre-merge check.

Flakiness containment by design: every rung bounded by its own budget inside the wall, every
failure a status rather than an exception, per-host politeness made process-global, per-run
caching, and a `rung=` field so prod flakiness is a query rather than a guess.

## Operator decisions

- **D1. Fire the step-0 diagnostic?** It is free and it is the only thing standing between us
  and either building or dropping the impersonation rung. Needs a push (blocked for me) and a
  dispatch. I can write the workflow file for review.
- **D2. Posture on TLS impersonation, if step 0 validates it.** Every other rung respects the
  site's expressed intent; impersonation is the one where we misrepresent what we are. Ranked
  weakest interference first: Wayback (no request to the host), url_context (a fetcher the site
  can and does control via robots.txt), local PDF decode (a body the host served us),
  impersonation (a Chrome fingerprint after the edge refused our real one). Ships behind its own
  flag either way, so this is a switch rather than a code change.
- **D3. Honour robots.txt `Google-Extended` for rung 7?** Recommended yes, as a free per-host
  pre-check recorded as its own terminal status, so the skip is greppable and the hypothesis
  stays falsifiable. It is a cheap filter, never a complete predictor — Manifold was refused
  with `Allow: /`.
- **D4. If step 0 says pure IP reputation, do we change egress?** New question this plan raises
  and does not answer. Options and their costs are not yet scoped; flag if you want them.
- **D5. Is a Wayback snapshot admissible as primary grading evidence, and what age bound?**
  Recommended: yes, with a mandatory "as of <timestamp>, N days before this forecast" line and
  a `stale_data` withhold past the bound, mirroring the Datawrapper 30-day guard. The bound's
  value is a judgment call: 30 days was calibrated on daily-republishing trackers and a
  question resolving on a weekly series needs tighter.
- **D6. Per-route caveat wording.** `design:578-581` asks whether a url_context read may render
  under the "primary grading evidence" heading. Recommend generating one caveat sentence per
  `route` instead of one binary decision, since Wayback (deterministic, stale) and url_context
  (model-mediated, current) fail that promise differently.
- **D7. Grounded search to 3.8** needs one paid verification call. Yours to fire, optional.

## Decisions taken (operator, 2026-09-03, inline)

- D1 fire the free GHA diagnostic: **yes**. D2 impersonation behind a flag if validated: **yes**.
- D3 Google-Extended pre-check: **only if it actually blocks url_context**. Determination: two tiny
  url_context probes (a small plain HTML page on a Google-Extended-disallowed host vs an allowed
  host) bundled into the D7 verification call; if the disallowed one retrieves, drop the pre-check.
- D4 egress change: **too complicated**, parked in FUTURE.md (low priority).
- D5 Wayback admissible: **yes, if very clearly stated as stale** (as-of date in the section lead,
  withheld past the age bound).
- D6 per-route caveat sentence: **yes**. D7 grounded search → `gemini-3.8-flash`: **yes**, one live
  verification call, thinking pinned to `medium`.
- Spend reconciliation: operator reports **$63 over the last 90 days, concentrated on June 18 and
  June 24**. Consistent with the prod reconstruction ($10-33) plus the June benchmark runs that PR
  #47 ("Benchmarking + Use metaculus gemini tokens", merged 2026-06-18) shipped and that this
  archive cannot see; PR #49 `goog-cost` (merged 2026-06-26) is the "made some changes" the
  operator half-remembers — it moved the gap-fill resolver off grounded Gemini.

## Sequencing

Phase 1, no gates, independent of the egress question — the URL-extractor fix, meta-refresh plus
ARIA rewrite, local PDF extraction with its own byte cap and passage selection, the Chromium
wait-condition fix, the process-global per-host gate, the two free url_context hardening items
(status logging, retry options), the reader model change, and the `GEMINI_USAGE` marker.

Phase 2, after step 0 — impersonation rung if validated, or the egress decision if not.

Phase 3 — Wayback with unwrap guard and age disclosure; Tier-1 rendered rung; derived-API
registry with XHR-harvested endpoints.

Phase 4 — url_context last, behind its flag and the robots pre-check, with per-route caveats.

Each phase ends green on `make test`, `make lint`, `make typecheck`, `make deps`,
`make lint_imports`; `/forge` after phases 1 and 3; the subagent QA pass after phase 3; the
paid smoke test once, at the end.

## Rejected, do not re-propose

From `design:449-453`: "find URLs in the JS bundle and fetch them" (the registry exists so a
crafted page cannot choose our host). From `design:321-325`: renaming status tokens, which are
telemetry identifiers. From `FUTURE.md:2449-2451`: Firecrawl and Olostep, rejected in favour of
this DIY ladder. And from this plan: building the impersonation rung on laptop or EC2
measurements, which cannot see the failing environment.
