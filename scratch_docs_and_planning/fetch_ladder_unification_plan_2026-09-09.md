# One fetch ladder for both callers (plan, 2026-09-09)

Written for the fall-season merge that has to land before 2026-09-28. Branch `mantic-competition`,
tip `faf4581`. Repo root `/Users/flatljan/personal/metaculus-bot`; every path below is relative to
it. The operator asked for this in one sentence: "of course we should modularize and have full
parity between the agentic route and the resolution fetcher. both ways, full rung."

This repo is a Metaculus forecasting bot. Two separate code paths fetch web pages. The
**resolution-source fetcher** (`metaculus_bot/research/resolution_source.py`, 3,518 lines, called
Tier 1 in the older docs) fetches the URLs a question names as its grading source. The **gap-fill
v2 agentic loop** (`metaculus_bot/research/agentic/`) gives a driver model a `fetch` tool and a
`read_document` tool for URLs the model picks itself. They already share every transport:
`research/http_fetch.py` for aiohttp, `impersonated_fetch.py` for the curl-cffi retry that presents
a Chrome TLS fingerprint, `rendered_fetch.py` for headless Chromium, `url_context_reader.py` for the
paid Gemini read, and `robots_policy.py`. What differs is which rungs each one calls and what it
does with the bytes.

Read `scratch/cost_pass_2026-09-09/fetch_gap_inventory/INVENTORY.md` and its `ladder_table.md`
first (both gitignored, both readable with `cat`). They are today's rung-by-rung comparison with
file and line receipts, and every frequency number in this plan comes from them. Then read
`docs/research.md` "Resolution-source fetcher" and `docs/agentic_gap_fill.md` "The fetch ladder"
for the documented behaviour, and `scratch_docs_and_planning/fetch_ladder_plan_2026-09-03.md` with
`impersonate_rung_plan_2026-09-04.md` for the decisions already taken. The inventory predates the
five rung-port commits named under "What the split costs today", so read its loop column as the
state before `21f3122`.

## What the split costs today

The loop lacks six of the resolution fetcher's rungs: the meta-refresh hop, the ARIA-table rewrite,
the inline chart-data read, the Datawrapper dataset hop, the derived-API feed reuse and Wayback. It
also runs a single default trafilatura extraction where the fetcher runs a calibrated two-pass
policy. The fetcher lacks four things the loop has: a run cache, an ask-directed BM25 passage digest
for long HTML, windowed pagination, and a throttle-phrase check. Every one of those is a pure
function or an existing transport, so what we have is duplication and not a capability gap. The
measured cost sits mostly on the loop's side: across 121 archived transcripts, 33 of its 80
successful Chromium renders served under 500 characters, which is the exact shape the fetcher's
ARIA rewrite, chart read and harvested-feed check are built for. On question 43949 a cited page
served about 80,000 characters with none of the resolving figures in them, while its annual series
sat in a `data-chart` attribute the fetcher reads and the loop ignores.

One caution before you start. On the same day this plan was written, five commits ported most of
those rungs into the loop: `21f3122` (Tier 1's HTML extraction steps, the chart read and the
meta-refresh hop), `8eba338` (Wayback with the capture date surfaced), `246fe68` (the harvested
JSON feed when a render is empty), `924cfc2` (the tool description steering the driver off FRED,
Kalshi and Yahoo Finance) and `8f1a56b` (docs). Absorb that work rather than redoing it: those
calls become the shared classifier's, and step 4 below shrinks to the derived-feed REUSE half.
Read the constraints section at the end before planning a line of the extraction.

## Module boundary and entry point

Build a new package `metaculus_bot/research/fetch_ladder/` with five modules, and reduce both
callers to adapters. A package beats one file here because `resolution_source.py` is already
oversized, and `FUTURE.md` carries its split as an open item on exactly these seams.

| Module | Holds |
|---|---|
| `policy.py` | `LadderPolicy`, a frozen dataclass of the per-caller knobs, plus the two presets built from `constants.py`. |
| `context.py` | `LadderContext`, today's `FetchContext` renamed, carrying the wall origin, the wall-clock instant, the caller's ask, the per-question rung budget and the rung attempts. |
| `classify.py` | The one classification path for a body: content-type routing, the ARIA rewrite, the two-pass extraction, the chart read, the PDF read, the chrome and JavaScript-wall floors, the throttle check, the meta-refresh hop. |
| `rungs.py` | The seven escalation rung bodies, unchanged in behaviour. |
| `ladder.py` | The direct fetch, the escalation dispatcher, the run-scoped cache, and the single entry point. |

The status vocabulary stays where it is, in `research/resolution_fetch_result.py`. Its name becomes
a mild misnomer once both callers share it, and that is the right price to pay: every string in that
module is a telemetry contract, and the repo rule is to add tokens and never re-spell one. A module
rename is available later as its own mechanical commit, and it is out of scope here. The entry
point is one coroutine:

```python
async def fetch_url(url: str, *, policy: LadderPolicy, ctx: LadderContext) -> FetchResult
```

`FetchResult` already carries the text, the route that produced it, the status, the status reason,
the HTTP status, the content type, the per-rung attempt records and the chart fields. Add three
optional fields with defaults, so an archived record and an all-direct fetch stay byte-identical:
`capture_date` for an archival read, `harvested_json` for the JSON responses a render recorded, and
`bytes_read`. Nothing in `FetchStatus`, `FetchStatusReason`, `FetchRoute` or `RungSkipReason` is
renamed. The loop's own `PlainFetchResult` and its status strings survive as its adapter's output,
mapped from `FetchResult` in one function, because `provenance._METHOD_TO_TIER` and the conclude
gate's fetch floor both read them.

## The policy knobs

| Knob | Resolution-source preset | Gap-fill preset |
|---|---|---|
| Total budget | `RESOLUTION_SOURCE_WALL_TIMEOUT` (45 s) less the rung margin | The `fetch` tool's 90 s, and 25 s for the document ladder |
| Per-hop timeout | `RESOLUTION_SOURCE_HTTP_TIMEOUT` (20 s), clamped to the remaining wall | The same 20 s |
| Rungs enabled | All, with the two expensive rungs declining on the fast path | All, with the two expensive rungs declining under their own floors |
| Presented characters | 6,000 per URL, 18,000 per question | 8,000 per window, paginated by `start_char` |
| Digest query | The question's title plus its resolution criteria | The driver's own `ask` per call |
| Wayback | Withheld past `RESOLUTION_SOURCE_WAYBACK_MAX_AGE_DAYS` (30) | Capture date surfaced, the driver judges freshness |
| Per-question caps | 2 Wayback snapshots, 2 paid reads | The same two caps, which the loop lacks today |

Four knobs are the same for both callers and stay plain constants. Those are the two byte caps
(5 MiB for a page, 40 MiB for a declared PDF), the robots pre-check before the paid read, the
platform self-reference refusal, and the shared run cache. Three rows above carry the design work. The
digest query becomes a policy field, which is what lets the fetcher digest a long cited HTML page
it would otherwise read head-first; the digest itself is the LLM-extractive one under "Decisions,
resolved". The per-question paid-read cap is new for the loop and lowers its spend. The Wayback row
is the one genuine coupling, and it needs a paragraph.

That 30-day bound is calibrated on a page a question cites as its grading source. A month-old
capture of the page a question grades on is still evidence about that page, and a URL the driver
chose carries no such guarantee. So the loop's preset surfaces the capture date and its age and
lets the driver decide. `wayback.wayback_lead` already writes that sentence, so the text is free.

## Rung order, one list

Cheapest first. Each rung declines by returning nothing, in which case the previous outcome stands.
The order below is today's `FetchRoute` order with one step added in front, and the third column is
why each rung sits where it does. Only rung 0 is new code. The run cache exists in the loop already
and is new only to the fetcher, and rungs 2 through 7 all exist today on at least one caller.

| Order | Rung | Why here |
|--:|---|---|
| 0 | Known-API translation | Free, exact, and no page fetch at all. A seat only: `policy.known_api` is `Callable[[str], Awaitable[FetchResult \| None]] \| None`, default `None`. A sibling agent owns the registry and the SEC EDGAR client. |
| 1 | Run cache | No network. Successes only. A block, error or throttle is never cached, because the driver is told to retry those. |
| 2 | Direct plain GET | The baseline. The meta-refresh hop, ARIA rewrite, chart read, two-pass extraction, local PDF read and throttle check all live inside its classification, so none of them is a separate rung. |
| 3 | Impersonated retry | Free, one GET. Four of four Akamai-fronted federal URLs that refused our client answered it 200 from a GitHub runner on 2026-09-04. Triggers on a host's 403 alone. |
| 4 | Derived API feed | One GET against an endpoint a render already recorded for that host, which saves a second launch. |
| 5 | Headless Chromium | Rescued 6 of 8 archived JavaScript walls. A 100 to 300 MB launch, contended process-wide, floored at 12 s. |
| 6 | Wayback | The one free route whose egress is not ours. Rescued 3 of the 22 URLs still failing from a residential address, and reads no JavaScript wall. |
| 7 | Paid Gemini read | The only rung that spends money, and the only one whose product is a model's answer. Last for both reasons. |

The Datawrapper dataset hop stays where it is, a second phase after the page fan-out. It produces a
second result for one URL, and the loop's tool contract returns exactly one outcome per call. So the
loop reaches it by having its adapter append the dataset to the page text. That is the one place the
two callers still diverge, and it is a presentation difference and not a rung.

## Telemetry

No new marker. `RESOLUTION_SOURCE_FETCH` and `RESOLUTION_SOURCE_ESCALATION` become both callers'
markers, with one optional keyed field appended at the tail:
`caller=resolution_source|gap_fill_v2`. It is optional and tail-positioned for the same reason the
existing `reason`, `route`, `failure_class`, `exc` and `server` fields are. An archived line still
parses, an absent field keeps meaning "does not apply", and a keyed tail group cannot claim its
neighbour's value. The archive stays continuous with no deprecation window and no double logging,
and the loop gains the per-URL fetch record it has never had. The marker name reads as a slight
misnomer for a loop fetch, and names are contracts here, so the misnomer is the correct price.

Each caller emits its own marker lines, and the ladder emits none. The resolution fetcher emits at
its per-question aggregation point, because that is the only place the question id exists. The
loop's adapter emits after each tool call with `question=None`, exactly as its three existing event
markers do, so a join goes through the run id. `scripts/telemetry/markers.py` needs the one
optional group added to both specs, and `docs/telemetry_markers.md` needs the field description.
The loop's `GAP_FILL_V2` counters, `AGENTIC_FETCH_THROTTLED`, `AGENTIC_FETCH_LOCAL_DOC` and
`AGENTIC_URLCONTEXT_ROBOTS_SKIP` are untouched.

## Migration, in order

Each step below is one commit with its own tests, and each reverts on its own. Steps 1 through 4
and step 6 are strictly safer or purely additive, so build them without pausing for a check-in.
Step 5 is the one that changes what a forecaster reads or when a fetch gives up, and it carries the
three operator decisions at the end of this document. Do the steps in this order, because each one
shrinks the surface the next has to reason about.

1. Extract with no behaviour change. Move the ladder bodies into the new package. Keep every name
   the two test suites patch reachable where they patch it, or move the patch target and every call
   site in one commit. `_extract_main_text` is the one that matters: about 20 sites patch it across
   `tests/test_agentic_tools.py` and
   `tests/resolution_source/test_resolution_source_extractor_policy.py`. A stale patch target stays
   green while proving nothing, and this repo has shipped that bug before.
2. Point the resolution fetcher at `fetch_url` with the preset that reproduces today exactly. The
   existing suite is the gate: 17 test modules under `tests/resolution_source/` carrying 472 test
   functions, none of which should need an edit beyond an import.
3. Point the loop at `fetch_url` with a preset that encodes today's gaps, every ported rung still
   off. Its 170 tests in `tests/test_agentic_tools.py` are the gate.
4. Turn on, one rung per commit, what the loop lacked: the ARIA rewrite, the chart read, the
   two-pass extraction, the meta-refresh hop, the derived-feed reuse, then Wayback. The first five
   only ever add content to a page the loop already read, and Wayback only turns a failure into a
   read.
5. Turn on, one per commit, what the fetcher lacked: the shared run cache, the throttle check, then
   the LLM-extractive digest for long cited HTML, which replaces the BM25 digest on both callers.
6. Delete the duplicates. The loop's module-global `_FETCH_HOST_SEMAPHORES` folds into
   `http_fetch.host_semaphores()`, which closes a defect the 2026-09-03 plan flagged: the two paths
   share the politeness helper and not the map, so six concurrent questions can still hit one host
   at once. Then drop the loop's dead `_fetch_plain`, `_fetch_one_hop`, `_try_impersonated_fetch`
   and `_try_rendered_fetch`, plus its five reaches into `resolution_source` privates.

## Tests

The CI gate is a synthetic rung-order corpus. It holds one case per pair of direct status and
status reason, each with a fake session that answers a fixed way. Each case asserts the rung
sequence, the final route and the final status under BOTH presets. This corpus
is what makes the extraction provably behaviour-preserving. CI green is the gate and a local green
run is not, so nothing here may assert a path derived from the developer's environment.

The pre-merge local check is an archive replay. Write a script under `scripts/` that replays the
archived outcomes and prints a diff table of route and status, before against after. The fetcher's
per-URL records live in `backtests/research_archive/raw/*.jsonl`, carrying the fields `url`,
`status`, `http_status` and `content_type`. The loop's transcripts live in
`backtests/research_archive/by_qid/*.jsonl`, where each tool result opens with `status:` and
`method:` headers. Leave it out of CI, because that archive is gitignored and expires at 90 days.

Three more test groups finish the work. Per-rung unit tests reuse the existing fakes in
`tests/resolution_source_fakes.py`, `tests/playwright_fakes.py`, `tests/http_fakes.py` and
`tests/agentic_fakes.py`. Bounds tests assert that no rung exceeds its own timeout or byte cap
under either preset, and that a rung below its floor records a `wall_budget` skip without dialing.
The two-caller cache test has the fetcher read a URL, has the loop then ask for the same URL, and
asserts no second request. That cache holds the FULL extracted text and each caller applies its own
character cap at presentation, which is what lets the loop paginate a page the fetcher capped. The
gates to run are `make test`, `make lint`, `make typecheck` at zero errors, `make lint_imports` and
`make deps`. The autouse egress guards in `tests/conftest.py` keep every one of them free of spend.

## Effort, and what to leave alone

Expect about 2,000 lines moved, 400 lines new, 40 new tests and roughly 25 files touched. The blast
radius is the fetch and deadline surface, which is strictly-safer-changes territory under the repo's
standing rules, so steps 1 through 3 must be pure moves whose diffs a reviewer can read as such.
Missed deadlines cost real forecasts this quarter, so a change that is not obviously safer belongs
in `FUTURE.md` instead of in this merge. Add no hardening either. The DNS pins, connection-time IP
assertions, landing-host checks and the WebSocket block stay as they are. Do not re-implement them
for the shared path, do not extend them, and do not cite them as grounds for a new branch. They
were built for a threat this deployment cannot realise, which is a standing operator ruling.

Four more prohibitions close the list. Add no per-host branch except through the known-API seat. Do
not change when the fast path declines the browser or the paid reader. Leave both benchmarking
leakage guards returning an empty string when `is_benchmarking` is true. Re-spell no status token,
route token or skip reason. Each of the four is a rule this repo earned the hard way, and
`AGENTS.md` carries all four under "Guards and safety" and "Proportion".

## Decisions, resolved 2026-09-09

The operator resolved all three the day the plan was written, so none of them blocks the build.

1. **The throttle-phrase check on cited pages: ship it.** The loop classifies a 200 whose body is a
   rate-limit interstitial as throttled. The resolution fetcher has no such check, calls the same
   page a JavaScript wall, and then spends a Chromium launch on it. Sharing the check gives the
   fetcher a new `FetchStatus` member and a new escalation seam. The receipt is question 45191,
   where two throttled ogimet.com reads reached the driver as successes.
2. **A digest replaces head-first truncation on long cited pages: ship it, with an LLM doing the
   extraction and BM25 demoted.** Today a cited HTML page over 6,000 characters is read from the
   top and its tail is unreachable. The operator does not trust the deterministic BM25 digest as
   the primary mechanism and prefers a cheap model with a grounding check. The design is below.
3. **Wayback freshness for a driver-chosen URL: surface the capture date and let the driver
   judge.** The 30-day bound stays on cited grading sources only, as the policy table says.

The page digest, as agreed. A new paid support role, `page_digest_extractor`, reads a long page's
text plus the question (and, in the loop, the driver's ask) and returns the verbatim passages that
bear on the question, in ranked order. The operator chose the model on 2026-09-09:
`openai/gpt-5.6-luna` at reasoning effort `medium` ("luna is dirt cheap and medium will still be
fast enough"), with `google/gemini-3.8-flash` as the noted alternative. Both halves of that choice
are constants in `constants.py`, `PAGE_DIGEST_EXTRACTOR_MODEL` and `PAGE_DIGEST_EXTRACTOR_EFFORT`,
since that file is one of the two `tests/test_model_name_locations.py` allows a model id in. Every
call is tagged `role=page_digest_extractor`, so it lands in the `CREDIT_ROLE_SPEND` ledger like
every other role. The grounding check is the hallucination guard, and it is literal: a returned
passage is accepted only when it is a substring of the page text after whitespace normalisation,
and a passage that fails is dropped and counted. The page's opening passage is always kept ahead of
the ranked ones, so a reader still sees what the page is.

BM25 keeps two jobs. It is the free pre-filter that cuts a very long page to a few thousand tokens
before the model reads it, and it is the fallback when the call fails, returns nothing grounded, or
exceeds its per-call budget inside the 45 s wall. Set that per-call budget from luna-medium's
measured latency in today's probe, about 17 to 33 s of wall at medium effort on 40k-token prompts,
which is why the pre-filter must cut the page first for the call to fit inside the fetch wall. That
fallback path is today's behaviour, which is what makes the timing change strictly safer. Cost is
fractions of a cent per long page, and about 28 percent of the loop's plain reads hit the length
window per the inventory. Three optional tail-keyed fields ride the fetch marker:
`passages_returned`, `passages_grounded`, `fallback_used`.

## Constraints from the 2026-09-09 rung ports

The agent that ported the rungs (`21f3122` through `8f1a56b`) reports five constraints the plan
above states loosely or not at all. Treat each as a requirement of the extraction.

First, a meta-refresh target is a classification OUTPUT that only a redirect-loop owner may
consume. Today `_plain_body_outcome` and `_plain_html_outcome` return `PlainFetchResult | str`,
and the impersonated rung needed an explicit decline when handed the `str`. In `classify.py` make
the hop a typed `NextHop` so every consumer must handle it, with explicit decline branches in the
impersonated and Wayback rungs, which own no redirect loop.

Second, the loop's adapter must preserve five facts of its tool contract. A status of `ok` means
content was read, and nothing else: `provenance._harvest_verification_tiers` grants the `fetched`
tier on status alone, so `empty`, `blocked`, `error`, `throttled` and `document_needed` are never
`ok` with text attached. `method` is a token in `provenance._METHOD_TO_TIER` for a real read
(`wayback` and `derived_api` were added there, additively) and absent for a non-read. `http_status`
is set only by a host's response and never by our own refusals, because it triggers the
impersonated retry and Wayback, and a self-produced `blocked` carries `None` precisely so those
rungs cannot be handed it. `escalate_rendered` is the thin-content signal, and a chart block pins
it off, because a render loses client-side `data-chart` attributes (question 43949). Links resolve
against the document URL after a client-side redirect.

Third, the loop has no per-question object today. State in `tools.py` is module-global or per
call, and `build_gap_fill_tools(question_topic)` is the only per-question closure, which is why the
Wayback rung shipped without a per-question cap. Construct `LadderContext` once per question at the
seam `agentic_gap_fill.run_gap_fill_v2` and capture it into the tool handlers, or the caps in the
policy table have nothing to count on.

Fourth, the derived-API rung has two halves. REUSE dials an endpoint a prior render on the host
recorded, which is Tier 1's rung 4 and what step 4 still owes the loop. HARVEST serves the JSON the
render just captured, costs no request, and belongs inside the rendered rung's empty-DOM
classification, where `246fe68` put it. The two render memo scopes (`_RENDER_MEMO_SCOPE` is
`gap_fill_v2` against Tier 1's `resolution_source`) may merge only after step 4 has the loop on the
same classifier, because until then "rendered to nothing" means different things to each.

Fifth, the test seams the migration must keep reachable, or move with every call site in one
commit: `agentic_tools.render_page`, `_fetch_plain_with_impersonated_retry`, `_fetch_plain`,
`_try_wayback_fetch`, `_try_rendered_fetch`, `_read_response_body`, `_get_session`,
`is_public_http_url`, `fetch_impersonated`, and `resolution_source._extract_page_text` with
`_extract_main_text`. Two housekeeping items ride the extraction commit: the 86 pre-existing
receipt-comment findings across `tools.py`, `fetch_outcomes.py`, `provenance.py`, `derived_api.py`
and `tests/test_agentic_tools.py` move into the docs in the same commit as the code they annotate,
and the pending `tools.py` split is absorbed by the package layout rather than done separately.

Sixth, added after the EDGAR agent's archive read: three archived "unsupported content type"
refusals were the loop's own allowlist and not the hosts. `fetch_outcomes._TEXTUAL_CONTENT_TYPE_TOKENS`
admits exactly `text/plain`, `text/csv` and `application/json`, so api.weather.gov answering
`application/geo+json`, Treasury's yield-curve feed answering XML, and the mass-shooting tracker's S3
bucket answering JSON under an unlisted type were all refused at HTTP 200 with the resolving data in
the body. The unified classifier must treat structured-data types (JSON and every `+json` suffix,
XML, CSV) as readable content routed to the harvested-JSON and derived-feed path, the way Tier 1's
`is_json_content_type` already admits `+json`; neither path reads XML today. One item outside this
plan for the prediction-market provider's owner: the repo dials `api.elections.kalshi.com` while
Kalshi's current docs name `external-api.kalshi.com`, so that venue's host may have drifted.
