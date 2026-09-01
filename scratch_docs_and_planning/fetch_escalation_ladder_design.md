# Fetch escalation ladder for cited resolution sources: design

This is a design for the operator's separate approval (next-season bundle item 13, written
2026-09-01). Nothing in it is implemented; the bundle ships this document and no code. Every
number below comes from a file named beside it, and every live observation carries its date.
Where a claim rests on belief instead of on something I read or ran this session, the sentence
says so. A reader who has never seen this repository should be able to follow it end to end.

## Summary

The bot fetches the URL a Metaculus question names as its resolution source and puts the page
text in front of every forecaster as "primary grading evidence"
(`metaculus_bot/research/resolution_source.py`). Between 2026-07-26 and 2026-08-28 that fetch
failed on 40 of 96 cited URLs (41.7%), and on 23 of the 71 questions that cited a URL, every
cited URL failed (`scratch/next_season_bundle_2026-09/item13/archive_fetch_status_tally.txt`).
The failures fall into a few stable classes. Federal sites behind an Akamai edge answer our CI
runners with HTTP 403 every time: cdc.gov 3 of 3, bls.gov 4 of 4, and in the raw archive
fsis.usda.gov 3 of 3. Single-page applications answer 200 with an empty shell: ocearch.org 2 of
2, kasa.go.kr 2 of 2. Two hosts answered HTTP 202, which the status table files under `error`
beside socket timeouts. And rigcount.bakerhughes.com timed out on all four direct fetches in one
run while Google's own fetcher read it fine.

I propose three additions above today's direct fetch. Each one fires on a fetch status the
provider already records, runs inside the provider's wall clock, and records why the direct
fetch failed in a way that separates our fetcher from our egress from the host.

1. A meta-refresh rider on the direct fetch. cdc.gov's cited surveillance URL is a 340-byte stub
   that redirects with `<meta http-equiv="refresh">`, which our redirect loop doesn't follow. The
   target page holds the resolving numbers, but trafilatura strips their labels. So the rider has
   to carry label and value together: question 44873 gets a self-describing "48 plus the District
   of Columbia", and its sibling 44874 would get a bare "2".
2. A derived-API rung for JavaScript shells. The OCEARCH tracker's own JS bundle names its API
   base and map id in cleartext, and one unauthenticated JSON GET returns the resolving shark
   count (398). This has the same shape as the Datawrapper CSV hop that shipped 2026-08-25, and
   it should be built the same way: a small registry of known vendors, shape-validated ids, and
   the existing SSRF boundary applied to every derived URL.
3. A url_context rung. Gemini reads the URL from Google's network through the same `url_context`
   tool that gap-fill v2's `read_document` already uses in production. It runs on the personal
   Google credential, so it survives a drained donated OpenRouter credential. In the archived
   runs it's the only route that read Baker Hughes, and the only route besides search snippets
   that read cdc.gov.

My recommendation is to build the first two now and the third behind its own flag. The first two
are deterministic, spend nothing, and can be tested end to end from fixtures. The third should
wait for the two-call url_context probe that has sat unrun since 2026-08-03. At current volumes
the third rung would spend under a dollar a month. Its risks are latency inside a 45-second
provider wall and the question of whether a model-mediated read may stand as primary grading
evidence. Both are addressed below, and both are the operator's call.

## Where the fetcher stands today

`resolution_source_provider` (`metaculus_bot/research/resolution_source.py`) pulls URLs out of a
question's resolution criteria and fine print. It drops Metaculus self-references and the FRED
and Yahoo URLs that other providers own, caps the list at `RESOLUTION_SOURCE_MAX_URLS` (5), and
fetches the rest in parallel with browser-like headers, one request at a time per host. Each URL
gets a `FetchResult` whose `status` is one of nine tokens defined in
`metaculus_bot/research/resolution_fetch_result.py`: `success`, `blocked`, `not_found`,
`js_wall`, `error`, `unsupported_type`, `ssrf_blocked`, `stale_data`, `empty_body`. The provider
renders the successes as `### <url>` sections under a "primary grading evidence" caveat, names
the failures in a trailing note, and hands provider diagnostics a per-domain map of outcomes
(`_fetch_result_sources`). The whole provider runs under `RESOLUTION_SOURCE_WALL_TIMEOUT` (45 s).
If that fires, the provider returns an empty string and every page it had already fetched is
lost. Each request has its own 20-second timeout (`RESOLUTION_SOURCE_HTTP_TIMEOUT`).

The provider already has one escalation, and it's the template for everything below. When a
fetched page's raw HTML embeds a Datawrapper chart, `_fetch_datawrapper_dataset` derives the
chart's live "Get the data" CSV URL and fetches it as a second network phase. Four properties of
that hop carry over to any new rung. First, the derived URL comes from a shape-validated id
through a fixed template (`datawrapper_live_data_url` in `metaculus_bot/research/http_fetch.py`
rejects anything but a 5-character alphanumeric id). It then passes the same `is_public_http_url`
preflight as a cited URL, and the session's connect-time `FilteringResolver` stays the real
anti-rebinding boundary. Second, the hop's budget is whatever the wall has left minus a 2-second
margin; it's skipped outright below a 3-second floor, and its own timeout returns the Tier-1
pages instead of letting the outer wall cancel the provider. Third, its result sits directly
after the page that embedded it, draws on its own character allowance so it can't evict cited
page text, and appears in diagnostics under its own label (`datawrapper:<chart_id>`). Fourth, it
serves live data or nothing: the body has to be row-shaped before any freshness stamp renders,
and anything older than `RESOLUTION_SOURCE_DATAWRAPPER_MAX_AGE_DAYS` or undatable is withheld as
`stale_data`.

Next door, gap-fill v2's `fetch` tool (`metaculus_bot/research/agentic/tools.py`) already runs a
ladder. It tries an in-process cache, then a plain aiohttp GET through the same SSRF guard, then
a headless Chromium render when the plain fetch extracts fewer than
`GAP_FILL_V2_MIN_CONTENT_CHARS` (500) characters, then `read_document` for PDFs and images.
`read_document` calls Gemini with `tools=[{"url_context": {}}]` on `GAP_FILL_V2_READER_MODEL`
using the personal `GOOGLE_API_KEY` (`_run_document_read_sync` in
`metaculus_bot/research/agentic/tool_backends.py`). It refuses any answer where Gemini's
`url_context_metadata` reports zero successful retrievals and logs
`AGENTIC_DOCUMENT_UNGROUNDED_SUPPRESSED` when it does. Three things stop that ladder from covering
the resolution source. The driver decides which URLs to read, and it only targets the resolver
when its own dry run says so. The loop runs on the donated OpenRouter credential, so on
2026-07-26 it died at step zero on both cyclosporiasis questions while the resolution-source
provider kept running (`scratch/residual_2026-08-31/dossiers/44842_verification.md` item 5). And
a 403 stops it cold: `fetch` returns `blocked` without trying Chromium or `read_document`
(`tools.py`, the `plain.status == "blocked"` branch), so the driver has to choose `read_document`
itself, as it did on 44842. Its output also lands in a findings artifact, away from the section
forecasters are told is the grading evidence.

## What the archive and the three worked cases say

I re-ran the 44873 verification's sweep over the research archive and extended it per host
(`archive_fetch_status_tally.py` and `raw_archive_fetch_tally.py` in
`scratch/next_season_bundle_2026-09/item13/`, local files only). Of 1,069 `latest/` records, 181
are schema-v2 artifact records, and 71 of those carry resolution-source outcomes for 96 cited-URL
attempts:

| status | attempts | share |
|---|---:|---:|
| ok | 56 | 58.3% |
| blocked | 17 | 17.7% |
| js_wall | 10 | 10.4% |
| error | 9 | 9.4% |
| not_found | 4 | 4.2% |

The raw archive (`backtests/research_archive/raw/`, 89 resolution-source records, 116 per-URL
results) adds the HTTP status behind each token. Every `blocked` is a 403 and every `js_wall` is
a 200. `error` splits into 5 with no HTTP status at all (timeouts and connection failures) and 4
with HTTP 202 (fts.unocha.org and ballotpedia.org, two each). Hosts that never succeeded across
two or more attempts: bls.gov (5 of 5 blocked), cdc.gov (4 of 4 blocked), ocearch.org (3 of 3
js_wall), fsis.usda.gov (3 of 3 blocked), tracxn.com (3 of 3 blocked), washingtonpost.com (2
timeouts), trueup.io, sagaftra.org and congress.gov (2 of 2 blocked each), kasa.go.kr (2 of 2
js_wall), and the two 202 hosts. The only mixed hosts are en.wikipedia.org (11 ok, 2 not_found)
and ogimet.com (2 ok, 2 js_wall). A few hosts fail the same way every time, and that stability
is what makes a status-keyed escalation worth building.

Three verified cases fix the design. On 44872 (OCEARCH shark count, resolved 398, spot peer
−38.81), the tracker page is a Mapotic white-label map. The plain fetch returned 200 with
133,670 bytes and zero occurrences of the counter, and a Chromium render produced 385,141
characters with no counter either. But the page references a single JS bundle whose cleartext
config reads `baseApiUrl:"https://www.mapotic.com/api/v1/"` and `ocearchMapoticMapId:3413`, and
`https://www.mapotic.com/api/v1/maps/3413/public-categories/` returns `Sharks: 398` as
unauthenticated JSON (`scratch/residual_2026-08-31/dossiers/44872_verification.md` item 1). I
re-checked on 2026-09-01. The shell is 200 and still names `main.f98087fa12338672.js`. The bundle
is 2,092,964 bytes of `text/javascript` and still carries both config strings. The JSON endpoint
answered in 0.65 s with 2,474 bytes and `Sharks: 398`
(`scratch/next_season_bundle_2026-09/item13/mapotic_public_categories_2026-09-01.json`). The
value the question resolved on sat one derivable GET away the whole time.

On 45218 (Baker Hughes rig count, spot peer +35.19), the resolution-source provider recorded
`rigcount.bakerhughes.com: error` after 20,107 ms, which is the HTTP timeout. The dossier read
that as a Tier-1 defect. The verification found the run's own gap-fill v2 transcript: the driver's
plain `fetch` hit `SocketTimeoutError` on four different URLs on that host, and only
`read_document`, a fetch made from Google's network, read the page
(`scratch/residual_2026-08-31/dossiers/45218_verification.md` item 5). Raising our timeout or
fixing our parser would have changed nothing, because the host wasn't answering our egress. This
is the case that forces the failure-reason recording below.

On 44873 and 44874 (cyclosporiasis states and deaths, same CDC page), the cited URL
`www.cdc.gov/cyclosporiasis/php/surveillance/index.html` was 403 on all three archived attempts.
One of the three was a healthy full-roster run, so the block has nothing to do with the dry
donated credential (`scratch/residual_2026-08-31/dossiers/44873_verification.md` item 3). When the
verifier's laptop got a 200, the body was a 340-byte stub with a `<meta http-equiv="refresh">` to
`cyclosporiasis/cases/index.html`, which our fetcher classifies as `js_wall` (200, zero extracted
text). Following the refresh yields `success` with 5,397 characters containing the resolving
phrase, so the rider is real. But trafilatura renders the stat block as `17,180 2 48 plus the
District of Columbia` with every label gone and the hospitalization count dropped entirely
(items 4 and 6 of that verification). On 2026-09-01 the same two cdc.gov URLs answered my laptop
with 403 from `server: AkamaiGHost` under the repo's own `BROWSER_HEADERS`
(`scratch/next_season_bundle_2026-09/item13/cdc_surveillance_403_akamai_body_2026-09-01.html`),
where the verifier got 200 with the same headers a day earlier. The edge judges the client, and
our code was identical both days.

The url_context route has its own record. Across 102 gap-fill v2 loops that ran between
2026-07-20 and 2026-08-28, the driver issued 160 `read_document` calls, and the
`AGENTIC_DOCUMENT_UNGROUNDED_SUPPRESSED` marker fired 42 times over all `read_document`
invocations in that window, driver-issued and auto-escalated together
(`scratch/next_season_bundle_2026-09/item13/telemetry_read_document_tally.txt`). So at least 118
of 160 driver reads retrieved something. The 42 misses name congress.gov 4 times and trueup.io 3
times, both hosts that also 403 our direct fetch, and cdc.gov once. Google's fetcher is a
different egress with a different reputation, which helps on Akamai-fronted federal sites and on
Baker Hughes. A host that blocks non-browser fetchers by fingerprint can still block Google's.
The rung gives us a second route, and the design has to record which hosts it fails on so the
list of hosts nobody reaches can grow from evidence.

## The ladder

Each URL runs the rungs in order, and each rung fires only on the status the previous one
produced. A rung that succeeds ends the chain for that URL. The chain for one URL reads:

1. Direct fetch, as today.
2. If the result is `js_wall` and the raw body carries a meta refresh, follow it as one more
   redirect hop and re-run the direct fetch on the target.
3. If the result is still `js_wall`, or is item 11's `no_resolving_content`, try the derived-API
   rung.
4. If the result is `blocked`, an egress-class `error`, or a `js_wall` or `no_resolving_content`
   the derived-API rung couldn't resolve, try url_context.

A result from a later rung sits directly after the direct-fetch result it escalated, the way a
Datawrapper dataset sits after its parent page. It carries a `route` field naming how it was
obtained (`direct`, `meta_refresh`, `derived_api`, `url_context`), so the section lead, the
diagnostics map, and the telemetry all say the same thing. A URL that fails every rung keeps its
direct-fetch status and gains the escalation record, so the failure notice forecasters see can
say which routes were tried. Each of the three additions has its own trigger, its own budget,
and its own failure token, described next.

### Rider on the direct fetch: meta refresh

When an HTML 200 extracts to fewer than `RESOLUTION_SOURCE_JS_WALL_MIN_CHARS` characters and the
raw body contains `<meta http-equiv="refresh" content="N;URL=...">`, the fetcher should treat the
URL as a redirect hop. Concretely, `_resolution_html_outcome` detects the tag on the raw decoded
body it already scans for Datawrapper embeds, resolves the target with `urljoin` against the
current URL, and returns it as the next hop instead of a `js_wall` result. The hop then goes
through exactly what a `Location` header goes through in `_resolution_redirect_outcome`: the
`is_public_http_url` preflight, the Metaculus self-reference refusal, and the shared
`MAX_REDIRECTS` cap (5, `http_fetch.py`). The refresh delay value means nothing to us and should
be ignored. A refresh whose target is the same URL, or a body that carries the tag beside
substantial text, is a real page and should be left alone; the trigger is a near-empty extraction
paired with the tag.

Following the refresh isn't enough on its own, because the target page's stat block is an ARIA
table built from `<div>` elements. The Wayback capture of the CDC cases page (2026-08-29,
`scratch/next_season_bundle_2026-09/item13/cdc_cases_stat_block_markup_wayback_20260829234445.html`)
renders each figure as `<div role="row"><div role="rowheader">Deaths</div><div role="cell"><p>2
</p></div></div>` inside a `<div class="table dfe-table" role="table">`. Trafilatura's
`include_tables=True` handles `<table>` elements. On these divs it keeps the cell text that
happened to sit inside a `<p>` and drops both the row headers and the one value (`922`) that had
no `<p>` wrapper. That is the exact mechanism behind the unlabeled `17,180 2 48 plus...` strip.
So the rider needs a second half. Before extraction, rewrite ARIA-table markup
(`role="table"`, `role="row"`, `role="rowheader"`, `role="cell"`, and the
`columnheader`/`gridcell` variants) into real `<table>`, `<tr>`, `<th>` and `<td>` elements, so
trafilatura's table path keeps label and value on one row. Extracting label/value pairs straight
from the raw HTML and prepending them as a "Stat block" list would also work. Rewriting is the
better choice because it keeps one extraction path and lets the existing truncation and
tag-stripping code run unchanged. The test that matters is 44874's: a fixture built from that
capture must yield `Deaths` adjacent to `2` and `Hospitalizations` adjacent to `922`, and the
travel-associated block's `Deaths 0` must stay distinguishable from the domestic one.

Two limits should be stated plainly. From the CI runners this rider never reaches cdc.gov today,
because the 403 arrives before any body does. Its proven value is on hosts that serve the stub,
and its practical value on cdc.gov is that it tells the url_context rung which page to read. And
I don't know whether Google's fetcher follows a meta refresh. The 44842 transcript shows
`read_document` reading the surveillance URL on 2026-07-25, but I can't tell whether the stub
existed then. Running the rider before the url_context rung and handing that rung the refresh
target removes the question.

### Derived-API rung: read the page's own JavaScript config

A single-page application's shell is a 200 with almost no text, which the fetcher already labels
`js_wall`. The data the shell renders comes from an API that the shell's JavaScript is
configured to call. For white-label map and dashboard vendors, that configuration is a few
cleartext name/value pairs in the main bundle. The rung reads those pairs and issues the GET the
browser would have issued. It is registry-driven. A `DerivedApiVendor` entry names the vendor
and the regular expressions that pull the API base and the collection id out of a bundle. It
also fixes the id's allowed shape, the URL template(s) to build, the host the template may
produce, and a validator for the response's JSON shape. Mapotic is the first entry: config names `baseApiUrl` and
`<prefix>MapoticMapId`, id shape `\d{1,8}`, template `<base>maps/<id>/public-categories/`, host
`www.mapotic.com`, and a validator that requires a JSON array of objects each carrying `name.en`
and `stats.pois_count`. A second template for `pois.geojson/` isn't needed for the count and
costs a far larger body; leave it out until a question needs per-animal rows.

The flow for a `js_wall` (or `no_resolving_content`) result:

1. Scan the raw HTML for same-origin `<script src>` references and any inline `<script>` text.
2. For each registered vendor, try the inline text first, then fetch at most two same-origin
   bundles in document order. Each bundle fetch goes through the standard fetch path with the
   5 MiB `RESOLUTION_SOURCE_MAX_RESPONSE_BYTES` cap (the OCEARCH bundle is 2.09 MB) and is read
   only when the response is `text/javascript` or `application/javascript`.
3. On a config match, build the derived URL from the vendor's template, run it through
   `is_public_http_url`, and GET it under the vendor host's politeness semaphore.
4. Validate the JSON shape. A body that fails validation is withheld.
5. Render the raw JSON capped at `RESOLUTION_SOURCE_PER_URL_MAX_CHARS` under a lead such as
   "Derived from the page's embedded Mapotic configuration (map 3413): live `public-categories`
   JSON, fetched <timestamp>. The cited page itself rendered no readable text."

Raw JSON passthrough is deliberate. The Tier-1 JSON branch already renders API bodies verbatim
(the CISA KEV feed is the sizing example in `constants.py`), so forecasters are used to reading
a JSON body in this section. A deterministic body under a provenance lead beats a summary the
fetcher invented, and an eight-entry categories array with names and counts needs no
translation. The Mapotic categories body is 2.4 KB, well inside the cap.

Freshness needs a different guard than Datawrapper's. That hop reads a static CDN file whose
`Last-Modified` is the dataset's publish time, so an old stamp means stale data. An API response
is generated per request: the Mapotic body I fetched carried `Last-Modified` equal to the
response time and `server: cloudflare`. The right check here is the shape validator plus the
fetch timestamp in the lead. There's no publish date to gate on, and pretending there is would
withhold every response. What the rung must never do is fall back to a page-pinned or cached
variant of the API, for the same reason the Datawrapper hop refuses the versioned `dataset.csv`
route.

Infogram, the embed behind the 44554/44556 Nebraska polling miss, is the case the registry
approach doesn't reach. The embed URL `e.infogram.com/_/<id>` serves 636,542 bytes with 38 chart
definitions whose `data` arrays are empty and one `atlas_google_drive` live provider. The numbers
live in a Google-Drive-backed feed the shell resolves at runtime
(`scratch/residual_2026-08-31/dossiers/44554_verification.md`, the Infogram wall row). No URL
template I can derive from the shell reaches that feed. Item 11's `no_resolving_content` status
will at least make the failure visible. Whether a url_context read of the embed URL recovers the
numbers is an experiment for the validation phase; I wouldn't assume it.

### url_context rung: read from Google's network

This rung applies to a URL whose direct fetch ended `blocked`, ended `error` with no HTTP status
(a timeout or connection failure), or ended `js_wall` or `no_resolving_content` with no derived
API. The provider asks Gemini to read the URL with the `url_context` tool alone. The call mirrors
`_run_document_read_sync`: the personal `GOOGLE_API_KEY`, `tools=[{"url_context": {}}]`, no
`google_search` tool (so the call can't wander onto other pages), a client-side HTTP timeout, and
the shared `extract_url_context_telemetry` reader to count successful retrievals. Zero
retrievals is a failure and gets its own status, `ungrounded`, terminal to this rung the way
`stale_data` is terminal to the Datawrapper hop; the text is discarded because it can only be
recall. The ask is fixed and shaped by the resolution criteria, which are passed in: quote
verbatim the passages that state the quantity or condition the criteria name, report the page's
own as-of or last-updated date if it states one, report the section headings you saw, and add
nothing you didn't read. The model is the one `read_document` already runs in production. The
constants comment calling that model id unverified predates the archived transcripts that show it
reading Baker Hughes and cdc.gov (see the adjacent observations at the end).

The rendered section has to say what it is. A lead such as "Read via Gemini url_context
(Google's fetcher) because our direct fetch returned `blocked` (HTTP 403 from an Akamai edge).
The text below is the model's verbatim extraction from the page." puts the forecaster on notice
that a model stood between the page and the bundle. The repo's rule that character caps apply to
raw passthrough and never to LLM-emitted research exists to keep briefings whole. This text sits
in the raw-evidence section under a per-URL cap, and the ask bounds it anyway, so I'd apply
`RESOLUTION_SOURCE_PER_URL_MAX_CHARS` to it like any other section and let the truncation marker
fire if it must. Whether a model-mediated read may sit under the same "primary grading evidence"
heading as a direct fetch is the operator's call (open question 4). If the answer is no, the
section can render under its own caveat line and still count as `success` for diagnostics.

## Which statuses are seams and which are terminal

The classification principle is simple: a status is a seam when a different client, a different
route, or a derived URL could plausibly read what the host holds, and terminal when the host has
stated the content is gone, when the body carried nothing from any client, or when one of our
own safety caps fired as designed. The `error` row splits on information the result already
carries (`http_status` is on `FetchResult`) plus one field the fetcher currently throws away, the
exception class. No token is renamed. `resolution_fetch_result.py` says every status string is a
telemetry token and changing one is a breaking change, so the split lives in additive fields,
and the only new status token is `ungrounded`.

| status | class | why |
|---|---|---|
| `blocked` (403, 406, 429) | seam | the host refused our client; a different client may be admitted (cdc.gov, bls.gov via url_context) |
| `js_wall` (200, under 100 extracted chars) | seam | the content exists behind script; meta refresh, derived API, or a rendering fetcher can reach it |
| `error`, no HTTP status | seam | a timeout or connection failure is an egress or host event, and url_context is a second egress (45218) |
| `error`, HTTP 202 | seam | an accepted-but-not-served response is a challenge page in practice; treat like `blocked` (mechanism unverified; 2 hosts) |
| `no_resolving_content` (item 11) | seam | the page rendered prose but the resolving figure sits in an embed; derived API or url_context may reach it |
| `error`, other HTTP status (5xx, redirect chain exhausted, oversized body, malformed redirect) | terminal | a 5xx is the host's own failure; the other three are our caps working as designed |
| `not_found` (404, 410) | terminal | the host states the resource is gone; another fetcher would fetch the same 404 |
| `empty_body` | terminal | a 200 with nothing in it carries no information from any client |
| `unsupported_type` | terminal for now | covers both PDFs we choose not to read and bodies we couldn't decode; zero occurrences in the archive, so no evidence to split it yet (open question 6) |
| `ssrf_blocked` | terminal | a private, loopback, or unresolvable host must never be handed to any fetcher, including Google's |
| `stale_data` | terminal | Tier-2 only; the freshness guard refusing to serve old data is the feature working |
| `ungrounded` (new, url_context only) | terminal | Gemini retrieved nothing; the answer would be recall |

Two rows deserve a note. The HTTP 202 row rests on two hosts and on my reading that a 202 with
no served content is an anti-bot challenge page; I haven't confirmed the mechanism, so the row
should be revisited once the failure-class field below has recorded a few more. The
`unsupported_type` row is terminal by default only because nothing in the archive has hit it. A
cited PDF would land there today, and url_context reads PDFs well on the agentic path, so this
row is the most likely to move.

## Recording why a fetch failed

The 45218 lesson is that `error` alone can't tell a fetcher bug from an egress property from a
host outage. The fix is to record the evidence at the moment the failure happens instead of
reconstructing it from transcripts a month later. I propose three additive fields on
`FetchResult`: `failure_class` in {`fetcher`, `egress`, `host`, `unknown`}, `exception` (the
class name, e.g. `TimeoutError`, `ClientConnectorError`, `ServerDisconnectedError`), and
`server_header` (the response's `Server` value when there was a response). Classification at
fetch time is mechanical:

- No response at all is `egress` or `host` and can't be separated yet, so it's recorded as
  `egress` with the exception name and revised below.
- A 403 or 202 whose `Server` header or body names a WAF edge (`AkamaiGHost`, `cloudflare`, an
  `x-reference-error` header, an "Access Denied" title with a reference number) is `egress`,
  because the edge is judging our client.
- A 403 from an origin server with no edge signature stays `unknown`.
- A 5xx, 404 or 410 is `host`.
- An oversized body, an exhausted redirect chain, an unsupported type, or a `js_wall` on a body
  with substantial script but no text is `fetcher`: our code chose not to, or couldn't, read what
  the host served.

The revision step is what makes the field worth having. When the url_context rung later succeeds
on a URL whose direct fetch was `egress` or `unknown`, the direct result's class becomes `egress`
with `escalation_recovered=true`. When url_context also fails, the class becomes `host` if Google
reported a fetch error and stays `egress` if Google was blocked too. Over a season that turns the
per-host table above into three lists: hosts we can't reach but Google can (escalate), hosts
nobody reaches (stop spending on them), and hosts our own parser fails on (fix the parser).

Every field should ride item 19d's `RESOLUTION_SOURCE_FETCH` marker, which would grow `route=`,
`failure_class=`, `exc=`, and `server=` segments so the archive can answer the 45218 question by
query. Each escalation attempt should emit its own line, `RESOLUTION_SOURCE_ESCALATION:
question=<id> url=<url> from_status=<status> rung=<meta_refresh|derived_api|url_context>
outcome=<success|<status>> wall_s=<n>`, with a `MarkerSpec` in `scripts/telemetry/markers.py`.
The same fields belong in the raw research archive, which already stores the full `FetchResult`
list per run. One mislabel is worth fixing on the way: `is_public_http_url` returns False on a DNS
failure, so a host that doesn't resolve is recorded as `ssrf_blocked` (its docstring says so). A
dead domain and a private-IP attempt are different facts, and the `failure_class` field gives the
dead domain a home (`host`) without touching the status token.

## Budget, latency, and cost per rung

The provider runs under a 45-second wall, and the orchestrator's own comment says why raising it
costs little. `resolution_source` is one of the "cheap hard-capped providers" that run
concurrently with the primary, whose configured worst case is 600 s and measured worst case
110 s (`metaculus_bot/research/orchestrator.py`, `_select_research_providers`). Nothing under
110 s lengthens the research phase in practice. The archive shows how much of the 45 s a direct
fetch uses (`scratch/next_season_bundle_2026-09/item13/rs_latency_tally.txt`). When every cited
URL succeeds, the median is 0.7 s and the worst 10.2 s. When every cited URL fails, the median is
0.3 s (a 403 is instant) but 3 of 23 ran the full 20 s timeout. Questions cite 1.35 URLs on
average and never more than 3 in this window.

| rung | trigger | network and model calls | observed or bounded latency | marginal cost |
|---|---|---|---:|---|
| direct fetch | always | 1 GET per URL, redirects in-band | p50 0.7 s; 20 s on timeout | none |
| meta refresh | `js_wall` with the tag | 1 extra GET, counted as a redirect hop | one more request; same 20 s cap | none |
| derived API | `js_wall` or `no_resolving_content` | up to 2 bundle GETs (2.1 MB observed) plus 1 JSON GET (0.65 s observed) | a few seconds; each under the 20 s cap | none |
| url_context | `blocked`, egress `error`, unresolved `js_wall`/`no_resolving_content` | 1 Gemini call per URL on the personal credential | 12 s observed on cdc.gov (44842); bounded by a 55 s client timeout in the agentic path | cents per call |

The cost figures rest on the billing notes in `CLAUDE.md`, which were verified there against
ai.google.dev on 2026-07-17; I didn't re-fetch them. url_context carries no per-request fee, the
retrieved document bills as input tokens, and the reader model's tokens run $1.50 per million in
and $9 per million out. A 6,000-character page is about 1,500 input tokens, so a read costs well
under a cent even with the prompt. Volume is bounded by the failure rate. The 40 non-ok attempts
over this 5-week window come to roughly 8 escalations a week, so the rung would spend under a
dollar a month at that rate, far from the prepaid-credit behaviour the `CLAUDE.md` notes warn
about (exhaustion shows up as 429s). Because the call carries no `google_search` tool, my reading
of the billing notes is that it draws nothing from the 5,000 grounded-prompts-per-month pool. The
probe below records `n_web_search_queries` per call, which is the cheap way to confirm that.

The latency risk lives inside the 45-second provider wall. If a direct fetch burned its 20 s
timeout and a url_context read takes 12 to 30 s, the current wall leaves no room. The Datawrapper
pattern (inner budget equal to the wall minus elapsed minus margin, skipped below a floor) would
then skip the escalation on exactly the timeouts it exists for. Two knobs follow. First, mirror
the Datawrapper budgeting for every rung, with a floor of at least 15 s for url_context, so a
too-late escalation is skipped and logged instead of started and cancelled. Second, raise
`RESOLUTION_SOURCE_WALL_TIMEOUT` to 90 s whenever url_context reads are enabled. That is still under the
primary's measured worst case, so the expected phase cost is zero, and it leaves a timed-out
direct fetch with about 65 s for the read. The close-derived time budget already thins research
on short windows. Its fast path keeps `resolution_source` because the provider is cheap, so the
escalation rungs should switch off whenever `fast_path` is set (the orchestrator holds that flag
at `_select_research_providers(fast_path=...)` and can pass it into the provider factory), and
the provider should also skip url_context when the remaining research-phase budget is below the
floor.

## SSRF posture for derived URLs

Whoever wrote the question chose the cited page, and whoever runs the site wrote its HTML and
JavaScript, so every URL the ladder invents is attacker-influenced. The module comment in
`resolution_source.py` states the threat: fetches run on GitHub's runners, where a request to the
instance metadata service or an internal host would exfiltrate runner state into the research
prompt and the public Metaculus comment. The existing boundary already covers the shape of every
new request. A meta-refresh target and a derived API URL go through `is_public_http_url` before
any I/O, the same preflight a `Location` header gets. Both are fetched on the same aiohttp
session whose `TCPConnector` resolves through the `FilteringResolver`, so a rebinding host is
caught at connect time no matter what the preflight saw. The rule for the implementer is that no
derived URL may take a code path that bypasses `_fetch_one_hop`'s session; a second session
without the resolver would reopen the hole.

The derived-API rung adds a constraint the redirect path doesn't need. A redirect target is one
URL the server chose. A JavaScript bundle is megabytes of strings. A rule of "find URLs in the
bundle and fetch them" would let any page we're pointed at direct our runners to arbitrary public
hosts, and would put those hosts' bodies in front of forecasters as grading evidence. That is why
the rung is registry-driven. The vendor entry fixes the template and the host it may produce; the id is
shape-validated before it touches the template, exactly as `datawrapper_live_data_url` does; and
a bundle that names `baseApiUrl:"https://attacker.example/"` produces no request, because the
template's host comes from the registry and never from the match. The bundle fetch itself is
restricted to same-origin `<script src>` targets, to JavaScript content types, and to the 5 MiB
cap, and inline `<script>` text is preferred over any fetch. The url_context rung has a different
exposure: Google's fetcher can't reach our runner's network, but we still shouldn't send a
private, userinfo-bearing, or unresolvable URL into a prompt. So the same preflight runs before
the call and `ssrf_blocked` stays terminal. Tests for all of this are offline and cheap; the
validation plan lists them.

## Prerequisite: the url_context positive-control probe

`scratch/urlctx_probe_2026-08-03/probe.py` (in the main checkout) makes exactly two paid Gemini
calls on the personal `GOOGLE_API_KEY` and has never been run. The positive control builds the
repo's real research prompt around a question that names a Wikipedia URL and calls
`invoke_gemini_grounded` with both `google_search` and `url_context` enabled. The negative control
runs the production `gemini_search_provider` on an ordinary question with no URL. For each call
the probe captures the SDK's `url_context_metadata`, the grounding-chunk and search-query counts,
and whether the provider's `### URL Context Fetches` marker appeared, and it writes
`probe_results.json` beside itself. Its two spies only observe: the raw-log spy stashes the SDK
response instead of writing to `backtests/`, so the probe can't pollute the research archive.

The probe is a prerequisite for a specific reason. `FUTURE.md` records that 0 of 271 archived
Gemini grounded-search sections carry the url_context marker, so on the grounded-search path the
tool has never demonstrably fired in production, and the same note says a positive control is
needed before blaming the wiring. The `read_document` path is different: it calls Gemini with
`url_context` as the only tool, and the archived transcripts show it retrieving pages (at least
118 of 160 driver-issued reads). The design above builds the rung on the `read_document` shape for
that reason. The probe still settles three things the rung depends on. Does a url_context call
with a URL named in the prompt retrieve it under the repo's own client and prompt construction?
Does such a call report zero `web_search_queries` when the search tool is also present, which is
the billing question? And what does the SDK return when retrieval fails, which fixes the
`ungrounded` classification? The cost is two grounded calls, or 2 of the month's 5,000 free
grounded prompts, and the operator decides when to fire it. I recommend firing it before any
rung-3 code is written, and extending it with a third call that runs the `read_document`-shaped
request against `https://www.cdc.gov/cyclosporiasis/cases/index.html`. That third call measures
the Akamai question directly: does Google's fetcher get through where our runners don't?

## What would justify each rung, and what would retire it

Two facts justify the meta-refresh rider: 44873's measured recovery (`js_wall` to `success`,
5,397 characters, the resolving phrase present) and a cost of one request. The rider would lose
its case if the Wayback record showed the stub pattern confined to cdc.gov and cdc.gov stayed 403
from CI. In that world it would only ever help by naming a target for the url_context rung. The
exact strip the verification quoted, and the 922 that vanished, justify the ARIA-table half, and
a fixture from the archived capture proves it offline.

One question worth 38.8 spot-peer points, a mechanism that still works today, and a shipped
precedent whose properties transfer justify the derived-API rung. The evidence against it is
that the registry has one vendor. The offline replay in the validation plan is the honest test.
Of the 12 archived `js_wall` results, the 3 on ocearch.org resolve through Mapotic; the other 9
(kasa.go.kr and ogimet.com with two each, and one each on portwatch.imf.org, bsky.jazco.dev,
forest-fire.emergency.copernicus.eu, dcas.dmdc.osd.mil and companiesmarketcap.com) need their
shells inspected for a vendor before anyone can say what a second registry entry would cover. If
none of them has a derivable API, the rung ships with one entry and earns its keep on the next
tracker question, which is a narrow justification.

The archive's dominant failure class justifies the url_context rung. 23 of 116 raw attempts are
403s concentrated on federal and finance hosts, cdc.gov alone touched three questions in the
2026-07-26 to 07-28 cluster, and Google's fetcher has read cdc.gov and Baker Hughes in archived
transcripts while our runners couldn't. Its per-call cost is negligible. Its real costs are
latency inside the wall and the epistemic question of a model-mediated read. It should be
retired, or restricted by host, if the `failure_class` revision step shows Google blocked on the
same hosts we are (congress.gov and trueup.io already look that way in the
`AGENTIC_DOCUMENT_UNGROUNDED_SUPPRESSED` tally), or if the probe shows retrievals failing under
our own client and prompt.

## Validation plan under the cost gate

Everything except the probe is offline and free. The plan follows the pattern the existing suites
already use: `tests/test_resolution_source_provider.py` and
`tests/test_resolution_source_datawrapper.py` monkeypatch `_get_session` on `resolution_source`
to serve canned responses, stub DNS through `is_public_http_url`, and run the real fetch path,
and `tests/test_agentic_tools.py` does the same for the agentic ladder. The fixtures for the new
rungs already exist on disk in `scratch/next_season_bundle_2026-09/item13/`: the 423-byte Akamai
403 body from cdc.gov with its `server` header, the CDC cases page's ARIA stat block from the
2026-08-29 Wayback capture, the Mapotic `public-categories` JSON with its response headers, and
the OCEARCH shell's `<script src>` line. The 340-byte meta-refresh stub is quoted in the 44873
verification, and the 2.09 MB bundle should be reduced to a synthetic bundle carrying only the two
config strings. The benchmarking guard stays as it is: the provider returns an empty string when
`is_benchmarking` is set, so no rung can run in a backtest.

The tests I'd require before any rung is switched on:

- The meta-refresh stub, served from a canned session, yields a second hop to the target and a
  `success` whose text pairs `Deaths` with `2` and `Hospitalizations` with `922`.
- A page carrying the refresh tag beside substantial text stays `success` unchanged, and a
  refresh to a private IP or to metaculus.com is refused.
- The OCEARCH shell plus a synthetic bundle yields a derived Mapotic result placed after the
  shell's `js_wall` result and rendered under the provenance lead.
- A bundle naming a foreign API base produces no request (asserted by counting session calls), a
  bundle over 5 MiB is dropped, and a JSON body that fails the shape validator is withheld.
- The url_context rung, with `generate_content` stubbed, handles a response reporting one
  successful retrieval (rendered under its lead), a response reporting zero (status `ungrounded`,
  text discarded), and a timeout (the Tier-1 result survives and the escalation marker records
  the failure).
- Every rung is skipped when the remaining wall budget is under its floor and when `fast_path`
  is set.
- `FetchResult`'s blank-success guard still holds for every new construction site.
- `_fetch_result_sources`, the failure notice, and the new markers carry the `route` and
  `failure_class` fields, and the marker parse tests cover both.

Two free replays give the numbers the decision needs. Over the raw archive's 116 per-URL
results, count how many would have been eligible for each rung by status and HTTP code (23
blocked, 5 egress errors, 4 HTTP 202, and 12 js_wall are the ceilings), and inspect each archived
js_wall shell for a registry vendor. Over the telemetry archive, join
`AGENTIC_DOCUMENT_UNGROUNDED_SUPPRESSED` hosts against the resolution-source failure hosts to
seed the list of hosts where Google is blocked too. Neither needs a network call. The one paid
check is the probe, extended with the cdc.gov read, at a cost of three Gemini calls. The bundle's
optional `test_bot_basic.yaml` dispatch (about $2.60, one question, publishes) would exercise
the full provider live, but it only tests the rungs if the chosen question happens to cite a
failing host, so I wouldn't count on it for this design.

## Open questions for the operator

1. Run the two-call url_context probe now, and extend it with a third `read_document`-shaped
   call against the CDC cases page? Without it the url_context rung's billing and Akamai
   assumptions stay assumptions.
2. Raise `RESOLUTION_SOURCE_WALL_TIMEOUT` from 45 s to 90 s when the url_context rung is on?
   The orchestrator's numbers say the phase cost is zero in expectation. The alternative is a
   floor that skips the rung on exactly the timed-out fetches it targets.
3. Ship the meta-refresh rider and the ARIA-table rewrite unflagged, as redirect and extraction
   correctness fixes, and put the derived-API rung and the url_context rung behind two flags
   (`RESOLUTION_SOURCE_DERIVED_API_ENABLED`, `RESOLUTION_SOURCE_URL_CONTEXT_ENABLED`), both off
   until the validation above is green? Or one flag for all three?
4. May a url_context read render under the "primary grading evidence" caveat, or does it get its
   own caveat line? The forecaster prompts tell models to weight the resolution snapshot
   heavily. A model-mediated extraction is a different kind of evidence, and the lead text says
   so, but the caveat is the operator's promise.
5. Should the derived-API registry stay Mapotic-only until a second vendor shows up in the
   archive, or is the js_wall replay (9 non-OCEARCH shells) worth doing before the code lands?
6. `unsupported_type` covers cited PDFs, which url_context reads well in the agentic path. Split
   the token (a breaking change to a telemetry string) or leave PDFs terminal until one costs a
   question?
7. Is a Google-mediated read of a federal site acceptable at all as the bot's evidence, given
   the sites are refusing automated clients? Baker Hughes and cdc.gov publish these figures for
   the public and the model quotes them verbatim, but this is a posture question for the
   operator.

## Adjacent observations outside this design

While reading the code against the receipts I found four things worth a note. The comment on
`GAP_FILL_V2_READER_MODEL` in `metaculus_bot/constants.py` says the model id is unverified on
the native AI Studio SDK until a paid smoke test; the 45218 and 44842 archived transcripts show
`read_document` retrieving pages on it in production, so the comment is stale. The status table
maps HTTP 202 to `error`, which files a challenge page with socket timeouts; the `failure_class`
field above resolves that without renaming the token. `is_public_http_url` records a DNS failure
as `ssrf_blocked`, a misleading name for a dead domain. And the agentic `fetch` ladder returns
`blocked` on a 403 without trying Chromium or `read_document`. That is a deliberate choice (the
driver can call `read_document` itself), but it means the agentic ladder is no automatic
escalation for the 403 class, which is why this design doesn't lean on it.

## Receipts

- Plan: `scratch_docs_and_planning/next_season_bundle_2026-09_plan.md` §3 item 13, item 11,
  item 19d.
- Cross-dossier: `scratch/residual_2026-08-31/DOSSIER_SYNTHESIS.md` §6 pattern 11 and §7
  ("Tier-2 escalation" bullet); `scratch/residual_2026-08-31/SYNTHESIS.md` §6 free item 2 and
  paid item 2.
- Cases: `scratch/residual_2026-08-31/dossiers/44872_verification.md` (Mapotic; its probe
  `44872_verify_fetchability.py` and stdout), `45218_verification.md` (Baker Hughes, item 5),
  `44873_verification.md` (meta refresh, items 3 to 6 and 8b; scripts
  `44873_verify_metarefresh.py`, `44873_verify_fix_efficacy.py`), `44842_verification.md`
  (read_document recovery timing, item 5), `44554_verification.md` (Infogram).
- Code: `metaculus_bot/research/resolution_source.py`, `resolution_fetch_result.py`,
  `http_fetch.py`, `agentic/tools.py`, `agentic/fetch_outcomes.py`, `agentic/tool_backends.py`,
  `gemini_search.py`, `url_context_telemetry.py`, `orchestrator.py`; `metaculus_bot/constants.py`
  (`RESOLUTION_SOURCE_*`, `GAP_FILL_V2_*`, time-budget block); `docs/agentic_gap_fill.md`;
  `FUTURE.md` (resolution-source Tier-2 entry and the url_context verdict paragraph).
- Probe: `scratch/urlctx_probe_2026-08-03/probe.py`.
- This document's tallies and fixtures: `scratch/next_season_bundle_2026-09/item13/` (four
  scripts with their `.txt` outputs, the CDC 403 body and stat-block markup, the Mapotic JSON and
  headers). All tallies read local archive files only. The four live checks were plain GETs of
  public pages from a laptop on 2026-09-01 and touched no LLM, research, market, or Metaculus API.
