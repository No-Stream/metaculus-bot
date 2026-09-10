# One fetch ladder for both callers: progress log

Branch `fall/ladder` in the worktree `/Users/flatljan/personal/metaculus-bot-wt/ladder`, cut from
`mantic-competition` at `6f2f051`. The plan this executes is
`scratch_docs_and_planning/fetch_ladder_unification_plan_2026-09-09.md`; read it first, then this
file for what has actually landed and what was decided along the way. A successor with no context
can resume from the current status below; the later sections retain the migration history.

Free gates only. Nothing in this work may spend money: no `main.py` run, no backtest, no probe, no
GitHub Actions dispatch. `make test` is network-blocked by an autouse fixture and is always free.

## Current status: migration complete, production wiring follows merge

The recovered branch completed steps 4, 5 and 6 on 2026-09-10. The verified code commit is
`4b2a368d58acfeb25ee8c8ee7101e297a70b2f57`; HEAD was unchanged across the final gate.
`make test` passed 9,854 tests, with 39 skips and 5 live deselections, in 284 seconds. Ruff,
format checking, basedpyright, deptry and all six import contracts passed. An independent review
of the complete diff against `6f2f051` found no remaining unaccepted material production bug.
The requested `/forge` command was unavailable here, so independent source review and executable
regression checks supplied that review gate.

Step 4 enabled remembered derived-feed reuse for gap-fill. Step 5 added the complete-read cache,
shared throttle verdict and injectable async digest seat. The cache preserves each caller's
verdict, cap, query and acquisition route; it never reuses an old attempt list or paid answer.
The digest receives the wall remaining after extraction, preserves source text when selection is
empty, and keeps PDF page labels. A short flat document uses a separate local query-match check,
so model grounding counters cannot decide whether a local answer is available.

Step 6 removed the obsolete agentic transports and classifiers. The earlier suggestion that all
14 old browser tests were redundant was checked and corrected: unique setup, concurrency,
subrequest, teardown, link and caller-specific decline assertions were migrated before deletion.
The separate host-semaphore scopes remain the approved decision.

Review regressions now cover cached robots directives, cache eviction during presentation,
acquisition routes after failed rescues, throttle propagation across source-host rungs, gap-fill's
impersonated thin/empty browser rescue, and unreadable PDF parse reuse. Resolution-source's
unreadable impersonated response retains its original offsite fallback. A throttle from Wayback
declines that archive rung because it describes the archive host, not the cited host. The recorded
rendered-document limitation remains accepted; it was not silently expanded or repaired.

Next: finish the full archived trigger/order replay, merge into `mantic-competition`, then bind
the existing known-API and `page_digest` implementations there. The digest seat intentionally
remains injectable/default BM25 on this intermediate branch. Final production wiring needs its
own offline integration and CI-equivalent gate before any paid smoke is proposed.

## Design decisions taken before the first commit

These are the calls the plan left open or under-specified. Each one is reported to the lead.

### The package has eight modules, not five

The plan's table names `policy.py`, `context.py`, `classify.py`, `rungs.py` and `ladder.py`. It
gives no home to the SSRF guard (`is_public_http_url`, `_ip_is_disallowed`,
`resolve_vetted_public_ip`, `_hop_refusal`, `_landing_refused`, `_vetted_hop_target`) or to the
aiohttp session and per-host gate (`_get_session`, `_sem_for_host`), and those are exactly the
names that make the extraction a cycle problem: `rendered_fetch` and `impersonated_fetch` both
reach into `resolution_source` for them through function-scoped imports whose stated reason IS the
cycle. So the package gets a leaf `guard.py` holding all of them, plus its own `import socket` for
the DNS stub the test package patches. `rendered_fetch` and `impersonated_fetch` then import that
leaf at module scope and their two `# noqa: PLC0415` markers are deleted, which is the direction
the operator's 2026-09-09 exemption-marker ruling asks for. `FUTURE.md` item 3 nominated this
exact seam as the one that removes cycles rather than adding them.

A seventh module, `direct_fetch.py`, holds the bounded redirect loop and the single hop. The plan
put the direct fetch in `ladder.py` beside the dispatcher, which is a genuine import cycle: three
rungs (the derived-API feed, the Wayback snapshot, the paid rung's robots pre-check) issue their
request through `_fetch_direct`, so `rungs.py` would import `ladder.py` while `ladder.py` imports
`rungs.py`. Putting the redirect loop in its own layer under the rungs removes the cycle with no
injected callback and no function-scoped import. The module is named `direct_fetch` rather than
`direct` because three rungs and the dispatcher all have a local named `direct` for a direct
fetch's RESULT, and the shorter name would shadow it.

An eighth module, `digest.py`, holds the deterministic BM25 digest wrapper and the `LadderDigest`
result the `policy.digest` seat returns.

The dependency order is one-way: `guard`, `context` and `classify`, then `direct_fetch`, then
`rungs`, then `ladder`.

### No re-export aliases on `resolution_source` for moved names

A name left behind as an alias while its implementation moves is the trap this repo has shipped
before: the patch lands on the alias and the real call path is untouched, so the test stays green
and proves nothing. Every moved name is repointed at every patch site and every by-name import in
the same commit. A stale dotted patch target raises at patch time, which is the loud failure we
want. The three deliberate re-exports that already exist (`strip_markdown_escapes`,
`looks_like_csv_rows`, `format_resolution_sections`) are unaffected because none of those names
moves into the package.

### Step 1 is two commits, not one

The plan makes step 1 a single commit. A single commit moving about 2,000 lines and repointing
about 250 patch sites is not reviewable as a pure move, which is the property step 1 exists to
have. So it splits at the leaf boundary, and each half reverts on its own:

- **1a** moves the leaves and the per-URL bookkeeping: `guard.py`, `context.py`, `classify.py`.
  `resolution_source.py` keeps the spine (the hop loop, the rungs, the dispatcher, the provider)
  and imports the new modules.
- **1b** moves the spine: `policy.py`, `rungs.py`, `ladder.py`, and the `fetch_url` entry point.
  `resolution_source.py` is left as the fetcher's adapter.

### `FetchResult` gains a fourth additive field: `links`

The plan lists three (`capture_date`, `harvested_json`, `bytes_read`). The loop's driver is handed
the page's outbound links (`ToolOutcome.links`), collected during HTML classification by
`fetch_outcomes._extract_links_from_html`. Once classification is shared, the links have to ride
the result or the loop's adapter would have to re-parse HTML it no longer holds. `links` defaults
to an empty list and is populated only under `policy.collect_links`, which the fetcher's preset
leaves off, so an archived record and an all-direct fetch stay byte-identical exactly as the plan
requires.

### Three policy knobs carry what would otherwise be a per-caller branch

The two callers assemble their text differently, and the difference is entirely (a) whether the
unreadable-embed disclosure leads, and (b) whether the body is capped in classification or at
presentation. Both are policy fields rather than a `if caller == ...` branch:

| Knob | Resolution-source preset | Gap-fill preset |
|---|---|---|
| `per_url_max_chars: int \| None` | `RESOLUTION_SOURCE_PER_URL_MAX_CHARS` | `None`, uncapped; the loop windows at presentation |
| `disclose_unreadable_embeds: bool` | `True` | `False` |
| `thin_content_escalation_chars: int \| None` | `None`, escalate on status alone | `GAP_FILL_V2_MIN_CONTENT_CHARS` |

The third is the loop's `escalate_rendered` rule, which fires on a success shorter than 500
characters with no chart block, where the fetcher escalates only on `js_wall` or the `thin_page`
shape of `no_resolving_content`. Keeping it as a knob is what makes step 3's preset reproduce the
loop's behaviour exactly.

The plan's own note that the shared cache holds the FULL extracted text is why the cap has to be a
knob at all: with `per_url_max_chars` set, classification calls the same
`resolution_presentation` helpers it calls today, so the fetcher's text is unchanged.

### The policy rides the context, so no rung signature changes

`FetchContext.rung_budget_s` is patched as a CLASS attribute with a zero-argument lambda at twelve
sites, and about twenty rung functions already take `ctx`. Threading a policy argument through all
of them would edit every signature and every direct-call test. So `LadderPolicy` becomes a field on
`LadderContext` defaulting to the resolution-source preset, which leaves every existing
`LadderContext()` construction working unchanged and every rung reading `ctx.policy`. `fetch_url`
keeps the plan's exact signature, `fetch_url(url, *, policy, ctx)`, and is the one place that pairs
them, binding the policy onto the context it passes down.

The aiohttp session and the per-host semaphore map are per-question in the same way the rung budget
is, so they ride the context too; that is what lets `fetch_url` keep a signature with no session
parameter without creating a session per URL, which would change the fetcher's connector limits.
Step 2 verifies this holds and reports if it does not.

Steps 1a and 1b deliberately did NOT add these fields: `rung_budget_s` still reads module-level
constants exactly as it did, so both commits stayed pure moves with no semantic risk.

### The driver's per-call ask rides a replaced context, not a new argument

`fetch_url(url, *, policy, ctx)` keeps exactly the plan's signature. The loop's `read_document`
ask is per call while `LadderContext` is per question, so the adapter derives a child context with
`dataclasses.replace(question_ctx, query=ask, rungs=[])`, which is structurally what `_aux_ctx`
already does for a request a rung makes on a page's behalf.

### Step 2: which knobs are WIRED, and which are only declared

The plan and the step-2 brief list nine knobs plus two seats. Wiring all of them at once would
have gone inert in places, so step 2 applies one rule: a knob is read from the policy in this step
only when doing so is a value substitution with no new branch, no live test patch site on the
constant it replaces, and no rung signature change. Everything else is declared on
`LadderPolicy` with the fetcher's value and its read site left alone, so the step that
needs the difference wires it.

WIRED in step 2:

- `total_wall_s` and `rung_wall_margin_s`. `LadderContext.rung_budget_s` and the paid rung's
  client-side ceiling (`rungs._url_context_rung`) now read them off the policy, and the two
  constant imports are gone from `context.py` and `rungs.py`, so a stale read raises rather than
  going quiet. This is the knob the whole step exists for: without it step 3 cannot give the loop
  its 90 s `fetch` wall or its 25 s document ladder.

DECLARED, read site untouched, with the reason:

- `per_url_max_chars`. Eleven live `monkeypatch.setattr` sites name
  `RESOLUTION_SOURCE_PER_URL_MAX_CHARS` on `resolution_presentation` (nine) and on `classify`
  (two). Reading the cap off a frozen dataclass would make all eleven inert while staying green.
  Step 3 wires it together with those patch sites, by adding an optional cap parameter to
  `_page_text_with_leads` and `_lead_then_capped_body` rather than writing second copies.
- `wayback_max_age_days`, `disclose_unreadable_embeds`, `thin_content_escalation_chars`,
  `collect_links`. Each one's wiring is a branch only the second caller can take (an unbounded
  archive age, a suppressed embed lead, a thin-content escalation, link collection). Adding a
  branch for a caller that does not exist yet is what this repo's proportion rule refuses, so
  they arrive with the caller in step 3.
- `render_memo_scope`. A pure substitution, but the only read that needs it sits in
  `_render_or_record_the_skip`, which takes no context, and `_RENDER_MEMO_SCOPE`'s three-line
  rationale comment would have to move into a new file, where it is no longer covered by the
  operator's verbatim-move exemption. Step 3 or 4 takes it with the rendered rung it already
  touches.
- `caller`. The `caller=` marker field is step 3's, when a second caller exists to distinguish;
  the field carries the token so the marker work is a one-line read.

### Step 2: the two seats

`known_api` is consulted inside `fetch_url` ahead of everything else, which is where the run
cache will also sit once step 5 adds it (rung 0 stays in front of it). A non-None seat returning
a `FetchResult` short-circuits the whole ladder, and rung 0's result is
returned as it stands: `fetch_url` does not run it through `context._stamped_with_route`, so the
registry's own result carries its `route` (a new `FetchRoute` token is an ADD, which the telemetry
rules allow, and a rename never is).

`digest` is the page-digest seat. `digest.py` holds `LadderDigest`, the `DigestFn` protocol and
`bm25_digest`, the free deterministic default the seat's None means, wrapping `document_text`'s
`digest_pdf` for a document and `digest_text` for a flat page. Nothing calls the seat yet, and
that is deliberate rather than unfinished: today's only digest runs inside
`classify._parse_and_digest`, which is ONE `asyncio.to_thread` hop for the pypdf parse and the
BM25 selection together, precisely because the selection is 96-235 ms of CPU per document and
additive across concurrent questions. An async seat cannot run inside that thread, so routing the
PDF path through it splits the hop and moves measured CPU back onto the event loop inside the
2 s rung margin. That is a deadline change, not a strictly-safer one, so step 5 (which turns the
digest on for long HTML anyway) is where it belongs.

There is no run cache in `fetch_url`: not an empty one, not a disabled one, not a flag. Step 5
adds it.

### Step 2: the session and the host-semaphore map ride the context

`LadderContext` gains `policy`, `session` and `host_sems`, all defaulted, so every existing
`LadderContext()` construction and every direct `_fetch_one(session, url, host_sems)` call is
unchanged. `_fetch_one`'s signature does not move, which is what keeps the thirteen test modules
that call it positionally working with no edit. `fetch_url` resolves `host_sems` None to the
process-wide map (`http_fetch.host_semaphores`), binds the policy with
`dataclasses.replace`, and opens a session only when the context carries none, which is the shape
the loop's per-call plain fetch will need in step 3.

### Step 2: the six test-side repoints, all mechanical

Two were `monkeypatch.setattr` sites and four were dotted reads inside assertions. Both patches
were on `fetch_ladder.context.RESOLUTION_SOURCE_WALL_TIMEOUT`
(`test_resolution_source_datawrapper.py`, the hanging-dataset teardown and the below-floor hop
skip); they now scale the preset itself, `monkeypatch.setattr(ladder_policy,
"RESOLUTION_SOURCE_POLICY", replace(..., total_wall_s=0.4))`, which the adapter reads at call
time, so the patch reaches the context (verified: with the preset at 0.4 s the context's remaining
budget goes negative exactly as before). The four reads of `RESOLUTION_SOURCE_RUNG_WALL_MARGIN_S`
(two in `test_resolution_source_fetch.py`, one in `test_resolution_source_precision_budget.py`,
one in `test_resolution_source_url_context_rung.py`) now read
`RESOLUTION_SOURCE_POLICY.rung_wall_margin_s`, so they track the number the ladder actually uses.
No test was weakened, skipped or deleted, and every one of the seven failures the move produced
was an `AttributeError` at the patch or read site, which is the loud failure the removed imports
were meant to cause.

## Commits

Baseline at `6f2f051`: `make test` 9,759 passed, 41 skipped, 5 deselected, exit 0. Every commit
below holds that count, which is the evidence no test was added, removed or weakened.

- **`1eca925` step 1a, the leaves.** `guard.py` (324), `context.py` (273), `classify.py` (941) out
  of `resolution_source.py`, which fell from 3,518 to 2,055 lines. `FetchContext` renamed
  `LadderContext`. About 250 patch sites and by-name imports repointed; no alias left behind. The
  guard becoming a leaf removed a real circular import, so `rendered_fetch` and `impersonated_fetch`
  now reach it at module scope and two function-level-import exemption markers are gone. Purity
  proof: all 70 top-level definitions byte-identical, comments included, once module prefixes and
  the rename are normalised. Gates: test 9,759 passed, lint clean, typecheck 0 errors, 6 import
  contracts kept, deptry clean.
- **step 1b, the spine.** `direct_fetch.py` (154), `rungs.py` (1,156), `ladder.py` (142);
  `resolution_source.py` fell to 669 lines and its monolithic-file exemption marker was deleted
  because the file no longer trips the 1,000-line threshold. 84 patch sites, 29 dotted attribute
  reads and 27 by-name imports repointed. Purity proof: 31 of 31 definitions byte-identical,
  comments included, the only differences being formatter line-wrapping where a module qualifier
  pushed a call past 120 columns.

- **step 2, the policy and the entry point.** New `policy.py` (76) with `LadderPolicy` and the
  `RESOLUTION_SOURCE_POLICY` preset, new `digest.py` (78) with the digest seat, `ladder.fetch_url`
  as the one public entry point, and `fetch_resolution_sources` pointed at it. `ladder.py` grows to
  150 lines while its comment blocks shrink to one line each: the dispatcher's rung-ordering
  rationale now lives in `docs/architecture.md` "The shared fetch ladder", which also carries the
  policy knob table and replaces the stale ladder rows in "Where the pieces live". Equivalence
  evidence beyond the suite: a throwaway harness drove `fetch_resolution_sources` over seven shapes
  (a plain success, a 403 rescued by the impersonated retry, a JavaScript wall rescued by the
  browser, a cited PDF, a blocked page rescued from the archive, a blocked page read by the paid
  rung, and the Datawrapper second phase) and dumped every `FetchResult` field, every rung attempt
  and every `RESOLUTION_SOURCE_FETCH` / `RESOLUTION_SOURCE_ESCALATION` line before and after; the
  two dumps are identical once the harness's own generated `Last-Modified` stamp is normalised.
  Six new tests in `tests/resolution_source/test_fetch_ladder_entry_point.py` cover the entry
  point itself: the wall pair reaching a rung through the per-hop `ClientTimeout`, a context with
  no session getting one opened and closed, a context with no host map contending on the
  process-wide one, and rung 0 both short-circuiting and declining. Gates: test 9,765 passed /
  41 skipped / 5 deselected exit 0 (9,759 at the baseline plus those six), lint clean, typecheck
  0 errors, 6 import contracts kept, deptry clean, and each of the eight package modules plus
  `resolution_source` and `rendered_fetch` imports first in a fresh interpreter. Committed after the
  lead reshaped the digest seat (below) and pinned it: 9,766 passed with the seventh new test.

  **The digest seat is the sibling's exact shape.** The step-2 draft gave `policy.digest` a richer
  signature (`pdf`, `top_k`, `max_chars`, `source_url`) returning a rendered block, which would have
  needed an adapter around `page_digest.digest_page`. The lead reshaped it so the sibling drops in
  with no adapter: `DigestFn` is `async (text: str, query: str, *, budget_seconds: float) ->
  DigestPassages`, and `DigestPassages` is a read-only Protocol with the five `PageDigest` fields
  (`passages: list[str]`, `passages_returned`, `passages_grounded`, `fallback_used`, `method`). The
  digest hands back ranked PASSAGES rather than a block because how many characters a reader sees
  is the caller's knob, so the ladder renders the block at presentation. The BM25 default,
  `digest.bm25_digest`, returns the top-K `select_passages` windows with both counters equal to the
  selection size and `method="digest_local"`. A test in `test_fetch_ladder_entry_point.py` mirrors
  the sibling's signature and result field for field and assigns it to `DigestFn`, so a drift in
  either fails `make typecheck` on this branch rather than at the merge. Wiring, later:
  `replace(GAP_FILL_POLICY, digest=page_digest.digest_page)`.

  **The known-API seat needs a four-line adapter, by design.** `KnownApiFn` is
  `Callable[[str], Awaitable[FetchResult | None]]`. The sibling exposes a sync
  `translate(url) -> KnownApiCall | None`, the backends that run a call, and
  `adapters.to_fetch_result(result, *, url, route="known_api")`; the wiring step composes those
  three into one coroutine and assigns it to `known_api`. The seat is not shaped as `translate`
  because the ladder must not know the backends exist.

### One real find in 1b, worth a reader's attention

Eight tests failed on logger scoping, not on behaviour. `caplog.at_level(level, logger="...")`
raises the level on that logger alone, so an INFO record emitted from a moved module is dropped by
the root logger's WARNING default. Every failing site asserts a message that now comes from a new
module, so the `logger=` argument was repointed; one impersonate-rung module reads both the rung's
own lines and the fetcher's marker lines, so its logger constant became the package prefix. No test
was weakened. One of them, an absence assertion, had gone fully inert and now passes for the right
reason. The moved log lines do change their `%(name)s` prefix in production run logs, which the
archive cannot see: `scripts/telemetry/markers.py` matches every marker with `re.search` on the
message text and its own docstring says a spec is agnostic to the log-line prefix.

## Step 3 cannot be a preset: the design, CONFIRMED by the lead and built as `b059275` + `e0a8164`

The plan makes step 3 "point the loop at `fetch_url` with a preset that encodes today's gaps". A
preset cannot do it. The fetcher's classifier does not merely READ a body, it JUDGES it, and its
judgment discards the text. Measured on this branch:

| extraction | content share | classifier verdict | characters published |
|--:|--:|---|--:|
| 87 | 0.706 | `js_wall` | 0 |
| 212 | 1.000 | `no_resolving_content` / `thin_page` | 0 |
| 394 | 0.967 | `no_resolving_content` / `thin_page` | 0 |
| 914 | 0.986 | `success` | 914 |

The loop serves every one of those as `ok` with the text, and 27 of its test fixtures extract to
under 100 characters with 6 more between 100 and 400. So a third of its pinned behaviour is text
this classifier refuses to publish.

**The seam is one predicate, not eight flags.** The READ survives the verdict: when the fetcher
withholds a short page, `_extract_page_text` still holds the extracted text and only the verdict
throws it away. The 0.38 content-share metric is NOT a divergence, because the loop adopted it in
`21f3122` and already honours `chrome_metric_withheld`. On the HTML path the whole difference
reduces to the 400-character chrome floor plus which of `embed_shell` / `js_wall` / `thin_page`
an unreadable page earns. So `policy` gains ONE seat, the verdict: the fetcher's is today's
`looks_like_page_chrome` and `_no_content_verdict` unchanged, the loop's is "any non-empty
extraction is content, under `GAP_FILL_V2_MIN_CONTENT_CHARS` escalates".

Five things ride with it, each independent of the verdict shape:

- **`links` and `escalate_rendered` become additive `FetchResult` fields.** The loop's driver
  contract needs both. `links` is JSON-native and a few KB per URL, so the research archive
  absorbs it; `raw_log` serialises every field with `dataclasses.asdict` against a
  200,000-character cap.
- **The PDF parse must NOT ride `FetchResult`.** `PdfText.pages` is the whole document, and the
  measured receipt file is 833,450 characters, so one such field truncates the entire question's
  archived payload to a preview. The parse stays in the existing per-URL side channel
  (`local_document._DOCUMENT_CACHE`), and the verdict decides what the text is: the digest block
  for the fetcher, the full joined text plus its truncation note for the loop, whose pagination
  and `[p.N]` label tests both need the latter.
- **A non-matching ask must not withhold on the loop's `fetch` path.** Confirmed: a document read
  in full whose digest matches no query term comes back with `passages=0` and its text absent
  from the block. The fetcher withholds that as `no_matching_passage`; the loop serves the text
  and lets its own three-condition rule decide.
- **The self-reference refusal must have its `http_status` nulled.** The fetcher's hop refusal
  carries the redirect's 301 or 302, and handing that to the loop would trigger the impersonated
  retry and the Wayback rung on a URL we refused ourselves, which is the bypass two of the loop's
  tests exist to prevent.
- **The loop's robots.txt reads must not run the floors.** They are 33 to 45 characters and go
  through the same plain fetch, so the floors would read every host as "no directives" and
  quietly open the paid rung on hosts that disallow it.

## Two later steps also diverge from the plan

**Step 6's host-semaphore fold should be dropped.** `FUTURE.md` item 5 already ruled on this exact
merge and blocked it: the loop's rendered rung holds its host gate across a Chromium launch of up
to 35 s, and merging the maps puts that hold in front of a fetcher request whose 45 s wall
discards every page already fetched, so the merge waits on either the bounded acquire or the
wall-degradation fix and needs the queueing measured rather than assumed. Folding it as briefed
makes the fetcher's deadline behaviour worse on the surface where missed deadlines cost real
forecasts this quarter. Recommendation to the lead: leave both maps, record the re-look in
`FUTURE.md`. The rest of step 6 is already done, since `1eca925` and `2e97e1d` repointed all five
of the loop's reaches into `resolution_source` privates; one stale prose reference remains at
`agentic/fetch_outcomes.py:227`.

**Step 4 has almost nothing left.** The loop already has the inline chart read, the meta-refresh
hop, the two-pass extraction, the ARIA rewrite (inside the extraction, which is why grepping
`agentic/` for it finds nothing), the Wayback rung and the harvest half of the derived-API rung.
Only the derived-feed REUSE half is genuinely absent, and most of that arrives with the shared
read. Size it after step 3 rather than assuming now.

## Step 3 as built: four commits

`b059275` (3a, the additive half), `e0a8164` (3b, the switch), `c9127b9` (the loop's marker tests
against the registered specs) and `0c01f88` (the fetch preset's wall pinned to the tool's ceiling).
Green on the full gate set at the tip `0c01f88`: `make test` 9,811 passed / 41 skipped / 5 deselected,
lint clean, typecheck 0 errors, 6 import contracts kept, deptry clean. The count is the baseline's
9,766 plus the corpus's 21, four new tests here, and the 20 the replay-script sibling added in
`ba328f1` / `06666e3`; the loop's own suite is otherwise unchanged in size, because what step 3 did
there was move seams.

### The verdict seat's shape

`fetch_ladder/verdict.py`, a leaf importing only `constants`, `document_cache`, `document_text`,
`rendered_fetch` and `resolution_fetch_result`. It holds a `LadderVerdict` Protocol with four
methods, two frozen implementations (`ResolutionSourceVerdict`, `GapFillVerdict`) and their two
module instances, plus the content floors that moved out of `classify.py` to keep the dependency
one-way: `looks_like_js_wall`, `looks_like_page_chrome`, `content_share`, `_no_content_verdict`,
`_pdf_unreadable_reason` and the extraction record, renamed `PageExtraction`. `policy.verdict` is
the ONE field, as the lead approved.

The four methods, and why it is four rather than the two the design named:

- `unread_route(content_type)` — the branch a content type earns with no body read, or None. Only
  the gap-fill verdict uses it, for a declared image, whose bytes buy nothing a local rung can read.
- `body_route(content_type, body)` — which branch a body takes. This is the addition the design did
  not anticipate, and it is load-bearing: the fetcher routes on the Content-Type header with the
  `%PDF-` sniff inside the fallback branch, while the loop routes on the BYTES first. Without it the
  loop loses `test_pdf_magic_bytes_behind_html_content_type_are_read_locally` (a mislabeled document
  read locally), its declared-image escalation to `read_document`, and its permissive
  empty-Content-Type textual read. All three are pinned behaviour, so the read's ROUTING is a
  verdict decision even though the read itself is shared and byte-identical.
- `html(extraction, chart_block, unreadable_embeds)` — the fetcher's two functions unchanged; the
  loop publishes any non-empty extraction and calls an empty one `js_wall`.
- `document(pdf, query, max_chars, source_url)` — runs inside the parse's own single thread hop
  (`classify._parse_and_read`). The fetcher's is today's digest-or-withhold; the loop's joins the
  pages, holds the parse in `research/document_cache.py` and never withholds on a non-matching ask.

`escalate_rendered` is NOT a verdict method: the rule ("a non-success, or a success under the floor,
with no chart block") is uniform, so `classify._escalates_on_thin_content` computes it from
`policy.thin_content_escalation_chars` and the verdict returns only the status and the text.

### The presets, and where each value comes from

`RESOLUTION_SOURCE_POLICY` is unchanged in every knob it already had. Three gap-fill presets:

| Knob | `GAP_FILL_FETCH_POLICY` | `GAP_FILL_DOCUMENT_POLICY` | `GAP_FILL_DIRECT_POLICY` |
|---|---|---|---|
| `total_wall_s` | 90.0 (the `fetch` ToolSpec ceiling) | 25.0 (`_LOCAL_DOCUMENT_BUDGET_S`) | 90.0 |
| `rung_wall_margin_s` | 0.0 | 0.0 | 0.0 |
| `rungs_enabled` | impersonate, rendered, wayback | impersonate, rendered | none |
| `verdict` | `GAP_FILL_VERDICT` | same | same |
| `render_memo_scope` | `gap_fill_v2` | same | same |
| `per_url_max_chars` | None | same | same |
| `wayback_max_age_days` | None | same | same |
| `wayback_extra_trigger_statuses` | `{unsupported_type}` | same | same |
| `wayback_needs_host_refusal` | True | same | same |
| `impersonate_dial_wall_s` | `RESOLUTION_SOURCE_HTTP_TIMEOUT` | same | same |
| `disclose_unreadable_embeds` | False | same | same |
| `thin_content_escalation_chars` | `GAP_FILL_V2_MIN_CONTENT_CHARS` | same | same |
| `collect_links` | True | same | same |

`rung_wall_margin_s` is 0.0 because the old code's admission rule was literally
`deadline_monotonic_s - now < RESOLUTION_SOURCE_IMPERSONATE_MIN_BUDGET_S` against a deadline of
`monotonic() + _LOCAL_DOCUMENT_BUDGET_S`. With `total_wall_s = 25` and no margin,
`rung_budget_s()` is `25 - elapsed`, so the rung's own floor check reproduces that decision exactly;
any margin would decline a dial the loop admits today. The loop's outer bounds (the ToolSpec ceiling
and the acquisition `wait_for`) are the cut, which is what a margin would otherwise be protecting.

`impersonate_dial_wall_s` is the eleventh knob and it exists for the strictly-safer rule. The old
loop capped the WHOLE retry at one plain hop's `RESOLUTION_SOURCE_HTTP_TIMEOUT`, tightened by its
caller's deadline; the shared rung hands the transport its entire remaining rung budget, which for
the fetcher is one question's worth and for the loop's 90 s `fetch` would be up to 90 s of dialing
on a slow redirect chain. The knob restores the 20 s cap for the loop and leaves the fetcher's
behaviour byte-identical.

### The mapping table, verbatim

Status, then route for a success; `research/agentic/ladder_adapter.py` is the one copy.

| `FetchStatus` (+ reason) | loop `status` | loop `method` | text the driver reads |
|---|---|---|---|
| `success` | `ok` | route: `direct`/`meta_refresh`→`plain`, `pdf_local`, `impersonate`, `derived_api`, `rendered`, `wayback`, `url_context`→`document` | the ladder's text |
| `js_wall` / `empty_body` / `no_resolving_content` | `empty` | `plain` | "Plain fetch returned no extractable text." |
| `unsupported_type` + `undecodable_body` | `empty` | `plain` | "Plain fetch could not decode the body as text." |
| `unsupported_type` + `image_needs_reader` | `ok` | `document_needed` | the use-`read_document` placeholder |
| `unsupported_type` (other) | `error` | `plain` | "Unsupported content type: X" |
| `unreadable_document` (any reason) | `ok` | `document_needed` | the same placeholder |
| `blocked`, `http_status` set | `blocked` | `plain` | "Fetch blocked with HTTP N." |
| `blocked`, `http_status` None | `blocked` | `plain` | the platform-block message, hosts named |
| `ssrf_blocked`, url == requested | `blocked` | `plain` | "Blocked non-public or unsupported URL." |
| `ssrf_blocked`, url != requested | `blocked` | `plain` | "Blocked non-public redirect target." |
| `not_found` | `error` | `plain` | "Fetch failed with HTTP N." |
| `error` + `oversize_document` | `error` | `oversize_document` | the too-large-to-read message |
| `error`, `http_status` a 3xx | `error` | `plain` | "Malformed redirect from URL" |
| `error`, `http_status` 200 | `error` | `plain` | "Fetch body exceeded the size limit." |
| `error`, other `http_status` | `error` | `plain` | "Fetch failed with HTTP N." |
| `error` with `exc` | `error` | `plain` | "Fetch error: ClassName" |
| `error`, nothing else | `error` | `plain` | "Redirect limit exceeded." |
| `stale_data` / `ungrounded` | `error` | `plain` | the cited host's own status; unreachable under these presets |

Three new `FetchStatusReason` tokens carry cells nothing else could: `oversize_document`,
`image_needs_reader`, `undecodable_body`. One new `RungSkipReason`: `rung_not_enabled`, with its own
`_rung_counts` key, which a suite guard requires (every skip reason must move a counts key).

### Condition (d): the fetcher-visible delta of nulling a self-produced `http_status`

`guard._vetted_hop_target` no longer takes an `http_status` at all, and both refusals it builds
(`ssrf_blocked`, and `blocked` + `metaculus_self_ref`) carry None. Verified against every rung's
trigger: `rungs._impersonate_rung_applies` keys on 403 and a 301/302 was never in that set;
`_rendered_rung_applies` keys on `js_wall` / `thin_page` / `escalate_rendered`;
`_WAYBACK_TRIGGER_STATUSES` and `_url_context_rung_applies` key on the status and reason, not the
number. So NO fetcher rung decision changes, which the corpus's `blocked_self_ref` row proves: it
still reads `('wayback', 'route=wayback', 'status=blocked')` before and after. The only visible
change is the marker: `RESOLUTION_SOURCE_FETCH ... http=302` becomes `http=n/a` for a refused hop,
which is the honest record. Two fetcher tests changed, both by dropping the argument rather than an
assertion: `test_the_terminal_site_keeps_its_status_strings` and the meta-refresh hop's caller.

### Patch sites moved

Fetcher side: 11 `RESOLUTION_SOURCE_PER_URL_MAX_CHARS` monkeypatches (9 on `resolution_presentation`,
2 on `classify`) became `capped_ctx(cap)` in `tests/resolution_source_fakes.py`, which hands the
ladder `replace(RESOLUTION_SOURCE_POLICY, per_url_max_chars=cap)` on the context. All 11 verified
LIVE by breaking the helper three ways (999,999 fails 10 of them; 500 fails the eleventh, the
no-marker one, which is one-directional by construction). 6 moved-name imports repointed to
`verdict` (`looks_like_js_wall`, `looks_like_page_chrome`, `content_share`, `_PageExtraction` and two
constant reads), 42 marker-line literals gained ` caller=resolution_source`, and 5 document-cache
call sites moved to `research/document_cache.py`.

Loop side, in `tests/test_agentic_tools.py`: 46 `_fetch_plain` stubs and 29 `_fetch_plain` SUT calls
went to `_serve_direct` / `_fetch_direct_only`; 24 `_try_rendered_fetch` stubs and 3 direct calls to
`_serve_rendered` or `rungs.render_page`; 7 `_try_impersonated_fetch` calls to `rungs._impersonate_rung`
or through `fetch`; 3 `_fetch_plain_with_impersonated_retry` to `_serve_direct` plus
`rungs.fetch_impersonated`; 18 `_read_response_body` patches to `classify.read_body_capped`; 5
`_try_wayback_fetch` sites to `_serve_direct` with the archive URL or `rungs._wayback_rung`; 1
`fetch_impersonated` and 2 `render_page` to their `rungs` twins; and the `_wayback_applies` predicate
test to `rungs._wayback_rung_applies`. Zero patches remain on a dead loop function.

### Trigger-set comparison

**Impersonate: identical.** Both callers read `impersonated_fetch.IMPERSONATE_TRIGGER_STATUSES` at
call time and both require `blocked`. The loop's `_RETRYABLE_FETCH_BLOCK_STATUSES` {403, 406, 429} is
the set that MAPS to `blocked`, which is the fetcher's `_NON_OK_FETCH_STATUS` blocked rows exactly,
not the retry trigger. No knob needed.

**Wayback: three differences, two knobs.** The loop fires on `error` or on `blocked` with a host
status; the fetcher on {`blocked`, `error`, `not_found`} with no status test. (i) The loop's `error`
covers the fetcher's `unsupported_type`, so the gap-fill preset adds it
(`wayback_extra_trigger_statuses`). (ii) The loop declines a `blocked` it made itself, which two of
its tests pin, so `wayback_needs_host_refusal` restores that. (iii) `not_found` needed nothing: the
loop maps a 404 to `error` with the status set, so both callers already tried the archive. One
universal exclusion was added on top for both callers, `image_needs_reader`, because an archived copy
of an image is no more readable than the live one; the fetcher's verdict never produces that reason,
so it is a no-op there.

### Content types: no loop test was rewritten to pin admission

The one test pinning a refusal, `test_unsupported_content_type_is_error` on `application/zip`, still
passes: the gap-fill verdict routes an unlisted type with no `<html` in it to `unsupported`, and the
adapter renders the same message. The `+json` and empty-Content-Type admissions the plan's sixth
constraint names are simply untested on the loop side, so nothing needed rewriting. What DID need
re-expressing was the parity test `test_the_body_classification_is_the_same_whichever_transport_read_it`,
now `direct_fetch._fetch_direct` against `rungs._impersonated_body_outcome` compared as whole
`FetchResult`s, because that is where the one copy of the classification rule now lives.

### What step 4 turns out to be

Delivered already by the shared read, so step 4 owes none of it: the ARIA-table rewrite, the
calibrated two-pass extraction with its line-shape metric, the inline chart read, the meta-refresh
hop, the local document read with its parse gate, the Wayback rung with its per-question cap, the
`+json` and empty-Content-Type admission, and per-rung wall floors on everything.

Step 4 is complete. Handoff status is **step 4 done**. `GAP_FILL_FETCH_POLICY.rungs_enabled` now
includes `derived_api`, so a feed an earlier render remembered for a host can be reused by the gap-fill
loop. The rung, its per-host memo
and its budget gate were already live; the policy's `rung_not_enabled` skip was the only thing that
kept reuse off. The regression in
`tests/test_agentic_tools.py::TestGapFillV2DerivedApiOnEmptyRender::test_a_remembered_endpoint_is_gotten_for_a_second_same_host_url_before_render`
renders the first dashboard URL, then verifies the second same-host URL GETs the remembered endpoint
and does not launch a second render. Its companion test drives both caller presets through the real
rendered rung and pins their separate `gap_fill_v2` and `resolution_source` memo scopes. The rung
order corpus now includes the enabled derived-feed attempt for the four gap-fill cases that reach
the browser ladder.

Free verification on 2026-09-10: 273 focused ladder and agentic tests passed, including all 21 rung
order cases; the shared rendered/derived suites added 152 passes. Repository Ruff and basedpyright
(`make lint`, `make typecheck`) passed with zero findings. No paid run, network dispatch or push was
performed.

The Datawrapper dataset hop stays a second phase of the fetcher's provider, as the plan says.

### The step-6 deletion list

Production, all unreferenced today: `tools._fetch_plain`, `_fetch_one_hop`,
`_plain_response_outcome`, `_plain_body_outcome`, `_read_response_body`, `_body_too_large_result`,
`_try_impersonated_fetch`, `_fetch_plain_with_impersonated_retry`, `_try_rendered_fetch`,
`_derived_api_outcome`, `_try_wayback_fetch`, `_wayback_applies`, `_RENDER_MEMO_SCOPE`, `_host_gate`
(once `_FETCH_HOST_SEMAPHORES` has no other reader); and in `fetch_outcomes`,
`_plain_redirect_outcome`, `_vet_hop_target`, `_plain_html_outcome`, `_plain_textual_outcome`,
`_non_ok_status_result`, `_content_type_is_document`, `_content_type_is_pdf`, `_content_type_is_image`,
`_body_is_document`, `_RETRYABLE_FETCH_BLOCK_STATUSES`, `_TEXTUAL_CONTENT_TYPE_TOKENS`,
`_HTML_CONTENT_TYPE_TOKENS`, `_FETCH_MIN_CONTENT_CHARS`; and `local_document.pdf_fetch_result`,
`oversize_result`. Deleting them takes `tools.py` from 1,079 lines to well under the monolithic
threshold and clears the last 5 smell findings there, all of which sit on those functions.

Tests to delete or migrate with them: 13 in `tests/test_agentic_tools.py` still call
`agentic_tools._try_rendered_fetch` as their subject
(`test_rendered_fetch_drains_routes_and_guard_tolerates_teardown_race`,
`test_try_rendered_fetch_uses_playwright_objects`,
`test_rendered_fetch_launches_bounded_by_global_semaphore`,
`test_rendered_fetch_route_guard_blocks_private_redirect_target`,
`test_rendered_fetch_skips_launch_when_host_not_pinnable`,
`test_rendered_fetch_launches_with_host_resolver_pin`, the two in
`TestRenderedRungSalvagesATimedOutNavigation`, the four in `TestRenderedRungTimeoutAtTheV2Wrapper`,
and `test_links_resolve_against_the_documents_landing_url`), plus one in `tests/test_rendered_fetch.py`
at line 483. Every claim they make is transport-level and already covered for the shared path by
`tests/test_rendered_fetch.py` (89 tests: the pins, the off-host landings, the DOM-read bound, the
deadline, teardown, the memos, and the Tier-1 rung's skip reasons) and by
`tests/resolution_source/test_resolution_source_rendered_rung.py`, so the deletion step can most
likely drop them rather than migrate them. Verify that claim per test before deleting, not in bulk.

Also for step 6: about a dozen tests are still named `test_fetch_plain_*` while driving
`_fetch_direct_only`; renaming them is churn with no coverage change and belongs with the deletion.

### Deviations from the plan and the design, all reported

1. **The verdict seat has four methods, not two.** The body ROUTING is a verdict decision, because
   the two callers disagree about whether a body's bytes or its header decide its branch, and three
   pinned loop behaviours depend on the bytes winning. Reported above.
2. **Eleven policy knobs, not nine.** `impersonate_dial_wall_s` (the strictly-safer rule) and the two
   Wayback knobs are the additions; `rungs_enabled` was in the brief.
3. **`_WAYBACK_TRIGGER_STATUSES` stayed in `rungs.py`** rather than moving onto the verdict, so the
   `tests/resolution_source/conftest.py` autouse fixture that empties it to decline the rung keeps
   working. The per-caller difference rides two policy knobs instead.
4. **`resolution_source.py` was touched in three places, not one**: the marker emitter (now two calls
   into the shared formatter), a `rung_not_enabled_skips` key in `_rung_counts` (a suite guard
   requires every skip reason to move a counts key), and the import block. Its module docstring was
   not touched, as briefed.
5. **A shared marker formatter, `research/fetch_markers.py`.** Two emitters spelling the same
   contract by hand is the drift the telemetry rule exists to prevent, so the format string has one
   home and each caller keeps its own logger and its own `question=` ref.
6. **`research/document_cache.py` is a new module**, because the gap-fill document verdict has to hold
   a parse and `fetch_ladder/verdict.py` cannot import `agentic/`. It is the same cache, moved out of
   `local_document` with all five call sites repointed.
7. **`document_text.disclosed_page_text`** moved out of `local_document` for the same reason, and
   `digest_pdf` / `digest_text` / `_truncate_digest` now take `max_chars: int | None`, where None is
   unbounded, rather than forcing the caller to invent a cap.
8. **One capability lost, recorded in `FUTURE.md`**: the loop's own browser rung escalated a rendered
   page whose Content-Type was a document to `read_document`, and the shared rung classifies every
   rendered DOM as HTML. Restoring it means a caller-dependent branch inside the rung for a case
   nobody has measured, which the proportion rule refuses.
9. **`fetch_url` still opens the aiohttp session before the SSRF preflight.** No egress happens, and
   moving the preflight up would mean resolving DNS twice per URL, so it stays. One loop test that
   proved "no session was opened" now proves "no request was issued", which is the same claim one
   layer out.

### The comment sweep, and what is left

`classify.py` went from 25 findings to ZERO: the extractor-policy calibration, the classification
path's three ordering receipts, the raw-body markup-strip rule and the transport-failure bucketing
all moved into two new `docs/architecture.md` subsections with one-line pointers left at the code.
`fetch_outcomes.py` went from 6 to 0 (the throttle receipt and the platform-refusal reasoning are
now `docs/agentic_gap_fill.md` sections). `tools.py` went from 24 to 5, and all 5 sit on the dead
functions step 6 deletes, as does its monolithic-file finding. `ladder.py`, `policy.py`, `verdict.py`,
`ladder_adapter.py`, `fetch_markers.py` and `document_cache.py` carry none.

Left for the operator's scheduled repo-wide sweep, following the step-2 precedent for a file only
lightly edited: `rungs.py` (46), `resolution_source.py` (32), `resolution_fetch_result.py` (20),
`http_fetch.py` (24), `document_text.py` (16), `resolution_presentation.py` (6), `guard.py` (5),
`local_document.py` (4), `direct_fetch.py` (3), `tests/test_agentic_tools.py` (34). Every one of
those counts is unchanged from the baseline at `6169965`; step 3 added no finding anywhere.

## Step 5a: shared process-run cache

`fetch_ladder/run_cache.py` now owns one bounded 50-key LRU of complete typed read
artifacts below caller verdict and presentation. HTML, decoded text, local PDF, derived-feed,
rendered and Wayback successes can be replayed; paid `url_context` answers, errors, empty reads and
throttle interstitials cannot. Known API remains ahead of the cache. Hits apply the current body
route, verdict, PDF query, disclosure, links and cap; incompatible entries fall through to a fresh
fetch, and a newly rejected or thin direct artifact enters the current caller's escalation without
repeating the direct request. Redirects have requested-URL and final-URL aliases.

The loop's former text/link LRU is deleted. Its pagination now reruns `fetch_url`, receives the full
shared artifact under the uncapped gap-fill policy and labels the result `method=cache` through the
additive default-false `FetchResult.cache_hit` field. Parsed PDFs remain solely in
`research/document_cache.py`; neither a parse nor raw PDF bytes ride a `FetchResult` or the run
cache. Presentation runs off the event loop and inside the current URL's remaining wall. A timed-out
presentation becomes that URL's error result so a concurrent sibling survives.

Next: step 5b throttle status, then step 5c digest through the existing `policy.digest` seat.

### Step 2: the smell findings in the two files it edited are FIXED, not carried

The 2026-09-09 operator ruling on the smell scanner (move the prose to documentation, keep one line
of why with a pointer, never delete the why) applies to a pre-existing finding in a file you touch,
and the Stop-hook gate enforces it. So the eight findings in `context.py` and `ladder.py` are gone:
`QuestionRungBudget`'s per-question receipt, `browser_escalation_gate`'s two paragraphs, the paid-read
cap comment, `LadderContext`'s field-by-field prose, the `_RUNG_WALL_SKIP_PHRASE` and
`_BUDGET_GATED_RUNGS` derivation receipts, `_escalate_unresolved`'s conventions and `_fetch_one`'s
two carry-forward comments all moved verbatim in substance into two new `docs/architecture.md`
subsections, "What the dispatcher returns, and what it carries forward" and "The per-URL context and
the per-question budget", each with a one-line pointer left at the code. Proof the trim touched no
code: the docstring-stripped AST diff of both files against `2e97e1d` names only the intended
identifiers (`Any`, `LadderPolicy`, `RESOLUTION_SOURCE_POLICY`, the three new fields, the two wall
knobs; and in `ladder.py` only `fetch_url` and its imports). The other moved files (`rungs.py`,
`resolution_source.py` and the three test modules this step barely touched) keep their findings for
the scheduled repo-wide sweep, since nothing in them was rewritten here.

Known follow-ups, none blocking: the adapter docstring and logger scoping follow-ups are complete
(`93c723b`; replay_audit confirmed six methods and seven assertions); `rungs.py` at 1,155 lines
trips the advisory monolithic-file threshold, left as one module because
the plan specifies it as one; and every comment-density and long-docstring finding in the moved
files is left verbatim under the operator's scheduled repo-wide comment sweep, which is its own
change (the two files step 2 created carry no findings).

## Step 5b: shared 200 interstitial throttle

The throttle phrase predicate now runs in the shared read-artifact presentation path before either
caller verdict, for HTML and structured/plain text bodies. PDF bodies bypass it. A match returns
`FetchStatus="throttled"` with blank text and only the phrase plus stripped character count needed
by the existing `AGENTIC_FETCH_THROTTLED` marker. Fresh and cached reads use the same `present()`
function, and throttled artifacts are never captured or written to the process-run cache.

Direct throttles terminate before browser and offsite escalation. A rendered throttle returns with
`route=rendered` and the rendered attempt's `outcome=throttled`. The agentic adapter carries loop
`status=throttled`, `method=throttled`, and blank text while retaining the source rung for the
marker. HTTP 429 remains the existing `blocked` outcome. Focused real-body tests cover direct HTML,
raw text, PDF exclusion, HTTP 429, rendered route/attempt metadata, marker fields, cache retry, and
the replay corpus row.
