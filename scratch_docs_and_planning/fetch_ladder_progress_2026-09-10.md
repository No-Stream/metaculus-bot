# One fetch ladder for both callers: progress log

Branch `fall/ladder` in the worktree `/Users/flatljan/personal/metaculus-bot-wt/ladder`, cut from
`mantic-competition` at `6f2f051`. The plan this executes is
`scratch_docs_and_planning/fetch_ladder_unification_plan_2026-09-09.md`; read it first, then this
file for what has actually landed and what was decided along the way. A successor with no context
can resume from the "Next" section at the bottom.

Free gates only. Nothing in this work may spend money: no `main.py` run, no backtest, no probe, no
GitHub Actions dispatch. `make test` is network-blocked by an autouse fixture and is always free.

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

## Step 3 cannot be a preset: the design, awaiting the lead's confirmation

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

## Next

Step 3 as designed above, once the lead confirms the verdict seat. Then step 4 (the derived-feed
reuse half, sized after step 3), step 5 (the run cache, the throttle check, the digest through the
`policy.digest` seat), step 6 (the deletions only, without the semaphore fold).

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

Known follow-ups, none blocking: `resolution_source.py`'s 125-line module docstring still describes
a ladder that has moved out from under it, and it is now purely the fetcher's adapter, so the
rewrite is due and deliberately not bundled into step 2; seven tests inherited from 1a assert on
records from a logger they never scoped and pass only because the root logger sits at WARNING;
`rungs.py` at 1,155 lines trips the advisory monolithic-file threshold, left as one module because
the plan specifies it as one; and every comment-density and long-docstring finding in the moved
files is left verbatim under the operator's scheduled repo-wide comment sweep, which is its own
change (the two files step 2 created carry no findings).
