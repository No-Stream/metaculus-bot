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

## Next

Step 2: introduce `policy.py` with `LadderPolicy` and the resolution-source preset, add
`ladder.fetch_url`, and point `fetch_resolution_sources` at it. Then step 3 (the loop's preset),
step 4 (turn on the rungs the loop lacked, one commit each), step 5 (the run cache, the throttle
check, the digest), step 6 (fold the host-semaphore map, delete the loop's dead private fetch
functions and its reaches into `resolution_source` privates).

Known follow-ups, none blocking: `ladder.py` trips the comment-density ceiling because the
dispatcher's rung-ordering rationale now sits in a small file, so that prose moves to
`docs/architecture.md` with a one-line pointer; `resolution_source.py`'s module docstring still
describes a ladder that has moved out from under it, best rewritten once step 2 makes it an
adapter; seven tests inherited from 1a assert on records from a logger they never scoped and pass
only because the root logger sits at WARNING; and `rungs.py` at 1,156 lines trips the advisory
monolithic-file threshold, left as one module because the plan specifies it as one.
