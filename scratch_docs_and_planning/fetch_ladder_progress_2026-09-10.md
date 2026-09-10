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

### The package has seven modules, not five

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

A seventh module, `digest.py`, holds the deterministic BM25 digest wrapper and the `LadderDigest`
result the `policy.digest` seat returns.

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

### `LadderContext` carries its own wall, so `rung_budget_s()` keeps its signature

`FetchContext.rung_budget_s` is patched as a CLASS attribute with a zero-argument lambda at twelve
sites. Reading the wall off the policy would mean `rung_budget_s(policy)` and would break every one
of them. Instead `LadderContext` gains `wall_timeout_s` and `rung_wall_margin_s` fields defaulting
to today's constants, and the gap-fill preset's per-question context is constructed with the tool
call's budget. The method signature is untouched.

### The driver's per-call ask rides a replaced context, not a new argument

`fetch_url(url, *, policy, ctx)` keeps exactly the plan's signature. The loop's `read_document`
ask is per call while `LadderContext` is per question, so the adapter derives a child context with
`dataclasses.replace(question_ctx, query=ask, rungs=[])`, which is structurally what `_aux_ctx`
already does for a request a rung makes on a page's behalf.

## Commits

Nothing landed yet. Baseline gate at `6f2f051` recorded below before the first commit.

## Next

Step 1a is briefed and dispatched. After it: 1b, then steps 2 through 6 in the plan's order.
