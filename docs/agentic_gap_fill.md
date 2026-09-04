# Agentic gap-fill (gap-fill v2)

Gap-fill v2 is the bounded agentic research loop that runs after the main
research providers have built a question's briefing. It gives one driver LLM a
small set of research tools and a strict time/tool budget, lets it decide what
the briefing is missing or getting wrong, and returns a citation-only findings
artifact that gets appended to the bundle every forecaster reads.

It is the newest and largest research subsystem on this branch. It runs
alongside the older v1 gap-fill pass (`research/targeted.py`), not instead of
it. Both are on in production as of 2026-07-21 (v2 was authored 2026-07-17 and reached
`main` in merge `b4e9df0` four days later), and both feed the same bundle.

The code lives under `metaculus_bot/research/agentic/`, with a thin seam at
`metaculus_bot/research/agentic_gap_fill.py` that wires it into the research
orchestrator.

## Why it exists

The always-on providers (AskNews, native search, Gemini, and so on) build a
broad briefing, but they do not read it back and ask "what's missing, and is any
of this actually wrong?" A step-zero audit found that most of the bot's worst
misses were not gaps in coverage. They were the panel leaning on a briefing
claim that was stale, misread, or hallucinated. So v2's highest-value job is
verification, not just filling holes: it re-reads the briefing through the lens
of the panel's own forecasting template, checks the two or three claims the
forecast leans on hardest against primary sources, and flags any that don't hold
up.

## How it works, end to end

1. The orchestrator finishes the first research pass and has a briefing bundle.
2. It kicks off v1 and v2 gap-fill concurrently (one `asyncio.gather`). v2 sees
   the bundle without v1's addendum, and v2's section is appended after v1's.
3. The v2 seam builds three prompts, the four tools, and a `LoopConfig`, then
   runs the loop.
4. The driver LLM does a private dry run of the forecast to decide what to
   research, then works through search/fetch/read tools, banking findings as it
   goes.
5. When it concludes (or the budget runs out), the loop renders the banked
   findings into a markdown section and returns it.
6. The orchestrator appends that section to the bundle under
   `## Agentic Research Findings`.

The whole thing is optional enrichment. If any of it fails, the forecast
proceeds on the first-pass research alone. That soft-fail contract is described
in its own section below because it is load-bearing.

## The driver: dry run, then research

The driver is briefed as a research analyst supporting a forecasting panel, not
as a forecaster. Its system prompt (`driver_prompt.py` `build_system_prompt`)
walks it through three steps:

**Step 1, private dry run.** The driver reads the question, its resolution
criteria and fine print, and the current briefing, then privately walks through
how it would forecast the question using the panel's own template. As it does,
it notes several kinds of target:

- **FILL** targets: facts the reasoning needed that the briefing doesn't
  contain, or contains only in a secondhand or stale form.
- **VERIFY** targets: the two or three claims the reasoning leaned on hardest.
  These get checked against primary sources, because a wrong load-bearing claim
  poisons every panelist.
- **RESOLUTION** targets: if the resolution criteria name a specific source,
  metric, or clause, the driver quotes the operative language and current value
  from the authoritative source itself, not from news coverage of it.
- **TIMING** is folded into every target: the question only resolves on events
  inside its window, so the driver pins the exact date of each candidate trigger
  and flags any event that pre-dates the question's open date.
- **BASE-RATE** targets: if the dry run leaned on a reference class, the driver
  decides whether to look up the real denominator and count. It researches
  conditional or niche or uncertain rates and skips common-knowledge ones.

The user brief (`build_user_brief`) is what gets these steps grounded in the
real question. It carries the question text, resolution criteria, fine print,
forecasting window, the panel's real per-question-type template skeleton, and
the full briefing bundle. The template skeleton is the actual `binary_prompt` /
`multiple_choice_prompt` / `numeric_prompt` output with only the research slot
replaced by a placeholder. Everything else in it (units, bounds, open/closed
bound notes, options) is the question's real values, because the dry run and the
later ghost forecast are only meaningful against the real template. The
placeholder itself carries the prediction-market section header, because the
panel's market-reading clause renders only when the research carries that header
and prod emits it on every question; without it the skeleton would show the
driver a template the panel never actually sees.

**Step 2, research.** The driver pursues its targets with the tools, follows
leads (a fetched page that references a more authoritative PDF is usually worth
chasing over a fresh search), batches independent calls in parallel, and records
findings as it confirms them rather than holding everything for the end.

**Step 3, conclude.** It calls `conclude` when every target is resolved or
confidently unreachable, or when the budget line tells it to. If the briefing
already covers everything, it concludes right after spot-checking the single
most load-bearing claim. Most questions need little; a few need a lot.

The user brief is frozen before the loop starts and the loop only ever appends
to the message list, so the brief acts as a stable prompt-cache prefix.

## The four tools

Built by `tools.py` `build_gap_fill_tools`. Each returns a `ToolOutcome`
(content markdown, links, a `method` tag, a status, a truncation flag). Each
also carries its own per-call timeout, set as `timeout_s` on its `ToolSpec` in
`build_gap_fill_tools` and scaled to how heavy the tool is; the loop enforces it
and turns a breach into an error `ToolOutcome` rather than a crash.

- **`search_news`** — recent and historical news via AskNews, using the same
  rate gate and concurrency semaphore as the primary AskNews provider. Returns a
  digest of matching articles with dates and URLs.
- **`search_web`** — semantic web search via Exa (direct SDK, not OpenRouter).
  Meant for official documents, datasets, reports, and primary sources the
  driver believes exist. Returns results with URLs and excerpts, which the
  driver is told to follow up with `fetch` since excerpts rarely verify a claim
  on their own. The lightest of the four, and the one on the tightest timeout.
- **`fetch`** — fetch a URL and return its main content as markdown plus
  outbound links. This is an auto-escalating ladder (detailed below), so the
  driver is told not to avoid a URL because of its format. Supports windowed
  reads: over-cap content is truncated with a `start_char=N` marker, and
  continuations are served from cache — a PDF is read here too, in full text, and
  paginates the same way. Its `ToolSpec` timeout leaves headroom above its own
  fetch budget for the document auto-escalation on the last rung.
- **`read_document`** — ask a specific question of a specific document, and get
  back the passages of it that bear on the ask. Acquisition-first: it runs the
  free rungs (this run's cache, then plain HTTP and headless Chromium) and
  answers from the page's own text with a deterministic BM25 passage digest
  (`method=digest_local`), and only where the ladder holds nothing usable — no text at
  all, or the one refused shape described below — does Gemini read the URL through the
  `url_context` tool on the native `google-genai` SDK (`method=document`). So it is
  for targeted extraction from long or complex
  documents and as a fallback when `fetch` was blocked, and the paid half of it
  is now reserved for hosts our own client cannot read — measured 2026-09-03,
  two of 47 archived fetch failures. Requires a precise `ask`: it is what selects
  the passages. Its deadlines nest: the `ToolSpec` timeout sits above a total
  budget the two rungs share (`_READ_DOCUMENT_TOTAL_BUDGET_S`), which caps local
  acquisition at `_LOCAL_DOCUMENT_BUDGET_S` and hands the reader whatever is
  left, itself bounded by Gemini's own read timeout
  (`_READ_DOCUMENT_TIMEOUT_S`), which sits above the HTTP timeout handed to the
  SDK — so the innermost one fires first and the driver gets a clean error
  outcome instead of a tool-level kill.

### The fetch ladder

`fetch` (`tools.py` `fetch`) tries progressively heavier methods and only
escalates when the lighter one comes up short:

1. **Cache.** A repeat fetch of the same URL, including `start_char`
   continuations, is served from an in-process LRU cache with no network call.
2. **Plain HTTP** (`_fetch_plain`). An aiohttp GET with browser-like headers,
   SSRF-hardened: a `is_public_http_url` preflight, a connect-time filtering
   resolver, and a bounded manual redirect loop that re-guards every hop.
   Trafilatura extracts the main text.
3. **Local PDF extraction** (`local_document.pdf_fetch_result`). A body that is
   a PDF — by content type or by magic bytes — is decoded with pypdf in a worker
   thread and served as its own full text (`method=pdf_local`), which paginates
   through `start_char` exactly as a long HTML page does. This is why the capped
   body read now happens BEFORE the document check: the classifier used to decide
   a PDF was unreadable without looking at a single byte of it. Measured
   2026-09-03, pypdf pulled 833,450 chars out of a 6.7 MB 220-page report in
   5.3 s while the paid reader returned nothing for the same file, so a declared
   PDF is read under the larger `DOCUMENT_TEXT_PDF_MAX_BYTES` cap; a body past
   even that is reported as `oversize_document` rather than escalated, since a
   document too big to read locally is also too big to be worth having a model
   retrieve. The parse is held for the run, so pagination and a later
   `read_document` on the same URL neither refetch nor reparse. A read that stopped
   early says so in the text it serves, led by
   `[Partial document read: N pages; stopped at the M-page read cap]` or
   `[Partial document read: N pages; stopped after M pages on the extraction time budget]`
   — the clause is `document_text.truncation_note`, shared with the digest header so the
   two wordings cannot drift, and without it the driver pages to the end, sees
   `truncated=False`, and can report an absence over pages nobody read. The parse itself
   contends for `http_fetch.pdf_parse_semaphore()`, two slots held loop-wide with the
   Tier-1 resolution-source PDF rung. It bounds concurrent parses and their pypdf arenas,
   not how many fetched bodies are resident.
4. **Headless Chromium** (`_try_rendered_fetch` in `agentic/tools.py`, a thin mapping
   onto this ladder's `PlainFetchResult` over the shared browser transport
   `metaculus_bot/research/rendered_fetch.py`, which the Tier-1 resolution-source
   rendered rung uses too). If plain extraction returns
   too little text (below `GAP_FILL_V2_MIN_CONTENT_CHARS`), the ladder re-fetches
   with Playwright's headless Chromium to run JavaScript. It waits for
   DOM-ready plus a fixed settle rather than for network idle, and salvages
   `page.content()` when the navigation itself times out: 4 of the 10 render
   rescues in the 2026-09-03 replay came from pages whose DOM was complete when
   `page.goto` raised. The rung's 35 s ceiling is unchanged; the settle comes
   out of the goto budget. The SSRF guard is re-applied to every request Chromium
   makes. If Playwright isn't installed, this rung logs a one-time warning and
   the plain result stands. A URL where Chromium ran and extracted nothing is remembered
   for the run and never rendered again, so the second launch a documented escalation
   would spend (a js-walled `fetch` the driver follows with `read_document`) is skipped.
   That is the ONLY outcome memoized: a `blocked`, `error` or `throttled` GET is not,
   because the driver is told to retry those URLs and caching them would suppress a retry
   the tool descriptions promise.
5. **read_document.** If the URL turns out to be an image, or a PDF with no text
   layer at all, `fetch` auto-escalates to `read_document` so the driver keeps
   its "handled automatically" promise without spending a second tool call. The
   `method` field on the result tells the driver which rung actually served it. The
   escalation passes `ladder_exhausted=True`, an internal argument saying the free rungs
   just ran for this URL, so `read_document` does not re-request a page (or re-download an
   image the plain rung classified off its Content-Type). It is deliberately absent from the
   driver-facing schema, which stays `(url, ask)`: the loop binds handlers with `**arguments`
   straight off the model, so an advertised — or merely hallucinated —
   `ladder_exhausted: true` would skip the free ladder and pay. `build_gap_fill_tools`
   wraps the real function to enforce that, the same way it hides `fetch`'s
   `question_topic`.

### The free digest, and the one shape it refuses

`read_document` answers from text the ladder already holds with a deterministic BM25
passage digest. It runs in a worker thread rather than on the event loop:
`select_passages` tokenises every window of the whole document and holds a counter per
window, which measured a 1,365 ms contiguous stall for six concurrent 400-page digests —
inside a research phase whose wall discards work that already succeeded.

The digest is refused — and the paid reader runs instead — for exactly one shape: held text
under `GAP_FILL_V2_MIN_CONTENT_CHARS`, with no PDF parse behind it, whose digest selected
NO passage. That is a JavaScript shell whose browser rescue already failed, and digesting
its navigation chrome stamped an unread page `fetched` — the one tier that supersedes the
briefing — while the tool description tells the driver a zero-passage digest means the
document does not discuss the ask. All three conditions are load-bearing: a thin-but-real
short page that matches the ask is still served free, a held parse is a real local read of
something a browser cannot help with, and a matching passage is the evidence that the text
is the page rather than its frame.

The paid rung's deadline arithmetic is fixed by design, and it can overrun. The wait is
`min(_READ_DOCUMENT_TIMEOUT_S, _READ_DOCUMENT_TOTAL_BUDGET_S − acquisition elapsed)`: 60 s
when acquisition failed fast, 40 s at the `_LOCAL_DOCUMENT_BUDGET_S` cap. The reader's own
in-thread ceiling is a FIXED 55 s (two attempts plus backoff, `tool_backends.py`), so past
about 10 s of acquisition the wait is the shorter of the two, and `wait_for` cannot cancel a
`to_thread` worker — a worker can outlive the wait by up to 15 s and finish a billed call
whose answer is discarded. What it cannot do is start a NEW billed request after the wait
fires: the last attempt begins by 28.5 s in, inside the 40 s floor. Sizing the attempts off
the variable wait instead would cut one attempt to 19 s on the handover path and fail reads
that succeed today, so the timeout values are unchanged and the overrun is documented rather
than traded away.

### The robots pre-check on the paid read

Before the paid `url_context` read — and only there; the free rungs are unaffected —
`read_document` fetches `<scheme>://<host>/robots.txt` once per host through the same
SSRF-guarded plain fetch (`_fetch_plain`, under `robots_policy.ROBOTS_FETCH_TIMEOUT_S`, the
bound the Tier-1 resolution-source reader shares), with the verdict cached process-wide and
filled single-flight, so concurrent callers on one host share one read. Only the
`Google-Extended` group is honoured,
because that is the product token Gemini's retrieval obeys: a host disallowing it refuses
the fetch server-side, so the read is spend with a known-zero return, which is what makes one
free request worth it. `urllib.robotparser` cannot express that —
`can_fetch("Google-Extended", url)` falls back to the `User-agent: *` group when no
Google-Extended group exists, which would skip the paid read on every host that merely
disallows generic crawlers — so the group parser is our own, in
`metaculus_bot/research/robots_policy.py` (shared with the Tier-1 url_context rung), and every ambiguity there resolves toward
PAYING rather than skipping (an unreadable robots.txt, an unmodelled rule shape, an absent
group all come back "not disallowed").

A disallow returns `status="robots_disallowed"` and earns no verification tier, because
nothing was read: only a `method` with an entry in `provenance._METHOD_TO_TIER` can be
stamped, which is also why a `throttled` fetch can never claim `fetched` and supersede the
briefing. It logs one `AGENTIC_URLCONTEXT_ROBOTS_SKIP` line (fields under Telemetry).

## The output artifact

`artifact.py` `render_findings` turns the banked findings into the markdown
section that gets appended to the bundle. It returns an empty string when there
are no findings and no pending leads.

Structure:

- `## Agentic Research Findings` header.
- `### ⚠ Corrections to the briefing` first, if any finding is flagged
  `discrepancy=true`. This block carries language telling the panel these
  findings contradict and supersede the corresponding briefing content. Putting
  it first is deliberate: a flagged briefing error is the single most valuable
  thing v2 can produce.
- The remaining findings grouped by topic, sorted.
- Each finding renders as Claim, Source, a blockquoted Quote, Date, and
  Retrieved how.
- A `Pending leads:` list at the end for things the driver couldn't verify (dead
  links, paywalls, no coverage) but wanted to flag rather than guess at.

### Detachment lint

Findings are supposed to state facts, never a view on how the question resolves.
`artifact.py` `detachment_lint` enforces this with a banned-register regex over
each finding's claim and topic fields. Banned phrases include likelihood and
verdict language such as "likely", "unlikely", "probably", "suggests",
"indicates that", "we believe", "we expect", "points to", "this implies",
"bullish", "bearish", "in our view", and "odds are". A finding that trips the
lint is rejected rather than banked, and the rejection is fed back to the driver
in the tool result so it can rephrase. The `lint_rejections` counter tracks how
often this happens.

Alongside the four research tools, the loop exposes its own internal ones
(`_INTERNAL_TOOL_NAMES` in `tool_schemas.py`): `set_research_plan` registers the dry
run's ranked gaps, and external tool calls come back as a nudge to plan first
until it has run; `record_findings` banks findings mid-run; `conclude` finishes
the loop, optionally banking final findings and leaving pending leads. The two
findings tools run their input through the same validation and detachment lint.
Internal calls don't count against the tool-call budget.

## The ghost forecast (telemetry only)

After the driver concludes, the loop asks it to privately complete the forecast
itself using the panel's template and its own findings, and to output only the
structured forecast block. This is the "ghost forecast." It is never shown to
the panel and never published. It exists so the run logs carry a signal of what
a forecast built purely on v2's research would have looked like, which is useful
for evaluating driver quality. The ghost phase runs only when the driver
concluded explicitly (not when the deadline cut it off) and is bounded by its own
`asyncio.wait_for` in `loop.py`, so a slow ghost call can't eat into the run.

`_run_ghost_phase` logs the parsed result twice: a lossy human-readable
`GHOST_FORECAST` line kept byte-identical for the already-harvested archive, and
an additive `GHOST_FORECAST_JSON` line carrying the complete forecast (every
percentile, not just the median) so `scripts/score_ghosts.py` can score numeric
ghosts. The JSON line is suppressed when no structured block parsed. The turn-one
plan emits the same pair as `GHOST_PRE` / `GHOST_PRE_JSON`
(`_set_research_plan_tool`) from the driver's pre-research dry run, so the
pre-versus-post delta measures whether v2's own research moved its own view.

## The bounds

The loop is anytime: it always emits whatever it has banked, even when it runs
out of budget. Three limits bound it:

- **Wall deadline** `GAP_FILL_V2_WALL_DEADLINE` (`constants.py`, env-overridable).
  A hard ceiling on the whole loop, enforced by an outer `asyncio.wait_for`. It
  sits inside v1's worst-case timing envelope, so running v2 concurrently with v1
  adds no research-phase wall-clock.
- **Max tool calls** `GAP_FILL_V2_MAX_TOOL_CALLS` (`constants.py`,
  env-overridable). Parallel calls each count against this cap. Steps, not calls,
  are where latency lives, so batching is encouraged.
- **Max steps** — `LoopConfig.max_steps`, which the seam doesn't override, so
  this one lives on the dataclass rather than in `constants.py` and takes no env
  var. A step is one driver turn.

There is also a **conclude threshold** `GAP_FILL_V2_CONCLUDE_THRESHOLD`. Once
fewer than that many seconds remain, or the tool-call cap is hit, `_tool_schemas`
stops offering the research tools and exposes only the internal ones
(`_INTERNAL_TOOL_NAMES`), which forces the driver to wrap up inside the wall
deadline. A budget line is appended to every tool result so the driver always
knows how much room it has left.

The loop also does light stuck-detection: an exact-duplicate tool call (same
tool, same normalized arguments) bumps a `dup_tool_calls` counter and gets a
gentle warning appended to its result telling the driver the result won't have
changed. There's no hard enforcement, just the nudge. One exception: a fetch that
came back throttled has its call key forgotten again as its tool message is
written, because a throttle outcome is never cached and its message asks the
driver to retry the same URL later in the run, so that retry really can return
something different and must not be told otherwise.

## Soft-fail and isolation (load-bearing safety property)

This is the most important property for anyone operating the bot: **gap-fill v2
can never crash a forecast.** A forecast built on first-pass research alone is
strictly better than no forecast, so every boundary in this subsystem degrades
to an empty string instead of raising. There are four layers of this:

1. **The seam** (`agentic_gap_fill.py` `run_gap_fill_v2`) returns `""` and makes
   zero LLM calls when the flag is off, when benchmarking, or for an unsupported
   question type, and it wraps the whole run in a broad `except` that logs and
   returns `""` on any error.
2. **The loop** (`loop.py` `run_agentic_loop`) wraps its body in
   `asyncio.wait_for` at the wall deadline. On timeout it marks `deadline_hit`
   and returns whatever findings it had banked. On any other exception it logs
   and returns the banked findings. `CancelledError` is the one thing it
   re-raises, so cancellation propagates cleanly.
3. **Each tool execution** is wrapped so that a tool timeout, a bad outcome, or
   any tool exception becomes an error `ToolOutcome` fed back to the driver,
   never a loop crash. The driver sees the error and can route around it.
4. **The orchestrator** (`orchestrator.py`) runs v1 and v2 in their own
   independent guards inside the gather, so a v2 defect (an import error in the
   agentic package, an unhandled raise) can never zero out v1's addendum, and
   vice versa.

The benchmarking guard deserves its own mention. When `is_benchmarking=True`,
v2 returns `""` before doing anything. Live search on a resolved question sees
post-resolution information, which would leak the answer, so v2 is hard-off in
backtests for the same reason the prediction-market provider is.

## Telemetry

Every run that reaches the loop emits one INFO line to the run logs
(`loop.py` `_log_completion`):

```
GAP_FILL_V2: model=... steps=... tool_calls=... searches=... fetches=... rendered=... reads=... dup_tool_calls=... deadline_hit=... concluded_early=... wall_s=... findings=... pending_leads=... lint_rejections=... provenance_rejections=... quote_mismatch_warnings=... plan_gaps=... plan_skipped=... conclude_gate_rejections=... error=...
```

`searches` sums `search_news` and `search_web`; `rendered` counts fetches that
went all the way to the headless-Chromium rung; `reads` counts `read_document`
calls; `concluded_early` is true when the driver called `conclude` before the
deadline. `error` carries the `repr` of whatever tripped the loop's catch-all
soft-fail and is `None` on both a healthy run and a deadline hit, which makes it
the one field that separates a step-zero crash from an idle run — the two emit
otherwise byte-identical `steps=0 tool_calls=0 findings=0` lines. Everything from
`provenance_rejections` onward postdates the original marker, so
`scripts/telemetry/markers.py` wraps that tail in optional regex groups and still
harvests pre-branch archived logs that end at `lint_rejections`. This marker is
grep-able in the durable `run_logs/` artifacts every workflow tees, so the driver
can be vibe-evaluated after the fact without pulling any research-archive JSON.

One event outside that line has its own marker, because it is invisible in the counters:
a fetch whose 200-OK body was the host's rate-limit interstitial rather than the page logs

```
AGENTIC_FETCH_THROTTLED: url=... method=... chars=... phrase=...
```

as a WARN from `tools.py`, harvested as `agentic_fetch_throttled`. It carries no `question=`
(the tool handlers run below the loop's `log_prefix`), so a join goes through the run id.
Such a fetch returns `status=throttled` and is never cached, so the driver's retry of the
same URL is a real request; `chars` and `phrase` are the two fields that say whether a fire
was a true throttle or the rule over-reaching. Receipt: q45191, where two throttled
ogimet.com fetches reached the driver as successful ones and its own retry was served the
cached refusal.

A second event outside the counters is a document read for free, which is what the
local-document rung exists to produce:

```
AGENTIC_FETCH_LOCAL_DOC: url=... method=pdf_local|digest_local chars=... pages=... passages=...
```

as an INFO from `local_document.py`, harvested as `agentic_fetch_local_doc` and likewise
without a `question=`. `pdf_local` is a `fetch` serving a PDF's own extracted text, which
paginates like a long page and therefore selects nothing (`passages=n/a`); `digest_local` is a
`read_document` answering the ask from BM25-selected passages of text we hold, where
`passages=0` is the reading that matters — the document does not discuss what was asked, which
in the block itself reads exactly like a successful read. `chars` is the text we HELD, not the
window handed to the driver, so it is comparable across both routes and against
`URL_CONTEXT_SIZE_GATE_TOKENS` (chars / 4).

The line fires only where a digest or a PDF's text was actually SERVED, so its absence is not
a measurement: a `read_document` whose digest was refused (the one shape above) or whose
ladder held nothing leaves no line at all, and the paid read that followed is visible only in
the reader's own spend. Count fires, never non-fires.

A third is the pre-check that skips a paid read the host would refuse anyway:

```
AGENTIC_URLCONTEXT_ROBOTS_SKIP: url=... host=...
```

as an INFO from `tools.py`, harvested as `agentic_urlcontext_robots_skip` and, like the two
above, with no `question=`. Non-alertable: a fire is a paid call NOT billed, not a defect.
`host` rides beside `url` because the robots verdict is cached and applied per host, so the
host is the unit any rate is computed over — and a suspiciously high rate is the signal that
the group parser is over-matching and withholding reads we could have had.

For a richer trace, the seam accepts an `archive_sink` callback. When the loop
actually ran, the orchestrator captures `{transcript, telemetry}` through it and
writes it into the research archive (`persistence.py`), including empty-findings
runs, whose telemetry is still worth keeping.

## Module layout

Everything lives under `metaculus_bot/research/agentic/`, with one seam file one
level up:

| File | What's in it |
| --- | --- |
| `agentic_gap_fill.py` (one level up) | The seam. `run_gap_fill_v2` owns prompt/tool/config construction and the outermost soft-fail boundary, keeping the orchestrator thin. |
| `agentic/loop.py` | `run_agentic_loop` and the turn loop: message management, the three internal tool handlers, per-call handler dispatch, the ghost phase, the `GAP_FILL_V2` completion marker, and the timeout/soft-fail wrapper. Everything that logs one of this loop's telemetry markers stays here so the markers keep their `...agentic.loop` logger. |
| `agentic/tool_schemas.py` | `_INTERNAL_TOOL_NAMES` (the loop's own tools, and their timeout) plus the JSON-schema builders for the tool list advertised each turn. |
| `agentic/loop_state.py` | `_LoopState` (the one mutable per-run record), the `_ToolCall` / `_ToolExecutionResult` per-turn records, the assistant-message parsers that produce them, and the budget arithmetic. |
| `agentic/provenance.py` | URL and quote normalization, the quote-grounding span logic, and the per-call harvesters behind the provenance gate and the W4 verification tiers. |
| `agentic/gates.py` | The W1 plan gate's nudge and gap coercion, the W2 conclude gate, the W3 `source_url` check, and W4 tier stamping plus idempotent findings banking. |
| `agentic/dispatch.py` | One assistant turn's tool calls in, one tool message each out: batch admission (plan gate, call budget, duplicate detection), provenance absorption, and the tool-message/rejection rendering. |
| `agentic/tools.py` | `build_gap_fill_tools` and the four tool handlers, including the escalating fetch ladder and its SSRF hardening. |
| `agentic/local_document.py` | The local PDF rung, the run's held-parse cache, the passage digest `read_document` serves, the url_context size gate, and the `AGENTIC_FETCH_LOCAL_DOC` marker. |
| `agentic/fetch_outcomes.py` | Response classification for the plain `fetch` rung: content-type and magic-byte sniffers, the outbound-link collector, the metaculus.com refusal, and the per-body-shape outcome builders including the throttle interstitial. |
| `agentic/tool_backends.py` | The outbound half of the tools: the AskNews and Exa clients with their retry ladders and concurrency caps, the Gemini `url_context` document read and its fixed in-thread ceiling, and the markdown formatting of what comes back. |
| `agentic/tool_descriptions.py` | The driver-facing tool descriptions and JSON parameter schemas — behavioral text, so a change here changes what the driver does. |
| `research/robots_policy.py` (outside `agentic/`, shared with the Tier-1 url_context rung) | The `Google-Extended` robots.txt group parser and per-host cache behind the pre-check on every paid read, written because `urllib.robotparser` falls back to `User-agent: *`. |
| `agentic/driver_prompt.py` | The three prompt builders: `build_system_prompt`, `build_user_brief`, `build_ghost_prompt`, plus the `SupportedQuestion` type. |
| `agentic/artifact.py` | `render_findings` (the output section) and `detachment_lint`. |
| `agentic/types.py` | The dataclasses and Pydantic models: `ToolOutcome`, `ToolSpec`, `Finding`, `GhostForecast`, `LoopConfig`, `LoopTelemetry`, `LoopResult`. |
| `agentic/llm.py` | `build_default_llm_call`, the litellm/OpenRouter binding with donated-key-first routing and personal-key fallback. |
| `agentic/__init__.py` | Package exports. |

## Configuration

All flags are read in `constants.py`. The enable flag uses the standard
`env_flag_enabled` helper, so it is off unless explicitly set to
`true`/`1`/`yes`.

Defaults are deliberately not reproduced here — read them off the definitions in
`constants.py`, which is the only copy that cannot go stale.

| Env var | What it controls |
| --- | --- |
| `GAP_FILL_V2_ENABLED` | Master switch. Off unless set; on in all four workflow yamls, live in prod since 2026-07-21 (`b4e9df0`). |
| `GAP_FILL_V2_DRIVER_MODEL` | The driver LLM. Picked by the 2026-07-17 blind 5-arm replay eval. |
| `GAP_FILL_V2_DRIVER_EFFORT` | Driver reasoning effort. |
| `GAP_FILL_V2_READER_MODEL` | The `read_document` backend model on the native google-genai path. |
| `GAP_FILL_V2_MAX_TOOL_CALLS` | Tool-call budget. |
| `GAP_FILL_V2_MAX_GAPS` | Ranked gaps `set_research_plan` accepts; the excess is dropped from the low-relevance end. Independent of v1's `GAP_FILL_MAX_GAPS`. |
| `GAP_FILL_V2_WALL_DEADLINE` | Hard wall for the whole loop, in seconds. |
| `GAP_FILL_V2_CONCLUDE_THRESHOLD` | Seconds-remaining threshold below which only `conclude` is offered. |
| `GAP_FILL_V2_MIN_CONTENT_CHARS` | Extracted-char floor below which `fetch` escalates plain HTTP to headless Chromium. |

The driver and reader run on separate credentials. The driver goes through
litellm/OpenRouter with donated-key-first routing (all eval candidates were
OpenAI/Anthropic models). The `read_document` reader uses the personal
`GOOGLE_API_KEY` on the native google-genai SDK, and `search_web` (Exa) and
`search_news` (AskNews) use their own personal keys.

One caveat worth flagging for operators: `GAP_FILL_V2_READER_MODEL`'s default id
(`gemini-3.8-flash`) was verified live on the native AI Studio SDK 2026-09-03, so
the constants file no longer carries an unverified-id caution. A wrong id still
soft-fails `read_document` (model-not-found becomes an error outcome), which
silently disables the directed-reading rung without breaking anything else, and
url_context retrieval fails the same quiet way on a host whose robots.txt
disallows `Google-Extended`. If `read_document` never seems to work, check the
reader model id first, then the target host's robots.txt.
