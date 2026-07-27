# Agentic gap-fill (gap-fill v2)

Gap-fill v2 is the bounded agentic research loop that runs after the main
research providers have built a question's briefing. It gives one driver LLM a
small set of research tools and a strict time/tool budget, lets it decide what
the briefing is missing or getting wrong, and returns a citation-only findings
artifact that gets appended to the bundle every forecaster reads.

It is the newest and largest research subsystem on this branch. It runs
alongside the older v1 gap-fill pass (`research/targeted.py`), not instead of
it. Both are on in production as of 2026-07-17, and both feed the same bundle.

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
later ghost forecast are only meaningful against the real template.

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
  continuations are served from cache. Its `ToolSpec` timeout leaves headroom
  above its own fetch budget for the rung-3 document auto-escalation.
- **`read_document`** — ask a specific question of a specific document. Gemini
  reads the URL directly via the `url_context` tool on the native `google-genai`
  SDK, handling PDFs, images, and JS pages. Slower and costlier than `fetch`, so
  it is for targeted extraction from long or complex documents, or as a fallback
  when `fetch` was blocked. Requires a precise `ask`. Its deadlines nest: the
  `ToolSpec` timeout sits above Gemini's own read timeout
  (`_READ_DOCUMENT_TIMEOUT_S` in `tools.py`), which in turn sits above the HTTP
  timeout handed to the SDK, so the innermost one fires first and the driver gets
  a clean error outcome instead of a tool-level kill.

### The fetch ladder

`fetch` (`tools.py` `fetch`) tries progressively heavier methods and only
escalates when the lighter one comes up short:

1. **Cache.** A repeat fetch of the same URL, including `start_char`
   continuations, is served from an in-process LRU cache with no network call.
2. **Plain HTTP** (`_fetch_plain`). An aiohttp GET with browser-like headers,
   SSRF-hardened: a `is_public_http_url` preflight, a connect-time filtering
   resolver, and a bounded manual redirect loop that re-guards every hop.
   Trafilatura extracts the main text.
3. **Headless Chromium** (`_try_rendered_fetch`). If plain extraction returns
   too little text (below `GAP_FILL_V2_MIN_CONTENT_CHARS`), the ladder re-fetches
   with Playwright's headless Chromium to run JavaScript. The
   SSRF guard is re-applied to every request Chromium makes. If Playwright isn't
   installed, this rung logs a one-time warning and the plain result stands.
4. **read_document.** If the URL turns out to be a PDF or an image (by content
   type or by magic bytes), `fetch` auto-escalates to the `read_document`
   backend so the driver keeps its "handled automatically" promise without
   spending a second tool call. The `method` field on the result tells the
   driver which rung actually served it.

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
(`_INTERNAL_TOOL_NAMES` in `loop.py`): `set_research_plan` registers the dry
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
`asyncio.wait_for` in `loop.py`, so a slow ghost call can't eat into the run. The
parsed result is logged as a `GHOST_FORECAST` marker.

## The bounds

The loop is anytime: it always emits whatever it has banked, even when it runs
out of budget. Three limits bound it (defaults in `constants.py`, overridable by
the matching env vars):

- **Wall deadline** `GAP_FILL_V2_WALL_DEADLINE`. A hard ceiling on the whole
  loop, enforced by an outer `asyncio.wait_for`. It sits inside v1's worst-case
  timing envelope, so running v2 concurrently with v1 adds no research-phase
  wall-clock.
- **Max tool calls** `GAP_FILL_V2_MAX_TOOL_CALLS`. Parallel calls each count
  against this cap. Steps, not calls, are where latency lives, so batching is
  encouraged.
- **Max steps** = 20 (`LoopConfig.max_steps` default; the seam doesn't override
  it). A step is one driver turn.

There is also a **conclude threshold** `GAP_FILL_V2_CONCLUDE_THRESHOLD`. Once
fewer than that many seconds remain, or the tool-call cap is hit, the loop
stops offering the research tools and exposes only `record_findings` and
`conclude`, which forces the driver to wrap up inside the wall deadline. A
budget line is appended to every tool result so the driver always knows how much
room it has left.

The loop also does light stuck-detection: an exact-duplicate tool call (same
tool, same normalized arguments) bumps a `dup_tool_calls` counter and gets a
gentle warning appended to its result telling the driver the result won't have
changed. There's no hard enforcement, just the nudge.

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
GAP_FILL_V2: model=... steps=... tool_calls=... searches=... fetches=... rendered=... reads=... dup_tool_calls=... deadline_hit=... concluded_early=... wall_s=... findings=... pending_leads=... lint_rejections=...
```

`searches` sums `search_news` and `search_web`; `rendered` counts fetches that
went all the way to the headless-Chromium rung; `reads` counts `read_document`
calls; `concluded_early` is true when the driver called `conclude` before the
deadline. This marker is grep-able in the durable `run_logs/` artifacts every
workflow tees, so the driver can be vibe-evaluated after the fact without
pulling any research-archive JSON.

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
| `agentic/loop.py` | `run_agentic_loop` and the whole driver loop: message management, tool dispatch, budget/conclude logic, the ghost phase, telemetry, and the timeout/soft-fail wrapper. |
| `agentic/tools.py` | `build_gap_fill_tools` and the four tool handlers, including the escalating fetch ladder and its SSRF hardening. |
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
| `GAP_FILL_V2_ENABLED` | Master switch. Off unless set; on in all four workflow yamls since 2026-07-17. |
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
is noted in the constants file as unverified on the native AI Studio SDK. A wrong
id soft-fails `read_document` (model-not-found becomes an
error outcome), which silently disables the directed-reading rung without
breaking anything else. If `read_document` never seems to work, check the reader
model id first.
