# Gap-Fill v2: Bounded Agentic Research Loop — Implementation Plan

**Status**: design agreed 2026-07-16 (operator + assistant, full-session discussion). Rev 2 — adds: Exa key wiring, fetch auto-escalation ladder, DIY-loop decision, additive two-flag rollout (v1 stays on), codex implementation split, DeepNews deferral. Rev 3 (2026-07-16, post-operator prompt review) — adds: discrepancy channel (§3.3 + Phase-1 amendment in §9), fetch `start_char` pagination (§3.2), Exa documented rate limits + 429 handling requirement (§2.1), fetch-politeness requirement for the rendered rung (§5). Rev 4 (2026-07-16, post-external-design-review) — adds: BTF-2 citation (§1), calibration-slope success metrics (§7), ghost model-dependence caveat (§3.4), dry-run scaffold guard + v1-lite duplicate-query telemetry + base-rate priority caveat (§3.1), inline source-tier tag (§3.3), traditional-provider tier-tagging deferral (§8). Not yet built.
**This doc is self-contained**: a fresh session should be able to execute from here without re-deriving the design. Read AGENTS.md first (cost gate, key routing, telemetry conventions). FUTURE.md carries a condensed pointer to this plan for sessions that don't find it.

---

## 1. What we're building and why

Add a **bounded agentic tool loop** as the second-pass research stage: a driver LLM that reads the question + resolution criteria + the fully assembled research bundle, privately dry-runs the forecast to discover what's missing or load-bearing, then iteratively searches / fetches / reads documents to fill gaps and verify key claims — emitting a detached, citation-only findings artifact appended to the bundle. Everything downstream (6-forecaster fan-out, median aggregation) is untouched. v1 gap-fill (`run_gap_fill_pass`) **stays on alongside it** for an overlap window (§4).

**Why (evidence base, researched 2026-07-15):**

- Metaculus's Fall 2025 bot-maker survey (39 respondents + behavior logs): research breadth was the strongest predictor of score (r = 0.42); winners made ~28 LLM calls/question vs 7; verbatim advice: research should be "agentic, not one-shot". Top-15 winners spent ~$1.40/question; median winner $0.90.
- FutureSearch (full research agent, ~$1/question, dozens of searches/page-reads per question) is #1 of 163 in Summer 2026 FutureEval. This repo (nostreambot) was 9th in Fall 2025, best open-source. Research depth is the identified gap to the leaders.
- Bridgewater AIA Labs ablation (arXiv 2511.07678): agentic search beat non-agentic search **on identical sources** (Brier 0.1140 vs 0.1174 / 0.1182 vs 0.1217 across two source configs).
- BTF-2 (arXiv 2604.26106): a strong prompt on good SHARED research (0.129 Brier) edged the best self-directed integrated agent (0.131), and the integrated benefit was model-dependent (only the Opus-class model clearly gained; Gemini was slightly better on fixed shared research). Our one-loop→detached-artifact→ensemble shape is that winning recipe; the lever is shared-research quality, not integration topology.
- Counter-signal that shapes the design: multiple survey respondents tried freeform "agent with a bag of tools" and reverted — the winning shape is a **structured bounded loop** (search → read → identify gaps → repeat), not an autonomous researcher. v2 is deliberately "gap-fill with hands", not a from-scratch researcher.

**v1's structural limits (why v2 exists, regardless of model quality):**

1. **No cross-gap iteration**: gap list frozen before any evidence arrives; each native-search resolver call is an island.
2. **No directed reading**: can't tell the black-box search "open *this* resolution source and quote the operative clause". (OAI native search is internally agentic — it follows its own leads, not ours.)
3. **No verifiability**: synthesis without a trace; hallucinated/misread sources invisible. Observability is a first-class goal even where it doesn't immediately move Brier.
4. **No follow-ups**: "page references a linked PDF / quarterly bulletin" is a dead end today. Follow-up pursuit is a major design focus — the driver prompt leans into chasing leads surfaced by fetch results.

## 2. Constraints (operator-set, hard)

- **Latency**: v2's hard deadline `GAP_FILL_V2_WALL_DEADLINE = 540` s (inside v1's worst-case envelope: `GAP_FILL_ANALYZER_WALL_TIMEOUT` 135s + resolver wave 420s ≈ 555s). Failing to submit a forecast is catastrophic; the loop must be **anytime** (§5) and **soft-fail to ""** exactly like v1. NOTE while both v1+v2 run (§4), research-phase wall-clock is max(v1, v2, other providers) since they run concurrently — verify the orchestrator actually parallelizes them; if v2 runs after v1 sequentially, latency stacks and v2's deadline must shrink.
- **Cost**: target ≤ ~$0.50/question marginal for v2. Driver tokens on the donated OpenRouter key (openai/anthropic/google models only — x-ai 404s on it). Exa on operator's key (~$5–8/quarter at $0.005/query). Gemini url_context reads on `GOOGLE_API_KEY` (bills Metaculus per AGENTS.md correction 2026-07-16; treat as cheap-not-free; flash-tier reader keeps it cents). Keeping v1 on adds its existing ~$0.15–0.25/question — accepted by operator for the overlap window.
- **Ensemble untouched**: 6 forecasters, median, clamps, aggregation — no changes. Composition screening (`scratch/ensemble_composition_2026-07-15/RESULTS.md`; memory `project_ensemble_composition_2026q3`) settled roster questions.
- **Cost gate**: all live runs ask-first per AGENTS.md. Phases 1–3 buildable + testable with zero paid calls (mocked tools); paid validation explicitly gated.

### 2.1 Exa key (funded 2026-07-16)

- Local: `~/.keys/exa_key` (standard key location). Wire into `.env` as `EXA_API_KEY` (or source the file).
- GHA: secret name **`exa_key`** → surface as `EXA_API_KEY` in workflow env (all four workflow yamls; Phase 3 wiring).
- Setup reference: `scratch_docs_and_planning/exa_setup_info_prompt.md` (vendor-provided, operator does not warrant it; canonical docs https://docs.exa.ai/reference/search-api-guide-for-coding-agents — fetch if anything looks stale).
- Config decisions from that doc: `type="auto"`, `contents={"highlights": True}` default (token-efficient) with optional `text={"max_characters": N}` when the driver needs fuller page context; `num_results` ~8; Python SDK `exa-py` (uv add), snake_case params. Known parameter footguns are listed in the doc's Troubleshooting section (no `useAutoprompt`, contents nested under `contents` on /search but top-level on /contents, `maxAgeHours` not `livecrawl`). Exa `/contents` endpoint (clean parsed content for known URLs, cache+livecrawl) is a possible future fetch-ladder rung — noted, not v1.
- **Rate limits** (fetched 2026-07-16 from https://docs.exa.ai/reference/rate-limits): `/search` **10 QPS**, `/contents` **100 QPS**, `/answer` 10 QPS (flat defaults for all users; docs don't break limits out by plan/tier, don't document burst behavior or the over-limit status code, and offer no self-service raise — Enterprise via sales@exa.ai). 10 QPS is far above our per-question usage, but a wide-question-fan-out run could brush it. **Handling requirement**: Phase 2's `search_web` adapter must implement bounded retry-with-backoff on 429 plus a small global concurrency cap, mirroring the AskNews semaphore pattern in `research/providers.py:82-210` — and since the docs don't specify over-limit behavior, treat any 429 defensively (backoff, then soft-fail the call) rather than assuming documented semantics.

## 3. Architecture

```
run_research (orchestrator)
  ├─ providers in parallel (AskNews, native search, Gemini grounded,
  │   financial, prediction market, resolution source)   [unchanged]
  ├─ bundle assembly + summarizer                        [unchanged]
  ├─ gap-fill v1 (run_gap_fill_pass)                     [stays, own flag]
  └─ gap_fill_v2(question, bundle)                       [new, own flag]
       └─ returns findings_markdown ("" on any failure)
forecaster fan-out ×6 → median                            [unchanged]
```

### 3.1 The driver loop — DIY, no framework

**Decision: hand-rolled loop, no agent framework** (LangGraph / PydanticAI / OpenAI Agents SDK / smolagents all considered and rejected). Rationale:

- The loop is ~200 lines of `while` + message-list management. Frameworks buy graph orchestration, checkpointing, multi-agent handoffs — none needed here.
- **Prompt-cache discipline requires exact control of the message array** (append-only, §3.6). Frameworks that rewrite/reorder/inject messages silently break prefix caching, and diagnosing that through an abstraction layer is worse than owning the loop.
- Observability was half the point: we want the raw transcript in our research archive, not a framework's trace format.
- litellm is already in-tree and speaks OpenRouter tool-calling.

**Implementation note**: `forecasting_tools.GeneralLlm` is prompt-in-text-out — it does not expose tool calling. Drive the loop with **`litellm.acompletion` directly** (`tools=[...]`, `parallel_tool_calls=True`), with a small helper mirroring `build_llm_with_openrouter_fallback`'s donated→personal key routing for raw litellm calls (check `fallback_openrouter.py` for reusable pieces before writing new routing).

**Scaffold — template dry-run**: driver briefed with the *actual forecaster prompt template* (per-qtype), the question incl. resolution criteria + fine print, and the full bundle. Instruction: privately attempt the forecast. Facts it reaches for and can't verify → **fill** targets. The 2–3 claims its reasoning leans on hardest → **verify** targets (primary-source checks; hallucination/misinterpretation defense). Reference classes its reasoning leaned on → **base-rate** targets: find the actual historical data (real denominator + count from a citable dataset) instead of leaving the panel on remembered rates — recorded as ordinary detached findings (operator addition 2026-07-16; forecaster-memory base rates are usually decent but unsourced, and sourced ones are both better calibrated and auditable). Priority caveat: step-zero (memory `project_step_zero_research_miss_audit`) says verify > fill, so base-rate research is a secondary rider on existing machinery — a live audit of remembered-rate accuracy is running (2026-07-16) to size whether it deserves more investment. Dry-run completes clean → **stop early** (encouraged; most questions are mechanical — don't burn credits for the sake of it). Scaffold guard: the dry run stays a plain "attempt the forecast, notice what you reach for" contract — do NOT add deliberative scaffolds (Bayesian-update rituals, propose-evaluate-select); those reliably hurt in the preregistered Schoenegger/Tetlock study (arXiv 2506.01578), and the dry-run's own calibration doesn't matter anyway.

**Loop mechanics**: sequential steps; **parallel tool calls within a step allowed and encouraged** (batch searches, fan-out fetches) — steps are where latency lives. v1-lite stuck-detection: the harness normalizes (tool, args) per call and counts exact/near-duplicate repeats within a run; `dup_tool_calls=N` joins the GAP_FILL_V2 marker line, and a duplicate gets a gentle warning appended to its tool result. ENFORCEMENT (semantic dedup vs primary-provider queries, MiroFlow-style rollback caps) stays deferred — see FUTURE.md. Implementation rides the post-Phase-3 polish pass, not the Phase-3 agent's scope.

**Caps** (constants.py, env-overridable): `GAP_FILL_V2_MAX_TOOL_CALLS ≈ 14` (parallel calls each count), `GAP_FILL_V2_WALL_DEADLINE = 540`, per-tool timeouts (fetch 20s plain / 45s rendered, search 15s, read_document 60s), `GAP_FILL_V2_CONCLUDE_THRESHOLD ≈ 90` (s remaining).

**Time enforcement is the harness's job**: every tool result gets a budget line appended (`[budget: 212s remaining, 9/14 calls used]`) — inside the new tool message, never edited upstream. Below the conclude threshold the harness rejects all tools except `conclude`. At the hard deadline the harness emits banked findings (§5).

### 3.2 Tools (four driver-visible; deliberately small)

| tool | backend | cost | notes |
|---|---|---|---|
| `search_news(query)` | AskNews via the **existing rate-limit/retry/dedup machinery** (`research/providers.py:82-210` — global semaphore, rate gate, backoff) — never raw SDK calls; shares the per-run budget with the primary provider | ~free | news modality |
| `search_web(query, end_published_date=None)` | Exa `/search` direct (`exa-py`), config per §2.1. `end_published_date` unused in prod — future backtest date-ceiling hook | $0.005/q | semantic/general modality |
| `fetch(url, start_char=0)` | **auto-escalating ladder, one driver call** (below) | free–cents | the workhorse; `start_char` pages through truncated content (served from fetch cache, no refetch) |
| `read_document(url, ask)` | Gemini flash-tier + `url_context` (google-genai) | cents | **directed reading**: PDFs, images, or "answer this specific question from this document" |

Plus `conclude(findings)` — the structured exit. **Excluded from v1**: probabilistic tools (opinionated quantities; artifact is opinion-free; the private dry-run doesn't need them), code execution, interactive browsing (§8), DeepNews (§8).

**`fetch` auto-escalation (semi-automation of the 3-round worst case).** The driver sees ONE tool; the harness runs the ladder internally within the single call:

1. Plain `http_fetch` + trafilatura → if content OK, return (`method=plain`).
2. If `js_wall` OR extracted body < `GAP_FILL_V2_MIN_CONTENT_CHARS` (~500) → retry with **headless Chromium** (playwright async, network-idle wait) + trafilatura → return (`method=rendered`).
3. If content-type/magic-bytes say **PDF or image** at any rung → route to the `read_document` backend with a generic "extract the main content relevant to [question topic]" ask → return (`method=document`).

Result schema: `{markdown, links[], method, status, truncated}` — concise markdown, hard char cap, outbound-links list (powers follow-ups). **Pagination**: `fetch` takes `start_char: int = 0`; the truncation marker is `[truncated at N of M chars — call again with start_char=N]` (extends the repo's `[truncated at N chars]` convention with total size + the continuation recipe). Repeat/continuation fetches are served from the existing fetch cache — a `start_char` follow-up re-slices cached extracted content, it never re-hits the network or re-runs the ladder. Worst case collapses from 3 driver rounds to 1, and a long document is now readable end-to-end in cheap continuation calls instead of being a dead end past the cap. `read_document` stays separately exposed for *directed* asks (driver knows the question it wants answered from a specific doc) — the ladder's rung 3 uses a generic ask, the tool call uses the driver's; for very long documents, `read_document` remains the smarter move than paging through many `fetch` windows.

**Tool descriptions are a deliverable, not an afterthought** (operator requirement): each description states exactly when to use it vs. its siblings ("`fetch` handles JS and PDFs automatically — do not pre-emptively use `read_document` for ordinary pages"; "`search_news` for events/reporting; `search_web` for facts, documents, data, niche sources"), includes one worked example each, and documents the result schema including `method` and the budget line. Operator reviews tool descriptions together with the driver prompt (§9).

### 3.3 The findings artifact (detached by construction)

Steering risk isn't just stating probabilities — **evidence selection is itself opinionated**. Output discipline is structural:

- Schema per finding: `claim / source URL / verbatim quote / date / retrieved-how / discrepancy: bool`. Organized by topic, never by supports/contradicts.
- **Source-tier tag** (2026-07-16, already sent to the Phase-3 agent; documented here): the driver prompt carries an optional inline source-tier tag reusing the forecaster prompts' `_SOURCE_PROVENANCE_LADDER` A–D vocabulary (prompts.py:234) — inline in claim text, optional, lint-exempt; NOT a schema field.
- **Discrepancy channel** (operator requirement, 2026-07-16): when the researcher determines a tier-1 briefing claim is faulty or misleading (wrong number, misread clause, hallucinated/misattributed fact, stale figure presented as current), the finding carries `discrepancy=true` and its claim states both what the briefing says and what the source actually says. The renderer surfaces all discrepancy findings in a dedicated `### ⚠ Corrections to the briefing` block at the TOP of the findings section, with explicit "this supersedes the corresponding briefing content" language, before the topic-organized findings. Detachment rules apply unchanged inside corrections — state what the source says, no resolution implications. `discrepancy=true` is reserved for briefing-vs-source errors; source-vs-source conflicts stay in the ordinary contradiction channel below.
- Contradictions **reported, not resolved** ("Reuters 07-12 says X; ministry release 07-14 says Y"). Resolving is the forecasters' job — six independent resolutions is the ensemble's point.
- Banned register: verdicts, likelihood language, "this suggests", recommendations. Detached, dense, citation-heavy. `artifact.py` enforces a lint-style check rejecting likelihood-register strings in claims.
- `pending_leads`: leads noticed but not pursued (budget) — free telemetry on whether the budget leaves value on the table.
- Rendered as `## Agentic Research Findings` section appended to the bundle. **Distinct header from v1's section** — both coexist during overlap (§4).
- Dry-run scratchpad is **logged, never bundled**.

### 3.4 Ghost forecast (telemetry only)

After the artifact is **finalized and frozen** (hard ordering, harness-enforced; ghost never touches the bundle), the driver emits its own forecast → `GHOST_FORECAST:` marker line. Never published.

Why: at resolution, ghost-vs-published-median scores answer "would an integrated research+predict agent beat detached-researcher + ensemble?" with real Briers. Also a **steering audit**: forecaster outputs correlating with the ghost above baseline = detachment leaking. Researcher does NOT become an ensemble member in v1 (referee-and-player: evidence selection and vote correlated by construction). Revisit on a quarter of ghost telemetry. Caveat: BTF-2 found the self-directed-search benefit is model-dependent (Opus-class gained, Gemini didn't), so ghost-vs-median results are conditional on the driver model — a mediocre ghost under a small driver doesn't kill the researcher-as-member idea; re-read before drawing conclusions.

### 3.5 Model choice (driver)

- **Dev harness**: `gpt-5.6-luna`, effort medium — cheap + reliable while building.
- **Vibe-eval candidates** (config-only swap via `GAP_FILL_V2_DRIVER_MODEL` / `GAP_FILL_V2_DRIVER_EFFORT`): luna-medium, terra-{low,medium}, sol-low, sonnet-5.0. All openai/anthropic → donated key.
- Physics: tool loops pay latency per step — (TTFT + thinking ÷ tok/s) × ~10–14 steps. tok/s ≈ luna 120 / terra 80 / sol 40 (+higher TTFT); sol is an unusually token-efficient reasoner at low effort → genuinely two-sided; decide by vibe-eval. Prior: v1 resolver bench 2026-07-09 (constants.py comment) — sol-low matched terra-low coverage in 20% fewer chars (single-call context, not loop).

### 3.6 Prompt caching (cost-critical)

- **Append-only transcript, no exceptions**: system prompt + question + bundle frozen at prefix; tool results only appended; budget line lives inside each new tool result. ~12 steps over a large stable prefix ≈ the difference between ~$0.15 and ~$0.60/question on the driver.
- OpenAI via OpenRouter: automatic prefix caching (>1024 tokens). Gemini: implicit. Anthropic: explicit `cache_control` breakpoints via litellm — wire only if sonnet-5.0 wins the eval.
- Verify cache hits empirically during vibe-eval (OpenRouter generation stats show cached-token counts); 0% hit rate = bug.

## 4. Rollout: additive two-flag overlap (v1 stays on)

- v1 keeps its existing flag (`GAP_FILL_ENABLED`); v2 gets `GAP_FILL_V2_ENABLED`. **Both 'true' in prod** for the overlap window (operator decision 2026-07-16: summer season winding down, both-on harvests comparison data points; cost accepted).
- Both sections enter the bundle (distinct headers). Data harvested: per-question artifact diffs (same question, v1 vs v2 output, from research archive) + resolution-scored both-on era.
- **Turning v1 off is a deliberate later step** (operator's call; reminder logged in FUTURE.md so it isn't forgotten). Kill switch symmetric: either flag flips independently, no deploy.
- v2 prod flip = new config era for residual analysis (note date in AGENTS.md at flip time).

## 5. Failure & latency posture

- **Anytime output**: findings bank in harness state as they complete; hard deadline emits banked findings, not "". Timeout costs depth, never the question.
- **Soft-fail contract identical to v1**: any unhandled error → log + return "" (forecast proceeds). Specific expected exceptions caught; unexpected → `logger.exception` + re-raise inside the worker, converted to soft-fail at the boundary (repo exception discipline).
- Per-tool timeouts prevent any single call from eating the budget; `search_news` inherits AskNews backoff behavior (waits, not crashes).
- **Fetch politeness**: v2's fetch reuses `http_fetch`'s per-host serialization + browser-like headers; the rendered (Chromium) rung must route through the same per-host gate, and Phase-2 tests must cover this (a rendered fetch and a plain fetch to the same host must not run concurrently).
- Playwright: browser launch failure (missing binary in CI, etc.) → ladder degrades to plain-fetch-only + WARN, never crashes the loop. GHA workflows need `playwright install chromium` step (Phase 3).

## 6. Telemetry & logging (rides existing plumbing — no new upload paths)

- Marker lines (stdout → tee → `run_logs/` GHA artifact, 90-day retention):
  - `GAP_FILL_V2: question=... model=... steps=N tool_calls=N searches=N fetches=N rendered=N reads=N deadline_hit=bool concluded_early=bool wall_s=... findings=N pending_leads=N`
  - `GHOST_FORECAST: question=... qtype=... value=...`
  - Existing `CREDIT_SPEND:` lines (shipped 2026-07-15) → watch first v2 runs.
- Full tool-call trace (queries, URLs, methods, statuses, truncated results, dry-run scratchpad) → per-qid research-archive JSON (`backtests/research_archive/`), pulled weekly by the existing sync job. Deterministic, no separate pull.

## 7. Evaluation ladder (backtests CANNOT evaluate this — leakage)

Evidence (verified 2026-07-15): date-filtered live search leaks resolution info on 41–55% of resolved questions (ACL 2026 audit); prompting models to ignore post-cutoff knowledge fails (arXiv 2601.13717); only clean fix is a frozen pre-resolution corpus (FutureSearch RetroSearch) — out of scope. **`make backtest_*` is uninterpretable for research-stage changes.** Instead:

0. **Step zero (free, before/during build)**: residual check on recent resolved misses — what fraction trace to research-comprehension failures (missing fact, misread resolution clause, stale info) vs judgment? Sizes the prize. performance_analysis dataset + comment rationales; read-only.
1. **Dev-time**: unit/integration tests with mocked tools (free); then vibe-eval on a handful of live questions via test_bot prod-mode (**cost-gated, ask first**) — driver candidates compared on artifact quality, latency, spend, cache-hit rate. Operator QAs artifacts by eye.
2. **Prod flip** after (1) test suite green and (2) operator QA/vibe-eval pass — operator explicitly accepts early flip; downside bounded by soft-fail + both-on overlap + kill switch.
3. **Both-on overlap data** (§4): artifact diffs + ghost telemetry (free, continuous) — including ghost-vs-median on calibration specifically, which tests integrated-research over-sharpening.
4. **Shadow forecasts** (later, cost-gated): ~50-question slice, forecaster fan-out twice (with/without v2 findings), publish one, score both at resolution. The rigorous verdict on Brier AND era-bucketed calibration slope/intercept — agentic-research wins in the literature optimize accuracy and can trade calibration for over-decisiveness ("more evidence → more decisive"); the detached artifact mitigates that structurally, and this metric ensures we'd see it if it leaks through anyway. Build only after v2 is stable.
5. **MiniBench + era-bucketed residual analysis** (existing practice): long-run verdict.

## 8. Deferred / explicitly out of scope for v1

- **Interactive browsing** (clicking, forms, BrowserUse): the 10 that costs 90. Not planned. (Headless-Chromium *rendering* IS in scope — it's rung 2 of the fetch ladder.)
- **AskNews DeepNews** (agentic iterative research over AskNews KG + Google/Wiki/X/Reddit; OpenAI-SDK-compatible endpoint): two integration options logged in FUTURE.md (low prio) — (a) `search_news_deep(query, max_depth)` as an optional heavy tool the driver escalates to; (b) upgrade `search_news` backend to DeepNews with depth param. **Blocked on operator checking limits/pricing** — it's a separate quota pool from the OpenRouter donated key, possibly subsidized. If cheap/free, solid case for (a).
- **Probabilistic tools in the loop**: excluded (§3.2). Dormant tool_runner path unrelated, stays as-is.
- **Researcher as ensemble member**: deferred pending ghost telemetry (§3.4).
- **Bagging backtest** (2–3 samples/model, ~tens of $, mostly donated): the remaining ensemble upside per composition screening; also fable-as-forecaster's first measured data. Independent; sequence after v2. Cost-gated.
- **Paid scrape/fetch APIs** (Firecrawl, Olostep — FUTURE.md's old candidates): rejected. DIY fetch ladder + url_context covers the need with control and observability; operator prefers DIY. Exa `/contents` noted as possible future ladder rung.
- **Deep-research APIs as providers** (o4-mini-DR, sonar-deep-research): rejected as primary path; DeepNews is the surviving candidate (above).
- **Researcher-side tier tagging for TRADITIONAL providers**: the provenance ladder lives only in the three forecaster prompts; `web_research_prompt` (and the AskNews summarizer prompt) don't tier-tag, so provenance can be lost through summarization. Follow-up: add a light tier-tagging instruction to the traditional researcher prompts, shipped AT the v2 prod flip so it rides the same config-era boundary. Not Phase 3 scope.

## 9. Build plan (codex implements, Fable orchestrates)

Implementation model (operator decision): **codex (GPT-5.x) agents write the code** from complete specs — Phases 1 and 2 are well-specified, test-backed, ideal codex territory. **Fable (main session) owns**: orchestration, the driver prompt + tool descriptions (§3.2/§3.3 — drafted in main context, **operator reviews before integration**), integration judgment calls, and QA/review of codex output (verify completeness — codex downscopes occasionally).

**Phase 1 — harness core** (codex agent): `metaculus_bot/research/agentic/`: `loop.py` (litellm-direct driver loop, caps, budget injection, anytime state, conclude enforcement, parallel-tool-call handling), `artifact.py` (findings schema, renderer, detachment lint), `types.py` (result schemas). Unit tests with a scripted fake LLM: early conclude, deadline mid-fetch, parallel calls, budget-line injection, cap enforcement, soft-fail, cache-friendly message-array assertions (append-only invariant tested).

**Phase-1 amendment (2026-07-16, post-operator-review — lands in the codex Phase-2-or-3 pass, since Phase 1 was specced before this feedback)**: `types.py`'s finding schema gains `discrepancy: bool = False`, and `artifact.py`'s renderer gains the priority rendering from §3.3 — discrepancy findings pulled into a `### ⚠ Corrections to the briefing` block at the top of the findings section with the supersedes language; detachment lint applies to discrepancy claims unchanged. Tests: rendering order (corrections before topics), empty-corrections case (no block emitted), lint-on-discrepancy-claim.

**Phase 2 — tools** (codex agent, parallel): `tools.py`: four adapters over existing machinery (`http_fetch`, AskNews rate-limited path, exa-py, google-genai url_context) + the fetch escalation ladder + playwright rendered-fetch rung. Per-tool timeouts, result schemas, char caps; `search_web` gets 429 retry-with-backoff + global concurrency cap per §2.1. Mocked-HTTP tests: js_wall→rendered escalation, thin-content escalation, PDF routing, truncation + `start_char` pagination (continuation served from cache, no second network hit; marker math correct), rendered-rung per-host politeness gate (§5), SSRF-reject passthrough, AskNews backoff, Exa 429 backoff, playwright-missing degradation. `uv add exa-py playwright` (+ dev browser install docs).

**Phase 3 — integration** (Fable subagent or main): orchestrator seam (add v2 call alongside v1; verify concurrent execution, §2 latency note); flags + constants block; marker telemetry + research-archive trace writing; ghost-forecast sequencing; workflow yaml env (`GAP_FILL_V2_ENABLED: 'false'` initially, `EXA_API_KEY: ${{ secrets.exa_key }}`, playwright install step). Integration tests: full pipeline with mocked tools; v1 path untouched when v2 flag off; both-on renders both sections; `make test/lint/typecheck` green.

**Prompts** (main context, before Phase 3 lands): `driver_prompt.py` — template-dry-run briefing per qtype, fill/verify/stop instructions, follow-up emphasis, detachment rules, tool descriptions. → **operator review checkpoint**.

**Phase 4 — validation (COST-GATED, ask operator per run)**: step-zero residual check (free) → single-question live smoke (luna) → vibe-eval matrix (luna/terra/sol/sonnet × a few live questions via test_bot) → operator reviews artifacts + latency + `CREDIT_SPEND` + cache hits → pick driver → **prod flip** (operator's call; new era noted in AGENTS.md).

**Docs**: AGENTS.md v2 section (pipeline, markers, flags) at prod-flip time. FUTURE.md updated now (2026-07-16) with the condensed position + DeepNews options + turn-off-v1 reminder.

## 10. Key file map (for the implementing session)

- `research/targeted.py` — v1 (stays on; wall-cap pattern to mirror)
- `research/orchestrator.py:116` — the call seam (v2 added alongside)
- `research/http_fetch.py`, `research/resolution_source.py` — fetch machinery, FetchStatus/js_wall, truncation conventions
- `research/providers.py:82-210` — AskNews rate-limit machinery (global semaphore, rate gate, retry) for search_news
- `metaculus_bot/constants.py:360-402` — v1 constants block (v2's adjacent)
- `metaculus_bot/prompts.py` — forecaster templates the dry-run briefing embeds
- `fallback_openrouter.py` — donated-key routing to mirror for litellm-direct calls
- `comment/markers.py` — marker-line conventions
- `scratch_docs_and_planning/exa_setup_info_prompt.md` — Exa config reference (+ canonical docs URL at top)
- AGENTS.md "Value extraction ladder" + telemetry paragraphs — logging/marker style to match
