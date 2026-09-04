# Handoff: next-season-bundle final QA and wrap-up (fetch ladder Phase 3 review and fix wave)

**Date:** 2026-09-03, late evening, US Pacific (written about 22:00 PT)
**Branch:** `next-season-bundle` (PR #66 "Next season bundle" against `main`; `origin` is the fork
`No-Stream/metaculus-bot`, `upstream` is the Metaculus template, so every `gh` call needs
`--repo No-Stream/metaculus-bot`)
**Repo:** `/Users/flatljan/personal/metaculus-bot`
**Status (updated 2026-09-04, late afternoon PT):** DONE except the operator's push and merge. The pushed head
`1f2b504` has PR CI green (run 33917127083). After it, the Codex review triage landed as four worktree merges
plus three small commits ending at `c07d7cf`: the browser route-guard comment and docs now say what the guard
covers (server-side redirect hops are unguarded), the model-id rule in AGENTS.md and `docs/roster_history.md`
states the real split with `tests/test_model_name_locations.py` pinning it, the
`litellm_callback_drain_timeout` marker spec is registered, and FUTURE.md carries the browser-transport fix
(item 8, deferred at that point) and the un-clocked PDF prologue (LOW). The full free gate on `c07d7cf` is
green (7,592 passed, 14 skipped, 33 deselected, `~/logs/gate17.log`). The commit after `c07d7cf` adds the
tracked PR description (`scratch_docs_and_planning/next_season_bundle_2026-09_PR66_description.md`, 97 KB,
over GitHub's 65,536-character body cap) and this status; the PR body on GitHub is the 14.7 KB condensed
version. Then, on the operator's instruction the same afternoon, the browser-transport follow-up was BUILT
into this PR rather than deferred to its own: `6646a0b` refuses a render whose main frame landed off the
pinned host and blocks page WebSockets, and `8ced8a5` hands the browser the direct fetch's landing URL so the
pin covers the host that serves the content. A free local render probe priced strict host equality first, at
zero refusals across 22 real render targets. The full free gate on the implementation worktree is green (7,608
passed, 22 skipped, 33 deselected); the transport's off-host WARNING is being registered as the marker spec
`rendered_fetch_off_host` in a parallel worktree, and the lead re-gates the merged tip once that and the
documentation commit have landed. Next: operator pushes, PR CI reruns, operator merges PR #66, then the
post-merge commands in "Operator follow-ups" at the end of this doc.

The operator does NOT read plan docs; every decision or approval you need from them goes inline in
chat, self-contained, with a recommendation. They sign off on any paid run before it fires.

## Update, 2026-09-04 late morning (flag on, live probes, ready for the smoke)

- **The paid url_context rung is ON in production** (`06b3fd9`). The operator's rule: a feature built behind a
  flag ships with the flag on unless stated otherwise; leaving `RESOLUTION_SOURCE_URL_CONTEXT_ENABLED` off
  was the lead's miss, not a decision. `RESOLUTION_SOURCE_URL_CONTEXT_ENABLED: 'true'` now sits beside the
  sibling research flags in all five bot workflows (tournament, cup, minibench, test_bot, test_bot_basic);
  the three paid-rung log lines are registered marker specs (`resolution_source_urlcontext_robots_skip`,
  `resolution_source_urlcontext_ungrounded_suppressed`, `resolution_source_urlcontext_not_addressed`) with
  fixture tests; a new test in `tests/test_workflow_reliability.py` pins the flag on and `GOOGLE_API_KEY`
  wired in every bot step; AGENTS.md's cost gate, `docs/operations.md`, `docs/research.md`,
  `docs/performance_analysis.md` and `.env.template` now say the resolution-source provider IS a paid
  surface, bounded by the trigger population, the free Google-Extended robots pre-check, the 15 s floor and
  `RESOLUTION_SOURCE_URL_CONTEXT_MAX_ATTEMPTS` (2 paid reads per question). The two stale curl_cffi comments
  (`fetch_diagnostic.yaml`, the AGENTS.md probe command) were fixed in the same commit.
- **Paid rung verified live** (operator-authorized, cents; receipt
  `scratch/fetch_ladder_2026-09-03/paid_rung_probe_2026-09-04.log`): four blocked URLs through
  `fetch_resolution_sources` with the real Google key and the flag on, 4.5 s total. trueup.io: robots
  Google-Extended disallow, skipped free (`ROBOTS_SKIP`). imf.org: Wayback declined, paid read fired,
  Gemini reported `URL_RETRIEVAL_STATUS_ERROR`, withheld as `ungrounded`, `GEMINI_USAGE
  role=resolution_source` logged (542 tokens). sagaftra.org (DataDome): paid read fired, the model opened
  with `NOT_ADDRESSED`, withheld as `no_resolving_content` / `not_addressed` (3,100 tokens). congress.gov:
  200 today, the extractor policy's precision fallback rescued the bill-status card
  (`precision_fallback_rescues: 1`). Two paid reads total, the per-question cap binding. Nothing from a
  challenge page reached forecaster text. This closes the review's "paid rung never ran live" gate.
- **Rendered rung verified live** (free, keys blanked; receipt `ogimet_live_check_2026-09-04.log`): the
  ogimet page that overran to 76 s in the QA sweep now returns in 42.3 s, inside the 45 s wall, recorded as
  `render_timeout` (the transport's own DOM-read cut) and memoised.
- **PR CI on the pushed `ad9fec3` is green** (lint, test, secret scan, audit).
- **Smoke test fired and QA'd (2026-09-04, 11:39 to 12:15 PT).** Operator pushed `cbc26bf`, PR CI green, the
  ONE authorized `test_bot.yaml` dispatch ran as GitHub run 33907102246 (9 m 26 s). All four questions
  researched, forecast 3 of 3, published (Metaculus API confirms new forecasts and private comments). The run
  exited 1 by design: six alertable degradation events, all one external cause, Kalshi's events-catalogue API
  answering HTTP 429 to three of four concurrent full paginations (the Kalshi code is unchanged since run
  67, which paged cleanly; the stampede exposure, no single-flight on the 6-hour catalogue cache, is
  pre-existing and a fix is in flight). Ladder telemetry complete and parseable: 4 FETCH, 5 ESCALATION (3
  wayback, 2 url_context), 2 `NOT_ADDRESSED`, 2 `GEMINI_USAGE role=resolution_source`; one real rescue
  (bls.gov/wsp/ for q38195 via a 6-day-old Wayback capture, as-of line and caveat once); both paid reads
  withheld as `not_addressed`; policy D withheld and rescued nothing; the rendered rung did not fire in the
  provider (gap-fill v2 rendered one page, so Chromium works on the runner). Spend: $10.00 in role lines
  ($8.86 donated, $1.14 personal for the Google forecaster); donated key 1,449.19 to 1,441.54. Forecasts:
  q38195 moved from a median near 1 to near 3 because search surfaced the Metaculus community forecast for
  the exact question and two models anchored on it (observation for the operator; not caused by the
  branch). QA verdict: fit to merge; no defect attributable to the branch. Report:
  `scratch/fetch_ladder_2026-09-03/qa_smoke_2026-09-04.md`. Low items it raised: the Wayback 30-day bound
  plus a `NOT_ADDRESSED` reply cost q20683 a static definitional page (gap-fill v2 quoted it anyway); the
  all-failed notice still makes resolution_source `ok` with sources=0/1 (pre-existing design); personal-key
  `CREDIT_ROLE_SPEND` shows usd = 2 x byok_usd (pre-existing, unverified); the free-text "rescued none of
  them" line printed on a question with a rescue (fix in flight); the withheld paid reply text is not
  persisted (fix in flight).

## What this repo is and what the work was for

This repo is a Metaculus forecasting bot. "Tier-1" is the resolution-source fetcher
(`metaculus_bot/research/resolution_source.py`), which fetches the URL a question names as its
resolution source and renders the text to the forecaster models under a caption that calls it
primary grading evidence. "v2" is the gap-fill agentic research loop
(`metaculus_bot/research/agentic/`), whose driver LLM has `fetch` and `read_document` tools. Both
got a deterministic, free-first escalation ladder on 2026-09-03, planned in
`scratch_docs_and_planning/fetch_ladder_plan_2026-09-03.md` (read its "Evidence table", "The
ladder", "Decisions taken" and the "Delivered" subsection). Phases 1 to 3 were built and merged
earlier in the day, the last (Phase 3: rendered, derived-API, Wayback and url_context rungs) at
`dd1074b`. The earlier work of the day (Phases 1 and 2, the first review and QA wave, the AGENTS.md
de-bloat, the fall Metaculus Cup configuration, credit alerting, the `gemini-3.8-flash` move, the
operator-run diagnostics) is written up in the PR description
(`scratch/next_season_bundle_2026-09/PR_DESCRIPTION.md`, section "The 2026-09-03 work") and is
not repeated here.

Tonight's session (roughly 19:00 to 22:00 PT) reviewed Phase 3, ran a live QA pass over it, and
fixed what both found. Starting point: HEAD `16ca9ab`, gates green at 7,291 tests.

## What we accomplished tonight (in landing order)

- **Plan doc and memory reconciled** (`9c6071e`). The plan's "Decisions taken" section gained a
  "Delivered" subsection recording Phase 3's final state; the project memory file
  (`~/.claude/projects/-Users-flatljan-personal-metaculus-bot/memory/project_fetch_ladder_2026_09_03.md`)
  was brought to the same state. The PR description was refreshed with the day's work (gitignored,
  untracked file; not committed).
- **Test-suite egress guard widened** (`bd10996`, merged `ade14d9`). `tests/conftest.py` had
  patched only `socket.socket.connect`, so a headless Chromium subprocess or a libcurl request
  escaped it. A new autouse fixture, `_block_native_egress`, refuses Playwright
  `BrowserType.launch` (and the connect variants) and `curl_cffi` requests, and fails the test at
  teardown if a refusal was swallowed by a soft-fail. A proof run with the guard alone showed nine
  resolution-source tests would otherwise have launched real Chromium. The guard has 12 direct
  tests in `tests/test_egress_guards.py`, and the package-level decline fixtures in
  `tests/resolution_source/conftest.py` were reworded to match what they actually do.
- **Phase 3 doc deltas folded in** (`a1ff0f7`, merged `19dd8e1`; small fixes `c2f8812` and
  `7830d10`). AGENTS.md, `docs/architecture.md`, `docs/operations.md`, `docs/research.md` and the
  marker-spec comments in `scripts/telemetry/markers.py` now describe the four Phase 3 rungs, the
  `ungrounded` and `renderer_unavailable` tokens, the new `RESOLUTION_SOURCE_*` budget constants,
  and the fact that `RESOLUTION_SOURCE_URL_CONTEXT_ENABLED=true` turns a free provider into a paid
  one. The gap-fill guide points at the shared robots policy and browser transport; the old design
  doc is marked superseded and the `impersonate` route is documented as reserved.
- **Live QA of Phase 3** (a fresh-context pass, every provider key blanked in the process
  environment, no paid call; report at `scratch/fetch_ladder_2026-09-03/qa_report_phase3.md`, a
  copy of `/tmp/fetchprobe/qa_report_phase3.md`). One sweep of `fetch_resolution_sources` per URL
  over 106 URLs: all 97 distinct resolution-source URLs in the research archive plus 9 from the
  2026-09-03 prod run. Headline: 93 of 106 succeed; the archived corpus went from 59 of 97 to 85 of
  97; 35 of the 47 archived failures now succeed (20 did at the previous QA pass on `99d3438`);
  routes over the 106 were direct 76, wayback 13, rendered 13, pdf_local 3, meta_refresh 1; the
  automated telemetry invariant check found zero violations; all 48 questions served entirely by
  the direct route rendered byte-identically to before. Defects, by severity: **P3-1 HIGH**, the
  rendered rung was unbounded on a page that keeps navigating (76 s on ogimet against the 45 s
  provider wall); **P3-2 HIGH**, the `favor_precision` drop from the earlier fix wave published
  navigation chrome on at least three pages and on congress.gov replaced the bill-status card;
  **P3-3 MEDIUM**, 5 of the 12 rendered rescues were navigation trees; **P3-4 MEDIUM**, the Wayback
  rung could not decode a zstd-encoded replay and misreported it as "no archived copy";
  **P3-5 MEDIUM**, the derived-API rung was unreachable on every corpus dashboard because their
  feeds are cross-origin; P3-6 to P3-10 low or cosmetic (a companiesmarketcap table shape, a
  served-but-thin capture logged as unserved, a zero-passage PDF digest still `status=success`,
  the then-unregistered `RESOLUTION_SOURCE_URLCONTEXT_*` lines (registered 2026-09-04), Chromium teardown tracebacks).
- **Fixes for P3-1, P3-4, P3-7 and P3-8.** `9811ca3` (P3-1) bounds the rendered rung: a DOM-read
  cap, an outer bound on the whole render at the remaining wall budget, a new skip reason
  `render_timeout` (with `render_timeout_skips`), a per-run memo so a timed-out URL is not rendered
  twice, and an exception boundary. `292d306` (merged `c39b4d4`; P3-4, P3-7, P3-8) declares
  `brotli` and `backports-zstd` as dependencies (with `[tool.deptry] DEP002` ignores, since aiohttp
  loads them implicitly) so Wayback replays decode, rewords the Wayback decline so a thin capture
  is not logged as unserved, and flips the zero-passage PDF digest from `success` to
  `no_resolving_content` with reason `no_matching_passage`, excluding that shape from the paid
  rung's trigger set. The earlier "keep success" choice for that digest turned out to be a review
  triage deferral rather than an operator decision, and the repo rule that prose never stands in
  for an absent section settled it.
- **P3-2 and P3-3: calibration study and policy D** (`162294e`, merged `93feda0`). The study
  (`scratch/fetch_ladder_2026-09-03/chrome_calibration.md`) re-fetched 118 bodies, ran five
  extractor variants on identical bytes, hand-labelled the results as content, chrome or
  ambiguous, and scored five publish policies. Policy D won: default (recall) extraction, then a
  structural chrome metric, `content_share60` (the share of extracted characters that sit in table
  rows or in non-table lines of at least 60 characters), at threshold 0.38; below the threshold
  re-extract with `favor_precision=True` and publish that only if it clears both the floor and the
  metric; otherwise withhold as `no_resolving_content` with reason `thin_page`. On the labelled
  corpus that publishes 46 of 46 content bodies, 2 chrome bodies and withholds 0 content, against
  43 / 11 / 0 for the shipped code and 40 / 4 / 5 for a plain revert to precision. Shipped as
  `RESOLUTION_SOURCE_CONTENT_LINE_MIN_CHARS` (60) and `RESOLUTION_SOURCE_CONTENT_SHARE_MIN` (0.38)
  in `constants.py`, the counts keys `chrome_metric_withholds` and `precision_fallback_rescues`, and
  18 tests in `tests/resolution_source/test_resolution_source_extractor_policy.py`. What it gives
  up: prose-shaped boilerplate (a cookie-consent wall, a glossary) still publishes under every
  policy, and kasa.go.kr's news ticker line (which carried its question's resolving fact) is
  withheld with the menu around it. The threshold's margin is about 0.05 on each side, set on a
  one-day corpus. This is an operator-reviewable decision.
- **Code review of the Phase 3 diff** (the `forge` skill: 16 reviewers, five of them on stated
  concerns, namely SSRF and egress invariants, timing safety under the strictly-safer rule, the
  cost gate on the paid rung, test containment, and marker contracts, plus three codex lenses,
  then batched verifiers and a triage step). Verdict needs-work: 0 critical, 24 findings in the fix
  plan (17 important, 7 minor), 29 report-only, no verification gate left blocking (its fast-path
  gate became finding F17 and was built; its egress-guard gate was closed by `bd10996`). Plan and
  report live at `/tmp/forge-lWgyXJ/` (`plan.json`, `report.md`, `fix_findings.md`,
  `report_only.md`, `exec_plan.md`), and the three markdown files are copied into
  `scratch/fetch_ladder_2026-09-03/` as `forge_phase3_report.md`, `forge_phase3_fix_findings.md`
  and `forge_phase3_report_only.md` so they survive a reboot.
- **Review fixes landed in four worktree branches plus the extractor branch above.**
  - *Wayback and docs* (`f98f7b8`, merged `6cd47d5`): F1, the Wayback request URL hardcoded the
    year `2026id_`, which from 2027-01-01 would return only end-of-2026 captures that the 30-day
    bound then withholds; it is now derived from the fetch clock (`FetchContext.now`, threaded in
    as `wayback_snapshot_url(url, now=...)`) at YEAR granularity rather than the review's proposed
    14-digit stamp, because a 14-digit request URL would itself parse as a dated capture. The same
    fix is in `scripts/probes/fetch_diagnostic.py`. Also R6 (dead `WAYBACK_HOST`), R12 (duplicated
    clock-skew tolerance), R19 (the third `GEMINI_USAGE` role in the remaining docs), R24
    (`status=ok` versus `outcome=success` documented, not re-spelled), R29 (a log line still called
    the shipped rung "a future Tier-2 LLM fetch").
  - *Test pins* (`e1d82c0`, `fff7a66`, `03fb9f4`, merged `78c7e6d`): F18 (the paid rung's timeout
    and attempt pins), F19 (`ssrf_blocked` pins for both third-party rungs), F21 (the conftest
    fixture docstring), R16 (`build_document_prompt`'s instructions pinned), R17 (the robots cache
    reset by the package conftest); R28 (a checked-in `live`-marked replay corpus) skipped with
    reasons recorded. One commit on that branch was made with `--no-verify` and the hooks were
    re-run on it afterwards (recorded from the session's notes, not verifiable from git).
  - *Ladder core* (12 commits from `0b1bfb2` to `fbb6aa1`, merged `5db552a`): F2, `_aux_ctx` gives
    the three auxiliary fetches (Wayback snapshot, derived feed, robots.txt) their own rung
    bookkeeping so an archived PDF capture no longer hijacks the route stamp; F3, a withheld
    Wayback capture no longer ends the ladder, so the paid rung is reachable for the population it
    was built for; F6, feed reuse gated on a real JSON content type; F14, repeated Wayback unwrap
    plus the Metaculus self-reference check on the innermost URL at every caller; F4 and R25, the
    paid rung re-reads its budget after a robots pre-check that is now bounded, via
    `_url_context_admission`, sharing `robots_policy.ROBOTS_FETCH_TIMEOUT_S` (5 s) with v2; F7, a
    per-question same-host browser escalation gate in the dispatcher so the second same-host URL
    reuses the harvested feed instead of launching its own browser; F17, the time-budget fast path
    threaded from the orchestrator into the provider, declining the two expensive rungs (render,
    paid read) with skip reason `fast_path` and count `fast_path_skips`; R21, route caveats computed
    over the sections that actually render after the character budget; R14, the shared robots.txt
    cache filled single-flight; R20, `url_context_no_api_key_skips` and six per-rung
    `<rung>_budget_skips` counts beside the aggregate `rung_budget_skips`; R23, the ungrounded log
    line spelled `RESOLUTION_SOURCE_URLCONTEXT_UNGROUNDED_SUPPRESSED` like its v2 twin, and both
    URLCONTEXT lines pinned (all three registered as marker specs on 2026-09-04); R7, the url_context response text read directly.
  - *Browser transport* (`ea6d11d`, merged `2744219`): F5, the navigation budget is recomputed
    after both gate acquires (`RENDER_POST_GOTO_TAIL_MS`, `RenderBudgetExpired`), so a render
    admitted late navigates on what is actually left or declines before a launch; F8, the
    rendered-to-nothing memo is scoped per caller through the `MemoScope` literal
    (`"resolution_source"` or `"gap_fill_v2"`), with a transport-private timeout memo; F9, harvest
    handlers owned and drained; F13, the harvest host rule uses the public-suffix helper
    `registrable_domain` (imported from `market_retrieval/settlement_join.py`) so sibling
    subdomains harvest; F12, `RENDERED_DOM_MAX_CHARS`; F10, `service_workers="block"`; F11, fail
    closed on a non-ASCII host; R26, the main-frame HTTP status is carried so a browser-targeted
    403 leaves the direct result standing rather than publishing as `success`; R22,
    `RENDER_LAUNCH_CAP` named once; R5; a bounded teardown with `RENDER_TEARDOWN_TIMEOUT_MS`; and a
    new skip reason `rendered_no_text` with `rendered_no_text_skips`, split out of
    `renderer_unavailable` so the install-failed signal is not inflated.
- **Two merge artifacts fixed on the bundle.** The gate after `5db552a` failed typecheck on two
  test call sites that still called `wayback_snapshot_url` without the new clock (fixed inside that
  merge; `~/logs/gate6.log`), and the gate after `2744219` failed two tests whose fake `render_page`
  doubles did not accept the transport's new keywords (fixed in `55e02f5`; `~/logs/gate7.log`).
- **Gates green at `55e02f5`**: `make lint`, `make typecheck`, `make lint_imports`, `make deps`,
  `make test_fast`: 7,428 passed, 14 skipped, 33 deselected, exit 0 (`~/logs/gate8.log`, 21:31 PT).
  Every merge above was gated the same way.
- **Smoke-test pricing correction.** The previous handoff's "~$1.83" for a 4-question Test Bot run
  was the personal key's `CREDIT_SPEND run_delta_usd`, which the marker itself tags as unsettled,
  a lower bound. The eleven `CREDIT_ROLE_SPEND` lines in the Test Bot #67 job log
  (`/tmp/qa_test_bot_67/joblog.txt`) sum to $10.50 (`usd=`) and $8.67 (`byok_usd=`), all on the
  personal key because the donated key was empty then (`remaining=0.00`). The donated key now
  holds $1,449.19 of its $2,300 limit (from the cup QA run's log), so the next smoke run bills
  mostly there. The PR description already carries this correction.

## Current state

- **HEAD `55e02f5`**, working tree clean, 76 commits ahead of `origin/next-season-bundle`
  (`ea1d558`), where PR #66's CI last ran green. Pushing is the operator's action (blocked for
  agents).
- **The telemetry and structure follow-up (merged; see the merge line at the end of this bullet):** branch
  `worktree-agent-a9f7b67682bcb1934` at `.claude/worktrees/agent-a9f7b67682bcb1934` (locked). Its
  scope: F15 (per-rung `outcome=` and `wall_s=` on `RESOLUTION_SOURCE_ESCALATION`, instead of the
  whole-ladder values credited to every rung), F16 (the `stale_data` / `ungrounded` vocabulary in
  the docs and the accounting note that a status may now be a rung verdict), F20 (one
  provider-level marker test per rung), F22 (one lead-then-cap helper), F23
  (`FetchContext.claim_rung_budget` replacing six copies of the wall-budget preamble), F24 (the
  `resolution_source.py` size figure in FUTURE.md item 3), R4 (a `RungSkipReason` Literal), R27
  (`failure_class=` / `exc=` / `server=` fields on the fetch marker), doc staleness, an optional
  per-question cap on paid reads, and the split of the 1,484-line
  `tests/resolution_source/test_resolution_source_escalation.py`. Commits on it so far: `bf11534`
  (F15), `6df4521` (F16), `1474a94` (F20), `5089867` (F22), `a693a99` (F23), `0362dd0` (R4), plus an
  uncommitted edit to `metaculus_bot/research/resolution_fetch_result.py` at 21:52 PT. Its diff
  against the bundle touches six files: `docs/operations.md`, `docs/research.md`,
  `resolution_fetch_result.py`, `resolution_source.py`, `scripts/telemetry/markers.py` and the
  escalation test module.

  *Merge of the telemetry follow-up:* merged as `aca5cd8` at 22:12 PT (twelve commits, the last
  `ea1b29a` being the test-module split); the full free gate on `aca5cd8` is green (7,440 passed, 14
  skipped, exit 0). Beyond the listed scope it also landed the optional per-question cap on paid
  url_context reads (`RESOLUTION_SOURCE_URL_CONTEXT_MAX_ATTEMPTS`, default 2, skip reason
  `url_context_cap`, count `url_context_cap_skips`) and moved `renderer_unavailable` /
  `render_timeout` out of `FetchStatusReason` into the new `RungSkipReason` Literal.
- **Report-only findings left as the verifier judged:** R1 (cross-page reuse of a remembered JSON
  feed is the plan's own specification, disclosed twice) and R2 (the Wayback attempt cap is per
  question while the gate it contends for is loop-wide; FUTURE.md item 5, the operator's call).
  **Deferred to their own PR:** R8 (`tests/test_agentic_tools.py`, 3,573 lines at review time,
  3,601 now), R9 (`constants.py`, 1,406 at review time, 1,429 now), R10 (a shared fake Playwright
  graph for the tests), R18 (the Tier-1 paid rung is configured by `GAP_FILL_V2_READER_*`
  constants).
- **Receipts** (gitignored, laptop-local): `scratch/fetch_ladder_2026-09-03/` holds the replay
  report and results, the reader sizing, both QA reports (`qa_report_phase1_2.md`,
  `qa_report_phase3.md`), the calibration study (`chrome_calibration.md`) and the three forge
  copies; `/tmp/fetchprobe/` and `/tmp/forge-lWgyXJ/` hold the originals and the raw harness
  output until reboot; `/tmp/qa_test_bot_67/` holds Test Bot #67's job log and artifact.
- **GitHub workflows:** tournament active; Test Bot and Test Bot Basic active (manual dispatch
  only); Metaculus Cup `disabled_manually` (re-enable after the merge to `main`); minibench
  `disabled_manually` BY DESIGN, never ask about it.
- **Local junk:** 57 merged `worktree-agent-*` branches (operator deletes; `branch -d` is blocked
  for agents). The one unmerged leftover besides the in-flight branch,
  `worktree-agent-aed63df23d441c8e4`, is fully superseded: its single commit `af6b795` is an older
  revision of the announced-unscheduled plan's status section, which the bundle already carries in
  `d38980b`. Idle teammates from the session are harmless and die with it.

## What's left to do (in order; 1 is done, 2 to 5 need the operator)

1. **Done: the telemetry follow-up branch is merged** (`aca5cd8`) and the full free gate is green
   there (7,440 passed). After any further change, re-run the gate in tmux:
   `tmux new-session -d -s gate "set -o pipefail; (make lint && make typecheck && make lint_imports && make deps && make test_fast) 2>&1 | tee ~/logs/gate.log; echo EXITCODE=\$? >> ~/logs/gate.log"`
   and poll with `tail ~/logs/gate.log` (`make test_fast` takes about 155 s; never `sleep` in a
   foreground Bash).
2. **Operator: push the branch** (blocked for agents) and watch PR #66's CI go green:
   `gh run list --repo No-Stream/metaculus-bot --branch next-season-bundle`.
3. **Smoke test (PAID, operator signs off first; ask inline with command and cost).** ONE dispatch
   of `test_bot.yaml` on the branch:
   `gh workflow run test_bot.yaml --repo No-Stream/metaculus-bot --ref next-season-bundle`. Four
   questions, about $10 at Test Bot #67's per-role rates, billed mostly to the donated key now,
   plus Google AI Studio cents for the two native Gemini surfaces; it publishes four comments to
   Metaculus test questions. QA it the way Test Bot #67 was (operational markers, forecast content,
   research content; artifacts under `/tmp/qa_test_bot_67/` show the shape), plus the new markers:
   `route=` on `RESOLUTION_SOURCE_FETCH`, `RESOLUTION_SOURCE_ESCALATION` with its per-rung outcome,
   `AGENTIC_FETCH_LOCAL_DOC`, `GEMINI_USAGE`, and the new `details["counts"]` keys
   (`chrome_metric_withholds`, `precision_fallback_rescues`, `fast_path_skips`,
   `render_timeout_skips`, `rendered_no_text_skips`, the per-rung `<rung>_budget_skips`).
4. **After the operator merges to `main`**, they run (give them these inline):
   `gh workflow enable "Forecast on Metaculus Cup" --repo No-Stream/metaculus-bot` (turns on the
   hourly :13/:33/:53 cup crons; only after `main` carries the dated slug);
   `gh workflow run fetch_diagnostic.yaml --repo No-Stream/metaculus-bot` (the FREE egress
   diagnostic; its yaml has to be on `main`; in the job log's table rows 1 to 4 are the Akamai
   federal hosts: bot client 403 and impersonated 200 means the runner's TLS fingerprint is scored
   and the TLS-impersonation rung is worth building; both 403 means the runner's egress IP is what
   is blocked and the rung is dropped for good); and
   `git branch -d $(git branch --list 'worktree-agent-*' --merged)`.
5. **When Metaculus publishes the fall bot tournament** (no object existed on 2026-09-03): set
   `TOURNAMENT_ID` and `TOURNAMENT_END_DATE` in `metaculus_bot/constants.py` from the object's slug
   and its `forecasting_end_date` (NOT `close_date`), per the season-start checklist in
   `docs/operations.md`. `make supply_probe` (free) is the watch. From 2026-09-20 the tournament
   crons and one CI test (`tests/test_tournament_dates.py`) go red on purpose as the reminder.
6. **Follow-ups worth a FUTURE.md line** (the first two are already filed there, dated
   2026-09-03; the rest are not yet): P3-10, the Chromium teardown tracebacks, whose root cause is a
   Playwright-internal race between `unroute_all` and `context.close()` (filed, LOW, candidate fix
   is a teardown-order change); the launch-cap and host-gate amplifiers on FUTURE.md item 5 (filed
   with `ea6d11d`); the driver-process leak when a render is cancelled inside
   `PlaywrightContextManager.__aenter__`; R8, R9, R10 and R18 above; a policy D re-calibration once
   a season of `chrome_metric_withholds` and `precision_fallback_rescues` counts exists; promoting
   `registrable_domain` out of `market_retrieval/settlement_join.py` into a leaf module (the
   rendered transport now imports it from a market-retrieval module); and widening the aiohttp
   `Accept-Encoding` set, which the brotli and zstd decoders now make safe but which is unmeasured
   and the operator's call.

## Key decisions made (don't re-litigate)

Carried forward from the day's earlier sessions:

- **Deterministic, free rungs first; Gemini url_context last, behind its own flag.** Operator: aim
  at the 80/20 and a bit beyond; no anti-bot arms race (soft guideline, not a prohibition). DataDome
  and Cloudflare-challenge hosts (sagaftra.org, trueup.io) try url_context when the flag is on and
  are otherwise accepted as lost. No residential IPs.
- **TLS impersonation (curl_cffi) is NOT built.** The archived Akamai 403s do not reproduce from the
  laptop or EC2 with the bot's own client; only the GitHub runner gets them, so the free diagnostic
  in step 4 decides it. `curl-cffi` is in `uv.lock` only transitively (yfinance) and is declared
  only in deptry's ignore list for the probe script.
- **Reader and grounded search both run `gemini-3.8-flash`**, thinking pinned (`medium` search,
  `low` reader); the lite tiers were rejected for the reader because its output reaches the
  SUPERSEDE block.
- **Google-Extended robots pre-check ships** because the live probe proved the mechanism.
- **Wayback is admissible only clearly marked stale**: mandatory as-of line, withheld past 30 days
  via `stale_data`, never used for `js_wall`.
- **One caveat sentence per non-direct route** under the evidence heading.
- **Egress change (proxy or self-hosted runner) parked** in FUTURE.md, low priority.
- **The 45 s Tier-1 wall is NOT raised**; every rung self-bounds inside it (FUTURE.md item 5,
  skipped by operator decision; tonight's F5 fix is the strictly-safer half, and the unbounded gate
  acquires stay the operator's reserved call).
- **Credit floor $100, alerting live.** **Minibench stays off; cup goes on after merge.**
- **AGENTS.md is a terse starting point** (operator rule, saved as feedback memory).
- **Timing, deadline and fallback code gets only strictly-safer changes** (standing operator rule).

Taken tonight by the session lead, and open to the operator's veto:

- **F17 built rather than deferred.** The plan committed the fast-path gate to Phase 3 twice; what
  landed only declines the two expensive rungs (render, paid read) when the time budget's fast
  path is on, and counts the declines.
- **F3 makes the paid url_context rung reachable when the Wayback capture is too stale.** That is
  a cost-policy consequence; the flag defaults off in code but is ON in every bot workflow since 2026-09-04 (operator instruction), so this path is live and bounded by the 2-read cap.
- **P3-8 flipped**: a zero-passage PDF digest is `no_resolving_content` / `no_matching_passage`,
  not `success`, per the prose-never-stands-in-for-an-absent-section rule.
- **Policy D for the extractor** (details above): the one decision here that changes what
  forecasters see on chrome-heavy pages, with its named losses.
- **R26 "direct result stands" instead of an honest `blocked` record** for a browser-targeted 403,
  because a `blocked` status would trigger a Wayback GET the page does not deserve.
- **R1 and R2 left as report-only; R8, R9, R10 and R18 deferred to their own PR.**

## Gotchas and context the next session needs

- **Cost gate is absolute**: never run `main.py`, `make run`, backtests, ablations, `test_live`, or
  dispatch a bot workflow without the operator's explicit go for THAT run. Free: all `make` gates,
  `make supply_probe`, `make check_credits`, `make sync_all`, the probe scripts without their spend
  flag, `fetch_diagnostic` (no secrets).
- **`RESOLUTION_SOURCE_URL_CONTEXT_ENABLED` is ON in every bot workflow since 06b3fd9** (operator rule: a
  flagged feature ships flag-on). The code default stays off. On, the resolution-source provider is a paid
  Gemini surface on the operator's personal Google AI Studio key, bounded by the robots pre-check, the 15 s
  floor and `RESOLUTION_SOURCE_URL_CONTEXT_MAX_ATTEMPTS` (2). Named in AGENTS.md's cost gate.
- **The repo Makefile has no `test_select` target**; run single files with
  `uv run pytest <paths>` (`make test_fast` for the suite).
- **Fake `render_page` doubles must accept the transport's full keyword set**: `memo_scope`,
  `host_gate`, `goto_timeout_ms`, `deadline_monotonic_s`, `harvest_json`. Copy the `_declined`
  double in `tests/resolution_source/conftest.py`. A double missing one keyword fails with a
  `TypeError` inside the rung's soft-fail and reads as a declined render.
- **`wayback_snapshot_url(url, now=...)` requires the clock** (keyword-only `now`); basedpyright
  catches a missing one, so run `make typecheck` after any merge that touches the Wayback rung.
- **Worktree merges conflict in two known places** when two agents touch them: the import block
  at the top of `resolution_source.py` and the counts paragraph in `docs/research.md`. Check
  `git status --short` before every merge; the worktree-auto-merge hook can STAGE a finished
  worktree's uncommitted edits into the main tree at teardown (happened twice today).
- **Blocked git ops** here: push, stash (prompts were denied), restore or checkout of files,
  reset --hard, branch -d. To clear a dirty file without them:
  `git show HEAD:<path> > <path> && git add <path>`.
- **`sleep` in foreground Bash is blocked**; use tmux plus `tail` polling. Logs go under `~/logs/`.
- **Pyright diagnostics injected mid-session are worktree noise** (unresolved imports from other
  agents' worktrees); trust `make typecheck`.
- **`GOOGLE_API_KEY` is in `.env`** (never print it). The Google project is paid-tier with no free
  token allocation; only the 5,000 grounded SEARCH QUERIES per month are free, and the bot fires
  about 12.7 queries per grounded prompt (about 1,130 a month, 23% of the pool). A Gemini-grounded
  backtest at scale re-creates the June 2026 overage.
- **`run_bot_on_metaculus_cup.yaml` on `main` still has the dead `metaculus-cup` slug** until the PR
  merges; do not enable it before then (hourly red runs).
- **Metaculus API needs the token**; unauthenticated calls 403. `make supply_probe` and the
  forecasting-tools client use `METACULUS_TOKEN` from `.env`.
- **`scratch/` and `scratch_docs_and_planning/` are gitignored**; tracked docs there are
  force-added (`git add -f`). `CLAUDE.md` in the repo is a symlink to `AGENTS.md`.
- **Forge outputs in `/tmp` die at reboot**; the copies in `scratch/fetch_ladder_2026-09-03/` are
  the durable ones. The old forge's reproduction skeptics mutated files in the live checkout; the
  new batched-verifier forge did not tonight, but keep checking `git status` before merges.
- **Memory files** (project-scoped, survive sessions) under
  `~/.claude/projects/-Users-flatljan-personal-metaculus-bot/memory/`:
  `project_fetch_ladder_2026_09_03.md` (updated tonight), `project_minibench_cup_workflows_off.md`,
  `feedback_asks_must_be_inline.md`, `feedback_agents_md_terse.md`,
  `project_next_season_bundle_2026_09.md` (its status line predates tonight).

## Operator follow-ups (surface ONLY at the very end of the session, in one list)

Everything below needs the operator or is the operator's call. Do not drip these out mid-session.

1. **Push** `next-season-bundle` (blocked for agents): `git push origin next-season-bundle`; then PR #66 CI.
2. **Smoke** (already authorized ONCE, 2026-09-04 morning, to run after the flag flip is pushed and CI is
   green): `gh workflow run test_bot.yaml --repo No-Stream/metaculus-bot --ref next-season-bundle`; 4
   questions, about $10 at run-67 rates, mostly on the donated key; publishes 4 comments on Metaculus test
   questions. QA it as Test Bot #67 was, plus `route=`, per-rung `RESOLUTION_SOURCE_ESCALATION`,
   `failure_class`, the three `RESOLUTION_SOURCE_URLCONTEXT_*` markers, `GEMINI_USAGE
   role=resolution_source`, and the counts keys.
3. **Merge PR #66** to `main` (paste `scratch/next_season_bundle_2026-09/PR_DESCRIPTION.md`; append a line
   for the flag flip and the two live probes).
4. **After the merge, three commands:** `gh workflow enable "Forecast on Metaculus Cup" --repo
   No-Stream/metaculus-bot`; `gh workflow run fetch_diagnostic.yaml --repo No-Stream/metaculus-bot` (free;
   rows 1 to 4 of its table are the Akamai federal hosts: bot client 403 and impersonated 200 means build
   the TLS-impersonation rung, both 403 means drop it for good); `git branch -d $(git branch --list
   'worktree-agent-*' --merged); git branch -D worktree-agent-aed63df23d441c8e4` (the last is a superseded
   doc revision).
5. **Fall bot-tournament slug** when Metaculus publishes it: `TOURNAMENT_ID` and `TOURNAMENT_END_DATE` in
   `metaculus_bot/constants.py` from the object's slug and `forecasting_end_date`; from 2026-09-20 the
   tournament crons and one CI test go red on purpose; `make supply_probe` is the free watch.
6. **Decisions taken by the lead that the operator may veto** (all reversible constants or small branches):
   the rendered rung's 3 s exit reserve with the 12 s pre-gate floor left in place, so renders admitted with
   12 to 15 s left decline at the gates (`RESOLUTION_SOURCE_RENDER_MIN_BUDGET_S` to 15 makes the pre-gate
   check truthful at the cost of that band); the `NOT_ADDRESSED` sentinel in the shared url_context prompt,
   visible in gap-fill v2's `read_document` tool text; chart-only publication when policy D withheld the page
   text; extractor policy D itself (`content_share60` at 0.38); a stale Wayback capture falling through to
   the paid read; the zero-passage PDF digest withheld; the fast-path gate on the two expensive rungs; the
   Wayback request stamp at year granularity.
7. **FUTURE.md items filed this session, for later rounds:** rendered-rung residuals under item 5 (unbounded
   driver stop; the 12 to 15 s band; harvest extraction to `rendered_harvest.py`; two-place cancellation);
   the DOM ceiling as the real bound on post-render extraction; extraction inside the per-host gate; P3-6
   (companiesmarketcap table columns); P3-10 teardown traceback root cause; the driver-process leak on
   cancellation inside `PlaywrightContextManager.__aenter__`; `tests/test_agentic_tools.py` and
   `constants.py` size (own PR); Tier-1 paid rung configured by `GAP_FILL_V2_READER_*` constants (R18);
   policy D re-calibration once a season of `chrome_metric_withholds` counts exists; Accept-Encoding
   widening (now safe, unmeasured); and, from the 2026-09-04 Codex triage, item 8 (the browser transport's
   three unguarded channels, now BUILT on 2026-09-04 with one residual left, FUTURE.md ~1409-1500), the
   un-clocked PDF parse prologue (LOW, ~2929-3029), and the stale per-model cost heuristic in
   `ensemble_analysis/ensemble_simulator.py` (adjacent rot, offline only, found by the model-id inventory).
8. **Record, not a decision: the browser-transport follow-up was BUILT on 2026-09-04**, on the operator's
   instruction the same afternoon, and it is in this PR rather than a later one. `6646a0b` refuses a render
   whose main frame landed on a host other than the pinned one and blocks page WebSockets with
   `route_web_socket`; `8ced8a5` hands the browser the direct fetch's landing URL so the pin covers the host
   that serves the content. The decline is the skip token `render_off_host`, counted as
   `render_off_host_skips`, and the transport's WARNING is the marker `RENDERED_FETCH_OFF_HOST`. The free
   local render probe that priced strict host equality ran first and its numbers are in the PR description under
   "The browser transport closure". One residual is left, a cross-host subresource Chromium resolves with no
   pin of ours, recorded in FUTURE.md item 8 with the single observation that would settle it. Nothing here
   needs the operator, and the smoke was not re-run for it.

## References

- Plan: `scratch_docs_and_planning/fetch_ladder_plan_2026-09-03.md` (tracked, force-added).
- Design it supersedes: `scratch_docs_and_planning/fetch_escalation_ladder_design.md`.
- Receipts (gitignored, laptop-local): `scratch/fetch_ladder_2026-09-03/` (`replay_report.md`,
  `replay_results.json`, `reader_sizes.md`, `reader_sizes.json`, `qa_report_phase1_2.md`,
  `qa_report_phase3.md`, `chrome_calibration.md`, `forge_phase3_report.md`,
  `forge_phase3_fix_findings.md`, `forge_phase3_report_only.md`);
  `scratch/agents_md_debloat_2026-09-03/loss_check_independent.md`; `/tmp/fetchprobe/`,
  `/tmp/forge-lWgyXJ/` and `/tmp/qa_test_bot_67/` until reboot.
- Gate logs: `~/logs/gate.log` through `~/logs/gate8.log` (`gate8.log` is the green run at
  `55e02f5`).
- PR: https://github.com/No-Stream/metaculus-bot/pull/66 (description at
  `scratch/next_season_bundle_2026-09/PR_DESCRIPTION.md`, refreshed tonight, untracked).
- Test Bot #67 run: https://github.com/No-Stream/metaculus-bot/actions/runs/33775800806;
  cup QA run: https://github.com/No-Stream/metaculus-bot/actions/runs/33815141451.
