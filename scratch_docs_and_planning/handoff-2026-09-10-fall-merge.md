# Fall 2026 merge: what landed on 2026-09-10, and what is left

**Date:** 2026-09-10, early afternoon Pacific. Supersedes `handoff-2026-09-09-fall-config.md`,
whose "What's left to do" list is now mostly done (see "Status of the 09-09 list" below).
**Branch:** `mantic-competition`, tip `6551ae1`, 64 commits past the 09-09 handoff tip `6f2f051`.
Tree clean, nothing pushed.
**Second branch, NOT yet merged:** `fall/ladder`, tip `74d2d56`, 14 commits, worktree
`/Users/flatljan/personal/metaculus-bot-wt/ladder` (own `.venv`, tree clean).
**Repo:** `/Users/flatljan/personal/metaculus-bot` (operator's laptop, Python 3.12, uv).
**Nothing is running.** Every agent from the 2026-09-09 and 2026-09-10 sessions has exited. The
10-minute supervision cron job was cancelled. No paid command ran today beyond the one bench below.

## What this work is

The repo is a Metaculus forecasting bot: it researches each question with several providers in
parallel, adds two gap-fill passes, asks three frontier LLMs, publishes the median as a comment, and
is scored on spot peer. The summer season closed 2026-09-06; the fall season's first question closes
2026-09-28. Everything here belongs to ONE merge that has to land before that date.

## Where things stand

### Merged into `mantic-competition` and verified

Four of the five planned chunks are in, each merged with `--no-ff` and gated afterwards:

| Chunk | Merge commit | What it does |
|---|---|---|
| Gap-fill v1 grade filter | `3143de5` | The analyzer grades each gap (`answerable_now`, `already_in_first_pass`, `same_need_as`); code drops the failing ones before the resolver spends. New `GAP_FILL_V1_TRIAGE` marker with per-reason drop counts. Cap stays 4. |
| Housekeeping | `4602c2c` | The two ablation broad-excepts removed with real fixes, `tests/test_cli.py` split into `tests/cli/` (137 tests, unchanged), the post-651 Mantic smoke reconciled across `docs/operations.md` and `FUTURE.md`, the Kalshi host check recorded. |
| Page digest | `50f1d1d` | `metaculus_bot/research/page_digest.py`: the `page_digest_extractor` role, luna at medium effort, literal grounding check, BM25 pre-filter and fallback, six constants with receipts, three optional marker fields. |
| Known-API registry | `b7c2ecf` | `metaculus_bot/research/known_api/`: `translate(url)` over 21 observed URL shapes, FRED / Yahoo / market-snapshot / SEC EDGAR backends, three `ToolSpec`s, two adapters, the two URL-parsing gaps closed, `SEC_EDGAR_CONTACT_EMAIL` passed into the four prod workflows. |

Both doc conflicts at the known-API merge (an `AGENTS.md` layout row and a `docs/research.md`
subsection inserted at the same anchor by two branches) were resolved by keeping both sides.

### Also landed on `mantic-competition` today

- **Era tagging for the fall read** (`5668116`, `26bf8fb`): the tagging table would have put all six
  fall questions in a `fall_target` sub-era, leaving the preregistered `fall_config` arm empty
  forever. Boundary chains replaced with data tables; the cost-pass merge is now one appended row.
  `era_gap` exits cleanly with a message when an arm has no scoreable records.
- **Private-comment reader** (`367b67c`): `performance_analysis/collector.py` and
  `scripts/backfill_research_from_comments.py` now list both public and private comments.
- **`same_need_as` fail-shut softened** (`502b09d`): a missing key reads as null; the two boolean
  grades stay required. Without this, an analyzer that omits null keys would have switched gap-fill
  v1 off entirely.
- **Test-isolation fix** (inside `847c6f1`): a root-conftest autouse fixture zeroes the five
  module-global degradation counters around every test. Two tests were leaking `provider_degradation`
  into `alertable_count` assertions. Note the commit message describes only an ablation comment
  sweep: a shared-index collision swallowed the fix's four test files. Amend was blocked; the content
  is correct and in the tree.
- **Ablation broad-excepts, all eight gone** (`09a0fbb`, `5bd681d`, `fe87fd7`, `c1090b7`, plus
  docstring commits). Three deliberate behaviour changes for the operator's awareness: a bug in a
  stacker or median call now aborts the stacker batch (it resumes from cache), a bug in the prune
  stage stops the stage rather than nulling every batch, and a bug in one forecaster loses that
  question's in-flight siblings rather than the whole run. Expected failures are still absorbed per
  question.
- **Comment-debt sweeps**, each verified AST-identical with docstrings stripped, in
  `metaculus_bot/ablation/{forecasters,run_stacker,prune,qa_iterate,claude_cli}.py` and five ablation
  test modules; 185 function-level imports hoisted; three silencing markers deleted by fixing the
  code they excused. Three `monolithic-file-loc` findings on the big ablation test modules are
  deliberately left.
- **Stale pointers retargeted** (`a65a9a0`, `bdea730`, `594a08b`, `016d1c1`, `4773f14`): every
  `main.py:NNN` and sibling line-number pointer now names a module and symbol. One adjacent fix the
  stop-gate required: `ensemble_simulator.py`'s blanket `except Exception` narrowed to `ValueError`
  (`f619477`), which its own scorers raise for every unscoreable case.
- **Comment privacy documented** (`6551ae1`): a new `docs/operations.md` section. This was settled
  today, see "Decisions" below.

**Gate state.** A full certification ran at `81e3a6e`: `make test` 10,032 passed, 33 skipped, 5
deselected, `make lint` clean, `make typecheck` 0 errors, HEAD unchanged across the run. Every commit
after that is a docs, comment or pointer change plus the one narrowed catch, each gated by its own
agent. A fresh full run was launched at `6551ae1` in tmux session `fall_gates`, log
`~/logs/fall_gates_2026-09-10.log`; read its tail and the `EXITCODE=` line before trusting the tip.

### The fetch-ladder branch, `fall/ladder`: half done, NOT merged

This is the plan in `scratch_docs_and_planning/fetch_ladder_unification_plan_2026-09-09.md`: give the
resolution-source fetcher and the gap-fill v2 agentic loop ONE shared fetch ladder instead of two
overlapping ones. Six migration steps, each one commit that reverts on its own. **Steps 1, 2 and 3
are done; 4, 5 and 6 remain.**

The decision log is `scratch_docs_and_planning/fetch_ladder_progress_2026-09-10.md` (644 lines, in the
worktree, tracked). It is written for a zero-context successor and its "Next" section is the resume
point. Read it before touching anything.

Landed, tip `74d2d56`:

- Steps 1a and 1b (`1eca925`, `2e97e1d`): the ladder extracted out of `resolution_source.py` into
  `metaculus_bot/research/fetch_ladder/` (`guard.py`, `context.py`, `classify.py`, `direct_fetch.py`,
  `rungs.py`, `ladder.py`). `resolution_source.py` fell from 3,518 lines to 669 and is now the
  fetcher's adapter. Purity proven definition by definition (70 of 70, then 31 of 31, byte-identical
  with comments once module prefixes and one rename are normalised).
- Step 2 (`66cb0b4`): `LadderPolicy`, the resolution-source preset, the single entry point
  `fetch_url(url, *, policy, ctx)`, and the fetcher pointed at it.
- Step 3 (`b059275`, `e0a8164`, `c9127b9`, `0c01f88`): the verdict seat. Full gates at that tip:
  `make test` 9,811 passed / 41 skipped / 5 deselected, lint clean, typecheck 0 errors, 6 import
  contracts kept, deptry clean.
- Plus the archive replay script (`ba328f1`, `06666e3`) and progress-file commits.

**Step 3's design was a genuine re-derivation, approved by the operator.** The plan said "point the
loop at `fetch_url` with a preset". That could not work: the fetcher's classifier both READS a body
and JUDGES it, withholding text under its 100- and 400-character floors, while the loop serves any
non-empty read to a driver that will fetch again. A preset would have meant about eight caller flags
on the classifier. What was built instead: the read is shared and identical, and the VERDICT is one
policy field. The fetcher's verdict is today's two functions unchanged; the loop's is "non-empty is
content, under 500 characters escalates". Five conditions were imposed and met (both callers
byte-identical to today, the PDF parse off the archived result, robots reads outside the verdict
floors, self-produced refusals carrying `http_status=None` so the retry rungs never see them, and
defaulted new `FetchResult` fields so archived records stay identical).

## What is left to do

1. **Finish `fall/ladder`, steps 4 to 6.** Work in the worktree, one commit per step, delegating each
   step's mechanical work to an opus subagent. The progress file sizes each one:
   - **Step 4 is one line**: add `derived_api` to `GAP_FILL_FETCH_POLICY.rungs_enabled`, turning the
     derived-feed REUSE half on for the loop, with a test that a remembered endpoint is GET-ed for a
     second URL on the same host and the two render memo scopes stay separate. Everything else the
     plan listed for step 4 already arrived with the shared read.
   - **Step 5**: the fetcher gains the shared run cache, then the throttle-phrase check, then the
     digest behind the `policy.digest` seat (one line calling the already-merged
     `page_digest.digest_page`).
   - **Step 6**: deletions only. The progress file lists every function and test by name (about 30
     production functions in `agentic/tools.py`, `fetch_outcomes.py` and `local_document.py`, and 14
     tests whose subject is a deleted private function). **The host-semaphore fold is dropped** by
     operator decision: FUTURE.md item 5 blocks it, because it would put the loop's 35-second
     Chromium hold in front of a fetcher request bounded by a 45-second wall. One FUTURE.md line
     recording the re-examination is owed.
   - Then: the `caller=` optional tail field on both fetch markers, the docs (`docs/research.md`,
     `docs/agentic_gap_fill.md`, `docs/architecture.md`, the `AGENTS.md` layout row, `FUTURE.md`), a
     final full gate run with HEAD pinned, the archive replay run once with its before/after diff
     table, and `/forge` on the whole branch diff.
2. **Merge `fall/ladder` into `mantic-competition`** (`git merge --no-ff fall/ladder`), resolve
   conflicts (likely `docs/research.md`, `docs/architecture.md`, `AGENTS.md`, `FUTURE.md`,
   `scripts/telemetry/markers.py`), run `make lint`, `make typecheck`, `make test` in the main tree,
   then `git worktree remove /Users/flatljan/personal/metaculus-bot-wt/ladder`.
3. **The wiring step, AFTER that merge.** One opus agent, seven small edits the sibling branches
   spelled out:
   - `resolution_fetch_result.py`: add `"known_api"` to the `FetchRoute` Literal.
   - `agentic/provenance.py`: add `"known_api": "fetched"` to `_METHOD_TO_TIER`.
   - `agentic/tools.py` `build_gap_fill_tools`: append `known_api.tools.build_known_api_tools(...)`
     (function-scoped import, justified as a real circular import).
   - Build the `policy.known_api` seat: `translate(url)`, dispatch to the backend, `to_fetch_result`
     on `ok`, else `None`. **It must fan a two-id FRED URL out over both ids**; no consumer does that
     yet.
   - `policy.digest`: call `digest_page(text, policy.digest_query, budget_seconds=ctx.remaining_wall_seconds)`.
   - `agentic/tool_descriptions.py` `FETCH_DESCRIPTION`: one clause that the three tools return a date
     window on demand.
   - `FRED_UNKNOWN_SERIES` gains `proposed_by=gap_fill_driver` (deferred with the wiring, in FUTURE.md).
   - No new exclusion is needed in the resolution-source fetcher: `select_fetchable_urls` already
     drops FRED and Yahoo quote URLs and keeps `kalshi.com`, so a cited Kalshi URL reaches the same
     rung-0 seat.
4. **Small leftovers flagged by the sweep agents**, none blocking: a stale `noqa` path comment could
   not be found anywhere else after the fix; `tests/ablation/test_ablation_qa_iterate.py` and two
   siblings still trip the monolithic-file threshold (deferred by FUTURE.md while branches merge);
   `resolution_source.py`'s 125-line module docstring still describes a ladder that moved out from
   under it; seven ladder tests assert on records from a logger they never scoped and pass only
   because the root logger sits at WARNING.
5. **The pre-merge smoke is the operator's call**, one `test_bot_basic.yaml` dispatch at about $2,
   fired ONCE when everything else is finished. Do not propose it early.
6. **After the merge to `main`**: enable the Mantic dispatcher job with
   `make cronjob_dispatch_setup ARGS="--apply --enable-mantic"` (paid dispatches, ask first), then
   watch `make dispatch_watch` and `make cost_report`. First fall residual read at 15 resolved
   records per the preregistration.

## Decisions made today (do not re-litigate)

- **Comment privacy: private stays, by Metaculus's own request.** The FutureEval Bot Tournament
  Resources Page (Metaculus notebook 38928, read 2026-09-10) says bots "should only leave private
  comments on questions. These will automatically be made public at regular intervals for FutureEval
  tournaments", and its rules ask bots to "use private notes as their comment type". So the summer
  comments reading public and the fall ones reading private is Metaculus's weekly flip, not a bug.
  Never pass `is_private=False`. Recorded in `docs/operations.md` "Comment privacy" and in memory.
- **Keep both gap-fill passes.** The section-strip bench (below) agrees with the earlier blind
  comparison: the passes are complementary by question type.
- **No 5-replicate bench rerun** (operator, 2026-09-10): the decision it would inform is already made.
- **Step 3's verdict seat** and **dropping the step-6 semaphore fold**, both above.
- **Lean gap-fill v1 by grade, never by a positional cap**; the resolver stays `openai/gpt-5.6-terra`
  at low effort; `NATIVE_SEARCH_CONTEXT_SIZE` stays as it is. All from the 2026-09-09 cost pass.

## The section-strip bench, for the record

`make strip_bench` ran to completion overnight: 63 resolved summer questions with both gap-fill
sections archived, four arms (full, minus v1, minus v2, minus both), three replicates, forecast by
`meta/muse-spark-1.3-contributor` over cached research on the personal OpenRouter key. 756 calls,
$1.96, exit 0. Output `scratch/probes/section_strip_bench_20260910T052824Z/SUMMARY.md`.

| Stripped | All types (n 63) | Where it helped | Where it hurt |
|---|---|---|---|
| v1 | +1.73 [-0.26, +3.92] | binary +3.57, discrete +3.43 | none |
| v2 | +2.94 [+0.33, +6.17] | multiple choice +7.46 [+1.75, +14.71] | numeric -2.07 |
| both | +4.13 [+0.78, +8.01] | discrete +5.57, multiple choice +11.60 | numeric -2.61 |

Deltas are full minus arm in peer-score points, so positive means the section was helping. A
Sonnet-tier model leans on spoon-fed facts more than the frontier trio, so these are an upper bound;
the production ensemble's own published forecasts score far above every arm on binaries (37.0 versus
14.5). The cheap-bench recipe itself is worth reusing and is saved to memory.

## Status of the 2026-09-09 handoff's list

| 09-09 item | State |
|---|---|
| 1. Read the bench results | Done, relayed, rerun declined. |
| 2. Read `API_TOOLS.md` | Done. The deterministic FRED, Yahoo and Kalshi providers DO fire in production (13 of 14 identifiers returned data; the provider gates on a classifier that answers "not financial" on about 9 of 10 questions, which is why the residual round read zero). No Kalshi host drift. Nothing blocked. |
| 3a. Gap-fill v1 grade filter | Merged, `3143de5`. |
| 3b. Shared fetch ladder | Steps 1 to 3 done on `fall/ladder`; 4 to 6 left. |
| 3c. Three API tools plus rung-0 translation, EDGAR wired | Registry merged, `b7c2ecf`; the wiring is item 3 above. |
| 3d. `page_digest_extractor` role | Merged, `50f1d1d`. |
| 3e. Prompt-rule fire-rate flag | Resolved 2026-09-09: not built. |
| 3f. Tagging pass writes `fall_config` | Done, `26bf8fb`. |
| 4. Post-merge Mantic dispatcher, watches, first fall read | Still owed, item 6 above. |
| 5. Housekeeping (ablation markers, `test_cli.py` split, FUTURE/operations reconcile, Kalshi drift, FRED terms) | All done except the FRED terms-of-use question, which is the operator's licensing call. |

## Gotchas that bit today

- **Agents died twice on `Could not load credentials from any providers`** (about 04:00 and again
  around 11:00 PT). It is transient and environmental, not a code fault; the work in the worktree
  survived both times. A dying teammate sometimes leaves NO failure notice, so check
  `git -C <worktree> status` and the progress file rather than trusting the roster.
- **One agent read for 45 minutes without writing a file and then vanished.** Brief a lead to
  dispatch its implementer within about 15 minutes of orienting.
- **A subagent's context can reach ~774k tokens on a step this size.** Use a fresh subagent per step.
- **Shared-index collisions are real**: two agents staging at the same time put one's files into the
  other's commit. Commit by path, one agent per file, and expect it anyway.
- **The smell scanner reports only the FIRST path when handed several files.** Pass one at a time.
- **pytest-randomly is NOT installed here**, so test order is deterministic and `-p no:randomly` is
  silently accepted; an order-dependent failure needs explicit node ordering to reproduce.
- Everything under `scratch/` is gitignored; `scratch_docs_and_planning/` is gitignored but its files
  are tracked, so add them with `git add -f`. `CLAUDE.md` is a symlink to `AGENTS.md`.

## References

- Ladder plan and decision log: `scratch_docs_and_planning/fetch_ladder_unification_plan_2026-09-09.md`,
  `fetch_ladder_progress_2026-09-10.md` (in the ladder worktree).
- Fall preregistration: `scratch_docs_and_planning/fall_2026_preregistration.md`.
- Previous handoff: `scratch_docs_and_planning/handoff-2026-09-09-fall-config.md`.
- Cost pass and its API-tools design: `scratch/cost_pass_2026-09-09/` (gitignored).
- Bench: `scratch/probes/section_strip_bench_20260910T052824Z/SUMMARY.md`, `~/logs/strip_bench3.log`.
- Gate log for the current tip: `~/logs/fall_gates_2026-09-10.log`.
