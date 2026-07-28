# Smoke-test readiness brief — 2026-07-27

Written at HEAD `3a7a70a`, working tree clean apart from two docstring edits made while writing
this brief (both committed alongside it).

## 0. Gates, verified in this session rather than taken on report

- `make lint` — passes, "All checks passed!", exit 0.
- `make typecheck` (basedpyright) — 0 errors, 0 warnings, 0 notes.
- `make test` in tmux — **4237 passed, 0 failed, 11 skipped, 5 deselected, 106s**. The baseline
  quoted in the brief for this phase was 4174; the surplus is new tests added by this session's
  own commits, not flakiness. The 5 deselected are the `live` marker, excluded by the
  `addopts = "-m 'not live'"` line in `pyproject.toml`.
- `make check_credits` — donated key `OAI_ANTH_OPENROUTER_KEY` reads limit $850.00, remaining
  $0.00, usage $4.39. Personal key `OPENROUTER_API_KEY` reads no limit, usage $23.60. The donated
  key is genuinely drained, so a smoke run exercises the real drained-key path.

No gate is red. Nothing was papered over.

## 1. Is the smoke test ready, and what will it prove?

**Ready to fire, with one hard precondition: the dispatch must name a ref that carries the fix.**
`origin/main` is at `73e4782` and local `main` is 64 commits ahead and unpushed. I confirmed by
grep that `origin/main:metaculus_bot/fallback_openrouter.py` still carries the incident-causing
veto `if "403" in msg or "forbidden" in msg or "moderation" in msg: return False` at line 322, and
has no `key limit exceeded` cue anywhere. `DONATED_KEY_STATE`, `classify_donated_key_state`,
`FORECASTERS_SURVIVED`, and `metaculus_bot/api_preflight.py` are all absent from `origin/main`
too. A dispatch against the default branch would spend the money reproducing the incident.

The preflight, which must produce a hit before dispatching:

```
git show <ref>:metaculus_bot/fallback_openrouter.py | rg 'key limit exceeded'
```

It hits on local `HEAD` and misses on `origin/main` (both checked today). The operator has said
they fire this workflow manually against a branch of their choosing, so this is a check-the-ref
step rather than a defect. Also note `gh` has no default repo configured here (`gh repo
set-default --view` reports none), and `origin` is the fork while `upstream` is the Metaculus
template, so every `gh` command needs `--repo No-Stream/metaculus-bot` or it silently targets
upstream and cannot see these workflows.

Given a correct ref, here is what each of the six intended observations is worth.

### 1. `PAID PERSONAL-KEY FALLBACK` warnings fire — CONFIRMED OBSERVABLE

The warning is emitted from `record_donated_key_fallback` in
`metaculus_bot/fallback_openrouter.py`, reached whenever `should_retry_with_general_key` returns
True. I re-ran the behavioral matrix at HEAD against the verbatim production body and that
function returns True on a reported 403 carrying the `KEY_LIMIT_EXCEEDED_CUE` phrase.

Expect **many** warnings on one question, not one. Two of the three forecasters route
donated-first (`openrouter/openai/gpt-5.6-sol`, `openrouter/anthropic/claude-opus-4.8`), and so do
the AskNews summarizer, OpenAI native search, the financial classifier, prediction-market keyword
extraction, gap-fill v1's analyzer and each of its resolvers, and gap-fill v2's driver once per
loop step. The gap-fill v2 driver is likely the largest single contributor because it calls once
per step up to its step cap.

Two absences are expected and are **not** symptoms of partial fallback failure:

- The third forecaster, `openrouter/google/gemini-3.1-pro-preview`, never attempts the donated
  key. `DONATED_KEY_BLOCKED_GOOGLE_MODELS` pins the `gemini-3.1-pro` prefix to the personal key,
  so `should_route_via_donated_key` returns False and the slot gets a plain `GeneralLlm` with no
  wrapper, no donated attempt, and no counter bump.
- The disagreement-crux analyzer and the stacker chain cannot fire at all.
  `.github/workflows/test_bot_basic.yaml` sets `NUMERIC_STACKING_ENABLED`,
  `BINARY_STACKING_ENABLED`, and `MC_STACKING_ENABLED` to `'false'`, and `forecaster.py` forces
  the spread-exceeds-threshold flag to False when the per-type gate is off, so
  `extract_disagreement_crux`, `STACKER_LLM`, and `STACKER_FALLBACK_LLM` are unreachable. The
  startup line still prints the analyzer's model name, which makes the role look live.

The parser (`PARSER_LLM`) may also make zero calls: `forecaster_runners.py` reads `outcome_type`
straight out of the structured block, and the value-extraction ladder reaches its LLM rung only as
salvage. On a healthy forecast neither path runs.

### 2. `DONATED_KEY_STATE: state=drained` — CONFIRMED OBSERVABLE

`credit_telemetry.py`'s probe ladder tests `limit_usd is None or remaining_usd is None` for
UNKNOWN, then `limit_usd <= 0` for ZEROED, then `remaining_usd > 0` for FUNDED, and falls through
to DRAINED. The live key reports limit $850.00 and remaining $0.00, so the strict `> 0` test on
remaining fails and it lands in DRAINED rather than UNKNOWN. The line fires once per run, at INFO,
inside the probe lock. It is downstream of an actual spend-cap 403 occurring, which will happen.

The telemetry archive corroborates that CI's key object is drained too, not just the local one:
every donated `CREDIT_BALANCE` row in `backtests/telemetry_archive/credit_balance.jsonl` from run
`30167100027` onward reads `remaining=0.0`. That was the residual worry, since the CI secret and
the local `.env` hold different key objects on the same pooled account.

### 3. All 3 forecasters survive — CONFIRMED OBSERVABLE, and now directly so

Grep `FORECASTERS_SURVIVED: question=... survived=n/N models=...` and check that `n == N`. Commit
`5329346` added this line to `_research_and_make_predictions` in `metaculus_bot/forecaster.py`; it
fires at INFO on every published question, unconditionally, and names the survivors read off each
prediction's own `Model:` prefix rather than the configured roster. I confirmed the line in a real
log by running the free offline end-to-end numeric test with INFO capture:

```
FORECASTERS_SURVIVED: question=70002 survived=3/3 models=claude-opus-4.8,gemini-3.1-pro-preview,gpt-5.6-sol
```

Two corroborating lines appear in the same log: `STACKING base combine: 3 pre-stacked outputs;
aggregating by median for final output` from `aggregation_pipeline.py`, whose count is the number
of surviving predictions, and `Degradation counters: forecasters_dropped=0, ...` from
`forecaster.py`. On a degraded run the base-combine line reads `single pre-stacked output;
returning as-is` instead.

Do **not** infer survival by counting `EXTRACTION_RUNG` lines. That was the only method before
this commit and it over-counts on numeric questions: `extract_numeric` logs its rung before the
unit-mismatch guard in `forecaster_runners.py` can raise and drop that forecaster, so three rung
lines are compatible with a drop.

One limit worth knowing: `N` is the configured roster length, so the line cannot tell you the
roster itself shrank.

### 4. AskNews briefing is summarized, not raw — OBSERVABLE BUT INDIRECT

Correct in code, weakly observable in this specific workflow. `test_bot_basic.yaml` does not set
`PERSIST_RESEARCH_ENABLED` (only the three prod yamls do), so no research bundle is archived for
later inspection. `RAW_RESEARCH_LOG_ENABLED` is set and captures the AskNews article dicts, but
that is the summarizer's input, not evidence the summarizer ran.

What to check instead, in order of strength:

1. The published Metaculus comment must **not** contain `⚠ RAW UNSCREENED ARTICLES`
   (`SUMMARIZER_SOFT_FAIL_BANNER` in `metaculus_bot/prompts.py`). That banner is prepended by
   `_degraded_to_raw_articles` in `metaculus_bot/research/orchestrator.py` whenever the summarizer
   soft-fails, so its absence is a positive signal.
2. `summarizer_failures=0` in the `Degradation counters:` line.
3. Absence of `AskNews summarization failed` and `AskNews summarization returned blank output` in
   the log.
4. Positively, the briefing in the comment should open with an evidence-age disclosure of the form
   `Newest directly-relevant article:` and may carry a `Screened out as not decision-relevant:`
   list. Both come from the 2026-07-18 summarizer prompt rules, so their presence means the
   briefing pass genuinely ran.

Worth considering adding `PERSIST_RESEARCH_ENABLED` to this workflow so a smoke run leaves an
auditable bundle, but that is a follow-up, not a pre-run fix.

### 5. Run exits zero — OBSERVABLE, and the exit code carries more meaning than expected

`credit_alerts_active()` reads `date.today() >= CREDIT_ALERT_RESUME_DATE`; I called it and it
returns False today, so `cli.py` sets the suppressed subset equal to the credit fallback count and
computes `alertable = bot_alertable + generic_fallback - suppressed_credit_fallback`. Because the
credit counter is a strict subset of the generic one (I re-verified: three increments, all inside
`record_donated_key_fallback`, the generic one unconditional and the two subsets inside `if`
branches), the fallback contribution nets to exactly zero when every fallback is credit-caused.
`cli.py` even has a dedicated INFO line for that shape: "every fallback was a suppressed credit
event, so this run stays green." The `CREDIT_FLOOR_BREACH` warning also fires and is suppressed.

The exit code is a stronger discriminator than the earlier audit suggested. Bot-side degradation is
never suppressed, and every lost forecaster is recorded through the single `_record_forecaster_drop`
write path into `alertable_count`. So exit zero implies zero forecaster drops, which for a
three-model roster implies all three survived. Items 3 and 5 cannot both read fine on a degraded
run.

The real risk to exit zero is unrelated degradation, and I sized it from the archive rather than
guessing. Of the 96 archived degradation-counter rows, 13 belong to runs that actually forecast (at
least one `extraction_rung` record). Nine of those 13 carry non-zero `bot_alertable`. Seven of the
nine are the drained-key incident itself: their `gap_fill_v2` error text is the verbatim
`Key limit exceeded (total limit)` 403, their two forecaster drops are exactly
`openrouter/anthropic/claude-opus-4.8` and `openrouter/openai/gpt-5.6-sol`, and every one of them
has donated `remaining=0.0` at both snapshots. Those would be green on fixed code. Only two of the
thirteen are genuine unrelated degradation, and both predate the drain: run `30152479270` (donated
balance $7.27, two forecaster drops plus two gap-fill v2 errors whose text is an OpenRouter 502
"Provider returned error") and run `30160981339` (donated balance $5.01, one forecaster drop, no
gap-fill error). So the honest background red rate is roughly 2 of 6 pre-drain runs, not 9 of 13.

If the run does exit 1, read the breakdown line in `cli.py` before concluding the fix failed. On a
correct ref, `bot=0` with the credit count cancelling the fallbacks means the fix worked and
something else broke; a non-zero `bot=` with two `error_other` drops on the two donated-key
forecasters means the wrong ref was dispatched. The workflow's step uses `shell: bash`, which
GitHub runs with `-e -o pipefail`, so a `sys.exit(1)` survives the `| tee` rather than being masked
by tee's zero status.

There are three `sys.exit` sites in `cli.py` (the alertable check, the floor breach when alerting
is active, and the deprecation tripwire) plus one pre-spend abort: `verify_metaculus_api_identity`
raises before any money is spent if the Metaculus API identity preflight fails.

### 6. `CREDIT_SPEND: key=personal run_delta_usd=` — NOT TRUSTWORTHY AS A NUMBER, now self-disclosing

This is a confirmed defect, and it was disclosed rather than fixed during this session. Read
`source=`, never the number. Commits `b0c92c7` and `3a7a70a` added a `source=` field to the
`CREDIT_SPEND` line and a sibling `CREDIT_SPEND_UNSETTLED` warning:

- `source=remaining_delta` (the donated key, which reports a limit) is reliable.
- `source=usage_delta_unsettled` (the personal key, which reports no `limit_remaining`) is a
  **lower bound** and frequently reads `0.00` on a run that spent real money.

The cause is an OpenRouter settlement lag, not `byok_usage` — that was the old docstring's stated
worry and two agents' initial suspicion, both wrong. Measured across 178 archived paired
personal-key runs: within-run deltas summed to $3.31 against $5.66 of true lifetime-usage growth,
so the marker captured 58%. The lag is proven, not assumed: the gap between each run's `phase=end`
usage and the next run's `phase=start` sums to $2.35, and $3.31 plus $2.35 is $5.66 to the cent.
The tightest cut removes the "maybe those runs were free" reading entirely: of the 25 paired runs
carrying at least one `extraction_rung` record, 7 read exactly $0.00, and
`gemini-3.1-pro-preview`, the slot pinned to the personal key, forecast in all 25.

So a single run's figure proves nothing. For the settled number, run
`uv run python scripts/reconcile_credit_spend.py` (free, offline, reads the telemetry archive) once
a later run has provided a successor snapshot. The current run always shows as unsettled until
another run follows it.

### Where the observations live afterward

The run-log artifact is named `logs-${{ github.run_id }}` (deliberately not `research-*`, so
`scripts/download_research.py` never picks up test runs) with 90-day retention, uploaded under
`if: always()` so it survives a non-zero exit. Read the six observations out of that artifact
during the run, then run `make sync_all` afterward so the markers land in
`backtests/telemetry_archive/` before the 90-day expiry.

## 2. What to fix before spending the $2.60

**Nothing in the code.** One procedural precondition, and it is the operator's call anyway.

Ranked:

1. **Would make the run mislead you — the dispatch ref.** Not a code defect, but the single thing
   that would turn $2.60 into a rerun of the incident with every log check reading as still-broken.
   Run the `rg 'key limit exceeded'` preflight above against whatever ref you dispatch. If you push
   local `main` first, dispatch against `main`; otherwise name a pushed branch that carries the
   fix.
2. **Cosmetic, optional, no bearing on trust.** Adding `PERSIST_RESEARCH_ENABLED: 'true'` to
   `test_bot_basic.yaml` would leave an auditable research bundle and make item 4 directly
   inspectable rather than inferable. Also, a note that 32 lines across four files carry a
   malformed `# noqa: HARNESS-SCAN-EXEMPT-*` directive that ruff warns about on a cold cache; see
   the uncertainty section.

No blocker was manufactured. The fallback and telemetry surface itself is trustworthy on the
evidence below.

## 3. What shipped in this wrap-up

Commits, oldest first:

- `9c92dca` "Document that a GHA bot-workflow dispatch spends the same as a local run"
- `e2f2849` "Say that the cost gate forbids an agent deciding to spend, not the operator directing
  one"
- `ad610cb` "Give the attribution parsers a deterministic CI floor" — the parse-attribution
  regression test now runs in CI, keyed on a checked-in 12-record miniature at
  `tests/data/performance_comments_mini.jsonl` (21 KB, distilled from the 27 MB local pull). The
  `.jsonl` extension sidesteps the repo-wide `*.json` ignore rule instead of adding a negation.
  Each miniature is admitted only under a hard faithfulness filter in
  `scripts/derive_mini_comment_fixture.py`: it must parse identically to its full-size source
  under every parser in `parser_outputs`.
- `0ec7b88` "Stop an empty parse from counting as a clean one" — a real defect found by
  independent verification of that commit. The coverage assertion counted a record that parsed
  nothing as a clean parse, so a parser returning `{}` for every comment would have passed at a
  1.00 ratio. Reproduced against the pre-fix file with an always-empty mutant. Empty parses now get
  their own bucket, discriminated by a plain `*Forecaster` substring so "no bullets to find" is
  exempt while "missed bullets that are right there" fails, and that bucket is removed from both
  sides of the ratio.
- `5c53857` "Refresh the gap-fill v2 telemetry marker and budget prose" — the gap-fill v2 budget
  prose disagreed with the code in five places, two actively wrong (a "60s fetch budget" that
  exists nowhere in the tools module, and an AskNews backoff comment transcribing the product of
  three constants as a fixed number). All three copies of the `GAP_FILL_V2` telemetry marker also
  stopped at `lint_rejections`, omitting six later fields including `error=`, which is the one
  field that distinguishes a step-zero v2 crash from an idle run.
- `a13ca9a` "Name the constants behind the retry and tool-budget comments"
- `f39f0a5` "Note that the 402 family skips the drained-vs-revoked probe"
- `c8f63ce` "Pin the soft deadlines against the model timeouts they are sized against"
- `99e42b5` "Pin every mini-fixture record to its exact multi-parser output" — closes the gap
  independent verification flagged: the fixture class asserted only a coverage ratio and shape
  metadata, with no per-record expectations. There is now a `_EXPECTED_PARSES_BY_POST` table
  pinning the full six-parser output per `post_id`, plus an assertion that the table's key set
  matches the fixture so a new record cannot slip through unchecked, plus an
  `--emit-expectations` mode on the derivation script so refreshing it is one command.
- `5329346` "Say in the run log how many forecasters survived" — item 3 above.
- `b0c92c7` "Stop a 0.00 personal-key spend from reading as no spend" — item 6 above, plus
  `scripts/reconcile_credit_spend.py`.
- `3a7a70a` "Record the demonstrably-spent cohort in the settlement-lag note"
- Plus the two docstring corrections committed with this brief: both
  `scripts/derive_mini_comment_fixture.py` and
  `tests/test_performance_analysis_parsing.py` described the faithfulness invariant as covering
  "every public per-model parser", which is literally false. `parse_per_model_reasoning_text` is
  public and exported in `metaculus_bot/performance_analysis/__init__.py`, and it necessarily
  diverges because the shrink elides exactly the prose it returns. I measured it: key sets
  identical on 12 of 12 records, full bodies identical on 0 of 12. The exclusion is right; the
  wording was not.

## 4. Regression checks re-verified at HEAD

I re-ran all three probes myself against `3a7a70a`, not against the reports.

- **The 11-row fallback and credit matrix: HOLDS, 11 of 11.** Every row matches expectation under
  both a `litellm.exceptions.APIError` carrying a real `status_code` and an
  `openai.APIStatusError` built over an `httpx.Response`, with a methodology guard that fails a row
  if `llm_status_code` returns anything other than the intended status. The verbatim production 403
  gives fallback True and credit True; the three moderation 403s give False and False; the 429
  echoing "402" gives fallback True and credit False; the two 404s give fallback True and credit
  False. No network: the probe constructs its own `httpx.Response` objects locally.
- **Invariant 1 — fallback routing never reads a live balance: HOLDS.** A name-agnostic transitive
  call closure across the whole `metaculus_bot` package from `should_retry_with_general_key`
  contains six functions and zero references to httpx, requests, urllib, aiohttp,
  `fetch_auth_key`, or `credit_telemetry`. The non-vacuity control trips as it should: the same
  scan of `is_suppressible_credit_error` does reach `classify_donated_key_state`, so the probe
  genuinely lives in the alerting predicate and not in the routing one.
- **Invariant 2 — no await after the threaded probe: HOLDS.** The single `ast.Try` in
  `record_donated_key_fallback` contains the one `ast.Await` (the `asyncio.wait_for` around the
  threaded probe). The four statements after it contain zero `Await`, zero `AsyncWith`, and zero
  `AsyncFor`, with three `Global` declarations and exactly three `AugAssign` nodes on the three
  counters.
- **Counter subset relationship: HOLDS.** Nine writes total across the module: three module-level
  initializers, three test-only resets, and exactly three increments, all inside
  `record_donated_key_fallback`. The generic increment is unconditional at function-body level; the
  credit and 404 increments sit inside `if` branches. That matches the arithmetic in `cli.py`,
  where the generic count adds and at most one subset subtracts.

One rationale comment is worth correcting eventually, though the invariant itself is right and
should stay. The comment justifies the no-await rule as preventing a bytecode-level race between
concurrent forecasters. On a single event loop an await is a cooperative suspension, so `+=` cannot
interleave; the genuine hazard is cancellation between the generic increment and the credit
increment, which would leave the generic bumped and the subset not, producing a false red rather
than a false green. Same rule, different reason. Not urgent, and it cannot affect this run.

## 5. Open items for the operator

Every git operation below is blocked for agents, so these need a human shell.

1. **Decide the dispatch ref and preflight it.** If you want to run against `main`:
   `git push` (blocked for me), then confirm
   `git show origin/main:metaculus_bot/fallback_openrouter.py | rg 'key limit exceeded'` hits, then
   `gh workflow run test_bot_basic.yaml --repo No-Stream/metaculus-bot --ref main`.
   Otherwise name a pushed branch that carries the fix. The `--repo` flag is required because no
   default repo is configured.
2. **Delete the `api-preflight` branch, optional housekeeping with zero bearing on the run.**
   Nothing of value on it is missing from `main`. Its two commits (`170c182`, `6f2d17a`) touch 13
   files, all of which exist on `main`, and only six added lines are absent — all in the block that
   reads `MetaculusApi.API_BASE_URL`, an attribute that no longer exists on the installed
   forecasting-tools, so the branch's version raises `AttributeError` at import. `main`'s
   `MetaculusClient().base_url` version is strictly better and additionally honors
   `METACULUS_API_BASE_URL`. The branch is checked out in a worktree, so plain `git branch -D`
   fails; the sequence is
   `git worktree remove /Users/flatljan/personal/metaculus-bot-preflight` then
   `git branch -D api-preflight`. Note "provably dead code" is conditional on `main`'s
   forecasting-tools pin: the branch pins the older version, where that attribute does exist.
3. **Optionally drop three redundant stashes, and leave the other twelve alone.** Verify by tree
   SHA, not index, because concurrent agents keep shifting the stack. Today:
   - `stash@{9}` (`WIP on main: 8987527 Make three silent research losses visible...`) is fully
     contained at HEAD; its patch reverse-applies cleanly and all 108 added lines are present in
     the files they came from.
   - `stash@{10}` and `stash@{11}` are byte-identical to each other (both tree
     `68da21a8548022ee7c63577d04fe2a4bfd371baa`, both 1 insertion and 14 deletions to
     `metaculus_bot/fallback_openrouter.py`) and are a strict regression of code now on `main`:
     they delete the guard and the `asyncio.wait_for` wall bound around the `/auth/key` probe.
     **Do not pop or apply either.** The missing wall bound is the load-bearing loss — the probe
     runs before the personal-key retry, so an unbounded version stalls recovery.
   - `stash@{0}` through `stash@{8}` are nine mutation-testing probes on
     `metaculus_bot/performance_analysis/parsing.py`, pushed by a concurrent agent this session.
     Leave them; they are someone's live scratch.
     Drop the highest index first, and re-run `git stash list` immediately beforehand:
     `git stash drop 'stash@{11}'`, then `'stash@{10}'`, then `'stash@{9}'`.
4. **After the run: `make sync_all`,** so the `logs-<run_id>` telemetry lands in
   `backtests/telemetry_archive/` before the 90-day artifact expiry.
5. **Still tracked, not for today:** the 2026-09-10 credit-alerting cliff. On that date
   `credit_alerts_active()` starts returning True and a drained donated key reddens CI again.

## 6. Genuinely uncertain

- **Whether the 32 malformed `# noqa: HARNESS-SCAN-EXEMPT-*` directives should be normalized.**
  Ruff warns on each when its cache is cold ("expected a comma-separated list of codes"), across
  `tests/test_ablation_run_stacker.py` (19), `metaculus_bot/research/prediction_market.py` (11),
  `tests/test_performance_analysis_parsing.py` (1), and `tests/test_ft_patch_pins.py` (1). Ruff
  still exits 0, so no gate is affected. Elsewhere in the repo the same pragma appears as a bare
  trailing comment with no `noqa:` prefix, which does not warn. I did not normalize them because
  the scanner that consumes the pragma lives outside this repo and I could not verify that
  stripping the prefix keeps the pragma visible to it. Someone who knows that scanner should
  decide.
- **Whether the smoke run will produce a non-zero personal-key spend figure.** Roughly a coin
  flip. The one prior `test_bot_basic` run did read $0.21, but 7 of 25 runs that provably forecast
  read exactly $0.00. Read `source=`, and if it says `usage_delta_unsettled`, treat the number as a
  floor.
- **A latent alerting-only tail risk on the `/auth/key` probe under concurrency.** When many
  fallbacks fire at once before the probe cache is warm, they queue on the probe lock while each
  holds its own `wait_for` budget. If the probe were slow enough for waiters to time out, they log
  a probe-failure line, mark the event non-suppressible, and stay alertable — reddening CI on
  exactly the condition suppression exists for. Recovery is unaffected, because routing was already
  decided textually and the personal-key call proceeds regardless. Nobody measured `/auth/key`
  latency, so the size of this window is unknown. Flagging rather than claiming.
- **The item-4 signal set for a summarized briefing.** The four checks listed are each sound
  individually, but none is a single unambiguous "the summarizer ran" marker. Adding
  `PERSIST_RESEARCH_ENABLED` to the workflow would replace inference with an artifact.
- **Test-count drift under concurrency.** Several agents were editing this tree while these gates
  ran. My 4237-passed reading is a snapshot of `3a7a70a` plus two docstring edits, and the working
  tree was clean at the time. A reader re-running later may see a different count as further
  commits land.
