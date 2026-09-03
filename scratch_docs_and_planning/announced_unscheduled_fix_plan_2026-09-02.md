# Plan: fix the announced-but-unscheduled-event miss class, and de-bloat the forecaster prompts and structured block

## Status (2026-09-03)

Implemented, reviewed and fixed. Items A, B, C, D, E and the §10 housekeeping all landed on
`next-season-bundle` over commits 7e7d449..8cd2295 (32 commits). `/forge` then ran on the full
7e7d449..8cd2295 diff and its findings were applied over a further 12 commits, 7e7d449..a598854
(44 in total): the six FIX findings were F1 (the multiple-choice prompt asked for percentages where
the schema demands decimals), F2 (the history-discharged receipt was stated as a flat fact rather
than the small audit sample it is), F3 (the pending Item C decision was invisible in FUTURE.md and
the revert recipe in AGENTS.md named the wrong deletion set), F4 (the numeric tie tolerance had no
test on the block rung it exists to serve), F5 (the driver template's comment overstated how often
prod emits the market header) and F6 (four copies of the same prompt-render test helpers), plus
sixteen of the twenty report-only items. Free gates green at a598854: `make lint` clean, `make
typecheck` 0 errors, `make lint_imports` 6/6 contracts, `make deps` clean, `make test` 6815 passed /
22 skipped / 5 deselected in 145 s. No paid call was made in either pass. One path §11 below names
has moved since: the wording pins are no longer in `tests/test_prompts.py`, which the forge pass split
into `tests/prompts/test_base_prompt_rules.py`, `test_structured_block.py` and
`test_research_clauses.py` with the shared builders in `tests/prompt_builders.py`.

Rendered prompt sizes came in above §7's targets by 12% (binary), 11% (MC), 8% (numeric) and 6%
(analyzer), because of the operator's own §7.2 keeps, principle 3's per-rule reason clauses, and
Item A's wording measuring 648 collapsed characters rather than the 450 §4 estimated. AGENTS.md's
"Prompts" subsection carries the measured figures, the method, and since the forge pass an explicit
record that the overshoot is a waiver rather than an oversight.

What the operator still owns: merging the bundle as ONE era boundary before the fall cup; the
optional `test_bot_basic` smoke run (~$2.60, publishes to Metaculus), which is the only way to
confirm that real forecasters still emit a well-formed STRUCTURED FORECAST block under the rewritten
prompts; the 2026-09-10 credit cliff; confirming Item C, which shipped on this plan's recommendation
and reverses by deleting the constant `_HISTORY_DISCHARGED_RULE`, its two interpolation sites and
only the history-discharged cases in the shared test class that also covers the approved
`_SOFT_CLOCK_RULE` (AGENTS.md carries the recipe); and the merged `worktree-agent-*` /
`worktree-wf_*` branches to `git branch -d`.

Receipts for this session: per-removal git traces in `scratch/prompt_debloat_2026-09-02/receipts.md`,
the wording pass over the result in `scratch/prompt_debloat_2026-09-02/wording_review.md`, and a PR
section describing both this work and the 2026-09-01 residual round appended to
`scratch/next_season_bundle_2026-09/PR_DESCRIPTION.md`.

Everything below is the plan as decided on 2026-09-02 and is left as the record of those decisions.

Written 2026-09-02, revised the same evening after two audits and an operator review of every item.
For a fresh session. Everything a reader needs is in this file or in the paths it names. Branch:
`next-season-bundle` (tip 7e7d449 at writing; the single branch over `origin/main`; prod runs from
`main`, so nothing here is live until the bundle merges).

## Where things stand (session handoff, 2026-09-02 evening)

- `next-season-bundle` is at 7e7d449, tree clean, all free gates green (lint, basedpyright 0
  errors, 6 import contracts, deptry, 6774 tests). It carries the whole 2026-09 bundle, the
  2026-09-01 residual round's six fixes, and 13 forge fixes. Nothing is pushed. It is the single
  branch over `origin/main` (eded193) and must merge as ONE era boundary before the fall cup
  (~2026-09-20). PR text from the earlier bundle work: `scratch/next_season_bundle_2026-09/PR_DESCRIPTION.md`
  (needs a paragraph for the 2026-09-01/02 additions).
- No agents, workflows, cron loops or tmux jobs are running. Merged worktree branches
  (`worktree-agent-*`, `worktree-wf_*`) remain for the operator to `git branch -d` (blocked op).
- Operator-side items still open: merge the bundle; the optional `test_bot_basic` smoke run
  (~$2.60, publishes); the credit cliff on 2026-09-10 (every run exits non-zero from then;
  refill the donated key, push `CREDIT_ALERT_RESUME_DATE`, or disable the tournament cron at
  season end); fall cup / minibench re-enable; Item C below.
- Receipts: residual round `scratch/residual_2026-09-01/` (SYNTHESIS.md is the judgment doc);
  audits under `scratch/failure_mode_audit_2026-09-02/`, `scratch/prompt_bloat_audit_2026-09-02.md`,
  `scratch/schema_bloat_audit_2026-09-02.md`. The project memory directory Claude Code keeps for this
  repo is current.
- This document is the work plan for the next session: Items A to E below, in the order in §11.

## 0. Ground rules

- Cost gate is absolute: never run `main.py`, `make backtest_*`, `make ablation_*`,
  `make test_live`, or anything that hits a paid API. Free gates: `make lint`, `make typecheck`
  (basedpyright, 0 errors), `make test` (~150 s, in a named tmux session with a polled log),
  `make lint_imports`, `make deps`.
- Timing / deadline / fallback code is high-risk in this repo; nothing in this plan touches it.
- Prompt text and schema shape shift the forecast distribution, so they are a config-era boundary.
  Everything here lands in the SAME merge as the rest of the bundle (one era boundary), before the
  fall cup opens (~2026-09-20).
- Every prompt rule is a NAMED module constant in `metaculus_bot/prompts.py`, interpolated into
  the binary and multiple-choice base prompts and never into the three stacking prompts, and pinned
  by tests in `tests/test_prompts.py` (presence in the right prompts, absence in stacking prompts,
  the benchmarking leakage guard, the example-block JSON validity test). When a rule is reworded,
  UPDATE its pin to the new wording; never delete a pin to make a test pass.
- Work in a git worktree per item, commit there (plain imperative messages, no AI attribution),
  then `git merge` the worktree branch into `next-season-bundle`. Do not push. AGENTS.md
  (`CLAUDE.md` is a symlink to it) is touched by every item, so merge sequentially and resolve the
  one-line conflicts by hand.

## 1. The operator's principles for this work (decide every judgment call by these)

1. **Forecast quality first.** The forecasters should spend their attention on the most accurate
   forecast, not on following rules. A rule is a shackle; keep it only if the pipeline REQUIRES it,
   it SCAFFOLDS the model's reasoning toward the forecast, or it CORRECTS a measured failure and
   there is no shorter form that keeps the correction.
2. **Remove repetition and shackles, keep the core meaning.** The substantive points of the prompt
   stay; each is said once, in one place. Do not change what the prompt asks the model to do,
   only how many times and how prescriptively it asks.
3. **The prompt must explain itself.** Read it as the model would: it should read as something
   hardened over months of questions, where each instruction carries its reason, not as a list of
   unjustified suggestions. So when consolidating duplicates, keep the one-clause "why" attached to
   the surviving rule (e.g. "because past misses traced to forecasters ignoring markets"), and where
   a receipted rule has no stated reason, add one short clause. Rationale is not bloat; repetition is.
4. **Structured-block slots are template scaffolding or the forecast itself, nothing else.** The
   block is written last, after the forecast is fixed, so a slot there cannot scaffold reasoning; it
   can only be the forecast (probabilities, percentiles), transmit a decision the pipeline needs
   (`outcome_type`), or echo prose for our post-hoc analysis. The third kind goes. We read
   rationales afterward.
5. **Template versus prompt.** Instructions that shape the STRUCTURE of the model's answer belong
   in the numbered analysis template ("Reproduce the following analysis template in your answer");
   policy and background belong in the guidance sections around it. When a checklist item is worth
   keeping, ask whether it is really a missing template step (usually yes; move it) rather than a
   post-hoc reminder.
6. **Trace the why before removing.** For every removal or merge, run
   `git log -p -S'<distinctive phrase>' -- metaculus_bot/prompts.py` and read the commit that added
   it (the two audits recorded most of these under "receipt / origin"). If the history shows a
   measured failure the surviving text no longer covers, keep a one-line version.

## 2. Inputs a fresh session must read first

- `scratch/failure_mode_audit_2026-09-02/AUDIT_SYNTHESIS.md` and `STATS.md`: the whole-archive
  audit (826 records, four lenses, 193 blind re-coded) that established the announced-event class.
- `scratch/prompt_bloat_audit_2026-09-02.md`: every instruction in the forecaster prompts
  classified (required / scaffolding / corrective-with-receipt / restrictive / stale), sizes before
  and after, contradictions (§5), exact shortened wordings (§6), inventory tables (§2 binary, §3 MC,
  §4 numeric, §8 gap-fill analyzer, §9 stacking). Item E executes it with the operator's
  overrides in §7 of this plan.
- `scratch/schema_bloat_audit_2026-09-02.md`: every structured-block field classified, consumers,
  what breaks on removal (§8). Item D executes it.
- `scratch/residual_2026-09-01/SYNTHESIS.md` §2 and `DOSSIER_SYNTHESIS.md` §4, and the dossiers
  for 43837, 44424, 44557, 45217 under `scratch/residual_2026-09-01/dossiers/`.
- `scratch/` is gitignored; these files exist only on this machine.

## 3. The problem Item A fixes, in plain words

Some questions ask "will X happen before date D" where the only reason to expect X by D is a
target date the responsible actor has announced but is not bound to: a launch "scheduled" for
August 23, a summit "planned", a tournament "expected this fall". The forecasters decompose the
forecast as P(the target lands inside the window) x P(X happens | target), then set the first term
near 1 because the target was announced.

Measured over the bot's whole resolved history (STRICT cohort, n=815):

- 52 records carry the shape (6.4%; 8.3% of binary questions); coders agreed on the label
  (kappa 0.74).
- On the 37 flagged binaries the bot published a mean 0.44 that the event would happen; it
  happened 3 times (8%). Thirteen records were above 0.5 and resolved NO; none went the other way.
- Flagged records score 18.7 spot-peer points worse than unflagged binaries (95% CI 5.9 to 33.4)
  and are wrong-sided 40% of the time against 18%.
- Soft targets WITHOUT the decomposition move score fine (+13.6), and deadline questions in
  general are calibrated (0.25 published, 0.25 realized). The defect is the specific move.
- Roster-wide: every vendor is biased upward by 0.27 to 0.47 on the shape and within 0.06 of zero
  off it. Zero fires yet in the 30 resolved live-triple records, so this is pre-emptive.
- Contrast: 45217 (German regulator approving Uber's Delivery Hero offer) had a statutory clock;
  members computed the date and scored +45. The discriminator is binding clock vs soft target.
- Caveat the wording must respect (audit record q42305): a weekly bulletin with a MEASURED
  publication lag is a binding clock in practice, and a near-1 timing term was correct there.

## 4. Item A: the soft-clock rule (the fix)

Add ONE named constant `_SOFT_CLOCK_RULE` to the binary and multiple-choice base prompts, next to
the outside-view / reference-class bullets in Phase 1. It is the single rule for this shape and
supersedes two of the four rules shipped on 2026-09-02 (Item B). Wording (the verbatim text below
measures 648 whitespace-collapsed characters, not the "about 450" this line first estimated, which is
most of why the binary and MC prompts landed above §7's targets; includes its reason, per
principle 3):

```
• A target date the responsible actor has not bound itself to — no statute, no contract, no
  published schedule it has a measured record of meeting — is evidence that a target EXISTS,
  not that it will hold. Price the probability that the target lands inside the question
  window as its own number, derived from that actor's record of slips and scrubs for this
  kind of event; an announcement, plan, tracker page or partner page does not raise it. Where
  a binding clock exists, compute the date from it and say which clock. (Announced-but-unbound
  dates are the bot's most consistent miss: forecasts averaged 44% on events that happened 8% of
  the time.)
```

No structured field for it; the number lives in the rationale.

Tests: presence verbatim in binary and MC prompts; absence from numeric and all three stacking
prompts; leakage guard unchanged. Docs: one bullet under "Prompts" in AGENTS.md naming the
constant, the receipt (52 of 815 records; 43837 / 44424 / 44557 vs 45217), and placement.

## 5. Item B: reverse two of the 2026-09-02 rules (operator confirmed)

| rule shipped 2026-09-02 | verdict | action |
|---|---|---|
| `_REMAINING_EXPOSURE_RULE` | Correction right, constant redundant: bullet 1 restates the conditional-hazard bullet already in Phase 1; bullet 2 restates the disjoint-union clause added to the union line in the same commit. | Remove the constant. Make sure the conditional-hazard bullet says "from now until the deadline, treating the elapsed event-free part as observed" (one phrase), and keep the amended union line ("union only over paths that cannot be the same event"). |
| `_ANCHOR_CONSISTENCY_RULE` | Bullet 1 is the fourth request to state the outside-view number; bullet 2 targets a failure whose measured cost is zero and would suppress good moves on winning questions. The whole-archive counterfactual of "publish your anchor" nets about zero with a parse-dependent sign. | Remove the constant. Carry the 15-point gap size into the existing "Anchor on your math" bullet ("a move of more than about 15 points needs a named reason"). |
| `_COUNT_IN_PERIOD_REFERENCE_CLASS` | Short, no duplicate, corrects a real class. | Keep verbatim. |
| `_LAST_REAL_USE_GAP_RULE` (gap-fill analyzer) | The gap type is right; the "one gap must ask" mandate displaces other gaps because the analyzer fills every slot regardless of instruction. | Fold the clause into gap type 6 of the analyzer's list (about 200 chars); drop the standalone block and its "must". |

`remaining_window_days` and its `WINDOW_DECLARED` marker go with Item D.

## 6. Item C (operator to decide; recommendation: ship): history-repeats

A member's own rationale names a reason the historical cadence has been discharged (the package
landed, the deadline passed, the rule changed), then still uses the old cadence as its central
estimate. Audit: 12.1% of rationales, about 7 spot per flagged record (95% CI 2.7 to 12.2), the
pattern failed in 83% of flagged cases (13 of 13 on the live triple). Coder agreement 0.59 and the
label is partly hindsight-contaminated, so treat the numbers as upper bounds.

One sentence, conditional on the member's OWN written acknowledgment, beside the base-rate bullets:

```
• If your own research names a reason the historical cadence has been discharged (its driver was
  met, the deadline passed, the rule changed), that cadence is a bound on your estimate, not its
  centre; state the post-change estimate and what it rests on.
```

With `_ANCHOR_CONSISTENCY_RULE` removed, the contradiction the audit flagged disappears. If the
operator declines, skip; nothing depends on it.

## 7. Item E: de-bloat the forecaster prompts (the operator's decisions, item by item)

Targets after Item E: binary about 19,300 to about 14,300 chars (with Item A and the kept
"three moves" paragraph), MC about 14,100 to about 10,700, numeric about 18,700 to about 13,700,
gap-fill analyzer about 4,700 to about 3,800. Re-measure with the audit's render script
(`scratch/_prompt_bloat_tmp/`); within about 5% is fine. Each receipted correction survives in
exactly one place, with its reason.

### 7.1 Remove as pure repetition (operator: yes)

- **Pre-open rule: keep TWO statements, drop the third.** This footgun has cost the bot badly, so it
  stays stated twice on purpose: the forecasting-window line ("events before the open date do not
  resolve YES unless ...") and the status-quo derivation's "name the specific POST-OPEN event".
  Remove only step 0a bullet 2 (the 447-char restatement with the 1945 example).
- **Numeric anchor-to-latest and trend:** keep one anchor statement (step 0's status-quo derivation)
  and one trend statement (step 3's trend continuation); remove the step-1 "Critical: centered near
  this value" push, the step-3 "status-quo outcome" restatement, and the step-7 trajectory check.
- **Market clause:** the rendered market table's legend (`market_retrieval/rendering.py`,
  `MARKET_SIGNAL_LEGEND`) owns notation (relation tiers, RESOLVED, `↳` rows, `[remaining N]`,
  `(Nd ago)`, `demoted from same-date:`); the prompt owns policy. Cut
  `_MARKET_RELATION_WEIGHTING_SENTENCE` from 1,908 chars to about 480 keeping its three policy
  clauses with their reason (extrapolate from an other-cut market rather than discount vaguely;
  liquidity governs a relation-vs-liquidity conflict because a thin price is noisy however tight
  its relation; a multi-outcome ladder is a distribution, never an equality constraint on a tail).
  Remove `_MARKET_LIQUIDITY_WEIGHTING_SENTENCE` (verbatim duplicate of the legend). Gate the whole
  market clause on the `## Prediction Market Snapshot` section being present (prod-neutral; removes
  about 3.5k chars from every benchmarking or market-less prompt).
- **Source-tier ladder:** the A to D definitions (574 chars) become one line naming the inline tags
  the research already carries; `[unverified attribution]` from four lines to two.
- **Odds check and small-delta check:** one check.
- **Numeric step 8:** drop the four lines that duplicate the schema notes or the bound messages
  ("think in ranges", "strictly increasing", "no scientific notation", open vs closed). State
  `outcome_type` once (the schema note is the definition; step 9 keeps a one-line pointer).
- **Checklists (binary/MC/numeric):** the items that re-ask for template outputs go (paraphrase the
  criteria, state the base rate, list evidence, red-team). The two unique items, the bait-and-switch
  check and the "X out of 100 times" consistency line (numeric adds units), MOVE INTO THE TEMPLATE
  as the final step rather than staying as a trailing checklist (principle 5). Drop numeric schema
  Notes bullet 1 (restates the "all 13" line printed six lines above).
- **MC:** remove "Use integers 1%-99% ... sum to 100%" (contradicts the decimal schema) and the
  exposure bullet 2 (MC has no union).

### 7.2 Loosen shackles (operator: yes, with two keeps)

- **KEEP the "three valid moves" reconciliation paragraph** (step 5b). It traces from real
  reasoning failures the operator has seen. Keep its substance; a light tightening of wording is
  fine, and it must state its reason (all hedging operates through the clauses so the criteria stay
  consumed as constraints). It is the mechanism for leaving the clause product, which makes it
  consistent with 7.4 item 1 below.
- **KEEP the meta-justification sentences** that explain why a step exists ("this forces the
  criteria to be consumed as structured constraints rather than a prose paraphrase"). Principle 3:
  the prompt should explain itself. Reversal of the audit's B17 removal.
- Remove `_ANCHOR_CONSISTENCY_RULE` and `_REMAINING_EXPOSURE_RULE` (Item B).
- Remove `_NULL_RESULT_READING` bullet 3 (unreceipted; pushed the wrong way on q43837). Bullets 1
  and 2 stay with their q44799 reason.
- **Forecastability (numeric): redesign, do not just delete.** The current step 9b (687 chars) asks
  for an output line `FORECASTABILITY: HIGH/MEDIUM/LOW` that nothing parses. The operator's intent
  is real: the model should decide whether the quantity is forecastable from current information
  (an administered or slow-moving series) or close to a random walk (a traded price, a volatile
  count), and if it is near-unforecastable, centre on the current value with a width taken from the
  series' realized variability and not expect movement it cannot source. Implement that as ONE
  numeric template step, "Forecastability and width", that also absorbs the kernel of the
  calibration-guidance paragraph (volatile quantities get wide distributions; stable, well-measured
  indicators anchor tightly; match width to what the reasoning supports, do not pad or sharpen out
  of disposition; log score punishes narrow-and-wrong far more than wide). No separate output line.
  The calibration paragraph then shrinks to a pointer, so the guidance lives in one place. Keep the
  even-handed wording the 2026-07 width audit settled on (no directional push toward wide or
  narrow); `tests/test_open_bound_guidance.py` pins the old FORECASTABILITY text and must be
  updated, not deleted.
- **Gap-fill analyzer:** drop the "Most questions have 0-2 real gaps; a few have 3-5" counts (the
  analyzer fills every slot regardless: 55 to 77% of records sit at the cap, and "3-5" is stale
  against `GAP_FILL_MAX_GAPS = 4`); keep "DO NOT invent gaps for completeness" with its reason
  (each gap is a paid search). Shorten ANSWERABLE NOW from 1,165 to about 800 chars keeping the
  mandate, the "already answered counts" carve-out, and the "never phrase as the resolution-date
  value" rule; shorten ORDER THE GAPS to two sentences; fold `_LAST_REAL_USE_GAP_RULE` into gap
  type 6. Note for the operator: the spend lever is the cap, not the prose; this plan does not
  change `GAP_FILL_MAX_GAPS`.

### 7.3 Do not touch

The status-quo derivation and the 5b clause-pricing table; the resolution-metric echo; the
bear/bull/red-team step; the "use your own expertise" permission; the numeric units, bounds and
scoring-rule paragraphs and the open-bound messages (medium-high risk per the July audit);
`_COUNT_IN_PERIOD_REFERENCE_CLASS`; `_NULL_RESULT_READING` bullets 1 and 2; the "Anchor on your
math" bullet (now the single anchor-adherence rule); the STRUCTURED FORECAST preamble and the "LAST
thing you write" line. Stacking prompts (prod-disabled): only the wording slip ("analyses above"
while they are interpolated below) if the file is being edited anyway.

### 7.4 Contradictions to resolve (operator agreed on all four)

1. One anchor-adherence rule, "Anchor on your math" (with the 15-point size). On multi-clause
   questions the 5b clause product is the anchor because it is the more specific computation, and
   the "three valid moves" are the permitted ways to leave it. State that linkage in one sentence
   where the product is computed, so the prompt says which computation anchors.
2. MC integers vs the decimal schema: decimals win; delete the integer line.
3. Numeric step 7 "My base rate was X% ... moving to Y%" is a probability template on a
   distribution question: reword to "my central estimate from the outside view was X; after the
   evidence it is Y because ...".
4. History-repeats vs "do not move off your number": resolved by removing the latter (Item B).

## 8. Item D: remove post-hoc fields from the structured block (operator agreed; schema audit §8 is the recipe)

Remove from the prompts' schema instruction and example blocks (keep the Pydantic field tolerant
where archived blocks carry it, so old comments still strict-parse):

| field | why it goes | what else goes with it |
|---|---|---|
| `remaining_window_days` (binary, MC) | added 2026-09-02, telemetry only by its own instruction | `_readable_window_days`, `_log_window_declared` in `forecaster_runners.py`, the `window_declared` marker spec, its docs in AGENTS.md and `docs/operations.md`, four test classes |
| `base_rate_anchor` (binary) | 213 chars restating the Phase 1 range; read only by `tool_runner` behind `PROBABILISTIC_TOOLS_ENABLED=false`; harvested marker has 0 rows | `_anchor_and_clause_telemetry_lines` and the anchor/clause markers; keep the schema field optional for archive parsing |
| `criteria_clauses` (binary) | 271 chars restating the step-5b table; same dormant consumer, 0 rows; the 5b prose stays | as above |
| `concentration` (MC) | dormant Dirichlet input; 7 of 19 fills echo the example's 20.0; a LIVE bug: q45189's gemini wrote `0.0`, the `> 0` validator rejected a valid ballot and MC has no strip-and-retry, so it fell to LLM salvage | drop from prompt AND make the validator lenient |
| `other_mass` (MC) | dormant; 15 of 20 fills are 0.0; option sets are exhaustive and sum to 1 | drop from prompt, lenient validator |
| `outcome_type` in `stacking_numeric_prompt` only | the stacker's vote is never read | the base numeric prompt KEEPS it |

Keep: `question_type`, `posterior_prob`, `option_probs`, `declared_percentiles`, base-numeric
`outcome_type`, the "ONLY authoritative source" preamble and the "LAST thing you write" line.

Hardening follow-ons (small, do them here): a misspelled `outcome_type` should cost one parser
call, not the whole numeric block; `_check_percentiles` should accept ties the way the ladder and
`sanitize_percentiles` already do.

## 9. Not planned (explicitly)

- Underconfidence / walking away from own analysis: 18.5% of rationales, cost about zero, coder
  agreement 0.34. No lever.
- Catastrophizing the extreme version of an event: 3.6% of rationales, coder agreement 0.27,
  shrinking since May. No lever.
- Any telemetry field or `clock_type` label in the block.
- Changing `GAP_FILL_MAX_GAPS`.

## 10. Housekeeping the same session should pick up

- FUTURE.md bookkeeping (ledger rows 23, 29 in `scratch/residual_2026-09-01/FOLLOWUP_LEDGER.md`):
  era-gap flag retired, "numeric-only lever" to numeric-dominant, anchor-screen rejection recorded,
  the mixture-branch "zero prod fires" line fixed (wrong for a third round).
- Forge report-only items worth doing when touching those files
  (`scratch/residual_2026-09-01/forge_report_fixes_2026-09-02.md`, "Minor / unverified"): three
  near-identical binary-post fixture builders in `tests/test_performance_analysis_extended.py`;
  `RESCORE_ATOL` duplicated between `rescore_diff.py` and `collector.py`; a third copy of the FRED
  number formatter in `resolution_chart_data.py`; `tests/test_resolution_source_provider.py` past
  2,000 lines; the supply-probe forfeit sweep logs nothing during its detail GETs.
- Before merge (operator): one `test_bot_basic` smoke run (~$2.60, publishes to Metaculus) is the
  natural live check of every prompt change in the bundle; the six forge verification gates fold
  into it.

## 11. Delegation, order, acceptance

- Order: D (schema) first, then E and B together (they edit the same prompt sections; one careful
  agent for all of `prompts.py`, or one per prompt file region with disjoint line ranges), then A
  and C (they add bullets into the Phase 1 region E just cleaned), then docs. Worktree per item,
  opus, TDD; merge sequentially into `next-season-bundle`.
- Before each removal, principle 6 (trace the why in git). Before each merge of duplicates,
  principle 3 (keep the one-clause reason).
- Done means: gates green on the merged tree; rendered prompt sizes re-measured and within about 5%
  of §7's targets; every receipted correction present exactly once with its reason (grep); every
  wording pin in `tests/test_prompts.py` and `tests/test_open_bound_guidance.py` updated rather than
  deleted; AGENTS.md "Prompts" section rewritten to name the constants that remain and to record in
  one sentence each what was removed and why (with the audit paths); `/forge` run on the new diff
  with FIX findings applied (in a Workflow, pass `stallMs: 600_000` on every `agent()` call, or run
  heavy stages as direct agents).
- Effect measurement: none possible offline. The first read is the residual round after the fall
  cup's first resolutions: compare the announced-unscheduled shape's incidence and spot on
  live-triple records against the 6.4% / minus 18.7 baseline, and the history-repeats rate against
  12.1%, by re-coding rationales with the audit's lens definitions. Those definitions are the bullet
  list at the top of `scratch/failure_mode_audit_2026-09-02/AUDIT_SYNTHESIS.md`, with the record
  prep and dump helpers beside it in the same directory; an earlier version of this line pointed at a
  coder brief inside a workflow script under `~/.claude/projects/.../workflows/scripts/`, which does
  not exist on disk.

## 12. Decisions already taken by the operator (2026-09-02 evening), for the record

- Item B reversal of `_REMAINING_EXPOSURE_RULE` and `_ANCHOR_CONSISTENCY_RULE`: confirmed.
- Item D schema removals: confirmed.
- §7.1 removals: confirmed, with the pre-open rule kept in TWO places on purpose.
- §7.2: "three valid moves" KEPT; meta-justification / rationale sentences KEPT (the prompt should
  explain itself); forecastability REDESIGNED as a template step, not deleted; gap-fill counts
  dropped, "do not invent gaps" kept.
- §7.4 contradictions: all four resolutions confirmed.
- Open: Item C (ship or skip; recommendation ship).
