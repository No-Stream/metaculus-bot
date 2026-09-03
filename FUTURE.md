# Future Ideas

Ideas for improving the forecasting bot, roughly ordered by expected impact and feasibility.

> **On `scratch/...` citations.** Entries below cite dated analysis receipts under `scratch/`.
> Those are the operator's local research artifacts and are deliberately NOT committed
> (`scratch*/` is gitignored) — they run to hundreds of megabytes of captured research text and
> per-question payloads. A citation names which analysis produced a finding and when, so the
> claim is attributable even though a reader of this repo cannot open the file. The same
> convention applies to `scratch/` paths cited from AGENTS.md and from code comments.

> **Status as of 2026-05-10** (closing residual on spring-aib-2026, n=189; receipts
> `scratch/analysis_2026-05/{analysis_synthesis,extended_hits_misses_postmortem,NEXT_SESSION_QUEUE}.md`).
> Two findings then reshaped priorities: (1) 17/20 worst misses were high-spread (>0.15) with a base
> model closer than the ensemble — models disagree on reference class and median pulls away from the
> closer minority (though "right model" attribution is post-hoc); (2) stacking treatment effect
> directionally measurable at +89.8% bootstrap (n=8, `analysis_stacking_historical_treatment.md`).
> Both are now largely superseded — stacking was later rejected/disabled, and the 2026-07-18 residual
> found the modal worst-miss has moved to consensus-with-zero-dissenters (evidence in the parked
> "Spread-triggered second forecast round" entry under Low-priority; the live lever aimed at it is the
> shared-reliance / consensus-fragility item under "Ideas reverse-engineered from high-scoring
> competitor bots").

> **Status as of 2026-07-18 (july15 branch — shipped state).** This branch flipped several
> long-gestating items live in prod (all four workflow yamls). This block is just the index;
> detailed status lives in the per-item entries below.
>
> **Every date in this block is an AUTHORING date on the july15 branch. All of it reached prod
> together in merge `b4e9df0` at 2026-07-21T17:07:37Z, which is the single era boundary** —
> nothing on `main` changed between the 2026-07-12 merge (`f084bf7`) and that one, so 07-15 /
> 07-17 / 07-18 / 07-20 are not separable eras, and no shift across the boundary can be
> attributed to any one item (AGENTS.md, era-bucketing → merge-date rule).
>
> - **Agentic gap-fill v2** is ON (`GAP_FILL_V2_ENABLED: 'true'`), authored 2026-07-17, running
>   concurrently with v1 during the overlap window.
> - **Time-series anchor (text)** is ON (`TS_ANCHOR_ENABLED: 'true'`), authored 2026-07-17; the
>   chart-image side-channel (`TS_ANCHOR_CHART_ENABLED`) stays OFF pending its A/B.
> - **Summarizer, native-search, and crux-analyzer models migrated sol→terra** (native-search +
>   crux authored 2026-07-17; summarizer 2026-07-18) — see the per-role entries below.
> - **Supporting infrastructure landed**: raw pre-summarization AskNews text archived as
>   `asknews_raw` in the research-persistence record (`research/persistence.py` +
>   `research/orchestrator.py`), OpenRouter credit telemetry (`credit_telemetry.py`, wired in
>   `cli.py`), and the era-bucketed numeric width monitor
>   (`performance_analysis/width_monitor.py`).

## High-priority

> **Priority audit 2026-08-25.** Every label below was re-checked against what shipped this round
> and against what is reachable before the tournament closes 2026-09-06 (no successor slug exists
> yet). One item survives here. Each demotion carries a dated one-line reason at the top of its
> entry, and answered/shipped items moved to "Resolved / shipped" near the bottom rather than being
> deleted. Note that the round's two largest *measured* levers sit in Near-term, not here: the
> deterministic numeric tail-consistency check (+11.93 baseline pts on the q44453 cohort) and the
> gap-fill v2 office-holder precedent rule (~+31 spot-peer pts on q44210). The fall-2026 season
> entry added 2026-09-03 below is an operator ACTION rather than an improvement lever, so it is
> outside that count.

### Fall 2026 season: the cup is configured, the bot tournament has no successor yet (added 2026-09-03, **operator action**)

Metaculus granted $1,500 of API credits on 2026-09-03 for the bot to compete in both the fall
Metaculus Cup and the fall bot tournament. The cup side is done in the repo: `METACULUS_CUP_ID` holds
`metaculus-cup-fall-2026` (project 33108, forecasting through 2027-01-01, API-verified), the cup
workflow is at full env parity with the tournament workflow and runs hourly at :13/:33/:53,
`FALL_CUP_CONFIGURED` is True so the dated reminder is discharged, and research records are labelled
by run mode so cup runs no longer archive under the bot tournament's slug. **Two things are still
open, both the operator's.**

1. **Enable `run_bot_on_metaculus_cup.yaml` on GitHub.** It is `disabled_manually` there, which no
   merge can change, so the crons fire nothing until it is switched on. There is no repo-side
   warning for this state; the way to notice is a supply-probe row showing cup questions with no bot
   forecasts.
2. **No successor to `summer-futureeval-2026` was published as of 2026-09-03** (id space above the
   summer tournament empty, four plausible slugs 404, forecasting-tools still pointing
   `CURRENT_AI_COMPETITION_ID` at summer), so `TOURNAMENT_ID` deliberately stays on the summer season
   rather than being guessed. **From 2026-09-20** (`TOURNAMENT_END_DATE` plus
   `TOURNAMENT_HARD_STOP_WEEKS`) `check_tournament_dates` raises, which reddens `--mode tournament`
   runs and the CI freshness test in `tests/test_tournament_dates.py`. That is the intended reminder,
   it does not touch the cup (cup mode never calls the check), and the fix when a fall bot tournament
   appears is to point the constants at it. `make supply_probe` is the free watch: it takes its
   default slugs from these constants and a dead slug renders as one error row without stopping the
   other rows. Details and receipts: `docs/operations.md` "Fall 2026 season".

### Triple-era September re-read (numeric watch + the era's whole scoreboard) (added 2026-07-20, **HIGH — operator-confirmed 2026-08-25**)

**Scope, per the operator: the checkpoint is the FULL triple-era read, not numerics alone.** Five
reads come due at the same checkpoint and share one cohort, so they run together (the fourth added
by the 2026-08-25 priority audit, when the ghost-scoring entry was demoted to Low and its re-read
folded in here; the fifth on 2026-09-02, for the two prompt rules the next-season bundle adds).
Read number 2 has since been answered ahead of the checkpoint and is kept below with its receipts,
so four remain live:

1. **The numeric ensemble delta** — the accepted +3.24 lean toward the retired 6-member roster,
   which is what this entry was originally opened for (detail and decision rule below).
2. **The all-types peer gap: RETIRED on 2026-08-31, re-confirmed retired on 2026-09-01. Stop
   citing the gap.** What the flag said when it was raised on 2026-08-24: the triple era's first
   scores came in mildly below the older eras, STRICT triple mean peer +4.28 (n=12) against
   post_flip's +11.6 (n=104), and the within-tournament cut (summer-futureeval spans both eras, so
   the question generator is held fixed) put the gap at −4.69 peer points after type-mix
   adjustment (raw −7.3; Mann-Whitney p=0.084 unmatched, 0.030 lag-matched), on 10 effective
   clusters and a resolved set that was by construction the era's short-horizon tail. It was
   raised as a flag to re-read on a bigger cohort rather than as a finding, and two rounds have
   now re-read it.

   The 2026-08-31 round retired it at n=20. The STRICT triple cohort read spot peer +13.8 with
   coverage-scaled peer +12.07, right at the post_flip level, and the within-tournament era cut
   measured +1.26 peer after type-mix adjustment, cluster bootstrap 95% interval [−6.78,
   +9.55], Mann-Whitney p between 0.44 and 0.95 under every collapse convention. The 2026-09-01
   round confirmed that at n=30 (28 effective clusters, 25 conservative): STRICT triple spot peer
   +17.46 against post_flip's summer-clean +14.33, a type-mix-adjusted gap of +3.66 spot with a
   cluster bootstrap interval of [−6.67, +13.90], and +5.79 on the coverage-scaled field the
   earlier round used, which sits inside that round's own reopen interval. The correct statement
   is "no measurable era difference".

   Two honesty notes travel with the retirement. Retired is not inverted: both rounds warn
   explicitly that reading these numbers as "the triple is better" would be the same small-sample
   error with the sign flipped, and the 2026-09-01 gap drifted positive mostly because the
   comparison arm fell, since the 19 new post_flip records are that era's long-horizon "before
   September" tail and scored +3.01 against +16.11 for the post_flip records already in hand. And
   the −70.1 single-record reading that prompted the second look was the dry-key incident
   scoring itself (q44874, coverage-scaled peer −70.1, published on one surviving forecaster),
   not the live bot; it sits in `DEGRADED_RUN_QIDS` and is excluded from every figure above.
   (Flag as raised: `scratch/residual_2026-08-24/SYNTHESIS.md` §1. Retirement:
   `scratch/residual_2026-08-31/SYNTHESIS.md` §1 and
   `scratch/residual_2026-08-31/dim_category-tournament.md`. Confirmation:
   `scratch/residual_2026-09-01/SYNTHESIS.md` §1 and
   `scratch/residual_2026-09-01/dim_category-tournament.md`.)
3. **The anchored/unanchored and v2-treated/untreated splits**, which are newly cheap to run:
   `performance_analysis/research_tags.py` (shipped 2026-08-24, `dece67f`) now stamps
   `anchor_present` / `anchor_confidence` / `gfv2_present` / `gfv2_loop_ran` /
   `research_source_class` onto every performance record straight off the research archive, so the
   treated-vs-untreated cuts no longer need re-deriving out-of-band each round. Both treatments
   went live in the same merge as the triple, so the era is the first place either can be measured
   against a fixed roster — and `anchor_present=False` must be read through
   `anchor_confidence`, since a trimmed comment-backfill record can read absent when it isn't.
4. **The gap-fill v2 ghost re-score** (`make score_ghosts`, free). The first read was a null at
   n=12 and joins the same resolutions this cohort waits on, so it rides along rather than being
   scheduled separately; the composition guardrail (most scored ghosts are byte-identical to the
   driver's PRE-research dry run) is in the demoted entry under Low-priority.
5. **The effect of the two Phase 1 prompt rules the next-season bundle adds**, `_SOFT_CLOCK_RULE`
   and `_HISTORY_DISCHARGED_RULE` in `metaculus_bot/prompts.py` (Items A and C of
   `scratch_docs_and_planning/announced_unscheduled_fix_plan_2026-09-02.md`; live from the
   bundle merge). **Item C is not an approved rule.** The plan reserves it for the operator in
   three places and recommends shipping; it shipped on that recommendation with the operator's
   final say still pending, so a reader landing here should not take it as settled. If the
   operator declines it, this watch item narrows to `_SOFT_CLOCK_RULE` alone and the
   history-repeats comparison below has nothing to measure. AGENTS.md carries the revert recipe. Nothing about either can be measured offline, so the first read is the residual
   round after the fall cup's first resolutions: re-code the live-triple rationales with the
   audit's lens definitions (the lens list at the top of
   `scratch/failure_mode_audit_2026-09-02/AUDIT_SYNTHESIS.md`; tables in `STATS.md`) and compare
   the announced-unscheduled shape's incidence and spot peer against the whole-archive baseline of
   6.4% of records and minus 18.7 spot (95% CI 5.9 to 33.4 worse), and the history-repeats
   rationale rate against 12.1%. Two cautions carry over from the audit. The shape fired on 0 of
   the 30 resolved live-triple records, so there is no pre-bundle live-era rate to compare against
   and the whole-archive figures are the baseline. And the history-repeats label had coder
   agreement 0.59 and was partly hindsight-contaminated, so 12.1% is an upper bound and a
   post-bundle rate should be read as one too, not as a point estimate.

Status: shipped-and-watch. The drop to the latest-per-vendor triple (`gpt-5.6-sol` /
`claude-opus-4.8` / `gemini-3.1-pro-preview`) shipped accepting a fragile numeric lean toward the
retired 6-member roster. Two adversarially-verified analyses
(`scratch/ensemble_3member_audit_2026-07-20/` + `scratch/ensemble_power_model_2026-07-20/`) found the
triple non-inferior on binary (delta +0.01, P(loss>1pt/Q)=0.35) and MC (−0.14, 0.26), but numeric
leans full by **+3.24 log pts/Q, 95% CI [−2.5, +9.1], P(full better)=0.88, P(loss>1pt/Q)=0.80** —
the interval spans zero, so not decisive, but it is the one type where the drop could cost score.
Operator explicitly flagged this as "honestly a bit risky" — hence HIGH.

**Why it's fragile, not merely uncertain.** The lean rests on 2 questions: the top-2 |delta| values
are 51% of the summed numeric advantage, and jackknifing them halves the mean (+3.12 → +1.56 on raw
deltas). So don't read +3.24 as a precise number; the mean-targeting Normal likelihood and the
coverage-verified (over-covering) CI already handle the heavy tail.

**Why waiting for more questions does NOT resolve it.** The numeric posterior SD is dominated
by *between-era* variance (tau ≈ 3.85 over 5 eras), not within-era noise. That sets a hard
floor of ±3.37 log pts/Q on the 95% half-width **no matter how many numeric questions
resolve** — every operator precision target (±1/±2/±3 pt) sits below the floor and is
unreachable by question accrual. The only lever that tightens numeric is more *independent
roster-stable eras* (the tau/√n_era term), each of which is months of fixed-roster operation —
and no successor tournament slug exists yet (probed 2026-08-24: HTTP 400 on every plausible
fall-2026 name), so the next era cannot start until a new tournament does. This is a
ship-and-watch bet decided on today's lean, not a measurement we can sharpen by collecting more
questions.

**Plan.**

1. **FREEZE the triple** (gpt-5.6-sol / opus-4.8 / gemini-3.1-pro-preview). Roster churn
   restarts the era clock and re-inflates the between-era variance floor that blocks any
   numeric measurement — the current lineage has ~0 resolved questions, so the frozen triple
   is the *first* modern-lineage era. This is the second roster change authored on 2026-07-20 (it
   supersedes the morning fable-5 → opus-4.7 swap, which rode the SAME merge and so never ran in
   prod at all); do not swap members mid-window.
2. **Checkpoint 2026-09-12 (moved from 2026-08-25); the read is DESCRIPTIVE — do not run
   `run_all.sh` as a decision gate.** The cohort is **complete at 37 numeric-family questions
   forecast** (18 numeric + 19 discrete) and merely pending resolution: 6 resolved as of
   2026-08-24, all 31 open ones scheduled ≤ 2026-09-04 (the 30th on 2026-09-02; observed
   scheduled→actual lag 0.6–3.6 days). Ceiling **n ≈ 35 ALL / 34 STRICT; 50 is unreachable by
   construction** (only 37 exist). The prior round's ~5x projection error — expected ~30 resolved
   by 2026-08-25, got 6 — came from using the submission rate as an accrual rate without applying
   the resolution lag. Run all four header items in ONE pass, since they read the same records
   and re-pulling is free: the triple-era numeric peer / log score and PIT against the post_flip
   cell (the "did the accepted lean show up as a visible loss" sanity check), the all-types
   within-tournament peer gap on the full resolved cohort (a re-confirmation only now, since
   header item 2 was retired at n=20 and again at n=30), the anchored/unanchored +
   v2-treated/untreated splits off the `research_tags.py` fields, and the ghost re-score. Each is
   descriptive at n≈35 with
   ~10 independent clusters — below the between-era floor above — so no fitted correction ships
   off them (AGENTS.md: fitted calibration layers need a decisive out-of-sample era test).
   (`scratch/residual_2026-08-24/dim_numeric-width.md` §2.)

   **Update 2026-09-01: the resolution-lag projection above missed again, for the fourth round in
   a row.** Of the 26 numeric-family questions sitting at post status `closed` on 2026-08-31,
   exactly 3 resolved into the 2026-09-01 wave (45215 discrete, 45362 and 45363 numeric) and 23
   remain unresolved, among them both pending degraded records (44875, 44876), all three pending
   TS-anchored questions (44803, 44943, 45172), the post-fix replay 45241 and the TS-routing
   target 45401. STRICT triple numeric-family n therefore went 10 to 13, not the 15 to 25 the
   projection expected, so expect the 2026-09-12 checkpoint to read well short of n≈35, and treat
   any date on this cohort as Metaculus's to set rather than ours to forecast.
   (`scratch/residual_2026-09-01/FOLLOWUP_LEDGER.md`, the three framing facts at the top.)
3. **Decision rule:** reintroduce the dropped numeric members ONLY if the frozen-era numeric
   delta *still* leans full with **P(loss>1pt/Q) ≥ 0.7 AND** the point estimate survives the
   top-2-question jackknife. Otherwise keep the uniform triple permanently — a per-qtype roster
   split is the config spaghetti the operator would rather avoid, and it only earns its cost
   under that condition.

**Cost context for the re-add decision.** The MEASURED per-question OpenRouter spend in the
triple era is **$0.38–0.41/question** — an OpenRouter-only lower bound; excludes Google AI Studio
prepaid (Gemini grounded search, gap-fill v2 document reads), the AskNews subscription, and Exa;
measured over 29 triple-era runs / 33 questions ($0.4082/question; $0.3836 in the ranked-markets
window). Receipt: `scratch/residual_2026-08-31/gemini_review/RECOMMENDATION.md` §3, "Cost, and a
standing figure that should stop being quoted". The earlier estimate that the 6→3 drop cut
per-question reasoning spend from ~$3.05 to ~$1.65 (three fewer xhigh forecasters) was never
measured and is an order of magnitude too high; it is superseded and must not appear in a re-add
decision. Removing grok did also end routine personal-key forecaster spend. A 4th reasoning
forecaster is plausibly +$0.10–0.15/question, but the measured total cannot yet be split into
forecaster versus research versus ranker, so that increment is an assertion until the per-role
ledger accumulates a season of runs: `CREDIT_ROLE_SPEND` (`metaculus_bot/credit_telemetry.py`,
"Per-role dollar attribution", shipped in the 2026-09 bundle) books every OpenRouter completion
under its role (`forecaster:<vendor>`, `native_search`, `gap_fill_resolver`, `parser`, ...), and
`scripts/reconcile_credit_spend.py --roles` tabulates it against settled per-run spend. Price a
re-add off those rows, not off this paragraph. A re-add must clear both the score bar above *and*
justify the cost it brings back.

**Coverage caveat on those role rows (added 2026-09-01, forge F1/R18).** The same bundle bounded
litellm's end-of-run callback drain at `LITELLM_CALLBACK_DRAIN_TIMEOUT_S` (10 s) so telemetry can
never hold a finished run hostage, and a drain that times out logs a
`LITELLM_CALLBACK_DRAIN_TIMEOUT` WARN saying in as many words that the `CREDIT_ROLE_SPEND` ledger
under-counts that run's last completions. That WARN is deliberately not a harvested MarkerSpec, so
the one thing that would explain a low `reconcile_credit_spend.py --roles` coverage ratio is
invisible in the telemetry archive. Register it if a coverage ratio is ever puzzling.

**Discrete-calibration note (2026-08-24).** The post-fix (`9f1175c`) discrete cohort has ZERO
max-step-bug exposure by composition: all four resolved triple-era discretes are fine-grid
percent/spread questions, and re-introducing the legacy 0.2 cap changes their CDFs by exactly
0.00 — so a "the fix holds" read is vacuous until a low-count integer question resolves post-fix.
The bug itself is now priced: **+179.9 baseline log points over the 10 measurable pre-fix
forecasts** (mean +18.0, median +4.1), 93% of it in three questions where the resolution landed
in the capped bin. And the 0.2 cap's own start date is unestablished — six pre-fix discretes
published max bins above 0.2, three before `dc31951` (2025-11-05) and three after — so don't
era-bucket discrete calibration earlier than the fix date on a "capped throughout" assumption.
(`scratch/residual_2026-08-24/dim_discrete-maxstep-counterfactual.md`.)

Receipts: `scratch/ensemble_power_model_2026-07-20/synthesis.md` (composed-delta model, power
floor, dated re-run plan) and `scratch/ensemble_3member_audit_2026-07-20/synthesis.md` (paired
bootstrap audit). Era boundary: **2026-07-21T17:07:37Z** (`b4e9df0`) = the triple config live in prod;
2026-07-20 is only its authoring date.

## Research-triage round 2026-07-16 (lit + repo survey + codebase verification)

> Provenance: eight parallel research agents surveyed the last ~year of LLM-forecasting
> literature (evaluation pitfalls, prompting, ensembling, calibration, research harnesses,
> market-lifecycle) and the open-source forecasting-agent repos (Gnosis, MiroFlow,
> predictionprophet, vox, TimeCopilot, ForecastBench, futuresim, Bench-to-the-Future),
> plus two codebase verifications and a rigorous era-bucketed calibration audit. Entries
> below are grouped by priority within this block; thread them into the topical sections
> above if they survive discussion. Every "why it helps" from a repo is a mechanism argument
> unless a paper number is cited. The killed items are logged at the bottom so they aren't
> re-litigated.

### Calibration is probably NOT a realistic improvement lever for us (added 2026-07-16, decision) — but promote the audit to a monitoring module

Rigorous era-bucketed Bayesian calibration audit (2026-07-16; scripts
`scratch/calibration_audit_2026-07-16/`, data `scratch/coherence_2026-07-15/perf_all_tagged.json`,
694 records). **Conclusions:**

- **Not cleanly over- or under-confident — it flips by era.** Bayesian logistic slope:
  fall-aib-2025 **1.70 [1.28, 2.20]** (under-confident), spring-aib-2026 **0.83 [0.58, 1.10]**
  (over-confident), summer n=15 uninformative. Opposite-signed eras cancel to a ~calibrated pooled
  slope (1.17) — why past pooled reads were contradictory. Even within spring the early/late split
  flips 0.64 → 1.25 (roster churn inside one tournament).
- **"More NO right than YES right" is a base-rate artifact, NOT directional miscalibration.**
  YES-rate 24%/33%/60% by era; the decisive per-side test (observed vs expected-if-calibrated,
  Beta-Binomial CIs) is consistent with calibration on both sides in every era. Puzzle resolved.
- **The killed spring YES-overconfidence finding reproduces as spring-local** (intercept −0.40
  [−0.80, 0.00], P(a>0)=0.017) and is untestable on the current post-flip roster (n=15); fall shows
  the opposite sign. Nothing to fit.
- **Chronically data-starved for the miscalibrations a fitted layer would target.** Power sim:
  detecting mild overconfidence (slope 0.85) has power ~0.25 even at N=400; a roster era yields only
  ~150–210 resolved binaries before it churns. We catch only *large* miscalibrations, and any fit
  spans opposite-signed eras — a drift bomb by construction.

**Decision (operator, 2026-07-16): calibration is not a realistic source of improvement given our
budget. Prefer SOTA forecasters over fitted calibration.** Do NOT ship isotonic / Platt /
directional-shrink layers (see the killed-calibration entries; the Beta-Bernoulli "calibrator"
from arXiv 2605.27668 degrades OOD in its own tables — cite it as anti-adoption evidence).

**If Platt is ever enabled, one published binary path would bypass it.** `_base_combine`
(`aggregation_pipeline.py`) returns from its `len(predictions) == 1` branch above the
`_apply_platt_calibration` tail, so a lone raw binary survivor is the one published binary value that
never reaches Platt. `_simple_aggregate` is its mirror: it calibrates a single prediction but applies
no single-survivor publish floor. Inert while `PLATT_CALIBRATION_ENABLED` is unset in every workflow,
which is the standing state and the reason this is recorded rather than fixed. Whoever turns the flag
on decides whether the k=1 floor and Platt compose, and in which order; the branch comment carries
the same note.

**DO promote the audit to a standing monitoring module (free, zero scar risk).** Complementary to
`performance_analysis/analysis.py` (point-estimate buckets, a single `bias_pp` scalar, PIT
coverage): the audit adds multi-era comparison, Beta-Binomial CIs per bucket, the slope/intercept
decomposition (separates confidence from directional bias, which `bias_pp` conflates), the YES/NO
base-rate-artifact test, the power sim, and partial pooling. Reuse the existing `_interpolate_pit`
(log-grid fix) + bucket plumbing. Cadence: on each roster change print era-bucketed slope/intercept
CIs + reliability table + base-rate check; **act only if a CI excludes the null AND reproduces
across ≥2 eras** — "inconclusive" is the honest default.

**Update 2026-08-24 (one retired citation + gate bookkeeping).** (a) **Retire the fall
[0.70, 0.90) "under-confidence signature" as a standing citation** — quoted by the 2026-07-16
audit's RESULTS.md and the 2026-08-02 round as the one real reliability deviation, its exact
Poisson-binomial p is 0.045 against 21 bins tested across four eras, i.e. exactly the single false
positive that multiplicity predicts (1.05 expected). The fall/spring *slope* flip is the genuine
finding and does not depend on that bin. (b) The post-flip binary YES/NO gate (n≥40) **fired at
n=41: verdict consistent with a base-rate artifact on both sides**, now in all four eras including
one at a ~54% YES rate — retire the gate rather than re-target it; post_flip is a closed era.
(c) Retargeted at `triple_era`, the prior-round n-gates (n≥40 YES/NO re-test, n≥25 dimension
re-run, n≥15 MC verdict) are **unreachable inside summer-futureeval-2026**: 4 STRICT triple
binaries today, ~9 projected by the 2026-09-20 hard stop, the tournament closes 2026-09-06, and no
successor slug exists yet — record them as awaiting a next tournament, not accrual.
(`scratch/residual_2026-08-24/dim_binary-mc-calibration.md`.)

### ~~Geometric-mean-of-odds base-combine vs MEDIAN~~ — RUN 2026-07-16, DECISIVE NULL (keep MEDIAN)

The one open aggregation variant the prior rejection never covered (binary geo-mean-of-odds as
the scored base-combine; MEDIAN is logit-invariant so only a mean-type pool can differ). Offline
replay, zero API (`scratch/geo_odds_2026-07-16/`, `geo_odds` arm + 5 tests in
`ablation/offline_replay.py`): NULL everywhere — no era's bootstrap CI excludes zero on log-score
or Brier (fall +0.32 [−1.34, +2.04]; spring +2.14 [−0.52, +5.34], Wilcoxon p=0.16; pooled +1.07
[−0.40, +2.68]), Brier opposite-signed to log-score (wash tell), and on the full-ensemble n≥5
filter fall FLIPS negative — confirming the documented "sharpening pools wash out on the diffuse
6-model ensemble" headwind. **Verdict: keep MEDIAN; geo-odds joins the settled dead paths (with
mean, stacking-as-default, coherence-weighting). Do NOT re-run** absent a materially different
ensemble regime (a confident small-N roster) or post-flip data with a different disagreement
structure. Arm + tests stay in the harness for cheap re-runs.

### Backtest statistical hardening + leak-aware replay (added 2026-07-16)

Two independent findings (papers-skeptic + eval-repos) plus a codebase check converge here.

**(a) Statistical rigor — AUDITED 2026-07-16, now a scoped consolidation program.** A methodology
audit found the recent scratch analyses strong-to-gold-standard (coherence phase2: two-way FE +
cluster-robust SEs; `ensemble_screen`: paired deltas + era-bucketing; calibration audit: Bayesian
posteriors + power sim) but the **standing pipeline is the thinnest machinery in the repo**, weak
in the two ways the house rubric names mandatory (pairing, era-bucketing). **Operator directive:
promote the scratch machinery into the standing pipeline.** Ranked gaps:

1. **`backtest.py` never pairs** (highest leverage): runs multiple bots on identical questions,
   persists per-qid scores (`backtest/analysis.py:205`), then reports pooled means + population SD
   with zero inferential content (`analysis.py:43-58,104-118`). Fix: join arms on qid, per-qid
   deltas, import the in-repo primitives `ablation/scoring.py`
   `bootstrap_mean_ci`/`bootstrap_median_ci`/`sign_test`/`wilcoxon_signed_rank`.
2. **`performance_analysis/analysis.py` emits pooled, CI-free, era-blind calibration — the
   actively-misleading item** (the pooled numbers that flipped three times). Fix: Beta-Binomial CIs
   on every bucket (`calibration_audit/binary_calibration.py:53-56`, a 4-line drop-in) + a
   first-class era key. This is where the calibration-monitor promotion lands.
3. **Per-model comparisons unpaired** in both standing surfaces — same fix as (1).
4. **Percentile bootstrap under-covers on heavy-tailed log-score deltas** — `ensemble_screen` needs
   the median/robust (or Bayesian) hedge ablation Path A already has.
5. **No clustering/effective-N outside coherence phase2** — mirror phase2's `cluster_bootstrap_mean`
   where a cheap cluster key exists.
6. **No multiplicity correction** in ablation (~15) or ensemble_screen (~48) — BH q-values or
   partial-pooling. Low priority.

**Explicitly NOT broken (don't make work):** `backtest/scoring.py` CRPS/log-score (correct),
`audit.py` (non-inferential by design), ablation Path B fold-std (stability diagnostic), plain
means on large-n bounded proportions. The problem is never "a mean was used" — it's "used as the
*comparison* with no pairing, CI, or era split." Reusable-machinery map: primitives
`ablation/scoring.py`; era-bucketing `ensemble_screen.py:109-140`; Bayesian CI drop-in
`calibration_audit/binary_calibration.py:53-56`; cluster-robust
`coherence_2026-07-15/phase2_lib.py:176-345`.

**(b) Leak-aware replay — plumbing exists, archive quality is the gap.** Default
`backtest_{smoke,small,medium,large}` targets **re-run live research at replay time**
(leakage-exposed; the leakage detector is a filter, not a fix). Frozen-replay already exists
(`--research-dir` → `make backtest_with_cache` → orchestrator skips providers), so code distance
to leak-free ≈ zero. The gap is archive quality, though less of one than this entry used to
claim: it read "only 19 of 921 `latest/` records are genuine GHA-captured payloads" off a
`latest/` that the merge bug had filled with comment reconstructions. `by_qid/` held **280
artifact records** the whole time, and the 2026-08-03 precedence fix (artifact beats
comment_backfill in `download_research.record_precedence_key`) promotes all 280 into
`latest/` — 100% of the 269 questions that have any artifact-era record. The remaining ~734
`latest/` records are pre-2026-05-29 questions where no artifact was ever written, so they
are correctly comment-sourced (trimmed, reconstructed `providers_used`), and uncached qids
still fall back to live fetch. So: (1) flip default targets to `--research-dir`, (2) keep
growing artifact coverage from prod runs. Cheap near-term RetroSearch (below).

**Framing takeaway (papers-skeptic, arXiv 2506.00723 + Vaticinus preprint):** the re-run backtest
is an optimistic **upper bound**; calibration-on-own-published-forecasts (elicited at forecast
time, leak-free) is the trustworthy axis. And **stop treating "community Brier − our Brier" as
edge-over-market** — it's an affine shift of Brier on a balanced panel (ρ=1.000 across 25 rounds),
zero edge.

### Forecaster-prompt audit — verified clean, no action (added 2026-07-16)

The preregistered Schoenegger/Tetlock result (arXiv 2506.01578) found two prompt scaffolds
reliably *hurt* accuracy via over-decisiveness: step-by-step "Bayesian reasoning" and
"propose-evaluate-select". Audit of `prompts.py`: **neither harmful form present** — the closest
(binary:394 anchor-and-adjust) is the benign outside-view move the study carves out. The three
base prompts (~11.7k/7.8k/12.8k chars) are focused, not bloated. **No action.** Follow-up
**RESOLVED 2026-07-16:** the `base_rate_anchor` / `criteria_clauses` binary-schema fields are NOT
dead — live-elicited since `30bca2f` reached main in merge `642b027` on 2026-07-11T16:37Z (authored
2026-07-08; the merge date is the era boundary, per AGENTS.md era-bucketing → merge-date rule, and it
is what makes the presence rate a clean 100% instead of a partial one — **58%** on the per-slot cohort
in `scratch/residual_2026-08-02/dim_ghosts-and-guards.md:118`, 78.9% on a per-comment recount; the
cohort-dependence is itself a reason to key on the merge date and read 100%); the "0/2203 rows"
finding was a data-window artifact (archive ended 2026-07-01). Wrong-mechanism claims in
`comment/markers.py` + `scratch/coherence_2026-07-15/synthesis.md` fixed same-day.

### Market-deference: time-to-close term MEASURED DEAD; liquidity fix SHIPPED 2026-08-03, matching fixes survive (updated 2026-08-03)

Applicability gate (offline archive mining) killed the headline half:

- **Time-to-close term: structural null — drop to bottom-of-low.** Of 285 matches with real close
  dates (n=64 questions, ~July 2026), **0.0% within 30 days of close** (median 185d), liquid∧near-close
  intersection **zero**. Structural, not small-n: a near-identical market closes ≈ when the question
  resolves, we forecast near open (`skip_previously_forecasted_questions=True`, `cli.py:119-148`), and
  the `as_of` filter drops markets closing before resolution. Revisit only if we add late re-forecasting.
- **~~(1) Blank liquidity labels~~ — FIXED 2026-08-03, and the logged diagnosis was WRONG.** The item
  read "fix the `total_volume`/`open_interest` fallback"; there was no fallback to fix. Kalshi's live
  API carries `volume_fp` / `open_interest_fp`, the bare names the builder read appear nowhere in its
  OpenAPI spec, and the third field it consulted (`liquidity_dollars`) is documented to always return
  `0.0000` — the fields were never "dropped", they were never read. The logged one-line fix would have
  made things WORSE: on long-horizon markets `volume_24h_fp` is 0.00 (only 189 of 1,504 live markets
  have any 24h volume), so scoring off it labels a deep market `thin`, which the prompt reads as
  "discount this as noisy". Sibling fixed in the same pass: Polymarket's `openInterest` lives on the
  EVENT (5/5 live events) while the parser read it off the nested market (0/42), so its OI column was
  dead on 142/142 archived rows. Guards: `tests/test_prediction_market_liquidity_contract.py` against
  committed live payloads, plus provider_health's `market_field_contract` signal, which reddens CI the
  same day a venue renames a field. **Two lessons worth keeping:** a "one-line fix" logged without
  reading the upstream schema sat open three weeks and got worse, and a blank label was never cosmetic
  — the shared prompt clause tells forecasters to weight market signals BY that label, so 100% blank
  was a live instruction to weight by nothing.
- **Surviving — two matching/deference fixes (top-of-low / bottom-of-medium):** (2)
  **Fuzzy floor (40) so loose "match" ≈ topical-adjacent**: 100% "match" but confidence never exceeds
  0.77; ≥0.7-confidence (the defer trigger) is only ~8% of questions, ≥decent-liquidity ~12% — raise
  the floor or add a confidence tier. (3) Optional mild deference nudge for deep+high-confidence matches
  (plumbing already exists; prompt/render-level).
- Mechanics: `## Prediction Market Snapshot` header renders only on ≥1 match (`orchestrator.py:401-403`),
  so header-absence ≠ provider-off — use `providers_attempted`; close dates captured for
  Polymarket/Kalshi/Manifold (PredictIt lacks them).

**Manifold's zero-contribution outage — FIXED 2026-08-03; keep the mechanism, it is undocumented
upstream.** Manifold contributed zero rows to any forecaster bundle for 17+ days while reporting
healthy in every channel (source token `none`, tally clean, alertable 0). `/v0/search-markets` treats
`term` as a **strict conjunction of content tokens** — one absent token returns `[]` — established by
appending a nonsense token to a query with a known hit (zeroes it) versus reordering present tokens
(does not), which rules out a relevance floor. The provider was sending 9-15-token question sentences
against a measured ~4-token satisfiability cliff. Fix: a relaxation ladder
(`manifold_relaxation_terms`) walking progressively shorter terms, stopping at the **first non-empty**
result — the longest satisfiable conjunction. Recall falls and precision rises monotonically in token
count, so that stopping rule is a derived optimum rather than a tuned threshold (hence no knob), and
it fires only after the precise queries return nothing, so the healthy path costs zero extra calls.
**Watch item:** `tests/test_prediction_market_integration.py` carries a live tripwire for the day
upstream switches to ranked search, after which the base queries suffice and the ladder can be
retired (harmless meanwhile).

**Rides along (low-value, operator 2026-07-16):** a bias-corrected **∆LL-over-matched-market diagnostic**
on the ~8% near-identical subset — do we add signal beyond the price or defer harder. NOT worth building
against the Metaculus CP (bot-only questions, CP is mostly-poor bots + null-hidden for our account).
Small subset → accumulates signal slowly.

### ~~JS-divergence diversity lens on ensemble screening~~ — BUILT 2026-07-16, verdict: NOT a selection signal (paper's heuristic INVERTS under median)

The "Diversity is the Strength of the AI Crowd" paper (arXiv 2606.29661) argues decorrelation
matters more than solo accuracy for roster decisions. Prototyped a JS-divergence metric on top of
`ensemble_screen`'s loaders (`scratch/js_divergence_2026-07-16/`, JS over
Bernoulli/option-simplex/202-bin PMFs). Findings: the metric is internally valid (Spearman(JS,
per-question score-corr) pooled −0.67, p≈1e-9 — high-JS pairs make independent errors), **but under
MEDIAN, decorrelation is INVERSELY related to marginal contribution** — Spearman(decorr,
removal-drop) = **+0.83** in the only rankable era (fall_6m): the most-decorrelated members (kimi,
grok) are the *least* load-bearing. Grok's independence is *incompetent* (off in a different
direction, not right-when-others-wrong) and MEDIAN discards outliers rather than harvesting them
(the paper's combiner was learned). **Verdict: do NOT promote JS as a roster-selection signal; the
marginal-contribution benchmark stays the decision instrument.** Keep the scratch code for two
uses: redundancy/near-clone detection (spring `opus-4.5|opus-4.6=0.017`, summer
`gpt-5.4|gpt-5.5=0.018` bits) and the error-complementarity Spearman as a validity check. Caveat:
the inversion is median-specific — a learned combiner/stacker could exploit decorrelation (relevant
only if the stacker-revisit lands).

### Revisit the conditional stacker with the AIA supervisor evidence (added 2026-07-16) — SUPERSEDED 2026-07-19

Folded into "Spread-triggered second forecast round (re-forecast, NOT stacker)" — parked at
Low-priority 2026-08-25 — which reframes the AIA (arXiv:2511.07678, supervisor 0.1125 vs
no-supervisor 0.1199 ≈ 0.0074 Brier) + BTF-2 evidence as motivating a targeted-research-fed
second BASE-MODEL round rather than a stacker revisit. The stacker-as-judge rejection still stands
(disabled in prod; our own benchmark found it counterproductive on the current ensemble).

### Time-series anchor for numeric questions — Phase A verdict IN; Phase B text anchor SHIPPED ON in prod (2026-07-21), chart still OFF

For numeric questions resolving on a fetchable series, render an empirical P10/P50/P90 band from
the series' own history into the briefing — **TS-as-anchor, not TS-as-answer**; a principled
version of the "status-quo / last-print anchor" finding. The shipped provider is deterministic
naive/empirical (no statsforecast dependency, no model selection).

**Phase A offline-replay verdict** (`scratch/ts_anchor_replay_2026-07-16/synthesis.md`). On 105
mapped class-A resolved numerics the empirical band beats our published CDF: paired relative skill
−1.17 (sign-flip permutation p=0.005 over 22 clusters; conservative CR2-t p=0.15 — cluster-fragile
but directionally robust). Decisive result is **coverage**: our published low tail is badly
miscalibrated (~3% below published P10 vs 10% target) while the anchor's P10 coverage is 0.18 and
P90 on target — sharper AND better-tail-calibrated. **CV model selection rejected**: the
must-beat-naive gate passed 76% but those picks beat naive OOS only 43% (worse than a coin flip),
so we render the naive/empirical band directly.

**Phase B text anchor SHIPPED ON, live in prod 2026-07-21T17:07Z** (authored 2026-07-17, landed in
merge `b4e9df0`; `TS_ANCHOR_ENABLED: 'true'` in all four yamls; chart
side-channel `TS_ANCHOR_CHART_ENABLED` stays OFF — its entry was parked at Low-priority
2026-08-25). Provider
`research/timeseries_anchor.py` + `ts_fetch.py` (deterministic routing + point-in-time/ALFRED-vintage
fetch + empirical bands); prompt clause `_ts_anchor_evidence_clause` (gated on section header).
Cleared its validation ladder (paid 3-arm smoke → `test_bot.yaml` eyeball → `make backtest_medium`,
leakage-safe because the provider date-ceilings the fetch to `open_time` under `is_benchmarking`) —
the FIRST prod research provider also measurable in backtest. The `TS_ANCHOR_ENABLE` config era
(2026-07-21T17:07:37Z, `b4e9df0` — the merge-to-main date, not the 07-17 authoring date) is tracked
by the width monitor; it is the whole july15 bundle, so an anchor-only attribution needs per-record
tagging rather than the era row — that tagging shipped 2026-08-24 (`research_tags.py`, `dece67f`) and
the remaining step is teaching `width_monitor.py` to split rows on it. Seed doc:
`scratch_docs_and_planning/ts_anchor_plan_seed_2026-07-16.md`.

**Applicability gate** (`scratch/ts_anchor_gate_2026-07-16/ts_labeled.json`): **53.2% (123/231) map to
a standard fetchable series** (fall 57% / spring 41% / summer 75%; strictest level-only reading 27%)
vs the ~10% skip bar. Class A is recurring templates (10Y yield, HY OAS, VIX, index/commodity returns,
gasoline, unemployment, CPI, payrolls, poll averages, TSA volume). Design notes:

- **Strong-value core = 63 level anchors** (macro prints, rates, spreads, poll averages, TSA).
  Spot-check: q43611/q43591 (poll averages) off in exactly the direction a live anchor corrects;
  q43647 (HY OAS) center-correct, would sharpen. (n=3, anecdotal.)
- **47 relative-return spread questions need the anchor fit on the SPREAD series** (center≈0 +
  historical-vol band), not a naive level forecast. The 13 max-functional questions (VIX/commodity
  highs) need window logic.
- **Net-new vs `financial_data`:** its allowlist misses HY OAS, gasoline, Brent, VIX, poll averages,
  TSA (4/5 spot-checked FRED series absent), and emits a raw last-6-obs table, not a fitted band.
- **Class B later-add:** Mauna Loa CO2 + Norwegian EV-share (held out of A on ingestion only; fold in
  once statsforecast wiring exists). Backtest measurement uses the gap-fill v2 eval-ladder pattern.

**The 53.2% is NOT a target for the shipped fetch layer — stop comparing the live routing rate to
it.** `SeriesSpec.source` is `Literal["fred", "yfinance"]`, and 13 of the gate's own 123 Class A rows
carry notes naming a publisher neither can speak (poll aggregator ×9, TSA daily throughput ×4), so
the fetch-layer-achievable slice of that corpus is 110/231 = 47.6%. Worse, the two populations
differ: on the 22 triple-era numeric-family questions, 11 resolve on bespoke one-off web trackers
(CDC surveillance portals, a Bluesky index, a robotaxi leaderboard, Copernicus burnt-area, state
election turnout) with no FRED/yfinance analogue at all, so that cohort's ceiling is 7/22 = 31.8%.
Read routing coverage as **routed / fetch-layer-reachable** (5/7 = 71.4% on the triple era before
the gasoline fix, 6/7 after) and treat the applicability gate as a scoping study, not a KPI. The
rate claims are n=22 and wide; the mechanism claims (which branch rejected which question) are
exact. Receipts: `scratch/provider_repair_2026-08-03/diag_d3-ts-anchor-routing.md`.

**Deferred, deliberately: poll averages and the two-leg FRED spread.** Both are reachable in
principle and both need new machinery, and both are where a wrong-quantity band is most likely — so
neither should ride in behind a keyword change.

- *Poll averages* (q44841, q44855, q44858, q44911 in the triple era; the "strong-value core" bullet
  above cites q43611/q43591) need a THIRD `SeriesSpec.source`. The blocking detail is that the free
  published CSV is poll-LEVEL (one row per poll) while the questions resolve on the tracker's
  weighted+adjusted daily trendline, so serving them means reimplementing a proprietary aggregation —
  and a mismatch produces a band centered on the wrong number while looking authoritative. The
  article page is also paywalled, so the link's stability is not ours. If built, the tracker's own
  trendline file is the only acceptable input, and `Derivation` needs a `trailing_avg` member gated
  the way the gasoline pair is: q44911 resolves on a 7-poll moving average and q44841 on a daily
  trendline value, which are two different derivations of the same polls — the CPI wrong-quantity
  hazard in a venue with no gate yet.
- *Two-leg FRED spreads* (q44868, the NOB spread: DGS30 − DGS10 in basis points) currently land in
  `route_question`'s `url_ambiguous` branch and skip, which is correct — there is no two-leg FRED
  route, and the yfinance two-ticker route builds a mean-zero relative-log-return in percentage
  points, a different quantity. Building it needs a spread route plus an explicit unit-scaling rule
  (percentage points → basis points). Two gates now hold that line, covering DIFFERENT cases: the URL
  branch's two-FRED-link ambiguity check (q44868's shape), and — added 2026-08-03 — a route-level
  wording guard (`_TWO_LEG_OR_CHANGE_RE`) for the same quantity mismatch when NO URL is cited, which
  the URL branch never sees; `5025568` then extended that guard ONTO the single-URL route as well
  (q45362) and gave it the percentage-qualified change vocabulary its name always implied. The
  magnitude backstop does not substitute for either: a 4.4-4.95 percent
  level band against a −50..50 basis-point displayed range scores INSIDE that range once the open-bound
  tolerance widens it (pinned by `test_the_backstop_alone_would_not_have_caught_those`). The guard
  changed zero routes across 264 corpus rows and closes the hazard on the pre-existing `10-year yield`
  keywords too, so it is not specific to the UST tokens. **Placement is load-bearing and was got wrong
  once:** implementing it as per-entry `exclude_keywords` removed DGS10 from the match list the
  AMBIGUITY check counts, so "the 10-year treasury yield versus the high yield spread" left HY-OAS as
  the sole match and routed to one leg — a correct ambiguous-skip converted into a wrong single-series
  anchor. The guard therefore runs AFTER the ambiguity check, where it can only turn a route into a
  skip, and entries whose own series IS a published spread (HY OAS) are exempt via
  `_SPREAD_NATIVE_SERIES`. Both properties are pinned by tests. A spread route must clear both gates
  deliberately rather than by loosening one.

**Registry families still absent but FRED-reachable** (same silent-coverage shape the gasoline gap
had, logged rather than left implicit): `USEPUINDXD` Economic Policy Uncertainty (q40196, q40195) and
weekly SPR stocks (q42815). And every non-`level` derivation carries the same two-entry hazard the
gasoline pair just fixed — `mom_pct` (CPIAUCSL) and `mom_diff` (PAYEMS) gate on quantity language
with no complementary level sibling, so a CPI index-LEVEL question or a payroll LEVEL question routes
nowhere. That is deliberate per `548ba88` (a wrong-quantity band is worse than none), but it means
those families go dark silently, exactly as point-in-time gasoline did. **`5025568` widened those
gates' positive vocabulary; it did NOT add the missing level siblings, so this hazard is unchanged.**

**Method note for any future registry claim: probe against `question_text + resolution_criteria +
fine_print` through `route_question` itself, and treat any title-only number as inadmissible for a
route-level claim.** A title-only check produced the wrong diagnosis for q45082 twice — its title
says only "gas price" (matching no keyword), while the keyword hit actually comes from the criteria's
"Regular Gasoline", so the logged "add a `gas price` keyword" one-token fix would have changed
nothing. The same method error then produced a wrong coverage claim for the UST-10Y widening: on
titles the two new tokens look like they recover 9 questions, but on real text all 9 already routed
through the URL branch (their criteria cite the DGS10 FRED link), so the tokens recover **zero**
observed questions and are wording robustness only. The applicability-gate corpus
(`scratch/ts_anchor_gate_2026-07-16/ts_labeled.json`) stores TITLES with no criteria, which is why
every gate-derived rate in this section is a scoping figure and not a routing measurement. Two
further traps in the tooling: a probe that calls `_entry_matches` directly skips `route_question`'s
URL branch entirely (so it over-credits keywords), and probes living under `scratch/` are gitignored
and do not survive a merge — re-derive rather than cite one.

### Necessary-condition / scenario decomposition scaffold (added 2026-07-16, bottom-of-medium)

Gnosis ThinkThoroughly (gnosis-repos): generate ~3 necessary preconditions + ~5 hypothetical
scenarios (incl. the negation) as sub-questions, research/score each, synthesize. Highest
ceiling of the "experiment" bucket, weakest evidence (no in-repo ablation; the PROPHET benchmark
found agentic-RAG scaffolding gave only marginal Brier improvement and naive RAG sometimes
*hurt*). Overlaps gap-fill v2's private dry-run (a lighter cousin). **Operator prior:** prompting
tricks mostly don't work — bots check the boxes you give them then put down what they want. File
it; gate hard on beating a flat forecaster in our own backtest before shipping.

### Lower-priority / logged (added 2026-07-16)

- **Embedding-NN market matcher + LLM resolution-equivalence check (low).** Replace/augment our
  fuzzy-string market matching (rapidfuzz) with embedding nearest-neighbor + similarity
  threshold (Gnosis pattern), plus an LLM "is this really the same resolution criterion" gate on
  top — catches paraphrase matches we miss, serves defer-to-market. Low-risk recall win, but low
  priority.
- **Consistency-check as an offline diagnostic (bottom of low).** The 0.85 consistency↔Brier
  correlation (arXiv 2412.18544) is a cheap no-resolutions-needed roster-health signal — probe
  members with negation/paraphrase variants, flag incoherent ones. **Diagnostic only** — the
  paper's own experiments show enforcing consistency does NOT reliably improve accuracy, and we
  killed coherence-*weighting* on 2026-07-15. Never in the prediction path.
- **Deterministic coherence projection (bottom of low).** KL/Brier projection onto monotone
  deadline/threshold ladders + MC-sum-to-1. Survives the era-test trivially (it's algebra, not a
  fit) but **rarely binds** — most AIB questions are standalone; sibling A/¬A/A⇒B ladders are
  rare (unlike a market book). We already enforce the two that apply (MC renormalize, CDF
  monotonicity). Low-yield; build a lightweight same-tournament ladder detector only if the
  ladder frequency turns out non-trivial.
- **ForecastBench external submission (bottom of low).** MIT code, open submission,
  contamination-free scoreboard vs superforecasters + other bots (arXiv 2409.19839). Genuinely
  useful for knowing where we stand leak-free — but **not free to forecast** (spends API on the
  submission question set), so it sits at the bottom.

### Study-only / longer-term (added 2026-07-16)

- **RetroSearch-style frozen point-in-time corpus** (FutureSearch, Bench-to-the-Future): live
  Google for ranking quality, filtered so the agent only ever receives pages already in a frozen,
  date-bounded per-question snapshot ("no tool returns a document from after the simulated now").
  The rigorous fix for backtest leakage. **Reproduce the design only** — RetroSearch code+corpus
  are proprietary; futuresim (openforecaster.github.io/futuresim) has NO license (study, don't
  copy). Cheaper reproducible variant: date-gated CommonCrawl News + LanceDB with query
  date-range control (both open). The near-term substitute is the frozen-artifact-replay in the
  backtest-hardening entry above — do that first.
- **Longitudinal trajectory scoring** (futuresim: sum of Brier-skill-score over timesteps;
  rewards correctness + calibration + timeliness). Suggestive that *timing* matters (TimeSeek),
  but no paper shows re-forecasting the same question over calendar time improves a bot's score,
  and it's an uncertain fit for single-shot Metaculus tournament scoring. Longer-term R&D.

### Answer to "should we go end-to-end agentic research?" — NO (shared, not integrated) (added 2026-07-16)

The strongest evidence (BTF-2, arXiv 2604.26106) argues against per-forecaster integrated
research+forecast pipelines: the single most accurate forecast was a **strong prompt on good
shared/fixed research** (0.129 Brier), edging the best self-directed integrated agent (0.131), and
integrated-vs-shared was **model-dependent** (only the Opus-class model clearly benefited from its
own search; Gemini was slightly *better* on fixed shared research). Our architecture (multi-provider
shared briefing + gap-fill + 6-model ensemble + median) already **is** BTF-2's winning recipe, at ~7
calls vs ~5N. The lever is **shared-research quality + a strong prompt**, not integration topology.
So: make the shared research more agentic (gap-fill v2,
`scratch_docs_and_planning/agentic_gap_fill_v2_plan.md`, SHIPPED + ON in prod since 2026-07-21 — that half is
done); do NOT rebuild into per-forecaster agents; if ever tried, give it only to the Opus-class slot.
Calibration caveat: these agentic-research wins optimize pass@1, and edge-over-consensus has a
*flat-to-negative* trend across model generations — "more search → more decisive" trades calibration
for sharpness, so track era-bucketed calibration alongside Brier. See "Longer-term → Agentic deep
research" below (superseded by this).

**Addendum 2026-07-19 (BTF-2 re-read).** BTF-2 (arXiv:2604.26106) strengthens the case that
per-predictor research is the literal ideal for the strongest research agent (Opus-class), which
scored worse on fixed evidence than on its own research (0.131 → 0.153; the effect was
model-dependent, with Gemini slightly better on fixed shared research at 0.143 → 0.141). But at
~$3-10/question for 6 frontier agents each running
live search, it stays not-planned. Read the design as a continuum — shared bundle → shared bundle
+ one agentic researcher (gap-fill v2, current) → per-predictor research loops — where the
affordable middle is more v2 rounds / parallelism, not a rebuild. The NO verdict stands.

## Near-term (worth exploring soon)

### Provider-degradation alerting: SHIPPED 2026-08-03, extend beyond prediction markets

`metaculus_bot/research/provider_health.py` gives a silently-degraded provider a same-day exit code:
the run publishes normally, then `alertable_count` folds in a per-run provider-degradation summand and
`cli.py` exits non-zero. Two rules today, both scoped to the prediction-market provider and both
100%-of-denominator conjunctions with no tunable float: `market_field_contract` (a declared liquidity
field dead across every POOL row a venue produced) and `catalogue_empty` (a prefetch reported
success and returned an empty catalogue). Replayed over 47 archived runs the rules fire on exactly the two
real degradations in 20 runs with zero confirmed false positives, and go silent on the fixed tree.
**Denominators shipped 2026-08-25 (`feee7f1`):** the marker now carries `venues_observed` /
`catalogues_observed` / `pool_rows` off `recorded_observations()`, so `findings=0` finally
distinguishes "measured, healthy" from the 96% of archived records that evaluated nothing — and a
partially-failing venue records its observation over the rows it did get instead of blinding the
field-contract signal on exactly the runs where one query flaked.

**The load-bearing design constraint, for anyone extending this: prod runs carry ONE OR TWO questions**
(median 1, verified on a 1011-run histogram of `{1: 1008, 2: 3}`). So a rate over "the questions in a run"
IS a per-question flag, and a per-question flag must never fire — a single question with no matching
market is normal, not a defect. Every denominator has to exist INSIDE one question (the pool rows a venue
produced, a catalogue's own size). A longer run then makes every signal strictly harder to trip,
which is the correct direction and means no retuning when a `test_bot` run carries four questions.

**A third rule, `venue_no_contribution` (Signal B), was DELETED 2026-08-04 — do not re-add it in
per-question form.** It fired when a venue returned zero candidates while ≥2 siblings answered, and under
ranked retrieval that conjunction is unsound: the enumerable venues enumerate whole catalogues, so their
candidate count is never zero — they could never be the flagged venue but always supplied the sibling leg
for free, leaving a search venue whose index legitimately returned `[]` satisfying the one leg meant to
exclude correct behaviour. Replaying all 59 archived `backtests/research_archive/raw/*.jsonl` snapshots,
the shipped rule fired on 45 healthy manifold runs and 26 polymarket ones; the narrowest surviving form
(search venues only, one sibling) still fired on 20 manifold runs. **The surviving intent is real but
cross-run:** a venue contributing zero across many CONSECUTIVE runs is a genuine defect and is
unjudgeable inside one question, so it belongs in a check over `backtests/telemetry_archive/` (the
`provider_degradation` marker plus the per-source tokens in the research archive) rather than in a
per-run rule. That is the shape to build if the signal is wanted back.

Worth extending to the other providers on the same denominator discipline: Gemini grounded search
(grounding metadata absent on 100% of calls), the resolution-source fetcher (every URL blocked/js_wall on
a question that had extractable URLs), gap-fill v1 (analyzer names gaps and every resolver search returns
empty). `PROVIDER_DEGRADATION_SUPPRESSED_UNTIL` (constants.py) ships EMPTY and is the dated per-venue lever
for a known-and-accepted degradation — the same contract as `CREDIT_ALERT_RESUME_DATE`, dated rather than
boolean so a stale acceptance cannot outlive the season unnoticed.

### TS-anchor follow-ups from the derivation-gating fix (added 2026-07-20)

Context: commit `548ba88` closed the wrong-quantity routing bug (Codex P1 on b6edd2b) — CPI
index-level/YoY/UK/Egypt questions were inheriting the US CPIAUCSL `mom_pct` band; PAYEMS and
GASREGW had the same keyword overreach. The fix gates non-level derivations on quantity language
and SKIPS when ambiguous. Two follow-ups deliberately left out of that commit:

- **`yoy_pct` derivation to recover the YoY-CPI family.** The gate now skips YoY inflation
  questions entirely (q41640-UK/q41634-Egypt-style, and US YoY variants). A `yoy_pct` derivation
  gated on YoY language ("year-over-year", "12-month", "annual inflation") would give that
  recurring family a correct-units anchor instead of none. Foreign-country questions need
  country-specific series (ONS/ABS per `scratch/ts_anchor_gate_2026-07-16/MAPPING_AUDIT.md`) or
  stay skipped — don't reuse CPIAUCSL. Same conservative registry pattern; the
  `require_any_keywords` machinery from 548ba88 makes this a small add.
- ~~**Split `timeseries_anchor.py`**~~ — **DONE** in `e4bddae`: the 1,293-line module became
  `timeseries_anchor.py` (372) + `ts_routing.py` (675, routing + registry) + `ts_render.py` (394) +
  `ts_estimators.py` (231), and the duplicated 8-field `_Route` construction folded into the
  `_route_from_entry` helper this entry asked for.

**Routing follow-ups added 2026-08-24 — all three SHIPPED 2026-08-25 in `5025568`**
(`scratch/residual_2026-08-24/dim_research-archive-qa.md` §2; the q45401 mechanism pinned in
`dossiers/44453_verification.md` C6). What landed, and what the round had asked for: (1) the
**PAYEMS derivation gate's vocabulary** now carries `increase` / ` add ` / `adds` — its old
`require_any_keywords = ("change", "jobs added", "added", "gain", "gained")` matched neither the
payroll family's title verb ("add") nor its criteria wording ("the increase in number of
employees"), so **q45401 was a live prod miss** at `kw_derivation_gate`; the same
narrow-vocabulary sweep was applied to the other non-`level` entries (CPIAUCSL `mom_pct` gained
m/m and monthly-inflation forms, the month-scoped gasoline tokens gained "in the month" /
"calendar month"). (2) the **change-vs-level guard now covers the single-URL route** — q45362 (S&P
500 percentage change) routed via its criteria's Yahoo link to a `^GSPC` LEVEL band and was saved
only by the magnitude backstop, a numeric heuristic a bounds-overlapping case would sail through;
the guard on that branch can only turn a route into a skip, so the keyword branch's load-bearing
post-ambiguity placement is untouched. Deviation worth knowing: moving `_TWO_LEG_OR_CHANGE_RE`
alone did NOT match q45362 — the regex had no change token despite its name — so it also gained
percentage-qualified change vocabulary, with bare `change` deliberately left out because the
`mom_diff`/`mom_pct` families route on it. (3) `route_question` now emits one
`TS_ANCHOR_ROUTE: qid=… decision=… series=… step=…` line per question plus a MarkerSpec — of the
era's 30 route-level misses only 2 had left any log line across 1,800 persisted run logs. Net
routing change over the era corpus was exactly two questions: q45401 skipped→routed, q45362
routed→skipped. Still true and still the reason to watch this: the anchor rendered on **0 of 42**
`ranked_markets`-era records (5/101 era-wide, all five FRED-mirrored series), so every 2026-08-03
and 2026-08-25 routing repair remains unexercised in prod — the `TS_ANCHOR_ROUTE` marker is now
how that gets measured instead of re-run offline.

**Calendar↔row conversion class swept 2026-08-25 (`c577231`)** — `horizon_steps` /
`_horizon_end_date` / `_realized_vol_line` / `_FREQ_UNIT` all converted on a hardcoded trading-day
density, which on a 24/7 series drew a 62-step band under a 90-day label. Detail in the resolved
entry "financial_data / ts-anchor: calendar time converted on a hardcoded trading-day density".

**`ts_render._fmt` swept 2026-09-01, same branch (added 2026-09-01, SHIPPED).** The bundle replaced
`:.4g` at the five FRED render sites (a Case-Shiller print of 331.893 was rendering as "331.9" on a
question whose displayed range was four index points wide) but left the anchor section's own
formatter rendering anything between 100 and 10,000 at ONE decimal, and dropping decimals entirely
above that — so one bundle stated two different values for one observation in two adjacent sections
(both providers append unconditionally, and q44944's archived record carries both). `_fmt` is now
fixed-point up to three decimals above 100 with trailing zeros stripped, `:.4g` below.

The taste call this entry flagged went the OTHER way than the entry proposed. The entry argued for a
separate level formatter at the "latest value" and history-table sites only, leaving P10/P50/P90
BAND estimates at one decimal as honest about an estimate's precision. The global three-decimal cap
shipped instead, because the band quantiles are exactly what a forecaster sizes an interval from and
one decimal costs up to 0.05 index points there, roughly 1-2 buckets on these questions. Three
rather than `research/number_format.py`'s `format_decimal_value` six keeps six-decimal fabricated precision off
an estimate.

### Agentic gap-fill v2: SHIPPED, ON in prod since 2026-07-21 (added 2026-07-16)

**FLAG STATUS: `GAP_FILL_V2_ENABLED: 'true'` in all four yamls** — flipped ON in the branch 2026-07-17
and live in prod 2026-07-21T17:07Z (merge `b4e9df0`; the merge date is the era boundary) after
the paid smoke, the blind driver eval (winner: gpt-5.6-terra effort=low, now the prod default),
and the Exa-alive confirmation replay (`scratch/driver_replay_2026-07-17/arm_terra_low_exa_alive/`).
Pending: turn v1 gap-fill OFF after the overlap window (operator must remember — **now gated on
quality, not just time, per the 2026-07-18 content-audit entry**); the 3.5-flash researcher switch
is undecided. Full design (source of truth): `scratch_docs_and_planning/agentic_gap_fill_v2_plan.md`
(rev 4).

Summary: a bounded agentic tool loop is the second-pass research stage — a driver LLM briefed with
the forecaster prompt template privately dry-runs the forecast to find fill/verify targets, then
iterates over four tools: `search_news` (AskNews rate-limit machinery), `search_web` (Exa direct,
key stored locally / GHA secret `exa_key`), `fetch` (ladder plain → headless Chromium → Gemini
url_context), `read_document` (Gemini flash url_context). Output: a detached citation-only findings
artifact appended to the bundle; a ghost forecast logged for telemetry only. DIY litellm-direct,
append-only message array for prompt-cache discipline.

**Rollout: BOTH v1 and v2 run in prod during an overlap window** (independent flags
`GAP_FILL_ENABLED` / `GAP_FILL_V2_ENABLED`, distinct headers); artifact diffs + resolution scoring
harvested from the both-on era. Turning v1 OFF is a deliberate pending step — nothing does it
automatically.

Evidence base (plan doc §7): Metaculus Fall 2025 bot survey — research breadth ~ accuracy r=0.42;
FutureSearch #1 with agentic research ~$1/question; Bridgewater ablation agrees; date-filtered live
search leaks resolution info on 41–55% of resolved questions, so **`make backtest_*` is
uninterpretable for research-stage changes** — eval is test_bot QA then early prod flip.

Post-review (plan rev 4): dup-query semantic stuck-detector deferred (v1 ships only a per-run
`dup_tool_calls` counter); traditional-researcher tier-tagging is a follow-up to ride the same era
boundary; era-bucketed calibration is in the v2 eval ladder.

**DeepNews (AskNews's agentic research product; we call only basic HOT+HISTORICAL today).** Two
options vs the v2 driver: (a) expose as an optional heavy tool `search_news_deep(query, max_depth)`
the driver escalates to when basic search comes up thin; (b) upgrade the `search_news` backend to
DeepNews with a depth param. **Blocked on operator checking DeepNews limits/pricing** (separate
quota pool, possibly subsidized for tournament participants — if cheap, solid case for (a)).

### Bundle content-audit findings: v1 retirement gate, AskNews reform, market header (added 2026-07-18)

Three findings from the 2026-07-18 bundle content audit
(`scratch/bundle_content_audit_2026-07-17/RESULTS.md`, Fable-judged per-section value/redundancy).

1. **Gap-fill v1 retirement risk — do NOT flip v1 off on the calendar alone.** v1 is the MOST
   load-bearing section per token (59% unique content; carried the decisive single-source fact in
   the majority of sampled questions). The ~$190/quarter retirement savings holds only if quality
   is preserved. **Gate:** compare v1 vs v2 findings on the first ~20 prod questions where both are
   present; flip only if v2 consistently surfaces the same decisive facts. **Status 2026-08-25: gate
   NOT met, and the cheap proxy came back uninformative.** The first scored ghost read is a null at
   n=12 with 7 of those 12 ghosts byte-identical to the driver's pre-research dry run (see the ghost
   entry under Low-priority), so it says nothing about whether v2's findings stand on their own. The
   v1-vs-v2 findings comparison still has to be done by hand over the both-on cohort.
2. **AskNews reform — shipped 2026-07-18.** Audit: 44% of the bundle by tokens, 57% padding, AND
   stale/directionally-wrong in 5/10 sampled questions while smaller sections had the right answer.
   "Longer is better" nudge removed (`5cfb6cd`); quality audit
   `scratch/asknews_quality_audit_2026-07-18/RESULTS.md` (R1-R5 + per-failure prevention matrix).
   Shipped in `asknews_summarizer_prompt`: **R1** recency-first ordering, **R2** supersession +
   quote-the-deadline-inputs arithmetic transparency, **R3** hard per-article relevance gate
   (off-topic DROPPED to a screened-out list), **R4** evidence-age disclosure opening the briefing,
   and an operator **proportionality rule** (length tracks decision-relevant content). The
   `asknews_raw` archiving hygiene rec is a separate in-flight change (orchestrator + persistence).
   R5 (fetch query enrichment) remains optional. Do NOT blindly halve the section — prioritization
   change + eyeball, not a cap.
3. ~~**Prediction-market header revision**~~ — **SHIPPED / INVALIDATED 2026-08-24.** The
   unconditional `STRONG EVIDENCE` header this item targeted no longer exists: the conditional
   preamble shipped in `6eed305` (live via `b4e9df0`), and the ranked-retrieval port (`bfd5df2`)
   replaced match-confidence gating with per-row `relation` tiers that the forecaster prompts
   weight directly — the cited `prediction_market.py:1184` is gone. Reprice note: the "reprice if
   it grows" trigger on the section's token weight has fired — the ranked-era snapshot runs
   **~1,804 tokens / 13.6% of bundle p50** (from ~847 / 7.0% gate-era and ~557 / 4.6% blunt-era),
   still cheap at ~$0.06-0.07/question and not an argument to cut. The **caution survives**: the
   `[PRE-WINDOW]` apparatus in the summarizer output is load-bearing (prevents pre-open events
   read as resolutions); do NOT remove. Live follow-ups on the ranked design are in the
   "Market-render follow-ups" entry (added 2026-08-24).
   (`scratch/residual_2026-08-24/dim_market-informativeness.md`.)

### Market-render follow-ups from the first ranked-era resolutions (added 2026-08-24; three of four SHIPPED 2026-08-25)

The ranked-retrieval port works — strict same-quantity-same-date matches went from 1/117 questions
to 3/41, top-tier rank-0 label precision is 63%, and forecasters demoted 4/4 mislabelled rank-0 rows
in writing — so the binding constraint had moved to RENDERING. Status of the four fixes
(`scratch/residual_2026-08-24/dim_market-informativeness.md`):

1. ~~**Rank child outcome rows by question relevance, not liquidity, before truncating**~~ —
   **SHIPPED, and superseded while shipping.** `4e342da` sorted children by price; `58175a7` then
   replaced the rows-compete-for-slots model outright — a multi-outcome family renders WHOLE
   (leading outcomes as full `↳` sub-rows, every remaining outcome named with its own price in one
   `[remaining N]` ladder row that collapses by forecast content rather than dropping). This was the
   one defect with a measured score consequence: 81% of `same_quantity_other_cut` parents were
   truncated at a median 27% of price mass, and on q45189 (the era's worst MC record, peer −6.04)
   the omitted rows were exactly the "Fine wins narrowly" brackets carrying 0.365 of the family's
   mass, so all three forecasters transplanted the one visible ≥50-point-margin price (0.585) into
   P(>70% of the vote) = 0.58 verbatim. Replayed over the 42 archived ranked-era snapshots:
   individually-named outcomes 792 → 1,572 of 1,839, families naming under 0.95 of their own summed
   open price 60 → 6, fabricated prices rendered 59 → 0.
2. ~~**One relation-vs-liquidity precedence sentence in the prompt**~~ — **SHIPPED** in `4e342da`,
   and now rule 2 of `_MARKET_READING_RULES` (`prompts.py`): when the relation and liquidity labels
   disagree the liquidity warning governs the price, so an other-cut extrapolation from a thin
   strike widens rather than shifts. q45189's anchor strike had $1,377 of volume, all three
   forecasters called it thin, and all three resolved the conflict in favour of relation. The
   constant it shipped as, `_MARKET_RELATION_WEIGHTING_SENTENCE`, was cut on 2026-09-02 to its
   three policy clauses because the other 1,908 chars re-taught the notation the rendered table's
   own legend already defines (audit `scratch/prompt_bloat_audit_2026-09-02.md`, item R1).
3. **Staleness guard on tier grading — SHIPPED NARROW 2026-09-01, one operator decision left.**
   `ranking.cap_stale_top_tier` now demotes a row graded `same_quantity_same_date` exactly one rung
   when its close precedes the question's `open_time` by more than `MARKET_STALENESS_TIER_CAP_DAYS`
   (60), recording the ranker's own grade in `MarketMatch.tier_cap_note`; the render also appends
   `(Nd ago)` to any already-passed close cell, the ranker prompt gained a `closes` recency
   tiebreaker, and the legend now says `close` is the trading close and not the settlement date.
   **The open decision is whether to widen the cap.** The q45163 dossier's own recommendation was
   stronger than what shipped — a stale market "cannot be `same_quantity_same_date` OR
   `same_quantity_other_cut`" — and widening it matters because 45163's offending row was graded
   tier 2, so the shipped pass would not have touched it (the `(Nd ago)` disclosure and the prompt
   bullet are what cover that case today). The reason it was not widened unilaterally: tier 3
   (`driver_or_consequence`) is described to the forecaster as context rather than an anchor, so the
   demotion would change how the prompt weights the row, and a resolved market on an adjacent cut is
   legitimately informative (it says what actually happened). It is a two-line change in
   `_tier_cap_note` if wanted. **Watch item either way:** the cap fires on nothing in the 102
   archived snapshots (only 9 rows are graded tier 1 at all, none stale), so the first
   `MARKET_TIER_CAPPED` line in a prod run log — harvested as `market_tier_capped` — IS the finding:
   it means the ranker called a long-closed market same-quantity-same-date. Read the archive for it
   after a few weeks of fall-cup runs; a continued zero is also an answer (the ranker does not make
   that mistake at tier 1) and would argue for the wider rule rather than against it.
4. ~~**Render an explicit zero-rows line**~~ — **SHIPPED** in `4e342da`: a deliberate zero-row
   ranking over a non-empty candidate pool now renders one sentence naming it as a considered empty
   result. Before that, q45200's ranker validly returned zero rows over a healthy 381-candidate pool
   and the section silently vanished, indistinguishable from a provider outage, while the forecaster
   prompt still shipped the weighting clauses for a table that wasn't there. Every *failure* path
   still renders nothing, which is what keeps the two readings distinguishable.

Also still worth one upstream check: the Kalshi `close` column is not a settle date (median +114
days vs the question's own resolve time; 14/78 rows at +300d or more). The forecaster-facing half of
that is now disclosed — the legend sentence added 2026-09-01 says outright that `close` is the
venue's trading close and not its settlement date — so what is left here is the upstream question of
whether to render a settle date at all. Re-read the informativeness question at ~41 ranked-era
resolutions (late September / early October), not ~09-01 (which buys only ~7).

5. **Retune `KALSHI_NO_PRICE_SPREAD` from prod telemetry (added 2026-08-25, owner: the next
   residual round).** The 0.40 book-width threshold that blanks a Kalshi midpoint as
   manufactured (58175a7) is a reasoned choice, not a measured one: the committed captures hold
   11 live two-sided books (real spreads 0.01–0.10, so 0.40 sits 4x above the widest observed),
   but the raw archive stores post-parse children with NO bid/ask, so prod incidence is
   unmeasurable offline. The `withheld=` field on the `MARKET_CHILD_RENDER` marker exists to
   turn it into a query — after ~2 weeks of ranked-era runs, read the withheld distribution:
   near-zero on liquid families and nonzero exactly on empty books means the threshold is
   right; a material rate on books that later traded near their midpoint means it is blanking
   real prices and should come down.
6. **LOW — the shared Manifold test payload is unfaithful (added 2026-09-01).** It carries no
   `outcomeType` key, so every pipeline test using it logs a venue-parser warning ("Manifold None
   market carries a market-level probability but no readable value"); live Manifold always ships that
   field. Adding `outcomeType: BINARY` would give that row a price where several tests currently
   expect none, so the fixture and those expectations have to move together — which is why it was
   left alone during a parallel fan-out that had other agents in the same file.
7. **LOW — the narrowed market-odds research bullet claims a snapshot that can be missing (added
   2026-09-01, forge R24).** `_OUTSIDE_VENUE_MARKET_ODDS_BULLET` (`prompts.py`, shipped by the
   bundle's item 5) tells the search providers not to report Polymarket/Kalshi/Manifold/PredictIt
   prices because "a dedicated live snapshot of those venues is provided separately", which is true
   whenever the prediction-market provider works and false on any question where it fails — now the
   only way that snapshot goes missing, since the flag is on in every prod workflow. Left alone
   because gating the sentence on the flag cures only the never-happens disabled case, and the
   item's own receipt (`scratch/residual_2026-08-31/market_odds_coverage.md`) measured
   search-indexed covered-venue prices as net harmful in the ranked era (one content-redundant
   retrieval against three stale prices that contradicted correct live snapshot rows), so
   re-inviting them on a provider failure is not obviously an improvement over saying nothing.

### Sentinel-value sweep leftovers: deliberate deferrals (added 2026-08-26)

The 2026-08-25/26 sentinel-value work (`scratch/residual_2026-08-24/sentinel_value_audit.md`)
closed all 7 HIGH findings, the MEDs and every LOW. Three items were deferred ON PURPOSE, each
because the honest fix is a new decision rather than a correction. Later rounds appended their own
deferrals below the numbered three, so the heading carries no count: the section holds six items and
the prose below names each one's position.

1. **A confident DISCRETE point mass is now WITHHELD, and building the spike is the faithful
   third option.** `sanitize_percentiles` no longer cluster-spreads a whole-set epsilon collapse,
   so a model declaring "the count will be exactly 3" at all 13 percentiles reaches the
   unit-mismatch guard with a zero span and is dropped. That is right relative to what it
   replaced (a fabricated ±6-unit hedge nobody stated, whose invented width was exactly what let
   it pass the guard), but a spike CDF IS expressible: `grid_step_constraints` relaxes the max
   step to 1.0 on a coarse discrete grid. Prod incidence was unmeasurable before — the old
   `Cluster spread applied` WARN was never harvested — and now is: watch
   `numeric_degenerate_declaration` in the telemetry archive, and revisit if it fires on real
   forecasts. Building the spike means deciding what mass a "point mass" declaration deserves
   at the neighbouring integers, which is a modelling choice, not a bug fix.
2. **The ft-fallback numeric builder is unavailable on log-scaled questions.** On a `zero_point`
   question, upstream's `get_cdf()` can itself raise on a float-epsilon overshoot of 1.0, so
   `create_fallback_numeric_distribution` cannot rescue exactly the question shape where PCHIP is
   most likely to have failed. It fails FAST (the forecaster drops, attributed), so it is a lost
   forecast rather than a fabricated one — which is why it was left. Fixing it means reaching
   into forecasting-tools' builder; note is at the function's docstring.
3. **Tier-1 resolution-source content quality still has only a length floor.** `success` now
   requires non-vacuous content (empty body, mojibake, and non-row-shaped datasets are refused),
   but a soft-404 template or cookie/paywall boilerplate served as HTTP 200 with >100 chars still
   renders as the resolution source. The audit proposes a negative-phrase check near the start of
   extracted text. Deferred as a new HEURISTIC with real false-positive risk: "no data available
   for this date" is legitimate text on plenty of genuine resolution pages, and a wrong reject
   costs the cited source outright. Forecasters can usually tell from the rendered text; the
   diagnostics side cannot, which is the actual gap to close.

A fourth, smaller observation from the 2026-08-26 live-QA rerun (pre-existing, not from this
wave): the agentic `_fetch_plain` textual allowlist (`text/plain`, `text/csv`,
`application/json` — unchanged since 56c0d2f) refuses FRED's fredgraph CSVs, which are served
as `Content-Type: application/csv`, with a clean "Unsupported content type" error, so such URLs
ride the fetch ladder's later rungs. One allowlist entry if agentic FRED reads ever matter.

A fifth, from the 2026-09 bundle's forge pass (added 2026-09-01, forge R8): `getattr(question,
"<attr>", None)` on attributes that always exist on `MetaculusQuestion` is licensed by the
`question: Any` typing rather than by any field that can actually be absent, and the forge pass
counted about 30 such sites across `metaculus_bot/research/` (a raw grep for `getattr(question, `
there returns 51, the difference being genuinely type-specific attributes like `open_upper_bound`
and `options`). The two guarding the new staleness tier cap were fixed inside the bundle; the rest
is one mechanical sweep and deserves its own PR, because the MagicMock question fixtures several
research tests build will need explicit attributes once those defaults stop covering for them.

A sixth, accepted deliberately rather than fixed (added 2026-09-02, forge R1 from the 2026-08-31
round; operator decision recorded in that round's SYNTHESIS decision table): **the ft-fallback
numeric builder's CLOSED-bound path skips `safe_cdf_bounds` entirely.** In
`create_fallback_numeric_distribution` (`metaculus_bot/numeric/pchip_processing.py`) the
`BoundSafeNumericDistribution.get_cdf` override returns upstream's CDF unchanged when neither bound
is open, so no min-step, max-step or endpoint enforcement runs on that output and no
`CDF_MAXSTEP_CLIP` marker can fire; the forge reviewer measured a 0.2034 bin against the platform's
0.2 cap on that path, and the stacker path publishes such output directly, so in principle an
over-cap CDF is POSTed and Metaculus rejects the whole submission. It also falsifies the "single
choke point" claim in `docs/numeric_pipeline.md`. Accepted because the failure is compound-rare: it
needs closed bounds AND a PCHIP failure AND stacking enabled, and stacking is prod-disabled today,
with zero measured fires in the archive. Restructuring the fallback path speculatively risks more
than it buys, and timing/fallback paths in this pipeline have cost real questions before. If
stacking is ever re-enabled, this becomes a real pre-flight item: route the closed-bound branch
through `safe_cdf_bounds` too, with the grid-scaled step constraints the open-bound branch already
computes.
(`scratch/residual_2026-08-31/forge_report_resids-sept1.md` R1;
`scratch/residual_2026-08-31/SYNTHESIS.md` decision table, decision (b).)

### Deterministic tail-consistency check on the numeric structured block (added 2026-08-24)

From the q44453 dossier (July payrolls, peer −11.24; the whole field missed the −23k print, so the
recoverable loss is width/skew, not the center): when a rationale derives a σ from market
thresholds or states a distribution family and σ, and then declares a left tail materially tighter
than that same σ, widen or flag it deterministically — gpt-5.4 fit N(84.8k, 96.7k) to two market
prices inside its own rationale and shipped a 61.9k-equivalent left tail; opus-4.8 quoted
"forecast error SD ≈ 50–60k" and shipped 52k with a right skew. The values are already in the
structured block, so "your left tail must be at least as wide as the σ you derived" is one
arithmetic pass with no new elicitation. Measured ceiling, from the adversarial verification
(quote THIS figure, not the dossier's +15.8, which fed gemini a wider spread than it published):
de-skewing every member to a symmetric normal at its own declared p50 and 10–90 width is worth
**+11.93 baseline points, taking spot peer from −12.15 to −0.22** — roughly break-even with the
crowd, from arithmetic the models had already done.
(`scratch/residual_2026-08-24/dossiers/44453_dossier.md` + `44453_verification.md` C1.)

### Anchor-date discipline, prompt-side: make a status-quo anchor state its date (added 2026-09-02, tracked at last; recommended by the 2026-08-24 round)

**What.** One forecaster-prompt rule: when a member derives a status-quo or time-series anchor (a
last print, a current polling level, a latest index value), it must state the DATE of the reading it
anchored on and prefer the newest dated read available in the bundle. Nothing computes off it and it
is not a clamp; it is an elicitation rule aimed at a specific failure.

**Why.** The stale-anchor pattern is the common mechanism behind a whole run of misses. The
2026-08-24 round named 44858, 44841, 44855, 44553 and 45115, with 45114 and 45174 as the saves that
show what correct handling looks like, and the 2026-08-31 round added 44554 (sol published 17
points below its own anchor on a polling-lead question, presumably off stale summer polls; that one
is a plausible instance rather than a traced one). A member that never says when its anchor was
measured cannot notice that the bundle carries a fresher number, and the reader of the rationale
cannot tell a current read from a remembered one.

**Status and the honest discount.** Unshipped, and until 2026-09-02 also untracked, which is why
this entry exists. The RENDER side is now half-covered: `bc9d9ad` dates every rendered latest value
and flags staleness, and the 2026-09 bundle added vintage / as-of bullets to
`web_research_prompt` asking research output to carry the publication date of dated claims. That
weakens the prompt-side case without retiring it, because those changes make the dates AVAILABLE to
a forecaster without making a forecaster state which one it used. Not a fitted layer, so no
out-of-sample era gate is needed to ship it; but do not quote 44553's +58 as an expected value, and
size it against the fact that the bundle already carries several prompt rules whose combined effect
is unmeasured until the fall cup resolves.
(`scratch/residual_2026-08-24/SYNTHESIS.md` §6 item 6;
`scratch/residual_2026-08-31/SYNTHESIS.md` free-recommendation item 10 and
`dim_consensus-dissent.md` on 44554.)

### ~~Anchor-overshoot self-consistency screen (the anchor half of the same check)~~: MEASURED and REJECTED 2026-08-31, priced again 2026-09-02. Do not build it.

This is the sibling half of the tail-consistency check above, and unlike that half it has been
measured and does not survive. The proposal was: flag or correct a binary forecast whose published
probability sits outside the member's own declared `base_rate_anchor` range, on the theory that
leaving your own stated anchor is an internal contradiction. Three findings kill it, and a fourth
fact retires the input: the 2026-09 bundle removes `base_rate_anchor` from the prompt.

**The premise is not a contradiction.** The prompt defines `base_rate_anchor` as the outside-view
range stated in Phase 1 and `posterior_prob` as the post-update posterior, so publishing outside
the range is licensed by construction: it means the inside view moved the model past its own base
rate. The screen measures the size of that update, not an inconsistency.

**The incidence is 3x what the screen was priced against, and the direction is INVERTED at the n
it can be read at.** Over the triple era's clean published blocks the rate is 32 of 119 = **26.9%**
(Wilson 95% 19.7 to 35.5), against the 8% a single earlier round had carried, and it is a stable
property rather than an era effect (post_flip 21 of 87 = 24.1%, Fisher p=0.75 between eras). On the
21 blocks that had resolved, blocks that LEFT their own anchor scored **+72.11 (n=3, none
negative)** while blocks that stayed inside scored **+25.60 (n=18, 4 negative)**, and all three of
the worst blocks (worse than −100) sat inside their anchors. All 3 resolved fires were hit-side, 0
miss-side. The per-slot rates are a style difference, not an error signature: gpt-5.6-sol leaves
its own anchor on 46.2% of blocks against opus-4.8's 7.5% (Fisher p=0.0001), and it is not a width
artifact, since sol states the WIDEST intervals (mean 0.174 against opus's 0.139) and still leaves
them most often, with the width-free distance-from-own-anchor-centre measure putting sol at about
3x opus.

**Priced field-free on the whole archive, "publish your anchor instead" is worth about zero with a
parse-dependent sign.** The 2026-09-02 failure-mode audit re-medianed every binary ensemble with
each member's stated anchor substituted for its published value: **minus 221 spot peer over 143
records** under the strict parse (anchors written as a bare number) and **plus 789 over 376
records** under the extended parse (leading number or range midpoint), with the **median record
delta 0.00 under both**. Most overrides do move away from the resolution, but the moves toward it
are individually larger (30 to 78 points each above the 15-point gap cut) and roughly cancel, so
the net is a variance trade with no expected gain. The largest single contributors point opposite
ways: q42304 would have gained +144 from keeping its anchor, q39747 (the US bailout of Argentina)
would have lost 170, because there the members' 0.10 anchor was wrong and their 0.55 publish was
right.

**What to do instead.** Nothing automated, and the elicitation this check reads is itself going
away. The 2026-09 bundle asks a member to state its outside-view number and to name the specific
evidence when the final probability lands more than about 15 points off it, as one clause on the
existing "anchor on your math" bullet rather than as a standalone rule; the audit endorses exactly
that and warns against strengthening it into a clamp or a shrink. The same bundle removes
`base_rate_anchor` from the block (Item D of
`scratch_docs_and_planning/announced_unscheduled_fix_plan_2026-09-02.md`), so from that merge on
there is no machine-readable anchor to compare a publish against at all, and reviving the screen
would mean re-adding the field first. The one cheap thing worth continuing is the control:
hand-run the single anchor-versus-published comparison on any new binary MISS dossier and record
which side it fires on, because the resolved-fire cell is n=3 and every one of them is hit-side.
Never propose it as a screen or a guard. Note also that any June-era hand-run of this check is
unmeasurable: `base_rate_anchor` did not exist in the archive before `30bca2f` landed on main
2026-07-11.
(`scratch/residual_2026-08-31/dim_ghosts-guards.md` §3c and
`scratch/residual_2026-08-31/SYNTHESIS.md` decision table;
`scratch/failure_mode_audit_2026-09-02/AUDIT_SYNTHESIS.md` §4.)

### Gap-fill v2 throttle handling: the request-spacing half the q45191 fix left (added 2026-09-02)

The fix that shipped makes a host's rate-limit interstitial a `status=throttled` failure that is
never cached, so the driver's retry of a throttled URL is a real request
(`research/agentic/fetch_outcomes.py` + `tools.py`; receipt:
`scratch/residual_2026-09-01/dossiers/45191_verification.md`). The duplicate-call note that
contradicted that retry has since been fixed as well, and needed no per-URL outcome memory:
`dispatch._append_tool_messages` discards a throttled call's `(tool, normalized-args)` key from
`state.seen_tool_calls` at emission, so the identical retry is neither counted in `dup_tool_calls`
nor told "its result will not have changed", while `max_tool_calls` still caps a throttle spin.
One adjacent thing is still left alone on purpose, because it touches timing rather than detection:

**No request SPACING.** Same-host fetches already serialize on the per-host `Semaphore(1)` v2
shares with Tier-1 (`tools._host_gate`), but Ogimet asked for 20 s BETWEEN queries and nothing
waits. So a parallel same-host batch still trips a spacing rule; it is now disclosed and retryable
instead of silently wrong. Adding a per-host minimum interval means new sleep/deadline logic on the
path the wall deadline already governs, which is the highest-risk surface in this package. If it is
ever wanted, the cheap version is a per-host "next allowed at" timestamp consulted inside the
existing gate, and it needs its own review pass.

Also unhandled, and cheaper to leave: `read_document` (rung 3, Gemini `url_context`) applies no
throttle check, so an interstitial the reader summarises would still come back as a document read.
Nothing in the archive shows that shape; the marker `agentic_fetch_throttled` is what would make a
case for it.

### Gap-fill v2: office-holder precedent verification rule (added 2026-08-24)

From the q44210 dossier (McDonald sworn in as SDNY US Attorney, peer −24.7 — the worst genuinely
new judgment miss of the 2026-08-24 window): when a "will X assume office / take effect / be
installed" question's fine print enumerates alternative appointment/effectuation mechanisms,
retrieve **how the current holder of that office actually got the seat**. The office's own
15-month-old precedent — Schumer withholds the blue slip, Trump installs the nominee interim the
SAME DAY under §546(a), district judges retain him at day 120, never Senate-confirmed — appeared
nowhere in the 30,670-char bundle, so all six models priced the mechanism that actually delivered
(24 days early) at 0.10–0.25 as a hypothetical "Trump pivot". The adversarial verification prices
this leg at **~+31 spot-peer points, ~80% of the recoverable loss** and ~4:1 over the
market-retrieval leg (which ranked retrieval has since fixed, making that half a retrodictive
validation target rather than an action item). A one-query verification target, exactly what v2's
dry-run-then-verify brief exists for; v1 gap-fill contributed zero text on that run.
(`scratch/residual_2026-08-24/dossiers/44210_dossier.md` + `44210_verification.md`.)

### Obscure-counter fetch re-bench (paid) — the open half of the Gemini `url_context` question (added 2026-06-28; verdict answered 2026-08)

**Verdict half — CLOSED as an executed negative.** Gemini's `url_context` tool (built to read
criteria-named resolution URLs directly) contributes nothing in prod on the grounded-search path:
the 2026-06-28 audit found 0/17 Period-B Gemini sections citing a direct `.gov`/`fred`/`cboe` URL
(all `grounding-api-redirect/` links), and the telemetry added the same day
(`gemini_search.py` `_extract_url_context_telemetry`, marker `### URL Context Fetches` or
`_url_context: none_`) reads **0 of 271 archived Gemini sections** carrying either marker (2026-08-02,
standing as of 2026-08-24). Live resolving values came from the gap-fill native-search pass or the
financial-data API instead — damning case q43650, where Gemini's snippet said 4.44–4.46% and gap-fill
returned the exact 4.48% that resolved. So the fetch gap is *masked by gap-fill*, not *closed by
url_context*: the resolution-source **Tier-2 precondition is satisfied** and the narrow named-URL
fetcher keeps its justification. Two residual notes: a paid positive control (does the tool fire at
all when forced?) would be needed before blaming the wiring rather than the model, and v2's
`read_document` uses url_context on a separate path where it demonstrably does fire
(`agentic_document_ungrounded_suppressed` telemetry).

**The open half (deferred, needs a small paid re-bench — clear cost):** the audit could NOT test the gap's
worst case — obscure non-API official counters/registries/dashboards (state policy trackers, CBP
tables, WHO-style dashboards, mesonet tables). Period B had zero; the only two clean research-side
fetch failures in the 40-tracker corpus (q43046 WHO extranet, q43139 IEM mesonet) were both this type,
both pre-current-stack. A ~10–15-question adversarial re-bench enriched for that archetype is the
highest-leverage missing evidence. Narrow design (parse criteria for a resolver URL → force-fetch, or
make the gap-fill analyzer treat a criteria-named URL as mandatory) sketched in
`scratch/research_audit_2026-06-27/SYNTHESIS_62.md` §4. FRED/Yahoo URL-extraction (shipped 2026-06-28)
already covers the API-backed financial subset.

### Resolution-source fetcher: Tier-2 LLM fetch + oversized-source summarization (added 2026-07-09)

Tier-1 deterministic fetcher shipped `66e31c0` (`research/resolution_source.py` +
`research/http_fetch.py`, `RESOLUTION_SOURCE_ENABLED`). Smoke-validated on 40 cached questions
(`scratch/resolution_source_smoke_2026-07-09/REPORT.md`): 24/40 (60%) get a non-empty snapshot vs
the 62.5% Tier-1 target, 30/45 URL success, 0 SSRF false positives, first-cited URL was the primary
grading source in all 12 multi-URL questions; remaining misses are known Tier-2 hosts (JS walls /
fingerprinting). **Flipped ON in all three prod yamls 2026-07-10** after a `test_bot.yaml` eyeball
(same commit added the per-URL truncation marker + dropped-section note).

**Truncation-cap study (2026-07-09, don't re-derive):** uncapped re-fetch of the 29 smoke URLs:
p25=697 / p50=2,201 / p75=5,206 / p90=67,041 / max=438,049 chars. Elbow at 6,000/URL (3,000 cap
truncates 48%, 6,000 truncates 21%, past 6,000 only whales that need summarization). **Shipped:**
`RESOLUTION_SOURCE_PER_URL_MAX_CHARS` 3,000→6,000, `RESOLUTION_SOURCE_TOTAL_MAX_CHARS` 12,000→18,000.

Follow-ups:

1. **MEDIUM — conditional summarization for oversized sources.** First-cited URL stays verbatim
   (provenance); URLs 2+ / whales (≥~10k chars) go through the cheap summarizer (`gpt-5.6-luna`, temp
   0, a rounding error at $0.10/$0.60 per 1M). ~5 whale sources per 40 questions no cap captures.
2. **MEDIUM — Tier-2 LLM fetch** for the js_wall/blocked slice (~15%; Masters.com, childmortality.org,
   UNICEF, Tesla IR, sagaftra.org). The per-URL `FetchStatus` (blocked / js_wall / and since
   2026-09-01 `no_resolving_content`, the embed-shaped 200 whose numbers sit inside an Infogram,
   Flourish or Tableau widget) is the seam. Those three are pages we could not READ; `empty_body`
   and `unsupported_type` are bodies that carried nothing and are NOT escalation seams.
   **Precondition:** ~~the Gemini `url_context` probe above~~ — SATISFIED 2026-08 (probe negative,
   0/271 sections carry the marker). *Note 2026-07-16:* the gap-fill v2 fetch
   ladder gives the driver this capability inside the loop, so the js_wall slice may get covered
   agentically first — re-assess after the v2 overlap window.
   **The first Tier-2 hop shipped 2026-08-25 (`5f27c46`) and is deliberately narrow, not this
   item:** a fetched page's embedded Datawrapper charts now also serve each chart's live "Get the
   data" CSV, which is what unlocked the poll-tracker family (qids 44858 / 44841, both misses that
   trace to a stale prose anchor while the live series sat one fetch away). Two properties worth
   carrying into any wider Tier-2 work: only the version-free `static.dwcdn.net/data/<id>.csv` route
   is fetched, because the page HTML pins a stale chart version whose `dataset.csv` keeps serving
   5-14-month-old snapshots as HTTP 200; and anything older than
   `RESOLUTION_SOURCE_DATAWRAPPER_MAX_AGE_DAYS`, or undatable, is withheld as a new `stale_data`
   status rather than served as live. The generic js_wall/blocked slice is still open.
3. **LOW (deferred) — now over the ceiling:** module split of `resolution_source.py` (extract
   `ssrf_guard.py`). The "~670 LoC" in the original note is stale — it is **1,175** as of 2026-08-25
   after the Datawrapper hop, and the shared embed-detection primitives already moved to
   `http_fetch.py`, so the seam is visible.
4. **LOW — the all-failed section notice now overclaims for one status.** When every URL fails, the
   rendered notice still says the resolving page "was unreachable", which is imprecise for a
   `no_resolving_content`: we reached it, got a 200, and could not read the embed. The same notice
   names the real status token beside it and the instruction to the forecaster is unchanged, so this
   is wording, not a defect. The fix is per-status phrasing in that one render string; it was left
   alone in the 2026-09-01 bundle rather than reworking shared render wording under an unrelated item.
5. **The 45 s wall-clock timeout throws away work that already succeeded (added 2026-09-01, forge
   R3; SKIPPED by operator decision — do not implement without them).** The provider's `_fetch`
   wraps `fetch_resolution_sources` in a single
   `asyncio.wait_for(..., RESOLUTION_SOURCE_WALL_TIMEOUT)` and returns `""` on timeout, and that
   return happens BEFORE `_log_fetch_outcome_markers`, `record_raw_research`,
   `record_provider_detail` and `format_resolution_sections`. So an overrun loses every per-URL
   `RESOLUTION_SOURCE_FETCH` marker, the diagnostics source map, the raw-archive payload AND the
   page text the URLs that did finish had already produced — the run cannot tell afterwards which
   sources it nearly had. The minimal fix bounds the page gather internally the way the Datawrapper
   hop already self-bounds: cancel the stragglers, materialise the unfinished URLs as `error` or a
   new status token, return partial results. Note that this also starts rendering partial snapshots
   where today nothing renders, i.e. a forecaster-visible change on a timing path, and timing and
   fallback paths in this pipeline have cost real questions before (q45085). Blast radius is
   unmeasured, so size it first:
   `rg --no-ignore 'wall-clock timeout' backtests/telemetry_archive/`.
   **The 2026-09-03 fetch-ladder bundle added an amplifier this item did not have when it was
   written (forge R4, F61/F39/F35; still SKIPPED, still the operator's call).** The per-host
   politeness map moved from a fresh dict per provider call to the loop-wide
   `http_fetch.host_semaphores()`, so same-host requests now serialize ACROSS the
   `DEFAULT_MAX_CONCURRENT_RESEARCH = 6` questions researching at once rather than only within one
   question. Each hold covers the GET, the body read and the trafilatura extraction (and, until the
   F45 PDF-parse split lands, the PDF parse), the acquire wait is unbounded, and each request may
   take `RESOLUTION_SOURCE_HTTP_TIMEOUT = 20.0` — so six questions citing one slow host (the
   worked cases q44873/q44874 both cite cdc.gov) can queue past the 45 s wall, at which point the
   question that lost the queue discards every page it had ALREADY fetched, not just the contended
   one. The archived timings say the tail is real rather than hypothetical: all-fail p50 0.3 s, but
   3 of 23 ran the full 20 s. Nothing measured this interaction when the map was widened; the
   politeness argument for widening it is unaffected. Three remedies, all the operator's to pick:
   the partial harvest this item already describes; or a budget-bounded acquire — wrap the whole
   hop in `asyncio.timeout(ctx.rung_budget_s())` while KEEPING `async with sem`, so a queued
   question gives up its turn instead of the wall taking the section (the reviewers flagged this as
   new timeout policy on a surface reserved on 2026-09-01, and a hand-rolled acquire/release on a
   process-wide permit risks a leak or a double release, which is why it is not "strictly safer");
   or pure instrumentation, a WARN when the acquire wait exceeds a few seconds, which measures the
   contention without changing any timing. A `Semaphore(2)` per host is NOT an option — it trades
   the politeness the widening bought for a bound it does not actually give.
   **Merging Tier-1's host map with gap-fill v2's own map waits on this.** v2 keeps its own
   module-global map (`research/agentic/tools.py`) and shares only the helper, so today a Tier-1
   fetch and a v2 fetch can hit one host at once — but v2's rendered rung holds its host gate
   across a Chromium launch of up to 35 s, and merging the maps would put that hold in front of a
   Tier-1 fetch whose 45 s wall discards finished pages. So the merge needs either the bounded
   acquire or the wall-degradation fix above landed first, and it needs the resulting queueing
   measured rather than assumed.
6. **MEDIUM: trafilatura silently drops MediaWiki collapsible boxes, and the surviving text can
   read as the inverse of the truth (verified 2026-08-24 on q44870; tracked here 2026-09-02).** On
   an English Wikipedia endorsements page the box renders as
   `<div class="endorsements-box mw-collapsible"><div class="endorsements-box-title">Declined to
   endorse</div>...`. Under our production `favor_precision=True` settings trafilatura drops the
   whole thing; under default/recall settings the list body survives but the box TITLE never does
   (tested across txt, txt+formatting, markdown and xml output, and reproduced on the live page, so
   it is systematic rather than a revision artifact). The consequences are worse than truncation and
   worse than a plain miss. First, a cited grading page reports `status: ok` while the section the
   resolution criteria point at is absent, so nothing downstream knows the evidence was lost.
   Second, what the extraction DOES deliver from that section is unattributed: endorser bullets
   arrive with the per-candidate box titles stripped, so a "Declined to endorse" entry can appear
   immediately after another candidate's endorser list and read as an endorsement, which is the
   exact inverse of the truth. Third, the obvious budget-side fixes do not work: a
   subject-anchored truncation window and honouring the URL fragment both operate on the extracted
   text, where the box's content does not exist, and no per-URL cap delivers it. The fix has to sit
   at the extraction layer: for `en.wikipedia.org` specifically, pull section wikitext or
   `action=parse` HTML through a purpose-built reader instead of generic main-content extraction;
   or, generically, keep a structured pass over `<div>`-titled list blocks before handing off to
   trafilatura. Same extraction-fidelity family as the Datawrapper hop in item 2, and NOT covered by
   items 1 to 3 above (conditional summarization would summarize the same defective extraction).
   (`scratch/residual_2026-08-24/dossiers/44870_verification.md` §2.1.)

### Percent-form block labels vanish silently in comment recovery (added 2026-07-15)

A numeric STRUCTURED FORECAST block whose `declared_percentiles` keys are percent-form ("2.5" …
"97.5") not fraction-form (0.025 …) is dropped by BOTH recovery rungs in
`performance_analysis/parsing.py` (strict `parse_structured_block` rejects the schema; the tolerant
salvage rung drops the keys on its `0 < pct < 1` guard). Historically harmless (prose "Percentile
2.5: X" lines rescued these, e.g. qid 43684 / grok-4.3), but post-2026-07 block-last-no-prose prompts
leave no fallback — that model's percentiles vanish silently from residual analysis. Fix: teach the
tolerant rung to detect a canonical-set×100 key match and rescale (validator + canonical sets already
exist: `_validate_percentile_labels`, `_CANONICAL_PERCENT_LABEL_SETS`). Watch
signal: a model whose per-question percentile coverage drops to zero in a post-flip pull while its
`EXTRACTION_RUNG` prod telemetry stays healthy.

**Half-closed — verified in code 2026-08-25.** `_validate_percentile_labels` now deterministically
divides an exact canonical-set×100 label set back down (with a WARNING) instead of rejecting it
(`f530968`), so a percent-form block that survives extraction IS recovered. The remaining hole is
one line upstream of that validator: `_numeric_percentiles_from_block_tolerant` still `continue`s
past any key outside `0 < pct < 1` (`performance_analysis/parsing.py:605`), so on the tolerant path
percent-form keys are dropped before the rescale can see them. Widening that filter and letting the
validator adjudicate is the whole remaining fix.

### Dependency CVEs after the forecasting-tools unfreeze (updated 2026-08-17)

`make audit` (osv-scanner over `uv.lock`) passes — clean apart from the five enumerated
cryptography waivers below, every one of them blocked by a single upstream cap. The old gated set
collapsed at the 0.2.54 → 0.2.92 unfreeze:

- **litellm — CLEARED.** 0.2.92 moved litellm 1.80.0 → 1.92.0, past every previously-gated litellm
  CVE (the 9.x severities plus the proxy-server RBAC / auth-bypass set). All litellm entries were
  removed from `osv-scanner.toml`.
- **cryptography — five accepted advisories, all blocked by one asknews cap (updated 2026-08-17).**
  The asknews 0.11.32 → 0.13.54 bump lifted its `cryptography<45.0.5` cap to `<46.0.7`, so
  cryptography moved 45.0.4 → 46.0.6, fixing two of the four then-open advisories. Two survived
  that bump — **GHSA-p423-j2cm-9vmq** (CVE-2026-39892, CVSS 9.8 buffer overflow on non-contiguous
  buffers, fixed 46.0.7) and **GHSA-537c-gmf6-5ccf** (CVSS 7.5 vulnerable OpenSSL bundled in the
  wheels, fixed 48.0.1) — and three more were disclosed against 46.0.6 on 2026-08-17:
  **GHSA-jwv3-5hgf-82ww** (X.509 path-building DoS, 8.7), **GHSA-g6cj-pr64-35w5** (PKCS#7
  decryption oracle, 8.2) and **GHSA-m2h6-j472-rp4c** (wildcard SAN name-constraint bypass, 6.9).
  All five are waived in `osv-scanner.toml` with per-advisory reachability reasons: cryptography is
  not a direct dependency and this repo imports it nowhere (it arrives via asknews, google-auth and
  pyjwt[crypto]), and we build no X.509 chains and decrypt no attacker-supplied PKCS#7. The cap is
  the only blocker — 49.0.0 and 50.0.0 both predate the lock-time `exclude-newer` window, and
  `uv lock --upgrade-package cryptography` leaves 46.0.6 in place. **Re-check asknews's METADATA,
  not just its version**: 0.13.56 still declares `cryptography<46.0.7`. The waiver list carries a
  `review-by: 2026-10-23`.
- **aiohttp — PATCHED, not waived (2026-08-17).** Three advisories disclosed concurrently
  (GHSA-cq5v-8q36-5273 at 7.1, GHSA-mfx4-hv73-q22v at 6.3, GHSA-mq44-7p77-q5h7 at 6.9) were all
  fixed outright by 3.14.1 → 3.14.3, which shipped well outside the `exclude-newer` window so no
  freshness override was needed. Side effect of re-locking: `exclude-newer` re-anchored to
  2026-08-10; no other package version moved.
- **pillow / transformers / pydantic-settings — CLEARED.** The `[tool.uv]` override/exclude-
  dependencies workarounds were deleted from pyproject.toml at the bump; none resolve to a
  vulnerable version anymore.
- **gitpython — EXCLUDED 2026-07-25.** It shipped five High argument-injection / unsafe-option /
  env-expansion advisories at 3.1.52 across four releases in six days, arriving transitively as
  streamlit ← forecasting-tools. Rather than track that treadmill, `[tool.uv]
  exclude-dependencies = ["gitpython"]` drops it (with gitdb + smmap, 217 → 214 packages) — the same
  mechanism and reasoning as the transformers exclusion in commit 836a5d0. It is unreachable for us:
  an import tripwire over `import forecasting_tools` + `import metaculus_bot.forecaster` never loads
  it, and streamlit's only use is a lazy `import git` inside
  `app_session._handle_git_information_request`, served to a running streamlit server we never start
  and swallowed into a debug log. **streamlit itself cannot be excluded** —
  `forecasting_tools/__init__.py` imports `benchmark_displayer`, which imports streamlit at module
  scope, so excluding it breaks every `import forecasting_tools` including `tests/conftest.py`.
  Note a trap found while fixing this: an `exclude-newer-package` pin written only into `uv.lock`
  does NOT survive re-resolution (any `uv lock --upgrade-package <anything>` silently reverted
  gitpython to the vulnerable 3.1.52), so lock-only pins are not a durable CVE control. Re-evaluate
  the exclusion at the next forecasting-tools bump.

Re-audit at the next forecasting-tools or asknews bump (re-run `make audit`, prune resolved IDs). If
one of the five accepted cryptography CVEs becomes actively exploited before asknews lifts its cap,
evaluate a `[tool.uv] override-dependencies` bump + re-validate the research pipeline. The scan
stays LIVE for anything not enumerated in `osv-scanner.toml`, and CI runs it on every PR.

### Promote the core pipeline to `basedpyright` strict

The 2026-06 uv migration wired `basedpyright` at **standard** mode repo-wide, zero errors. The
intent was **strict on the core pipeline**; deferred because against an unclean base strict surfaced
~900 findings (~450 low-value "partially unknown" noise at the untyped `forecasting-tools` boundary).
Now the base is clean the promotion is much smaller. Spec:

1. Add the `strict` path list to `[tool.basedpyright]`: `metaculus_bot/{numeric, research,
   probabilistic_tools, forecaster.py, aggregation_pipeline.py, stacking.py}`.
2. Resolve the resulting `reportUnknown*` findings (~447 last measured) by **annotating our own
   functions** — ~95% of the unknowns are our own under-typed signatures/locals, not the library.
3. `forecasting-tools` ships **no `py.typed`** despite inline annotations, so strict re-raises
   `reportMissingTypeStubs`. `--createstub` is NOT a clean fix (drops Pydantic attrs, 892→1011).
   Options: (a) thin hand-maintained `typings/forecasting_tools/` stub for imported symbols only;
   (b) a `reportMissingTypeStubs` override scoped to the strict execution environments (the global
   `= false` does not survive the strict promotion); (c) request `py.typed` upstream.
4. Re-add the local `basedpyright` pre-commit hook (removed during migration) + gate CI.
5. Do as a heavily-parallel workflow (one agent per core module).

### Re-run native-search model evaluation each quarter (added 2026-05-17)

**Current model (2026-07-17): `openai/gpt-5.6-terra` at `effort=low` + `verbosity=low`, 360s**
(`NATIVE_SEARCH_DEFAULT_MODEL` / `NATIVE_SEARCH_REASONING_EFFORT_DEFAULT` in `constants.py`). Slot
history: grok→gpt-5.5 (2026-05-17) → gpt-5.6-sol (2026-07-09) → gpt-5.6-terra (2026-07-17, blind
research-role audit `scratch/research_role_audit_2026-07-17/` — terra 1st, "MARGINAL EDGE", −42%
cost); effort low since 2026-05-20 for latency. Baseline
`scratch/native_search_bench_2026-05-17/comparison_v3.md`. As cheaper/better OpenAI search models
ship (or Anthropic/Google add native search to OpenRouter), re-run `python
scratch/native_search_bench_2026-05-17/run.py --question-id <new open Q>` and update
`NATIVE_SEARCH_DEFAULT_MODEL`. Bake into the quarterly review cadence.

### Gemini on the donated OpenRouter key: pro-preview still blocked by free-tier BYOK (updated 2026-06-16)

The donated key (`OAI_ANTH_OPENROUTER_KEY`) now serves most Gemini (verified live:
`gemini-3.5-flash`, `gemini-3.1-flash-lite` both SUCCEED). **Remaining blocker —
`gemini-3.1-pro-preview` only (our forecaster slot):** routed through a free-tier Google AI Studio
**BYOK** key on the donated account (`is_byok:true`), and Pro-preview has no Google free tier, so
quota is structurally 0 — every donated call 429s `RESOURCE_EXHAUSTED`.

**Resolution — SURGICAL pin:** `GEMINI_USE_DONATED_OPENROUTER_KEY=true` in all four yamls (flash uses
donated), but `gemini-3.1-pro-preview` is **pinned to the personal key** via
`DONATED_KEY_BLOCKED_GOOGLE_MODELS` in `fallback_openrouter.py` (`should_route_via_donated_key` →
`False`; no donated attempt, no 429, no fallback-counter bump). Without the pin, each donated→429→personal
fallback bumps the counter → `cli` `sys.exit(1)` → **red CI every run**. OpenAI/Anthropic on the donated
key are unaffected.

**⚠️ TEMPORARY WORKAROUND, tagged `TODO(gemini-3.1-pro-donated)`.** Remove the blocklist entry once
Metaculus fixes the BYOK routing (pick one, account-side): (1) enable Cloud billing on the BYOK GCP
project → Tier 1 quota; (2) remove the Google AI Studio BYOK integration so `google/*` uses native
OpenRouter Google credits; (3) disable "Always use for this provider" on that BYOK key. Does NOT help:
raising OpenRouter-side native limits (the 429 is Google-side). Raised with Metaculus support; after a
fix, re-verify with one live call and delete the entry.

### Separate outside/inside view stages

Split the single-prompt outside+inside view into separate LLM calls so the inside view genuinely
adjusts FROM an explicit base rate — could help the arithmetic-override problem (models computing
correct probabilities then ignoring them). Smingers does this with cross-pollination (outside view
from Model A → inside view for Model B, adding diversity). Prototype: first pass produces base rate +
reference class, fed explicitly to the second. Moderate-to-high effort.

### ~~Post-hoc isotonic calibration on binary predictions~~ — DROPPED 2026-05-10

The May closing analysis (n=109 binary) did NOT replicate the [0.20,0.30]-band NO-bias concentration
that the April-new n=27 cohort showed — the April finding was a small-N artifact, and the 20 worst May
misses span many failure modes. The original proposal: fit `sklearn.isotonic.IsotonicRegression` on
`(our_prob_yes, resolution)` behind a `USE_BINARY_CALIBRATION` flag, refit quarterly, log
raw+calibrated. **Killed**, and independently re-confirmed dead by the 2026-07-16 calibration audit
(don't ship isotonic/Platt/directional-shrink — see the top calibration entry). Defer any global
isotonic until a >50pp residual is observed in a predicted-prob band on N≥50.

### Probabilistic tooling for base forecasters — DORMANT / on the settled dead-paths list

Infra built (`probabilistic_tools/`, `tool_runner.py`, `structured_output_schema.py` — Beta-binomial
updaters, survival/hazard, log-pooling + Satopää, distribution fits with out-of-bounds mass,
Dirichlet-with-Other, NegBinom/Beta-binomial-ceiling for counts, consistency checks), 261 tests green,
gated behind `PROBABILISTIC_TOOLS_ENABLED`. **Currently DISABLED in prod** (all prod yamls `'false'`;
tier-2 scaffold removed from prompts via Workstream C2) and on the settled dead-paths list (benchmarked
+ rejected — don't re-recommend without new evidence). Activation guide (exact edits, parser-ordering
gotcha, A/B sequence): `scratch_docs_and_planning/probabilistic_tools_activation.md`. Original motivation
was two numeric representation failures the percentile elicitation can't express: NM1 (DOJ antitrust —
model wrote "~92%" of 0 in prose but only ~55% mass at-or-below 0; needs Beta-binomial-ceiling/NegBinom)
and NM3 (MSFT EPS — knew the GAAP-vs-adjusted tail risk, couldn't widen enough, resolved past P97.5;
needs out-of-bounds mass reporting).

### Status-quo / last-print anchor for slow-moving numeric trackers (added 2026-06-28, NEEDS BACKTEST) — partly shipped as TS anchor

**Finding (2026-06-28 period-split audit, Period-B n=17, ~5 numeric trackers — directional only):** on
numeric/discrete questions resolving on a slow-moving/mean-reverting tracker, research surfaces the exact
current value and the ensemble then *degrades* it with directional drift or asymmetric widening. The three
worst Period-B trackers all had the resolving value in research (all scored positively — "left points on
the table," not lost):

- **q43647 HY-OAS** (peer +52.4): research handed over 2.71; medians skewed UP to 2.73–2.80; truth below p40.
- **q43611 generic ballot** (peer +31.8): research surfaced Silver +6.8 + a mean-reversion base rate; ensemble
  extrapolated to ~6.7–7.2; reverted to 6.4.
- **q43591 Trump approval** (peer +14.3): research surfaced 38.5; ensemble drifted down to 37.8; ticked up to 38.6.

**Proposed change (prompt-only):** in `numeric_prompt`/discrete handling, when research surfaces a current
authoritative value for a slow-moving/mean-reverting tracker, default p50 + the bulk of mass to that last
print and require an EXPLICIT named justification before applying drift — a *rebuttable* default, not a hard
constraint. **Note:** the shipped TS anchor (see the TS-anchor entry) covers the fetchable-series subset of
this with an empirical band; this prompt lever is the broader, non-fetchable version.

**Why NOT shipped:** n=3; over-correction risk on genuinely-trending / count-toward-deadline numerics (the
model must classify slow/mean-reverting vs trending); prompt-behavior value is unprovable without a paid
backtest. **Backtest gate (clear cost first):** A/B on a numeric/discrete-heavy slice — improvement on
tracker questions AND no regression on trending/event-driven ones. Full evidence
`scratch/research_audit_2026-06-27/SYNTHESIS_62.md` §3.

### LLM-based forecast self-evaluation

After each forecast, run a cheap model to assess: research relevance, factual accuracy,
reasoning soundness, date/chronology correctness, resolution criteria interpretation.
Flag potential issues before submission.

Smingers found this invaluable for catching date confusion, hallucinated sources, and
reasoning failures. Implementation: easy (structured eval prompt + cheap model call).

### Hits-side reasoning prompt-test ideas (added 2026-05-10) — LOW PRIORITY, defer

Three prompt edits from the May analysis (`scratch/analysis_2026-05/analysis_hits_reasoning_patterns.md`),
hypothesis-generating not shipping recs (need N≥30 prompt-vs-prompt backtest):

1. **"State your Poisson lambda explicitly"** — 5/10 top hits used `P(≥1)=1−exp(−λT)` with stated λ, T.
2. **"Required-vs-observed pace section"** — 3/10 top hits used threshold-by-deadline arithmetic. Its
   Wikipedia hit/miss base is N=1 not N=2 (miss 42238 applied the math correctly, landed in a 16% tail).
3. **"Distrust briefing claims that contradict the question's open status"** — inverse of the April
   Klimt-sale hallucination (miss 42243, 4/5 models pulled by a fake datapoint). N=1, weakest.

Deferred: #1/#2 overlap the (now-dormant) probabilistic_tools Poisson/pace math — don't A/B in parallel.
Risk: prompt-length growth degrades simple questions; gate is "mean Brier improves, no per-cell regression
on easy/middle tier."

### Stacker prompt: tell it which models are reliable dissenters (added 2026-05-10)

May C5+C7 analysis: gpt-5.2 is a **contrarian signal source** (8/20 best on worst-misses, 0/20 on hits,
mid-pack Brier 0.150) — the high-disagreement signal the conditional stacker should up-weight;
claude-4.6-opus has the inverse profile. The stacker prompt strips model IDs (self-agreement bias) but
the *historical* "reliable dissenter on high-spread questions" pattern is signal we throw away.
Hypothesis: a small "historical dissenter quality" hint could improve stacked outputs on the high-spread
cohort. Blocked on: STACKER_OUTCOME marker fix + ≥30 stacked records under it (and the stacker is
disabled in prod). Defer.

### Per-forecaster critic/revision pass (added 2026-07-08, medium priority)

An unconditional adversarial critic reviews each forecaster's draft against a resolution-criteria
checklist BEFORE aggregation (window discipline, already-resolved-events-don't-count, listing/instrument
bar, blind-spot pricing); the forecaster answers point-by-point and re-issues, revision capped ±20%.
**Evidence:** Laertes (summer futureeval-2026 #4) runs this on all forecasters — qid 42024: its Forecaster
1 drafted 97% (our published number) and the critic reversed it to 4% after flagging the "resolving" event
fell outside the open window. GreeneiBot2 (spring #1) runs capped critique rounds. Two top bots converging
is a notable signal.

**Caveat:** the demonstrated evidence is on **degenerate** failures (pre-open-window traps, criteria
misreads); generalizing to non-degenerate misses is unproven. Cost ~1 extra LLM call/forecaster/question.
**Distinct from the stacker** (benchmarked + rejected): this is per-member BEFORE aggregation with a bounded
±20% revision, not a post-hoc aggregate rewrite. **Gate:** `make backtest_medium` mixed cohort, primary peer
score — (a) improvement/no-regression on the degenerate subset, (b) no regression on non-degenerate. If (a)
lands but (b) regresses, ship as a **conditional** critic (fires only on pre-open-event/listing-bar/mismatch
flags).

**Update 2026-07-08 (acid test + guard counterfactuals):** down-scoped. On the two non-degenerate consensus
misses (41800 / 42855), advisory critique captured ~0 points even when it diagnosed the exact defect — only
BINDING corrections (bounded numeric adjustment / floor / cap, mechanically applied) moved numbers. The
"sequence after free deterministic guards" premise is falsified: all three guards were buried on offline
replay (era sign-flips / top-5 concentration / fall-hostile fire rates — see the three guard entries in the
"Killed by July 2026-07-08" section). So the critic carries the full burden on its own paid backtest, and
its gate must include a fall-like era-stability check (harvesting the spring miss cluster is exactly what
damages the largest, best-calibrated era). Receipts: `scratch/residual_2026-07-08/ACID_TEST_VERDICT.md` §3,
`scratch/residual_2026-07-08/experiments/GUARDS_SYNTHESIS.md`.

### ~~Telemetry-first guard revival program~~: CLOSED 2026-09-02 (the field it accrues on is being removed) (added 2026-07-08, passive)

**Why it is closed.** This program was a passive bet that the `base_rate_anchor` /
`criteria_clauses` elicitation would keep accruing archive rows until the guard-revival conditions
could be tested exactly rather than by parsing prose. The 2026-09 bundle retires that elicitation:
Item D of `scratch_docs_and_planning/announced_unscheduled_fix_plan_2026-09-02.md` removes both
fields from the binary prompt's schema instruction and example block (the Pydantic fields stay
tolerant so archived comments still strict-parse), on the grounds that they restate prose the
rationale already carries, their only consumer is `tool_runner` behind
`PROBABILISTIC_TOOLS_ENABLED=false`, and their harvested marker holds zero rows. Once the prompt
stops asking, no new question can carry the field, so neither free check below can accrue and the
`base_rate_anchor` leg of the anchor-floor guard's revival condition (in "Killed by July 2026-07-08
residual + competitor analysis") is frozen where it stands: about 40 triple-era binary questions
carry the field against a threshold of 50, and only 21 of their 119 blocks had resolved as of
2026-08-31.
Reviving any of this now means re-adding the elicitation first, which is a prompt change and a
config-era boundary, so it is a decision rather than a wait. The `remaining_window_days` field and
its `WINDOW_DECLARED` marker, which shipped earlier on 2026-09-02, are retired by the same item on
the same day, on the grounds that its own instruction made it telemetry-only; they never appear
in this file, so there is nothing else here to update. The evidence below is kept because it is
what stops the idea being re-proposed from memory.
(`scratch/residual_2026-08-31/dim_ghosts-guards.md` §3b-3c for the archived counts;
`scratch/residual_2026-09-01/FOLLOWUP_LEDGER.md` row 29.)


Shipped `30bca2f` telemetry (`base_rate_anchor {low, high}` + `criteria_clauses` on `BinaryStructured`;
authored 2026-07-08, live on main 2026-07-11T16:37Z in merge `642b027` — that merge date is the era
boundary for any replay that splits on the telemetry's presence)
plus `PREDICTION_MARKETS_ENABLED: 'true'` make future guard replays exact rather than parser-based. No
code on the roadmap; passive. Note, superseded 2026-09-02 by Item D: the computed
`ANCHOR_OVERSHOOT_PP` / `CLAUSE_PRODUCT_DIVERGENCE_PP` markers and the
`_anchor_and_clause_telemetry_lines` helper that emitted them are DELETED (`tool_runner.py`'s module
docstring records which of its dispatch is unreachable as a result), and the binary prompt no longer asks for
`base_rate_anchor` / `criteria_clauses`, so no new raw JSON lands at all. The Pydantic fields stay
optional so archived blocks still strict-parse; the overshoot/divergence math is replayable offline
only over the blocks already in the archive.

Two free checks were the plan while the elicitation was live, and NEITHER can accrue any further, so
they are recorded here as closed rather than as work: (1) the structured-JSON presence rate per
forecaster, grepping the archive for the raw JSON keys to confirm every slot emitted them; (2) whether
the spring confident-overshoot cluster (42024 / 42304 / 41800 analogues) reproduced on the roster after
`30bca2f`, which would have settled all three guard-revival conditions at once. The elicitation ran from
the 2026-07-11 merge (`642b027`) to the 2026-09 bundle's merge, so the archive closes at about 40
triple-era binaries, and both checks are replayable offline over exactly those blocks and no more.
`clause_product_divergence_pp` was the intended first trigger keying on divergence-from-own-math, the
conditionality the three tested guards failed to achieve; that helper and both computed markers were
deleted with the fields, so reviving any of this means re-adding the elicitation first, which is a prompt
change and a config-era boundary.

MC [0-5%) low bucket — **measured NULL 2026-08-24; do not re-open as a gap.** Exact cluster-correct
test over the whole archive: 61 questions supply 117 in-band options carrying **2.83 expected
resolutions, 1 observed, two-sided p=0.367**; 0 of 4 eras exclude the null, and the sign is not
era-stable (fall over-resolves the band). Demonstrating the observed gap needs **~454 MC questions**
against 86 available, and the `ceab2df` before/after version is separately unanswerable (3
post-bullet MC, all also post-clamp). The once-proposed prompt line ("price clearly-dead NAMED
options near the 1% floor") has no evidential support and would now collide with the 0.01 clamp
floor, which binds on 4 of 18 post-clamp ballots. **Do not conflate it with the MC top-band
under-commitment, which is the one MC signal still worth watching**: combined ≥0.60 top bands
over-resolve in all 4 eras (the direction replicates 4 of 4), and as of 2026-09-01 the STRICT
cohort reads **35 questions, expected 26.11, observed 30, exact p = 0.1624**, with the sample size
needed for exact p<0.05 now **57**, i.e. **short by 22 questions, not the 9 the 2026-08-24 reading
implied**. The item moved AWAY from the gate rather than merely failing to move: the three new
top-band ballots hit 2 of 3 against the cohort's own 0.857 realised rate, which lowers the effect
size and raises the required n. **There is no accrual path inside summer-futureeval-2026** — the
latest scored MC submission anywhere is 2026-08-15 and the tournament closes 2026-09-06 — **but
there is one now outside it.** Updated 2026-09-03: the fall Metaculus Cup is configured
(`METACULUS_CUP_ID` = `metaculus-cup-fall-2026`, project 33108, forecasting through 2027-01-01) and
its workflow runs hourly, so cup MC ballots start accruing once the operator enables that workflow
on GitHub and the cup publishes its first questions (it held 0 as of 2026-09-03). Two caveats before
counting on it: cup questions are drawn for humans rather than for a bot benchmark, so the MC mix
may differ from the tournament's, and no successor to `summer-futureeval-2026` existed on 2026-09-03,
so the tournament side stays closed until one opens. Treat 22 more questions as reachable over the
fall rather than as arriving on a schedule. The 1% floor stays (operator 2026-07-09: sub-1% headroom
~+0.01 nats/question vs parser/clamp regression risk — not worth it).
(`scratch/residual_2026-08-24/dim_binary-mc-calibration.md` §3–4;
`scratch/residual_2026-09-01/dim_binary-mc-calibration.md` §5.3 and ledger row 11.)

### Clip-threshold sweep: the binary and MC floors are priced, neither moves (added 2026-09-02, standing residual dim, operator decides)

**The question.** The bot clamps every published probability (binary members into
`[BINARY_PROB_MIN, BINARY_PROB_MAX]` = [0.02, 0.98] before the median, MC option vectors into
`[MC_PROB_MIN, MC_PROB_MAX]` = [0.01, 0.99] then renormalised) and nobody had priced those floors. The
operator asked for a repeatable pass: over every resolved binary and MC question, what floor would have
maximised spot peer, and does the answer hold across lookback windows? It is now a tracked module,
`metaculus_bot/performance_analysis/clip_threshold.py`, and a standing Phase 3 dimension of the residual
playbook; the 2026-09-02 write-up is `scratch/residual_2026-09-01/clip_threshold/dim_clip-threshold.md`
(with `sweep_strict.md`, `sweep_all.md`, `RECONCILIATION.md` and three refutation passes beside it).

**Dated result (2026-09-02, STRICT cohort = 447 binary / 97 MC, excluding `known_bug`,
`degraded_run`, `partial_degraded`).**

- The live clamp has bound NO binary publish since it went live: the 70 strict binaries published after
  the 2026-05-18 widening flip span 0.034 to 0.925, zero at or below 0.02, zero at or above 0.98. The
  MC floor has bound one option (q45088). So for the live config the clip question is moot in both
  directions until the ensemble prices below 0.034 again.
- Raising the binary floor loses in every window and every era: c = 0.05 costs 217.48 spot-peer points
  pooled (81 records moved, 0 resolving on the clipped side), of which 214.76 is the retired pre-flip
  regime, 1.05 the post-flip era and 1.67 the triple era; the live-regime figure is 2.72 over 70
  records (0.039 per question), and that, not the pooled 0.4865 per question, is the decision number.
  Spot peer is proper, so a calibrated forecaster loses 91.19 of the 217.48 to the clip by construction;
  the floor pays only if the sub-c band is under-priced, and its break-even hit rate (3.08% at
  c = 0.05) sits above the Jeffreys upper bound on 0 of 81 (3.0%). 94 publishes at or below 5% produced
  0 YES against 2.60 expected under the bot's own prices.
- An MC floor is a tax on every question: c = 0.05 moves 68 of 97 records, 67 of them losing with the
  resolving option never lifted (renormalisation drag off the top option), 3.53 points per question
  (95% CI 2.40 to 4.71) and era-stable (3.40 / 3.94 / 3.21 across the three eras).
- Loosening is censored, not measured: a record published at its in-force floor destroyed the raw
  member value. Below 0.01 the maximally generous bound is +8.56 over 447 binaries under the
  published-value rule and +10.91 under the member-level rule (28 records had a clamped member in a
  median position), all pre-flip; +0.50 over 97 MC questions, and unshippable under forecasting-tools
  0.2.92 anyway. The one MEASURED comparison runs the other way: today's 0.02 floor priced on the 23
  pre-flip publishes below it costs 20.31 points, 23 of 23 resolving NO, era-confounded and not a live
  lever.
- Every out-of-sample fit returns the do-nothing candidate, so the era test is vacuous rather than
  passed; the out-of-bag value of "pick the best floor, then apply it" is 0 to minus 0.25 per question
  in every strict window. Ceiling-only tightening reads +39.70 at c = 0.05 pooled, all of it q42024
  (pre-flip, 0.97, resolved NO, one of the spring miss cluster the killed YES-side shrink was fitted
  on), every interval straddling zero, no triple-era record reaching 0.90: **do not pursue**, it is the
  hard-clip form of the layer killed on 2026-07-08.
- The only pro-tightening row in either cohort is unfiltered and is one record: q44874, a dry-donated-
  key publish at 0.03 on a single surviving forecaster that resolved YES, worth +120.40 at c = 0.10 on
  its own (triple era +107.91 in-window, minus 0.55 per question out of bag). The single-survivor publish
  floor `[THIN_PUBLISH_BINARY_FLOOR, THIN_PUBLISH_BINARY_CEIL]` = [0.05, 0.95] on this branch fires on
  exactly that shape: +51.08 over the 4 genuine k=1 publishes in the archive (all in
  `DEGRADED_RUN_QIDS`), zero on the other three, forward price about minus 2 per future k=1 publish at
  0.03 that resolves NO.

**Decision status: operator decides; the data supports leaving both floors where they are.** Nothing
supports tightening either floor (proper-score loss everywhere, no under-priced band), nothing can
measure loosening on the live roster, and the one shape that ever read pro-tightening already has its
instrument. Two judgment calls are the operator's: whether the widening flip's binary half (0.01 to
0.02) is worth revisiting given the +20.31 measured on pre-flip records (it cannot matter until a
post-flip publish comes within 0.014 of the floor), and whether the MC headroom figure this entry
supersedes is the one the repo keeps. **Reconciliation with the 2026-07-09 MC decision above:** same
verdict, independently reached, but the sweep's measured bound (+0.50 total, about +0.005 per question)
supersedes that entry's "about +0.01 nats/question" estimate (0.01 nats is 1 spot-peer point, so about
+97 over this cohort), two orders of magnitude apart in the same direction. Re-run each round with the
two commands in the dim doc; the first line to read is the header's live-regime line.

### File splits + shared fetch-primitive promotion (added 2026-07-18, low, standalone PRs)

Structure findings from the branch-review forge + structure reviewers
(`scratch/branch_review_july15/reviews/`). Each is a clean behavior-neutral refactor — keep
them OUT of feature work, land as their own PRs.

- **Files over the monolith threshold — re-measured 2026-08-25**, since the 2026-07-18 numbers are
  all stale in both directions. Closed: `research/timeseries_anchor.py` split in `e4bddae` (1,293 →
  372 + `ts_routing.py` 675 + `ts_render.py` 394 + `ts_estimators.py` 231). Still open and now
  bigger: `research/agentic/tools.py` **1,047** (was 784 — the search-vs-fetch seam is unchanged)
  and `tests/test_agentic_tools.py` **1,855** (was 1,055). New arrivals since that measurement,
  none of which has its own entry: `metaculus_bot/ablation/cli.py` **2,206**,
  `research/agentic/loop.py` **1,684**, `prompts.py` **1,613**, `research/resolution_source.py`
  **1,175** (the Tier-2 Datawrapper hop pushed it past its own deferred-split note, which still
  cites ~670), `performance_analysis/parsing.py` **1,135**. `forecaster.py` (1,049) has its own
  Medium-term entry.
- **Promote the shared SSRF/fetch primitives into `http_fetch.py`** — still open, and the coupling
  widened. `agentic/tools.py` reaches into `resolution_source.py`'s private functions at six call
  sites: `_sem_for_host` (221), `_get_session` (501), `_extract_main_text` (593, 835) and
  `_ip_is_disallowed` (717, 738). Hoist those four into the shared `research/http_fetch.py` as
  public primitives and have both call sites use them.
- **Give the anchor-chart `_session_charts` global a public accessor** — still open; it is a
  module-level dict at `research/timeseries_anchor.py:94` (moved by the split) mutated and read by
  qid. Expose a small get/set/clear surface instead of touching the global directly.
- **Hoist a public `metaculus_get` and repoint the three private copies (added 2026-09-01).** There
  are now three separate authenticated Metaculus GET helpers with the same `Authorization: Token`
  header shape and offset/limit paging: `performance_analysis/collector.py`'s private `_api_get`
  (three 429 retries, then a bare `RuntimeError`), `scripts/backfill_research_from_comments.py`'s own
  copy, and `scripts/supply_probe.py`'s `_get_json` (a larger rate-limit budget, and it raises
  `requests` exceptions rather than `RuntimeError` because its caller soft-fails per slug). The probe
  deliberately did NOT import collector's private helper — the contract does not fit and promoting it
  would have edited a shared file mid-fan-out — so the consolidation is this follow-up: one public
  helper in a shared module, parameterized on retry budget and raising `requests` exceptions, with
  all three call sites repointed.
- **Split `tests/test_telemetry_markers.py` by marker family, and re-measure this list first (added
  2026-09-01, forge R7).** That file is **2,283** lines and each marker family's tests are already
  independent, so splitting it is mechanical the next time anything touches it. The sizes in the
  first bullet are all pre-`276ecf2` (2026-08-27, "split the six monolith modules"), which closed
  more than it recorded: `ablation/cli.py` is **972** not 2,206, `agentic/loop.py` **683** not 1,684,
  `agentic/tools.py` **624** not 1,047, `parsing.py` **432** not 1,135. Meanwhile `prompts.py` grew
  to **1,691** and `resolution_source.py` to **1,294**, and three files over the ceiling are listed
  nowhere: `research/financial_data.py` **1,365**, this test file, and `tests/test_agentic_tools.py`
  **2,021**.

  **Re-measured 2026-09-02, and one of them is closed.** `tests/test_resolution_source_provider.py`
  (2,054 lines, the largest test file with no entry anywhere) is SPLIT: its three declared layers are
  now `tests/resolution_source/test_resolution_source_helpers.py` (528),
  `test_resolution_source_fetch.py` (1,070) and `test_resolution_source_provider_gating.py` (424),
  with the fake aiohttp session and the HTML page builders in `tests/resolution_source_fakes.py`
  (266) and the autouse DNS stub plus the page fixtures in the package's own `conftest.py`, following
  the `tests/ablation/` precedent. Same 147 tests, none rewritten. The Tier-2 Datawrapper suite MOVED
  into that package too (`tests/resolution_source/test_resolution_source_datawrapper.py`), dropping its
  own copy of the autouse DNS stub, its own question builder and a twice-inlined Infogram embed literal
  in favour of the conftest's and the fakes module's; same collected count. **`tests/test_prompts.py` is
  also SPLIT** (2026-09-02 forge R11, at 2,192 lines): its three surfaces are now
  `tests/prompts/test_base_prompt_rules.py`, `test_structured_block.py` and
  `test_research_clauses.py`, with the question stubs and prompt renderers in
  `tests/prompt_builders.py`. Same 166 tests, none rewritten, and the package needs no `conftest.py`
  because nothing in it wants a fixture. Everything
  else on this list grew again: `tests/test_telemetry_markers.py` **2,428**,
  `tests/test_agentic_tools.py` **2,235**, `prompts.py` **1,805**, `resolution_source.py` **1,382**,
  `research/financial_data.py` **1,033** (down, after `currency_pegs.py` and `fred_rendering.py`
  came out of it). Read the prompts.py and resolution_source.py figures as a snapshot taken while
  the 2026-09 bundle was still being edited. The re-measure keeps going stale because nothing
  updates it except a round that trips over it, so measure before acting rather than quoting these.
- **Dedupe the peg anchor when two tickers share one (added 2026-09-01, forge R15).** The bundle's
  peg-anchor block is decided per ticker inside `_fetch_yfinance_data`
  (`research/financial_data.py`), so a question naming two pegged crosses that share an anchor
  (SZL and LSL both anchor on `ZAR=X`) or naming the anchor itself beside a pegged cross renders
  that anchor block twice, at the cost of one extra yfinance call (free, and the render is the
  actual cost). Skipped as a corner case; the fix is a dedupe in the job builder, natural to do
  when the peg table moves out to its own `currency_pegs.py` per the split above.

## Medium-term (requires more exploration)

### Consider migrating scheduled runs off GitHub Actions cron (added 2026-07-19, MEDIUM)

The 2026-07-18 latency/completeness audit
(`scratch/residual_2026-07-18/followups/latency_completeness.md`) traced ~80% of the
submission-latency p90 creep (24 → 58 min) to GitHub Actions cron STARVATION: the scheduled
workflows fire ~36 times/day against a nominal 72 (GHA silently drops scheduled runs under
load). Queue-delay p90 is already 82 min against 90-minute question windows, so worst-case
queue (82) + pipeline (12 p90) breaches the close deadline — and a missed close forfeits the
whole spot score. Gap-fill v2 (up to 540s wall) stacks more pipeline time on top once the
july15 branch merges.

Time-sensitive because the failure mode is a hard forfeit, not a soft regression. No clean easy
migration exists (hence MEDIUM, not HIGH) — alternatives to scope: a self-hosted GHA runner
(kills queue starvation, adds ops burden), EventBridge Scheduler → `workflow_dispatch`, or a
small VM / fly.io cron firing `workflow_dispatch`. The new CLOSE_MARGIN watch
(`make close_margin_watch`) is the instrument to confirm the problem persists and to measure any
migration's effect.

**Update 2026-08-24 — DEMOTED, no longer time-sensitive.** Cron delivery recovered GitHub-side
(not from any config change of ours): 41% of nominal at the era's trough → 71% → **94%** by late
August, with open→submit p90 falling 69 → 36 → 17 minutes and **zero red-line margin breaches
since 2026-07-31**. The era still lost 5 of its 104 tournament questions, but only one to genuine
cron starvation; the four mechanisms are q44801 (2026-07-22 midnight cron gap — the one a
migration would have addressed), q45093 (2026-08-06, five consecutive run failures, cause
unrecoverable — no artifacts uploaded), q45374 + q45375 (the separate **2026-08-19 wedge
incident**: an apt hang inside `playwright install --with-deps` held the workflow concurrency
group ~18 of 24 hours; the step-level `timeout-minutes: 3` fix shipped this round, `41ee30f`),
and q45085 (2026-08-03, forecast at 3/3 then rejected HTTP 405 — the run started seconds before
close and submitted five minutes after). Keep the entry for a future season; `close_margin_watch`
remains the instrument.
(`scratch/residual_2026-08-24/dim_bot-health-deep.md`, `TELEMETRY_INVENTORY.md` §2–3.)

**Update 2026-08-25 — three of those four mechanisms now have a fix or a diagnosis, which is most of
why this stays demoted.** (a) The wedge class is closed generically: `865d4e0` caps EVERY step and
job in all five bot workflows plus `ci.yaml` / `claude.yml`, sized off the 200 most recent successful
runs, and raised the Playwright cap 3 → 5 min because `41ee30f`'s 3 min was sized on n=10 and turned
out to be 3.3x the measured worst rather than 5x. A hang now holds the concurrency group ~5 min
instead of ~18 hours, and the job cap (80 min) is deliberately above the bot's own contract
(`PER_QUESTION_WALL_CLOCK_DEADLINE` + `WALL_CLOCK_STACKING_MIN_BUDGET` = 3600s) so it can only fire
on a hang, never on a slow-but-recovering run. (b) q45085's class is gated: `b76428f` adds
`metaculus_bot/publish_gate.py` as layer 4 of `publish_hardening`, which skips the forecast AND
comment POSTs when `close_time` has passed (or the cached state is already CLOSED/RESOLVED), emits
`PUBLISH_SKIPPED_CLOSED` and counts it as ALERTABLE — a skip means latency cost us a question, which
is exactly what should redden CI. Deliberately no safety margin: ft's publish body sleeps ~8s, so a
question with seconds left can still 405, and widening the gate would start skipping publishes that
would have landed. Same commit stops retrying a 4xx outside {408, 429}. (c) q45093's
"cause unrecoverable" is diagnosed and is NOT ours: all five failures carry `steps=[]`, an empty
`runner_name` and a span of exactly 15 minutes — GitHub never allocated a runner, so no timeout can
shorten it, and the remainder of that loss is cron delivery (the day's last fire was 18:43 against a
21:00 close). Only q44801's midnight cron gap remains a migration-shaped problem.

### Split `forecaster.py` (1,049 LoC, still past the ~1000 ceiling) (added 2026-07-20, MEDIUM)

Status: deferred refactor from the 2026-07-20 forge review of the run-QA commits (finding F3). Its
original "after the july15 branch merges" blocker is gone (that merged 2026-07-21), and **half the
job is already done**: `9f5dc17` carved the drop telemetry, degradation counters and post-fan-out
stacking routing out into `stacking_route.py` (394 LoC), which was extraction seam (2).

What is left is seam (1): the gather / wall-clock / soft-deadline concurrency machinery
(`_forecaster_with_soft_deadline`, `forecaster.py:959`, plus the parallel fan-out plumbing). The
file re-measured **1,049 lines on 2026-08-25** — the carve-out bought ~17 lines net because research
stages kept accreting, which is the argument for doing seam (1) rather than re-measuring again.
Large-blast-radius (the hottest file in the pipeline), so still its own PR.

### Price the high→xhigh reasoning-effort premium via backtest A/B (added 2026-07-20, MEDIUM/low urgency)

**Update 2026-07-20 (evening): the A/B as originally scoped is largely moot.** Two same-day changes
gutted the stakes: the roster dropped to the 3-member triple (gpt-5.5 left), and the operator then
dropped the gpt-5.6-sol forecaster xhigh→high — sol was ~70% of forecaster reasoning spend, so
paying an unmeasured premium on the dominant-cost slot wasn't justified. The only xhigh slot left
in the forecaster path is opus-4.8 (~$0.40/q), so the live question is now narrowly "does opus-4.8
xhigh beat high?" — lower stakes, MEDIUM/low urgency. (The opus-4.8 stacker also runs xhigh but is
prod-disabled, so that's backtest-only exposure.)

Status: operator explicitly deferred 2026-07-20 — worth doing, but the backtest budget is
contended; revisit when budget frees up or before the next major effort-config decision.

**Motivation.** The 2026-07-20 reasoning-effort audit
(`scratch/reasoning_effort_audit_2026-07-20/synthesis.md`, built from the AIB spring-2026 metac
baseline-bot leaderboard) found default→high effort is clearly worth it: 8/8 within-model
contrasts positive, sign test p=0.0078, median +2.9 spot-peer points/question, and thinking
OFF→ON is the single largest knob (opus-4-5 flipped −1.9 → +2.9/Q). BUT the board has zero xhigh
variants, so the high→xhigh premium — which, at the 6-model roster this entry was written against,
we paid on 4 of 6 forecaster slots, ~64% of donated per-run spend per the 2026-07-19 credit audit
(now just the opus-4.8 forecaster after the evening sol drop) — is unmeasured. The audit's
diminishing-returns hint is a weak single-seed prior, not evidence.

**Proposed test (rescoped 2026-07-20 evening).** A paired A/B backtest, high vs xhigh arms on the
same questions, on the one remaining xhigh forecaster slot — opus-4.8 (originally scoped as
gpt-5.6-sol + gpt-5.5, both since dropped from the xhigh forecaster set); ~`backtest_medium` scale
(2 arms × 32 questions), rough cost $60-90 at recent per-question rates.

**Decision rule when run.** If xhigh ≈ high within noise, drop opus-4.8 to high too; if xhigh wins
meaningfully, keep it and reconsider re-raising the other slots.

### Research-output audit: temporal/provenance error sweep (added 2026-07-08, low priority)

Motivated by the qid 42304 INES miss: the then-native-search provider (`x-ai/grok-4.1-fast`, retired)
cited an undated NucNet archive article from 1 Feb 1999 (the 1998–99 Istanbul INES-3 accident) as a
"February 2026" Turkish event with a fabricated "reported March 1, 2026" date. All five forecasters
anchored on it (81% published; resolved NO; peer −115.9); the same phantom reached ≥2 top-competitor
stacks — a field-wide hazard of undated archive URLs, not a one-off. Idea: a free offline audit over
`backtests/research_archive/latest/` — sample research per provider, spot-check high-leverage claims
(dates, event existence, numbers) against cited URLs, classify error modes (temporal displacement,
fabricated dates, summarizer certainty inflation). Low priority (offending provider gone; prompt-side
date-stamping + single-source flagging are the nearer lever) but worth doing before any new provider swap.

Related dependency follow-up (from the 2026-06-01 desloppify pass) — **DONE 2026-07-23 (branch
`gapfill-tweaks`).** `forecasting-tools` moved off the pinned `0.2.54` to `0.2.92`, dragging litellm
to 1.92.0 and openai to 2.x. This is forecast-affecting; the pre/post `make backtest_medium` gate
was NOT run (paid) — coverage is the full offline unit/integration suite, and the operator-triggered
paid `test_bot` GHA run is the final live gate before prod. (The poetry→uv migration follow-up
shipped 2026-06.)

### Gemini grounding via OpenRouter — currently NOT supported (added 2026-05-17)

Goal would be: route Gemini Google-Search-grounded calls (currently in `metaculus_bot/research/gemini_search.py` via direct `google-genai` SDK + `GOOGLE_API_KEY` — the old `gemini_search_provider.py` path predates the subpackage move) through OpenRouter so the donated Metaculus credits cover them, freeing up personal Google API budget.

**Status as of 2026-05-17**: NOT supported. OpenRouter's web plugin and `:online` suffix expose native search ONLY for Anthropic / OpenAI / Perplexity / xAI. Gemini falls back to **Exa** (verified HIGH confidence: <https://openrouter.ai/docs/guides/features/plugins/web-search>). Migrating today would silently swap Google's grounded retrieval for Exa text-search — quality regression, not just cost optimization.

**Recheck periodically**: <https://openrouter.ai/changes> — if/when OpenRouter announces native Google grounding (or a passthrough for `tools=[{"google_search":{}}]`), revisit this migration. Until then, no action. **Last actually verified 2026-05-17 and not re-checked since (noted 2026-08-25)** — treat the "NOT supported" status as a dated reading, not a current fact, and re-verify before quoting it.

### Update analysis-CLI defaults to summer-futureeval-2026 (added 2026-05-17)

Tournament rolled spring→summer 2026-05-17; live `TOURNAMENT_ID` updated but **three CLI defaults stay
pinned to spring** intentionally (`ablation/cli.py:122`, `performance_analysis/collector.py:37`,
`performance_analysis/cli.py:18` — line numbers re-checked 2026-08-25) so analysis defaults to the
resolved dataset, not the freshly-opened one.

**The gate has fired — this is now an open flip, 2026-08-25.** The "~30+ resolved Qs" trigger is long
past: summer-futureeval-2026 carries 104 resolved tournament questions in the post_flip cell alone,
it closes 2026-09-06, and no successor slug exists yet, so summer IS the resolved dataset for the
foreseeable future while spring is the stale one. Flip all three defaults to
`summer-futureeval-2026`; also refresh the stale slug examples in the comment and message at
`tests/test_tournament_dates.py:126,131`. Cheap and free — and AGENTS.md already tells readers to
pass `--tournament` explicitly because the default is spring, which is the tell that the default is
now the wrong one.

### Mixture model parameterization for numeric questions — largely rejected

Ask LLMs to parameterize a mixture (2-3 components: mean, std, weight) instead of percentiles, for
smoother CDFs (Mantic reports good results). **Note:** a mixture path
(`NumericStructured.mixture_components` + router branch) was built and REMOVED 2026-07-08 (landed on
main in `642b027`, 2026-07-11) because percentiles+PCHIP beat it in every benchmark; the `mixtures.py`
library is preserved but dormant. **The removal was justified at the time as "zero prod fires", which
turned out to be wrong.** The 2026-08-24 counterfactual round proved one confirmed prod fire (q43826,
2026-06-06, gemini-3.1-pro, whose published CDF reproduces bit-exactly only through the mixture
branch) and one rejected attempt (q43913, 2026-06-11, gpt-5.4). The removal decision itself stands on
the benchmarks; do not cite "zero prod fires" as its evidence. Re-proposing the
LLM-parameterized-mixture form has to clear the benchmark bar.
(`scratch/residual_2026-08-24/dim_discrete-maxstep-counterfactual.md`, "Two side findings worth
carrying forward" item 1; the same correction is in AGENTS.md's
"Probabilistic tools" mixture-of-normals bullet. Corrected here 2026-09-02 after surviving three
residual rounds.)

### Aggregation strategy improvements

**Status 2026-05-29: STACKER DISABLED ON ALL TYPES (default off in code).** Numeric was already off
(ablation median > stack CRPS, p=0.042); binary + MC now off too (binary was a *tie*, p=0.496 — low-risk
default given compute cost, not a measured harm; binary/MC treatment effect unmeasured on the current
stack). Revisit when post-2026-04-27 marker-era resolved questions exist.

Item status (from analysis; prompt changes address the bigger issues):

- **Trimmed mean** (drop hi+lo, mean of middle 4 of 6) — untested, on backlog.
- ~~Post-aggregation shrinkage toward 50%~~ — **KILLED 2026-05-10** (May data didn't replicate the NO-bias
  at n=109; shrinkage costs well-calibrated extremes).
- **Spread-aware aggregation** — **SHIPPED as CONDITIONAL_STACKING** (April 2026; prob-range trigger
  justified by May ρ=0.616 disagreement-error).
- **Per-type weighting by historical performance** — LOW-PRIORITY, deferred to Q3+ (only gemini-3.1-pro fit
  the binary-vs-numeric asymmetry; revisit only when ≥2 active models show it on ≥100 binary AND ≥30 numeric).

### Per-model peer ranking: GPT-strong / Claude-weak on binary (2026-05-29, NEEDS BENCHMARK before acting)

Peer-equivalent recompute (`scratch/residual_2026-05-29/dim_peer_recompute.md`, validated exact vs
`spot_baseline_score`), binary spring-aib-2026 n≈150:

- **GPT carries the binary ensemble** (gpt-5.1 +19, gpt-5.2 +17 peer, CIs exclude 0); the **Claude pair
  is the binary drag** (opus-4.6 −9, opus-4.5 +2.2; confound runs the wrong way, so genuine).
- **INVERTS on numeric** — Claude strong (opus-4.6 +24), gpt-5.1 weakest — so binary-specific, not
  "Claude is bad"; a blanket cut hurts numeric.
- Counterfactuals (paired, in-sample): drop Claude pair → +5.94 binary peer [+1.0, +11.7]; drop opus-4.6
  alone → +3.66 (survives jackknife, most robust). GPT-only +10.6 hinges on 1-2 questions — don't quote clean.
- gpt-5.2 is NOT high-variance (lowest binary sd); wildcards are gemini-3.1-pro (sd 96), opus-4.6 (sd 80).

**Do NOT act without an intense OOS benchmark** — in-sample (n=62 paired), multiple-comparisons exposed,
epoch-confounded, roster already rotated. Credible reading: on binary the Claude pair is a measurable drag
and GPTs carry the ensemble. Next step: a prospective per-type model-inclusion benchmark (GPT-heavy binary,
retain Claude numeric), gated on OOS peer. (The stacker's own model opus-4.5 being a weak base binary
forecaster is part of why disabling the binary stacker is low-risk.)

### Domain-aware CDF spread tuning

**Status 2026-05-29: HOLD — measured on a STALE pipeline; re-measure on current version before ANY narrowing.**
Residual analysis (`scratch/residual_2026-05-29/dim_numericpit.md`, `numeric_width_version_confound.md`)
re-confirmed across two rosters that the *analyzed* CDFs are too wide (PIT std 0.24–0.26 vs ideal 0.289; 90%
coverage 92–98%) and a uniform k≈1.2–1.3 contraction would hit the ideal. BUT all analyzed forecasts (≤2026-04-13)
ran tail-widening at full strength (k_tail=1.25); prod flipped to k_tail=1.0 on 2026-05-12 (`b8d730f`) *after* the
data, so current prod is already narrower. **Applying k≈1.3 now would overcorrect** — over-width is in the *body*
while deep tails are already too thin (1.5–6% mass vs 10% ideal).

Before anything here: (1) PIT log-grid measurement-bug fix in `analysis.py::_interpolate_pit` (mis-scores
log/`zero_point` questions by up to 0.86 PIT) — since shipped; (2) add PIT std as a first-class `backtest.py`
metric; (3) `make backtest_large` for a current-version PIT measurement; (4) only then, if still over-wide, a
smaller narrowing factor with a per-side tail-mass floor. Direction (mild body over-width) may survive; magnitude
k=1.3 won't. Note: the 2026 finance carve-out **inverts** the old advice — financial questions were the *most*
over-wide (cov90=100%), so "exclude finance/markets" is wrong on current data. (Retire the old "PIT std 0.143"
figure — it was the April n=11 ~2nd-percentile draw; population ~0.24–0.26.) The older forecastability-conditional
framing is superseded by this version-confound note.

### Numeric-width history — receipts + ongoing monitor (added 2026-07-17)

Consolidated, git-verified history of numeric-width config, so future sessions never re-litigate from memory.
Short version: tails deliberately widened 2025-09 → 2026-05, turned off on main 2026-05-18, and the TS-anchor
clause (on main 2026-07-21) now pushes back toward sharpening.

The **Date** column below is each commit's AUTHORING date — this table is a commit history, which is what that
column is for. The two rows that also define config eras land on main later, and the era boundary is the
landing date: `b8d730f` (2026-05-12) → main 2026-05-18T17:21:19Z (`0e85e1b`); `3a7ba7d` and the Phase-8 fix
(2026-07-17) → main 2026-07-21T17:07:37Z (`b4e9df0`). See AGENTS.md, era-bucketing → merge-date rule.

| Date (authored) | Change | Value before → after | Commit | Source |
|---|---|---|---|---|
| 2025-08-21 | Numeric prompt: earliest "widen" language added | (none) → "aim to produce somewhat wider and less confident predictions" | `4bd8685` | `prompts.py` |
| 2025-09-06 | Numeric prompt: widen language strengthened | "somewhat wider" → "wider and less confident … penalties for narrow intervals are severe" | `0437f3e` | `prompts.py` |
| 2025-09-07 | **Tail-widening machinery introduced** — `widen_declared_percentiles` + constants born | function default `k_tail=1.25`, `span_floor_gamma=1.0`; `TAIL_WIDEN_K_TAIL=1.25`, `TAIL_WIDEN_SPAN_FLOOR_GAMMA=1.0` | `4c6481b` | `tail_widening.py` + `numeric_config.py` |
| 2026-03-30 | Numeric prompt: blanket-widen replaced by forecastability-conditional wording | "aim to produce wider … penalties for narrow intervals severe" → "produce wide, diffuse [for volatile] … anchor tightly [for stable] … penalties for overly wide intervals on predictable quantities also accumulate" | `3217aab` | `prompts.py` |
| 2026-04-27 | Numeric prompt: added "Hedge audit" anti-over-widening clause | (none) → "Only widen tails when you can name specific evidence creating that uncertainty, not because it feels safer" | `95c4fff` | `prompts.py` |
| **2026-05-12** | **`k_tail` 1.25 → 1.0 and `span_floor_gamma` 1.0 → 0.0** (function defaults AND both constants), plus `ValueError` guard on `k_tail<1` / negative params | `k_tail: 1.25 → 1.0`; `span_floor_gamma: 1.0 → 0.0` | `b8d730f` | `tail_widening.py` + `numeric_config.py` + `constants.py` |
| 2026-06-01 | Subpackage move (no value change): `tail_widening.py`/`numeric_config.py` → `numeric/` | rename only | `78c5182` | subpackage extraction |
| 2026-07-09 | Numeric prompt de-overfit prune; **kept** hedge-audit | large prune | `5c6640a` | `prompts.py` |
| 2026-07-17 | Numeric prompt: **TS-anchor clause** reframes widening as the failure mode ("SHARPEN, not another license to widen"; "cov@10 ≈ 0.03 vs 0.10 target") | (none) → anchor clause | `3a7ba7d` | `prompts.py` |
| 2026-07-17 | Numeric prompt: fixed the surviving blanket-widen contradiction — Phase-8 "keep tails far apart for unknown unknowns" scoped to *nameable* unknowns, not padded from generic caution or beyond a calibrated anchor's band | unconditional → conditional | (this branch) | `prompts.py` |

Notes on the story:

- **`k_tail`/`span_floor_gamma` have a two-point history, no intermediate values**: born 1.25/1.0 (`4c6481b`,
  2025-09-07), flipped 1.0/0.0 (`b8d730f`, 2026-05-12); function default and constant always changed together.
  Current: `tail_widening.py` 1.0/0.0, `numeric/config.py` `TAIL_WIDEN_K_TAIL=1.0`, `TAIL_WIDEN_SPAN_FLOOR_GAMMA=0.0`.
- **No constant literally named "PIT."** The "PIT widening 1.25 → 1.0" memory = the `TAIL_WIDEN_K_TAIL` flip; PIT is
  the calibration metric (`performance_analysis/analysis.py`) that drove it, not a separate knob.
- **The 2026-05-12 study** (`scratch_docs_and_planning/tail_widening_empirical_calibration.md` +
  `scratch/tail_widening_calibration_2026-05-12/`, 43 resolved numerics Feb–May): drop `k_tail` to 1.0 (every
  segment minimized |PIT std − 0.289| there; 1.25 moved PIT *away* from ideal); `span_floor_gamma` a no-op (→0.0);
  no per-category `k_tail` (CIs overlap); narrowing (`k_tail<1`) is a silent no-op (motivated the `ValueError`
  guard). Caveat: "revisit after ~150 resolved numerics."
- Prompt-language and `k_tail`-code changes are **decoupled** (separate commits).

**Ongoing monitor: `metaculus_bot/performance_analysis/width_monitor.py`** (read-only, free — not cost-gated).
Run: `uv run python -m metaculus_bot.performance_analysis.width_monitor --cached scratch/coherence_2026-07-15/perf_all_tagged.json`
(or `--tournament <slug>` for a live read-only pull; `--output-json <path>` to persist;
`--exclude-qids known_bug` drops the known-pipeline-bug cohort — 43746/43747 (open-bound arithmetic bug) plus
43913 (pre-`9f1175c` discrete max-step cap) — that every other residual dimension already
excludes, and renders the dropped count so the exclusion is never silent). Per config era it reports
central-80% / central-50% coverage with Jeffreys CIs, cov@10/50/90, PIT std (uniform ideal 0.289 — below ⇒ too
wide, above ⇒ too narrow), median relative band width `(P90−P10)/|P50|` as the raw sharpness metric, and
`band_miss (lo/hi)` — the out-of-band rate split by tail, which separates a band that is too TIGHT (both tails
elevated) from one of roughly the right width that is MIS-CENTERED (misses piled in one tail); `cov80` cannot
express that difference and the two call for opposite corrections.

Eras are the two width-relevant flips, keyed on **merge-to-main timestamps** (AGENTS.md, era-bucketing →
merge-date rule): `WIDENING_FLIP` = 2026-05-18T17:21:19Z (`0e85e1b`, k_tail 1.25→1.0; authored 2026-05-12,
`b8d730f`) and `TS_ANCHOR_ENABLE` = 2026-07-21T17:07:37Z (`b4e9df0`, the july15 bundle; authored 2026-07-17).
Both constants previously held their authoring dates, which manufactured a phantom one-record `ts_anchor` era
out of qid 44795 — a question whose own comment names the retired six-model roster that the very same merge
dropped. As of 2026-08-24 the `ts_anchor` bucket holds its first **6 real records** (5 STRICT), so a
three-plus-row table is now the expected shape. Note the bucket is the whole july15 bundle
(anchor + triple roster + `MIN_FORECASTERS_TO_PUBLISH` 3→1 + gap-fill v2), so a width shift across it cannot be
attributed to the anchor alone.

**Measured 2026-08-24 on 258 recovered numeric+discrete questions** (`width_monitor` over
`scratch/residual_2026-08-24/perf_all_tagged.json`, `--exclude-qids known_bug`). This RESTATES the
2026-07-17 table, whose pre-flip row (0.851 / 0.267 / 0.674) came from the monitor before its
era-boundary and PIT fixes — the same 197 frozen records now read 0.871 / 0.258 / 0.686, so a diff
against the old row is a measurement fix, not data movement:

| era | n | cov80 [95% CI] | cov50 | cov@10 | PIT std | med rel width |
|---|---|---|---|---|---|---|
| widening_on (k_tail=1.25) | 197 | 0.871 [0.821, 0.914] | 0.578 | 0.071 | 0.258 | 0.686 |
| widening_off (k_tail=1.0) | 45 | 0.728 [0.593, 0.845] | 0.489 | 0.089 | 0.304 | 0.458 |
| ts_anchor (= the july15 bundle) | 6 | 0.643 [0.286, 0.923] | 0.214 | 0.167 | 0.348 | 0.109 |
| no_timestamp (fall-2025, unattributable) | 10 | 0.955 [0.783, 1.000] | 0.682 | 0.000 | 0.222 | 0.138 |
| all | 258 | 0.847 [0.801, 0.889] | 0.558 | 0.074 | 0.268 | 0.615 |

Reading (2026-08-24): the apples-to-apples post-flip NUMERIC cell sits at PIT std 0.2860 against the
0.2887 ideal (spread multiplier 0.991x, fourth consecutive round inside the band) — `k_tail=1.0`
stands; the pooled num+disc 0.304 is driven by the discrete cell's high-lean, not width. One defect
found and fixed this round: **out-of-grid PIT censoring** — the interpolation clamped to
`cdf[0]`/`cdf[-1]`, which sign-flipped a below-bound resolution into a high PIT once the open-bound
unclamp made large `cdf[0]` legitimate (q44218 read 0.917 for a resolution below everything we
forecast); fixed in `1fe96c9`, and the correction moves post-flip numeric cov@90 0.206 → 0.176.

### Width post-ship watch + monitor attribution tagging (added 2026-07-18, medium)

Two follow-ups from the 2026-07-18 full-branch width audit (`scratch/width_audit_2026-07-18/synthesis.md`).
The audit found no era-stable width bias and motivated Option B, shipped `f4f7984`: delete the Step-7
hedge-audit narrowing push and Step-9b's LOW→wide IQR prescription from the numeric prompt.

1. **Post-ship over-sharpening watch (load-bearing gate).** Option B removed a one-directional narrowing
   instruction, so forward risk flips over-wide → over-sharp. Over the next ~15 numeric resolutions on the
   `width_monitor`: if `cov80` climbs back toward ~0.88+ WITH PIT std below ~0.25 (the pre-flip over-wide
   signature), re-add the hedge audit — but as a SYMMETRIC clause ("match width to reasoning; don't pad OR
   sharpen from disposition"), never the one-directional form just cut. Standing gate: n≥25 with theme
   diversity before fitting any width knob (the "~150 numerics" caveat governs a fitted change). Do NOT cite
   `cov@10 ≈ 0.03` as a too-wide signal here — that's a pooling artifact; the current-era value is 0.107 (on
   target). The ~0.03 the TS-anchor prompt clause quotes is the live-prod low tail, a different cohort.
   **Status 2026-08-24: NOT FIRED at n=5 of ~15** — cov80 0.600, PIT std 0.356, i.e. both halves fail
   in the OPPOSITE direction (too narrow), not the over-wide signature the trigger describes. At
   effective independent n ≈ 3 (three submission days) with a PIT-std bootstrap CI containing ideal,
   that is a direction to watch, not a finding; if the narrow lean survives to n ≈ 30 in
   mid-September, the move is the SYMMETRIC clause this item already specifies.
   (`scratch/residual_2026-08-24/dim_numeric-width.md` §3.)
2. **Tag monitor records with `anchor_present` / `gap_fill_v2_present` — TAGGING SHIPPED 2026-08-24
   (`dece67f`); the monitor does not consume it yet.** The `ts_anchor` era bucket is confounded (TS
   anchor + gap-fill v2 + native-search/crux terra swaps + the 6→3 roster drop +
   `MIN_FORECASTERS_TO_PUBLISH` 3→1 all reached prod in one merge, `b4e9df0` 2026-07-21T17:07Z) AND
   pools treated with untreated (only ~53% of numerics route to a fetchable series).
   `performance_analysis/research_tags.py` now stamps `anchor_present` / `anchor_confidence` /
   `gfv2_present` / `gfv2_loop_ran` / `research_source_class` onto every performance record straight
   off the research archive, which is the collection-time half. What is left is one consumer change:
   `width_monitor.py` references none of those fields (checked 2026-08-25), so era rows are still
   pooled across treated and untreated. Split them and the anchor-effect read is unblocked — read
   `anchor_present=False` through `anchor_confidence`, since a trimmed comment-backfill record can
   read absent when it isn't.

### Starved outer tails: the detector shipped, the publish-time WARN did not (added 2026-09-01, medium)

`scan_outer_tails` (`performance_analysis/outer_tail.py`, rendered by the width monitor's CLI)
landed 2026-09-01. It finds a defect distinct from the max-step smear: on an open bound the
declared outer tail can route past the displayed range entirely, leaving every in-range bin above
the members' declared p99 pinned at the platform's per-bin minimum step, so any resolution in that
band earns the same ~−219 floor score whatever the grid size. That makes it a cliff at a fixed
location rather than a band of the wrong width, which is why widening does not address it and why
the standing `k_tail` hold above is not in tension with the detector. Two items are open.

1. **No publish-time `STARVED_OUTER_TAIL` WARN, deliberately, and a second trigger to calibrate if
   one is wanted.** On DISCRETE questions — exactly where both motivating records live, q45218 and
   q44182 — `numeric.pipeline._build_discrete_distribution` overwrites `declared_percentiles` with a
   resample grid pinned to the raw bounds, so a detector reading that field at publish time would put
   the anchor AT the bound and quietly never fire on the cohort it exists for. (Same trap
   `log_open_bound_piling_diagnostics` dodges by taking the sanitized declarations as an argument.)
   Firing correctly on the published aggregate needs each member's sanitized declarations threaded
   from `forecaster_runners` to the aggregation site, which is new plumbing on the publish path. **The
   alternative that needs no plumbing is to locate the band WITHOUT the declaration — the terminal run
   of bins sitting at the minimum step.** That is a second trigger definition with its own
   calibration, so it was written down rather than improvised; the reasoning is in the code comment
   above `STARVED_OUTER_TAIL_FLOOR_MULTIPLE`. No telemetry marker is registered while the WARN does
   not exist.
2. **The prevalence is a watch item, and it stays gated on the `k_tail` hold.** The shape is
   SYSTEMATIC, not a per-question accident: 68 of the 417 measurable open-bound sides in the archived
   cohort fire, across 49 distinct questions, 19 of them starved on BOTH sides, and the distribution
   is bimodal with 44 sides sitting essentially exactly at the pipeline's own applied floor (~1.1x the
   platform minimum). So read a fire as "this question carries a cliff", not "something broke here".
   Whether that prevalence justifies revisiting the CDF construction itself is an operator decision
   and explicitly out of scope of the detector; nothing about it licenses a width change, since the
   whole point is that the band is at the structural floor rather than the wrong width. Receipts:
   `scratch/next_season_bundle_2026-09/item19/`.

### Ideas reverse-engineered from high-scoring competitor bots (added 2026-06-26)

Source: dissection of 12 high-scoring outputs from GreeneiBot2 / Preseen-Atlas / SynapseSeer
(a local corpus of competitor outputs, not committed; report + grounding-critic
verdict `scratch/competitor_analysis_2026-06-26/REPORT.md`). **Caveat: the corpus has NO resolution
outcomes**, so every "why it helps" is a mechanism argument, not outcome-validated. Only the
source-provenance trust ladder shipped (`prompts.py:_SOURCE_PROVENANCE_LADDER`); the rest is gated on a
benchmark.

**1. Stacker deviation cap from the base-model median (experiment; needs benchmark).** Bound stacker
output to within K of the base median (binary in prob points — ABSOLUTE, not their unstable-near-0/1
multiplicative "±20% of average"; numeric as a percentile/location shift), in
`aggregation_pipeline.py::_stacking_aggregate` just before `_apply_platt_calibration`, new caps in
`constants.py`. *Grounding:* GreeneiBot2 caps consolidated updates to ±20% of the panel average (line 1352).
*Why:* a cap might let us safely re-enable the numeric stacker MEDIAN beat (CRPS p=0.042). *Caution:* too
tight defeats the point of stacking — ablate, don't assume. **Gate:** `make backtest_medium` capped-stacker
vs MEDIAN-default.

**2. Shared-reliance / consensus-fragility audit (experiment; higher-risk, needs benchmark).** The hole:
`compute_spread` takes MEDIAN when the N agree and never asks WHY — if all 6 swallowed the same unverified
shared-research fact, that consensus is falsely confident and invisible to spread. The missing *mirror* of
the disagreement branch. Three operationalizations, cheap→expensive: (A) **prompt-only self-report** — add a
`load_bearing_claims: [{claim, verified, source}]` field to the structured JSON; surfaces shared reliance,
zero aggregation change, low-risk (could ship in a prompt session); (B) **deterministic cross-forecaster
flag** — one unverified claim in ≥K of N → mark fragile, FORCE the crux/stacker path even at low spread (the
stacker prompt `prompts.py:725` already has the language, just never runs on agreement; hard part is
clustering free-text claims); (C) **dedicated consensus-auditor LLM** on the low-spread branch. *Grounding:*
OUR idea, seeded not copied — competitors discount overlapping-rationale agreement (SpaceX line 2028, Iran
line 1330) and notice shared unverified figures (Metaculus-predictions lines 36/43) but respond by
downweighting+widening, NEVER by gating aggregation. *Caution:* (B)/(C) add cost to the COMMON cheap MEDIAN
path — the main reason they're deferred. **Gate:** benchmark (B)/(C); (A) is the low-risk start.
**Re-priced 2026-08-24 — still the most-supported lever, on weaker headline evidence.** The
discriminative control behind the pitch (material dissent toward truth is rarer on misses than on
hits) moved from 6%-vs-19% to **9%-vs-21%** on the enlarged post-flip cohort (rate ratio 0.30 →
0.62 raw, 0.43 after removing a pipeline-clamp artifact; the miss-side numerator is 2–3 events),
and split by question family it does NOT hold for binaries — pre-flip binaries invert to 1.81 on
the one cell with real power, though follow-the-outlier remains a losing strategy ex ante (−13.7
baseline pts/Q over 357 binaries). The premise is untouched — MEDIAN never asks WHY the members
agree — and q44882 is the cleanest new specimen: three models at 0.19/0.25/0.25 on a YES, with the
v2 ghost (a fourth model with its own research) independently at 0.25. On the 3-member roster the
spread gate also fires far less (caveat in the parked "Spread-triggered second forecast round"
entry under Low-priority), which raises this mirror-image lever's relative value.
(`scratch/residual_2026-08-24/dim_consensus-dissent.md` §5, §9.)

**3. Numeric "unverified-conflict → variance" rider (low priority; tension with our calibration).** When
the trust ladder can't adjudicate two candidate values for a load-bearing quantity, place mass across both
(widen the relevant percentiles) with a materiality gate. *Grounding:* Metaculus-predictions fact-checkers
facing 3,856,697 vs 3,895,701 widened rather than picked ("sigma 95000 to cover the discrepancy", lines
36/43). *Tension:* our CDFs are ALREADY too wide (`TAIL_WIDEN_K_TAIL=1.0`) and the hedge-audit penalizes
caution-widening; the defense is this widens for a NAMED, quantified reason (satisfies the hedge-audit
carve-out) but risks over-application. Decided 2026-06-26 NOT to ship standalone. Revisit only as a
tightly-scoped numeric-only A/B; **gate** on P10/P90 coverage not regressing.

**Also observed, NOT pursued (don't re-litigate):** binary-complement coherence guard (Yes+No=100%,
candidate for a future prompt session, report rec #4); interrogate-resolution-source-quality clause
(Preseen-Atlas line 2679 — partly covered by the trust ladder); numeric partial-resolution incorporation +
reporting-vs-outcome-uncertainty (report rec #5). **Explicitly rejected as conflicting with verified data
(do NOT re-recommend):** GreeneiBot2's one-sided anti-overprediction shave (our binary slope flips 0.83→1.66
across rounds, so a fixed shave helps one and hurts the next); blanket sigma-widening (CDFs already too
wide); parametric mean/sigma numeric representation (our percentile→PCHIP→CDF-space pipeline subsumes it);
the open-tail "spike" grid-compliance trick (we solve grid validity deterministically in `pchip_cdf.py`).

## Low-priority (parked — evidence kept, do not resurface without new evidence)

Entries here are parked on a dated decision rather than deleted. Each opens with why it was
demoted and what would have to change to revive it; the evidence below that line is untouched,
because it is what stops the idea being re-proposed from memory. Re-filed in the 2026-08-25
priority audit.

### Spread-triggered second forecast round (re-forecast, NOT stacker) (added 2026-07-19; PARKED at LOW 2026-08-25)

**Operator decision 2026-08-25 — moved off top-priority to LOW so it stops resurfacing.** Two
reasons, both about today's configuration rather than the idea's merit: stacking is prod-disabled,
so spread gates nothing we currently ship, and the measured trigger looked dead for binaries on the
3-member roster (0 of 4 would have fired). **The second reason is now known to be wrong**: at
n=11 the gate fires on 4 of 11 triple binaries, so the parking decision stands on the
stacking-disabled reason alone; see the trigger measurement below before quoting the LOW label as
evidence about the idea. Revive if the roster grows back or stacking is re-enabled, and re-derive
the binary threshold rather than inheriting it.

**Trigger measurement (load-bearing, read first; first priced 2026-08-24 on the live roster,
re-measured 2026-08-31 and again 2026-09-01 as the triple cohort grew).** The modal worst-miss
remains consensus-with-zero-dissenters under a shared briefing (2026-07-18, reconfirmed
2026-08-24), so **spread carries no directional information when the ensemble shares an
attractor**. On the frozen triple the gate does fire less often than the post-flip data implied,
but the type story the first measurement told was an n=4 artifact. The 2026-09-01 recount over the
STRICT triple cohort (`skipped_config_off` telemetry, 30 questions, the marker agreeing with a
recomputed spread on 31 of 31 comparable records) has the gate firing on **15 of 30 questions:
binary 4 of 11, MC 0 of 6, numeric 6 of 7, discrete 5 of 6**. So this is a **numeric-DOMINANT
lever on the current roster, not a numeric-only one.** The correction matters because "dead for
binaries" is the kind of statement that would stop the item ever being revisited, and the two
newest binary firings are the widest three-member binary spreads on record (0.38 and 0.33 against
the 0.15 threshold). MC is the only cell still empty. The three supporting figures the earlier
measurement carried are superseded by the same recount: triple binary spreads run **0.0100 to
0.3800** rather than 0.04 to 0.08, the three-member narrowing is **1.63x on the mean and 1.90x on
the median** rather than about 4x, and the binary at-threshold rate is **36.4% against 63.0% for
post_flip** rather than zero. The lever still addresses the disagreement subset only, so do not
expect it to fix the current worst misses; the consensus-miss counters are separate items
(gap-fill v2 verify / DISCREPANCY channel; cross-question coherence / resolution-metric
verification checks from the residual rounds).
(First pricing: `scratch/residual_2026-08-24/dim_consensus-dissent.md` §5b and
`dim_aggregation-stacker.md` §4. Correction and recount:
`scratch/residual_2026-08-31/dim_aggregation-stacker.md` §1 and
`scratch/residual_2026-09-01/dim_aggregation-stacker.md`, ledger row 20.)

**What.** On questions where forecaster spread exceeds the existing CONDITIONAL_STACKING
thresholds (~30% of questions per the one recorded estimate): run the EXISTING crux extractor →
EXISTING targeted search → append `## Targeted Research` to the bundle → re-fan-out every base
forecaster → MEDIAN over round-2. The stacker stays off entirely — this is a re-forecast, not a
judge.

**Evidence.** BTF-2 (arXiv:2604.26106): 8 independent Opus rollouts straddle 50% on 38% of 200
difficult questions (mean per-q σ=0.08), and the strongest research agent (Opus-class) scored
*worse* on fixed evidence than on its own research (0.131 → 0.153; the effect was model-dependent,
with Gemini slightly *better* on fixed shared research at 0.143 → 0.141) — so the mechanism is
disagreement-triggered EXTRA RESEARCH, not a smarter judge. AIA Forecaster (arXiv:2511.07678)
independently found disagreement → targeted-search its biggest aggregation lever. Crucially: ALL of
this repo's stacking rejections (n=88 ablation, stack_aug arms, trio-50q) tested "stacker LLM
rewrites the forecast"; a targeted-research-fed SECOND BASE-MODEL ROUND was never built or tested
here, so the rejection evidence does not apply to it.

**Wiring sketch.** `compute_spread` already runs on every prod question; the crux extractor +
targeted search are importable functions welded to the stacker path only by code arrangement
(`stacking_route.py` owns the post-fan-out routing); reuse the fan-out helper for round 2 and
re-aggregate via the existing `_base_combine` MEDIAN. Flag-gated (e.g. `SECOND_ROUND_ENABLED`) + a
telemetry marker. Open design decisions: round-2-only vs pooled (round-1+round-2) median; a tighter
round-2 soft deadline (~5 min) to fit the 58:30 per-question wall clock; interaction with
`WALL_CLOCK_STACKING_MIN_BUDGET`.

**Cost** (priced on the retired 6-member roster). ~8-14 extra LLM calls (1 crux + 1 targeted search
+ 6 re-forecasts + parsers) on the ~30% triggered subset → roughly +30-40% forecaster spend.

**Eval.** Ablation harness `--stages forecast` on frozen research, paired on the triggered subset;
effects <0.02 Brier are undetectable at our n, so this is a big-lever ship-and-watch bet with
era-bucketing.

### Run crux extraction on every question + always-on stacker (added 2026-05-17) — largely superseded

Original idea: always run forecaster fan-out → crux extract → targeted re-research → stacker (vs
the current ~30% high-spread trigger). **Cost** at gpt-5.5 high effort, ~250 Qs/tournament: crux
$14 + targeted search $19 + stacker $75 ≈ **+$80/tournament** vs disagreement-only. Open question:
do cruxes move predictions on uncontroversial questions, or just add latency? Paired — the
always-on stacker half is benchmark-rejected (stacker disabled in prod), and the "spend the crux
research on a real re-forecast" half is the parked entry above. Moved here 2026-08-25 alongside it.

### TS anchor chart image — enable + A/B (added 2026-07-17; demoted from HIGH 2026-08-25)

**Demoted 2026-08-25:** the A/B is paid, its offline-replay form needs research-sink schema v3, and
the cohort it would measure barely exists — the text anchor rendered on **0 of 42** ranked-era
records (5/101 era-wide, all five FRED-mirrored series), so a chart arm has nothing to move before
the tournament closes. Revive after the 2026-08-03 routing repairs are shown to be lifting live
anchor coverage.

Beyond the text anchor's prose band, the chart side-channel passes each base model a rendered
800×400 PNG of the series + projected band as a vision message. **Skeleton shipped env-gated OFF**
(`TS_ANCHOR_CHART_ENABLED`, `'false'` in all four yamls): `research/ts_chart.py`
(`render_anchor_chart`, deterministic base64 PNG), a provider hook stashing the chart per-qid for
single LEVEL questions only (max-window/spread deferred), and forecaster plumbing threading the b64
→ `self._research_images[qid]` → the three runners via
`VisionMessageData(..., image_resolution="low")`. Every roster model was verified vision-capable via
OpenRouter (checked on the then-current 6-member roster); stacker/summarizer/gap-fill/parser never
see the image.

**Cost:** ~$0.02/question at low resolution — cheap enough to A/B but not free, hence gated
separately so the validated text anchor ships first.

**A/B needs:** arms bare / stats (text) / stats+chart on numeric-anchored questions, real read is
era-bucketed residuals on the level/spread cohort. **Prerequisite for archived-replay: research-sink
schema v3** — the archive stores only the text bundle; to replay chart arms offline the sink must
persist the chart b64 (or the band + series slice to re-render deterministically). Until v3,
chart-arm measurement is live-only. **Cheapest first signal — 3-arm smoke:** one hard resolved
level-series question, run a cheap model 3× (bare / stats / stats+chart), eyeball whether the image
moves the distribution beyond the text. Paid — gate behind operator sign-off.

### Reconsider claude-fable-5 for the Anthropic slots (forecaster + stacker) (added 2026-07-20; demoted from HIGH 2026-08-25)

**Demoted 2026-08-25:** the triple is FROZEN through the September checkpoint (plan step 1 of the
triple-era entry — roster churn restarts the era clock and re-inflates the between-era variance
floor), so a forecaster re-add is blocked by a standing decision, and the stacker half is
prod-disabled backtest/ablation exposure only. Both roles stay tracked; neither is actionable this
season. The two former entries are merged here — same model, same trigger, same decision.

Fable-5 held a forecaster slot from 2026-07-15 (replacing opus-4.6) and the primary stacker slot
from 2026-07-07, and was pulled from BOTH on 2026-07-20 after returning `message.content=None` on
4/4 attempts for Q14333's numeric forecast (the question was dropped, published 5/6) and a
truncated no-JSON-block output on Q578 that needed rung-3 LLM-parser salvage — both in the
2026-07-19 test_bot run (receipts: `scratch/gha_test_bot_2026_07_19.md`). Suspected cause: fable-5
content classifiers refusing certain question content, surfacing as fast deterministic empty
completions (NOT timeouts). opus-4.7 took the forecaster slot (xhigh, mirroring opus-4.8) and
opus-4.8 the stacker; the `gpt-5.6-sol` cross-provider stacker fallback is unchanged.
**Fable-5's forecast quality was never the issue** — this is a reliability/refusal problem, not a
capability one, and if the refusals are narrow/rare it remains the strongest available Anthropic
tier.

**Revisit when:** the roster freeze lifts AND root cause is confirmed (e.g. replay fable-5 against
the exact Q14333/Q578 prompts via a cheap manual call to see whether the empty completions
reproduce and isolate the content trigger), or provider behavior changes. Reconsider both roles
together.

### Score the archived gap-fill v2 ghost forecasts — harness DONE, first read is a null (added 2026-07-18; demoted from HIGH 2026-08-25)

**Demoted 2026-08-25:** the harness shipped and the first scored read is in, so there is no
buildable work left here; the next read rides the September checkpoint as scope item 4 of the
triple-era entry (free, `make score_ghosts`). The one remaining lever is a paid prompt change.

**What shipped.** The v2 loop privately dry-runs a forecast per question and archives it
(`archive_sink=_capture_gap_fill_v2`, `research/orchestrator.py`). The payload carries the full
structured ghost (`gap_fill_v2.ghost`, pydantic `model_dump`) alongside transcript + telemetry, a
`GHOST_FORECAST_JSON` single-line marker (full 13 numeric percentiles + full MC option probs) is
harvested into the telemetry archive, and `score_ghosts.py` prefers the JSON records and scores
numerics through the existing CDF machinery, with a regex fallback for the pre-upgrade marker era
(the legacy `GHOST_FORECAST` marker exposed only a numeric median, so numeric ghosts were countable
but not scoreable).

**Interpretation guardrail (still load-bearing).** The ghost is a same-model (terra-low driver)
counterfactual, NOT a panel proxy — it measures whether the v2 findings alone, forecast by one cheap
model, land near truth, not whether v2 would improve the ensemble. Its decision relevance is to the
gap-fill-v1-retirement call (gate in the "Bundle content-audit findings" entry): v1 carries the
decisive single-source fact in most sampled questions, so a ghost score is the cheapest read on
whether v2 findings stand on their own before v1 goes off.

**First scored read, 2026-08-24** (`scratch/residual_2026-08-24/dim_ghosts.md`). 94 archived
ghosts, 12 joined to resolutions, 12 scored: pooled ghost-minus-published delta **+7.02** (median
+1.23, bootstrap CI [−6.7, +24.0], sign test p=1.0) — a null at n=12 whose positive point estimate
is one MC question (+78.5). The composition guardrail matters more than the number: **7 of the 12
scored ghosts are byte-identical to the driver's PRE-research dry run** (and those 7 recorded zero
corrections), so the pool mostly measures the driver's prior, not v2's research — read the
loop-moved subset separately (n=4, mean +24.29; +6.21 without the outlier). `score_ghosts.py` emits
that pre/post split as of `171da89`. Two premises corrected: "diff the ghost rationales" is
unanswerable as written — ghosts are block-only BY DESIGN (`_GHOST_PROMPT` asks for the STRUCTURED
FORECAST block alone; prose outside the block is 0 chars in 80/80) — and the conclude-gate question
is answered: it has never fired in 93 prod runs and tightening it is the wrong lever (it runs
before the ghost and cannot observe integration). To answer the integration question rather than
bound it, the lever is `_GHOST_PROMPT` (ask the ghost to name which findings moved it) — a prompt
change, so paid validation, operator's call.

## Longer-term (significant R&D)

### Agentic deep research (ReAct loop) — SHIPPED, live in prod 2026-07-21

The gap-fill v2 plan (near-term entry; `scratch_docs_and_planning/agentic_gap_fill_v2_plan.md`) IS this —
a bounded tool-loop second pass, ON in prod (`GAP_FILL_V2_ENABLED: 'true'`). The old cost blocker is
resolved by budget caps (~$0.50/q) + early-stop; selective activation via the template dry-run. Direction
confirmed by the 2026-07-16 lit survey: keep it a **shared** stage (one loop → detached artifact all
forecasters read), NOT per-forecaster integrated pipelines (BTF-2 arXiv 2604.26106 — strong prompt on
shared research edged the best integrated agent; integrated benefit was Opus-class-only). Watch
era-bucketed calibration alongside Brier (agentic research trades calibration for over-decisiveness).

### Prediction market integration — SHIPPED (strong evidence, criteria/date-matched)

Structured Polymarket/Kalshi/Manifold/PredictIt access shipped as the prediction-market snapshot
(`PREDICTION_MARKETS_ENABLED: 'true'` in prod). Framing: markets are STRONG EVIDENCE, not a footnote. The
forecaster prompts anchor on a market whose criteria + date MATCH, discount proportionally to a specific
mismatch, and when only the DATE differs, extrapolate the market's probability to our date with a simple
model (constant-hazard / base-rate-over-time) rather than a vague haircut. Superseded the old "not beholden
to them" language after a referendum miss where the bot dismissed a market sitting at the correct answer.

## Resolved / shipped — evidence retained (do not re-recommend)

Items whose question is answered or whose code landed. Re-filed out of the priority sections in the
2026-08-25 priority audit so the live sections hold only open work; the evidence stays because it is
what keeps each one from being re-proposed.

### Grok drop — RESOLVED 2026-07-20, superseded by the latest-per-vendor triple (added 2026-07-19)

Grok was removed entirely as part of the drop to the 3-member latest-per-vendor triple (authored
2026-07-20, live in prod 2026-07-21T17:07Z) — not the incremental 6→5 step this entry originally
scoped. The re-add question (should any dropped member, grok included, come back?) lives in the
"Triple-era September re-read" entry; analyses in `scratch/ensemble_3member_audit_2026-07-20/` +
`scratch/ensemble_power_model_2026-07-20/`.

A paired leave-one-out replay on 2026-07-19 found that dropping grok from the 6-model roster
IMPROVES binary accuracy: Δlog-score **+1.83 [+0.74, +3.00]** favoring the drop (n=184) — but that
signal is entirely **grok-4.3-lineage** (grok-4.5, the slot at the time, had too few resolved
questions to score). The 2026-07-20 power model corroborated the direction on the modern lineage's
predecessors: grok read as a drag on binary (−2.4 log pts/Q) and MC (−1.3), and a help only on
numeric (+2.0), with every CI spanning zero. Grok's numeric help is the one reason a re-add could be
considered — exactly the numeric lean the triple-era re-read tracks. If grok is ever reconsidered,
era-bucket the read (do NOT pool grok-4.3 and grok-4.5 evidence); the parameterized replay
`scratch/residual_2026-07-18/followups/grok_loo_replay.py` is free to re-run (offline, no API).

### Parser hardening + forecasting-tools upgrade path — ALL WORKSTREAMS DONE (added 2026-07-07)

Standing decision from the plan (`scratch_docs_and_planning/parser_hardening_and_ft_upgrade_plan.md`):
do NOT migrate forecaster calls to native `response_format` structured outputs (OpenRouter
silent-degradation footguns, load-bearing rationale channel, zero competitive precedent).

- **Workstream A / block-as-authoritative — DONE 2026-07-10.** Value extraction runs the
  deterministic four-rung ladder in `value_extraction.py` (block parse → json-repair → LLM-parser
  salvage → `ValueExtractionError`) for all question types and the stacker. Shadow-divergence
  logging served its purpose and was deleted the same day (`EXTRACTION_RUNG` telemetry replaced it);
  strict json_schema on the *parser call* (`structured_parse.py`) is now the ladder's rung-3 salvage.
  The old "wait for ~50 questions of shadow-divergence data" trigger was operator-waived.
- **Workstream B / ft unfreeze — DONE 2026-07-23.** `forecasting-tools` 0.2.54 → 0.2.92 (litellm
  1.92.0, openai 2.x). Both verified breaks fixed: the PCHIP subclasses override `get_cdf()` — not
  just `.cdf`, which 0.2.92 made a deprecated shim delegating to it — so our CDF machinery can't be
  silently bypassed, and the `fetch_hardening` / `publish_hardening` patches retarget
  `MetaculusClient` (with a `__wrapped__` single-retry policy). Threading
  `strict_validation=False` / `standardize_cdf=False` on our numeric distributions preserves our
  beyond-range open-bound percentiles past `_check_too_far_from_bounds`; the MC clamp moved to
  `[0.01, 0.99]` (ft's validator bounds) via the drift-free `clamp_and_renormalize_probs`; and
  `required_successful_predictions=0.0` keeps our own `min_forecasters_to_publish` guard the sole
  publish arbiter.

### financial_data / ts-anchor: calendar time converted on a hardcoded trading-day density — FIXED 2026-08-24 (`e6ae276`), class swept 2026-08-25 (`c577231`)

Found by the q44882 dossier (Ethereum >$2,200 in August, peer −7.24) and verified by reproducing the
printed block to the last decimal: `research/financial_data.py` annualized daily returns with
`np.sqrt(252)` on assets that trade 365 days/year, understating crypto volatility by **1.2035x** —
the printed "30-day annualized volatility: 37.0%" was truly 44.6%, and "37%" was cited 18 times
across the three forecasts as the low leg of every model's barrier range (all three independently
derived ~33% from the Polymarket contract, then averaged it against the buggy 37% and published 25%
on a YES). Two siblings in the same file: `_compute_period_returns` labelled row offsets as calendar
periods (crypto "1m" read +5.05% where the true 30-calendar-day move was +17.24%, contradicting
AskNews's figure in the same bundle; "1y" understated the drawdown by 17pp), and the "52-week range"
was `iloc[-252:]` ≈ 8.2 months (high understated 31%). Archive blast radius 2 of 1,064 records — ETH
here, where it hurt, and BTC on q43592, where the same understatement pushed toward a NO that
resolved NO and helped: a systematic under-dispersion bias that flatters status-quo answers and pays
for it when the status quo breaks.
(`scratch/residual_2026-08-24/dossiers/44882_dossier.md` + `44882_verification.md`.)

**Fix.** `e6ae276` infers periods-per-year from the series' own observed density (rows per calendar
day, split at 6/7 — a newly listed 24/7 symbol needs no registry edit) and drives the vol factor,
the period-return row offsets and the 52-week slice off that one basis; business-day series are
byte-identical to the old output (pinned by tests), `TestAnnualizationBasis` carries the
365-frequency fixture, and the revert check fails exactly the 24/7 test. `c577231` then swept the
class: four more calendar↔row conversions lived in the ts-anchor stack — `horizon_steps` (a 90-day
question became a 62-step horizon on a 24/7 series, a band ~20% too narrow), its inverse
`_horizon_end_date` (wrong by the same factor in the opposite direction, so the two CANCELLED and
had to move together), `_realized_vol_line`'s second `sqrt(252)` copy, and `_FREQ_UNIT`'s hardcoded
"trading-day" noun. The density read now lives once, as `ts_estimators.observed_periods_per_year`
with a `SeriesClock` carrying frequency AND density (`_detect_freq` reads the median day-gap, which
is 1.0 for a business-day series as well as a 24/7 one, so frequency alone is blind to a 1.45x
difference in rows per year). `mom_diff`/`mom_pct` now raise on a non-monthly source instead of
publishing a week-over-week change under a month-over-month label. Root-cause lesson worth keeping:
`e6ae276` kept the density read private to `financial_data` and a later commit hoisted only the two
CONSTANTS "so a correction can't miss a copy" — the constants were shared, the read that picks
between them was not.

### Open-bound out-of-range mass — EXERCISED AND PASSING (recorded 2026-08-24)

The prior round's ledger carried the W1 parser-unclamp (percentile values beyond open bounds are no
longer clamped, so `F(bound)` can express large out-of-bound mass) as still unexercised in a
resolved question. It fired, and it paid: **q44218** (US Strategic Petroleum Reserve) published
`cdf[0] = 0.9168` — 91.7% of mass below the open lower bound, because all six models' medians sat
at 265k–285k against a 300k displayed floor — the resolution came in at 293,426, below the bound,
and scored **peer +75.8**: the exact counterfactual of the 43746/43747 `known_bug` pair (`cdf[0]`
pinned at 0.0250, peers −77.8 / −66.2). Four of the nine post-fix open-bound records now have a
model declaring beyond a bound, and each published tail moved off 0.01/0.99 to express it. Two
honest limits: q44218 predates the W3 label widening, so it demonstrates the *pipeline's* ability
to carry below-bound mass rather than the W1 parser change specifically; and no post-fix record has
yet resolved beyond a bound in the direction its models declared. Side effect fixed the same round:
expressible out-of-bound mass broke grid-clamped PIT interpolation (the out-of-grid censoring
defect, `1fe96c9` — see the numeric-width history entry).
(`scratch/residual_2026-08-24/dim_numeric-width.md` §4.)

### One-liners

- ~~**Supervisor agent for high-disagreement questions**~~ — shipped as
  `AggregationStrategy.CONDITIONAL_STACKING`, whose stacker half has been prod-disabled since
  2026-05-29 (all workflow yamls pin `*_STACKING_ENABLED=false`), so it falls through to MEDIAN.
- ~~**Financial data tool access (yFinance, FRED)**~~ — shipped as `research/financial_data.py` (the
  entry's old `financial_data_provider.py` name predates the subpackage move).
- ~~**`OAI_ANTH_OPENROUTER_KEY` data-policy block for OpenAI native search**~~ — RESOLVED
  2026-05-29 when Metaculus enabled OpenAI on the donated key. `build_native_search_llm` routes
  through `build_llm_with_openrouter_fallback` (donated primary, personal fallback); verified
  end-to-end (grounded result, 404 fallback count = 0). The original block was a 404 "no endpoints
  matching your guardrail restrictions and data policy", and that matcher stays in
  `fallback_openrouter.py` as a safety net for the next provider migration.
- ~~**Second-pass web search + scrape pipeline**~~ — SUPERSEDED 2026-07-16 by agentic gap-fill v2.
  All three use cases (gap-filling, resolution-source reading, reopening inaccessible
  PDF/JS/paywalled sources) are covered by the v2 tool loop; Firecrawl/Olostep were rejected in
  favour of the DIY fetch ladder (plain → headless Chromium → Gemini url_context).
- ~~**Harden `BoundSafeNumericDistribution.cdf` fallback for coarse grids**~~ — DONE 2026-07-20 with
  the discrete-hardening pass: it computes `grid_step_constraints(len(base))` and threads the
  grid-scaled min/max step into `safe_cdf_bounds`, so the fallback matches the pipeline's resample
  path on a coarse discrete grid instead of clipping every bin to the 201-grid `max_step=0.2`.
  Regression test `tests/test_thirteen_percentile_e2e.py::TestFallbackCdfRespectsOpenBounds::test_fallback_coarse_grid_uses_grid_scaled_constraints`.
- ~~**Bundle section-content audit before any content cuts**~~ — DONE 2026-07-18. Operator
  directive was no willy-nilly trimming; a Fable-judged per-section value/redundancy audit is the
  prerequisite for any cut. Tokens `scratch/bundle_token_audit_2026-07-17/`, audit
  `scratch/bundle_content_audit_2026-07-17/RESULTS.md`; surviving follow-ups live in the "Bundle
  content-audit findings" entry.
- ~~**Summarizer model: bench sol-low vs terra-low**~~ — DECIDED 2026-07-18, switched to terra-low.
  The 2026-07-17 role audit kept sol (best synthesis/provenance; terra 2nd with one attribution
  blur, gap "MARGINAL EDGE"), but an operator value-call overrode it: AskNews is auxiliary (16%
  unique content vs native-search 54% / gap-fill 59%) so the frontier tier isn't warranted, and the
  AskNews quality audit (`scratch/asknews_quality_audit_2026-07-18/`) blamed 4/5 briefing failures
  on prompt-era issues rather than model tier. Terra: −43% cost, ~50s vs ~118s. Packets
  `scratch/research_role_audit_2026-07-17/`.

## Killed by May 2026-05 closing analysis

These were investigated and either failed to replicate or were superseded. Listed
here so future sessions don't re-recommend them without new data.

- **Drop or down-weight gemini-3.1-pro (binary axis)** — April rec #6 did not
  replicate at May n=62 (+0.024 vs ensemble, below threshold; net contributor on
  hits cohort 5 best vs 4 worst). Numeric weakness exists (rank 7/8) but is solved
  by per-type weights (deferred), not removal.
- **AI-capability "private preview is a leading indicator" prompt edit** —
  April rec #7 was N=1 (Anthropic Opus miss 43131); May still N=1 (same question).
  Don't ship a prompt edit on a single observation.
- **Time-of-tournament Brier rolling-avg analysis** — confounded by mid-tournament
  roster swaps (gpt-5.2 → 5.4 → 5.5; opus-4.5 → 4.6 → 4.7; gemini-3-pro → 3.1-pro).
  Cannot isolate bot drift from roster changes.
- **`nr_forecasters` difficulty stratification** — peer score already normalizes for
  difficulty better than `nr_forecasters` (which is more correlated with question
  attention/age than difficulty).
- **3D calibration grid (predicted-bucket × type × stage)** — ~132 cells with
  ~0.7 questions average. Noise dominates. Stick with the 1D predicted-bucket cut.
- **MC per-model audit** — per-model MC predictions don't survive in stored
  comments. Would require collector changes for n=24/round. Not worth it.

## Killed by July 2026-07-08 residual + competitor analysis

- **YES-side shrink / any fitted directional calibration layer** — the YES-overconfidence finding is
  real but era-local (spring-only; fall confident-YES ~100% accurate). All three stability criteria failed;
  a two-era fit degraded held-out fall +4.3e-3 Brier. Surviving hard rule: symmetric shrink was strictly
  worse in every era — never touch the NO side. `scratch/residual_2026-07-08/experiments/ARM_B_FINDINGS.md`.
- **Anchor-guard as a clamp/gate** — models publish outside their own base-rate anchor on ~88% of forecasts
  (normal outside→inside update); "flag it" gate precision ≤29% and sign flips across eras. Surviving
  fragment: overshoot MAGNITUDE >15pp degrades Brier monotonically in both well-powered eras → ship as
  telemetry + prompt nudge only (telemetry shipped 2026-07-08). `.../ARM_A_FINDINGS.md`.
- **Advisory (non-binding) critic passes** — GreeneiBot2's critique diagnosed our exact 41800 defect and the
  number never moved; laertes's correct 42855 recomputation was half-overridden. BINDING corrections
  (formula/floor/structural) captured 30–130 spot-peer points, advisory ~0. The critic-pass entry stays but
  binding-output only. `scratch/residual_2026-07-08/ACID_TEST_VERDICT.md`.
- **Fixed-direction haircuts** — GreeneiBot2's always-downward haircut helped 42855 but harmed 41800
  (extremized 12→10 on a YES). Any damping must push toward 0.5, not a fixed direction (mirrors the killed
  one-sided anti-overprediction shave).
- **"Median drowns the correct dissenter"** (re-confirmed dead) — 7/9 washouts framing is survivorship bias;
  median stays at the non-oracle frontier across two more pulls; even laertes's 41800 "win" discarded its own
  best member and was saved by its floor, not aggregation.
- **Anchor-floor guard on cheap tails** — median-band variant sign-flips (fall +1.68 / spring −2.19 total
  Brier at 10pp) because 76% of parsed anchor bands are degenerate points; union-band variant is
  sign-consistent but 100% top-5-concentrated (1–2 Qs/era) and catches 1/5 of known misses; same-side tail
  clamping hurts both well-powered eras. Revival was: ≥50 current-roster binaries with
  `base_rate_anchor` telemetry (`30bca2f`, live on main 2026-07-11T16:37Z in `642b027`) AND an
  era-stable, top-5<50% replay. **That condition can no longer accrue as of 2026-09-02**: the
  2026-09 bundle removes the `base_rate_anchor` elicitation from the binary prompt (Item D of
  `scratch_docs_and_planning/announced_unscheduled_fix_plan_2026-09-02.md`), so the count is frozen
  at about 40 triple-era binaries. Reviving this guard now starts with re-adding the field, which is
  a prompt change and a config-era boundary; see the closed "Telemetry-first guard revival program"
  entry. Note separately that the anchor-overshoot variant of the idea was measured and rejected on
  its own evidence in 2026-08-31 (its entry is beside the numeric tail-consistency check).
  `.../GUARD1_FINDINGS.md`.
- **No-market-no-extremize cap** — feasibility kill: market-presence signal exists in ~one era (fall 89%
  NO_SIGNAL, `## Prediction Market Snapshot` in exactly 1 archived binary), so era-stability un-testable;
  marketless fallback = Arm A's point-anchor drift bomb; strict form's gain is 3 spring questions and the
  gate itself adds ~nothing. Revival: ≥~50 binaries/era carrying the structured snapshot AND the market gate
  earning the delta. `.../GUARD2_FINDINGS.md`.
- **Signed deadzone haircut toward 0.5** (closed permanently) — extends the Arm B symmetric-shrink kill to
  thresholded/high-t forms: 0 of 24 (λ,t) cells help in both well-powered eras; fire rate ≥30% even at t=0.4
  (confident bot, median |p−0.5|≈0.32–0.38); LOTO on spring+summer degrades fall +1.12 Brier. Keeper: the
  signed-toward-0.5 direction constraint is a design rule. Revival: calibration profile qualitatively changes
  AND leave-one-era-out passes everywhere; don't re-grid λ/t absent that. `.../GUARD3_FINDINGS.md`.
- **Cross-cutting** — at mid-grid all three guards fire on the same spring miss cluster (42024 / 42304 /
  41800; often 42018 / 42577 / 42855 / 41644) — one lever measured three ways. The shipped `30bca2f` prompt
  changes target that cluster, so any future guard replay must never fit on pre-`30bca2f` eras — and the
  cut is at its MERGE date, 2026-07-11T16:37Z (`642b027`), not its 2026-07-08 authoring date, or three days
  of pre-change runs land on the wrong side (the roster-drift bomb with a prompt-change fuse; config-era
  bucketing already handles this).

## Killed / evaluated 2026-07-11 (blind-forecaster review + crowd-signal audit)

Context: blind judge pass over the 2026-07-11 `test_bot` run (`scratch/test_bot_july_11_2026 .md`) + a
live-API audit of surfacing crowd-signal informativeness in research packets.

- **Metaculus `similar-posts` endpoint as a research provider** — KILLED. Live audit, 23 source posts / 160
  rows (`GET /api/posts/{id}/similar-posts/`). Two dealbreakers: (1) community-prediction VALUE is `null` for
  our bot account on 160/160 rows; (2) returns ONLY open questions (160/160 unresolved) so the base-rate
  payload is absent. What remains (title + `nr_forecasters` + un-followable link) is decorative; match quality
  bimodal (good AI/US-politics/geo, garbage weather/sports/non-US — a Boston-Marathon source returned "World
  Bog Snorkelling" + a Zambian election), pads to 8 with no relevance score, rate-limits ~10 calls → 429.
- **Resolved-sibling base-rate lookup** (`?search=<kw>&statuses=resolved`) — general version KILLED (~110-call
  audit): **Metaculus null-outs `question.resolution` (+ description/criteria/CP) on every post this bot didn't
  itself forecast** (view-level AIB anti-cheating; no bypass via `with_cp`/`minimize`), so the Metaculus-wide
  resolved corpus is unreadable to our token. Endpoint mechanics work (`statuses=` plural; `actual_resolve_time__lt`
  + `order_by=-scheduled_resolve_time` silently broken). **Salvage (optional, LOW):** a
  `forecaster_id=275109&statuses=resolved` self-history lookup over the bot's own ~770 resolved posts —
  AIB recycles templates so sibling quality is excellent for recurring indicators / event-window binaries,
  naive title-as-query works, and it's backtest-MEASURABLE (date-filter `actual_resolve_time < open_time`, not
  a hard `is_benchmarking` disable). Ranked low — the indicator series are already pulled by the
  resolution-source + financial-data providers; unique value is structured prior-outcomes for repeated
  event-window binaries.
- **Crowd-signal informativeness surfacing** — WORTH DOING; market-liquidity half SHIPPED (snapshot now
  renders thin/decent/deep labels from total volume / OI / `uniqueBettorCount`, which the fetchers previously
  discarded — the old `vol`=24h-volume number was ~always ≈0 for long-horizon questions, misleading).
  `nr_forecasters` is free on the fetched object (`MetaculusQuestion.num_forecasters`, zero extra HTTP).
  Proposed labels: Metaculus <30 thin / 30–49 decent / 50–99 good / ≥100 high; real-money <$5k / $5k–50k /
  >$50k; Manifold bettors <20/20–100/>100. Aggregators SKIPPED: Metaforecast shut down; Adjacent News
  redundant (Kalshi+Polymarket only); PredictIt an optional politics-only price-only add; PMXT only if venue
  breadth becomes binding.
- **Open-bound tail-cramming on discrete/open-upper numeric** — FIXED (see the `OPEN_BOUND_PILING` telemetry
  + nominal-bounds prompt render, CLAUDE.md). gpt-5.6-sol / gpt-5.5 crammed ~20% / ~12.6% of mass onto the top
  displayed bin of Q38195 because "Respect the explicit bounds" read as a hard cap; both OpenAI models failed,
  Claude/Gemini/Grok handled it. Fix: prompt clarification + a WARN-only boundary-piling detector into GHA
  artifacts.

## Instrumentation bugs

**All three from the May 2026 closing analysis are FIXED (2026-05-10; 1187 tests pass).**

1. ~~STACKER_OUTCOME tri-state marker~~ FIXED — producer sets `_stacker_outcome[qid]` (primary /
   fallback_llm / fallback_median / skipped) at the END of each path; `_create_unified_explanation` emits
   the tri-state `STACKER_OUTCOME=` + legacy `STACKED=` markers (median-fallback previously silently emitted
   `STACKED=true`).
2. ~~Targeted-research header missing from comments~~ FIXED — `main.py:839` returned
   `research_report=research` not `combined_research` on the conditional-stacking branch; one-line fix.
3. ~~`audit.py::emit_synthesis` KeyError on numeric-mixed cohorts~~ FIXED — type-aware skip for non-binary
   entries in the spread section (`_rank_numeric` produces no `prob` key).

### New parser feature shipped — historical-header-aware detection

`parse_inferred_stacker_outcome` detects three older stacker-output body signatures (`## Stacker
Meta-Analysis`, `## Meta-Analysis`, `# Meta-Analysis and Synthesis`) plus the tri-state + legacy markers.
Unlocked the May 2026 stacker treatment-effect estimate (`analysis_stacking_historical_treatment.md`) —
first measurable signal (n=8 stacker-ran, −0.090 Brier vs counterfactual, P(helped)=89.8%).

### GitHub-runner egress reputation is the dominant cause of resolution-source 403s (added 2026-09-03; LOW priority by operator decision)

The archived `blocked` fetches on bls.gov, cdc.gov and fsis.usda.gov (Akamai edges) do NOT
reproduce from the operator's laptop or EC2 box with the bot's own aiohttp session and headers;
only the GitHub Actions runner gets 403 (`scratch/fetch_ladder_2026-09-03/replay_report.md`,
plan `scratch_docs_and_planning/fetch_ladder_plan_2026-09-03.md`). The free GHA diagnostic in
that plan's step 0 decides whether TLS impersonation helps from the runner. If it does not, the
only structural fix is an egress that is not a GitHub runner (an HTTP proxy on the operator's EC2
box, or a self-hosted runner). Operator 2026-09-03: "probably too complicated" — park here; the
Wayback and url_context rungs cover the class in the meantime. The same runner-side cause is
consistent with GitHub cron gaps (q45092 forfeited 2026-09-01: the 00:05Z fire saw 0 questions
and the next fire was 04:55Z, past a 2.8-hour window).
