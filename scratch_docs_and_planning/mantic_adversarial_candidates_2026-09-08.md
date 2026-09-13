# Mantic (Crucible): further adaptations for the adversarial question-writing setup

Research pass, 2026-09-08 (evening, Pacific). Read-only: no code, test, doc or config edits, no git
writes, no paid calls, no network beyond the local corpus. Branch `mantic-competition`, tip e37b8ae.

## Context for a zero-context reader

This repository is a Metaculus forecasting bot. For each open question it gathers research, runs
three frontier language models in parallel, aggregates their forecasts (the median; numeric and
date questions are aggregated pointwise in cumulative-distribution-function, CDF, space) and
publishes the result. Branch `mantic-competition` adds a second platform: Mantic's Crucible
tournament at competitions.mantic.com, a fork of the Metaculus platform that pays $3 per forecast
and is bot-only. Mantic's question writers are paid for the disagreement they induce between bots,
set the numeric and date ranges themselves, and in Series 1 about half of the date questions and a
fifth of the quantitative ones resolved outside the displayed range. Mantic scores an out-of-range
resolution against a fixed 5% reference (50 times the natural log of the mass placed there divided
by 0.05), so a forecast with 1% in that bucket scores -80.5 points where 5% scores 0.

Phase 1 (platform client, personal-keys-only policy) and Phase 2 (first-class date questions,
Mantic-optimised numeric fixes, Mantic-gated prompt clauses, telemetry, three crons) are built and
committed. Wave C (per-bin elicitation, a Mantic supply probe, fast-path alertability, a
median-versus-mean replay, the post-2026-09-20 reads of question 651 and of one resolved Series 2
score, the starved-outer-tail watch) is being designed by a separate agent. This document covers
what is left: candidates that are NOT built, NOT in Wave C, and NOT on the "accepted with no
change" list in `scratch_docs_and_planning/mantic_phase2_plan_2026-09-08.md`.

Sources mined: the ranked 25-item edge-case review and its raw hunter findings (five agents, 55
findings, workflow journal `wf_c25a69ef-230`), the Series 1 and Series 2 rules documents, Mantic's
OpenAPI spec and its diff against Metaculus, and the recorded corpus of all 556 public Mantic posts
(`mantic_research_2026-09-08/mantic_all_posts_2026-09-08.json`), which carries every competitor's
full published distribution on every resolved Series 1 question. Every number below labelled
"verified" was computed this session from that corpus or read from the code; anything else is
labelled as an assumption.

One method note that several findings rest on: I reimplemented the platform's resolution-to-bucket
mapping from the Metaculus backend source cached at `/tmp/mantic_probe/formulas.py` (the function
`unscaled_location_to_bucket_index`) and its baseline score formula, and reproduced Mantic's own
reported `spot_baseline_score` on 465 of 466 resolved Series 1 continuous questions to within 0.5
points. The one mismatch was not investigated. That reproduction is what lets the corpus stand in
for live scoring below.

## Ranked candidates

| Rank | Candidate | Expected impact per affected question | Share of Mantic questions affected | Recommendation |
|---|---|---|---|---|
| 1 | Mantic-only out-of-range tail floor at 5% (flip the HELD decision now) | +11 (quantitative) to +42 (date) if our tails stay at the structural 1%; 0 if the models already place mass beyond; worst case about -2 | every question with an open bound: 97% of date, 89% of discrete, 90% of numeric | BUILD NOW, pending operator go |
| 2 | Correct the shipped out-of-range base-rate wording (corpus accounting) | small; keeps the prompt's stated base rate honest (numeric is one in eight, not one in fifteen) | same population as rank 1 | BUILD NOW, folded into the prompt edit |
| 3 | Scaled-unit conversion example ("350B -> 350000000000") is wrong when the base unit is billions | tail risk: a member 1e9 times too large publishes ~98% above the ceiling, about -220 if the truth is in range; median-of-three absorbs one such member | 18 of 315 quantitative questions (5.7%) carry million/billion units | BUILD NOW |
| 4 | Scoring-grid sentence renders `precision` as a width on log-scaled quantitative grids | small; a false statement about bin width on log grids | about 9% of numeric questions were log-scaled in Series 1 | BUILD NOW, lowest rank |
| 5 | Day-bin edge convention: Mantic's resolver stamps calendar-date outcomes at 00:00 UTC 27% of the time, which the bucket formula scores in the previous day's bin | if the midnight convention held, every date-only forecast would be one bin late (near the floor); the corpus argues Mantic must intend the inside-day convention we use | every day-granularity date question (38% of supply is date) | BUILD AFTER LIVE DATA (this is Wave C item C5, sharpened) plus one free operator action |
| 6 | Starved outer tail: field evidence for Wave C item C6 | -225 on 5.2% of in-range date outcomes for the field (about -12 per in-range date question) | in-range date questions | BUILD AFTER LIVE DATA (already C6; evidence only) |

## 1. Mantic-only out-of-range tail floor at 5% (flip the HELD decision)

**Failure mode.** When all thirteen declared percentiles sit inside the displayed range, the built
CDF puts exactly 1% beyond each open bound (`numeric/pchip_cdf.py` clamps the evaluation grid to
the declared extremes; `docs/performance_analysis.md` records the 1% as structural). Mantic scores
that bucket against a 5% reference: 1% scores -80.5, 5% scores 0. Hits numeric, discrete and date
questions with an open bound. Phase 2 shipped the prompt clauses and the `oor_low` / `oor_high`
telemetry and HELD the mechanical floor until live telemetry shows whether the models place mass
beyond the bounds on their own.

**Evidence (all verified this session unless marked).**

- Out-of-range resolution rates, annulled questions excluded, Series 1: date 102 of 192 (53.1%),
  discrete 35 of 141 (24.8%), numeric 16 of 133 (12.0%), quantitative combined 51 of 274 (18.6%).
  The numeric figure is higher than the 6.6% the edge review used because Mantic stored seven
  numeric escapes as raw values outside the range rather than as `above_upper_bound` /
  `below_lower_bound` (posts 512, 460, 426, 396 and 305 on linear grids; 387 and 200 on
  log-scaled grids). The platform scored the five linear ones in the out-of-range bucket: my
  reimplementation reproduces their reported scores to 0.01 points.
- The rate rose as writers learned. Discrete out-of-range by post-id quartile: 10%, 34%, 24%, 28%.
  Date: 59%, 60%, 38%, 52%. Per-writer rates run from 0% to 65% (n about 20 each). Series 2's
  top-k writer scoring rewards exactly the high-variance strategies that produce escapes.
- Field counterfactual, the new input for the decision: applying a per-open-side tail floor to
  each of the eleven competitor bots' OWN published distributions on all 4,082 Series 1 forecasts
  (rescaling the in-range mass to keep the sum at one) and rescoring with the platform formula:

  | Floor | numeric mean delta | discrete mean delta | date mean delta | per-bot range |
  |---|---|---|---|---|
  | 2% | -0.07 | +3.45 | +1.85 | -0.4 to +7.6 |
  | 5% | -1.88 | +4.71 | +2.91 | -1.4 (SynapseSeer) to +9.1 (Panshul42) |
  | 10% | -5.59 | +4.24 | +3.22 | -3.5 to +5.7 |

  The floor only binds on the 3% to 8% of forecasts where the bot had under 5% beyond a bound and
  the outcome escaped, so the per-question average understates what it does for a thin-tailed
  bot. The two bots it helped most (Panshul42, AtlasForecasting-bot) have median in-range tails of
  3.1% and 0.6%; the two it cost at 5% (SynapseSeer -1.4, smingers-bot -1.0) run 5.3% and 4.9%.
  Our bot's structural 1% is thinner than every competitor's median.
- What the field actually does with tails on questions that resolved IN range: median total
  out-of-range mass 26% on date questions, 2.0% on discrete, 4.2% on numeric. The top bot by mean
  baseline score (preseen, +49.9 per question) runs a 2.15% median tail but 61% median mass on the
  realised bucket when a question escapes, so it reads the escape rather than flooring blindly.
- Our own archive (Metaculus): 96% of per-model declarations put zero percentiles beyond an open
  bound (edge review, verified from the hunter's measurement, not re-run here). Whether the Phase 2
  base-rate clause moves that on Mantic is unknown: this is the assumption the HOLD rests on.

**Impact arithmetic.** For a bot at the 1% floor with both bounds open, moving to 5% each side
gains 50 ln(5) = 80.5 when the outcome escapes and costs 50 ln(0.90/0.98) = -4.26 when it does not
(one open side: -2.0). Quantitative at 18.6%: 0.186 x 80.5 - 0.814 x 4.26 = +11.5 per question.
Date (open upper, 53.1%): 0.531 x 80.5 - 0.469 x 2.0 = +41.8 per question. Weighted by Mantic's
supply (38% date, 30% discrete, 26% numeric) that is about +22 per question if our tails stay at
1%, about +11 if the prompt clause fixes half of it, and exactly 0 if the models already place at
least 5% beyond (the floor is inert; it is a maximum, not an override of fat tails). The downside
is bounded by the field counterfactual: no bot type lost more than 1.9 points per question on
average at 5%.

**Mechanism.** In `metaculus_bot/numeric/utils.py`, `_pin_endpoints` (lines 50-60) pins the
aggregate CDF's open sides at 0.001 / 0.999. On a Mantic question (read via
`metaculus_bot/question_platform.py`, whose import from `numeric/` no import-linter contract
forbids; the date adapter carries `page_url`, verified in `numeric/date_axis.py`) pin them at
`MANTIC_OPEN_TAIL_FLOOR` / 1 - `MANTIC_OPEN_TAIL_FLOOR` instead, rescale the interior linearly
between the new endpoints, and let the existing min- and max-step repair run as it does now. The
constant lives in `metaculus_bot/numeric/config.py`. The per-member `oor_low` / `oor_high` fields
on `MEMBER_FORECAST` keep measuring the models' unfloored behaviour, and the `NUMERIC_AGGREGATE`
marker shows the floored publish, so the telemetry the HOLD was waiting for still arrives. Pins:
a Mantic open-bound fixture publishes exactly the floor when every percentile is inside and is
untouched when a member already exceeds it; a Metaculus fixture is byte-identical; question 651
(both bounds closed) is unaffected.

**Cost and blast radius.** About ten lines plus a constant and three pins, in the aggregate numeric
build path, which is not the timing or fallback surface. Metaculus behaviour cannot move (platform
gate). No Mantic question has been forecast except the one paid smoke on post 650, so there is no
Mantic era to break. It reverses a recorded HOLD, which is why it is an operator decision rather
than a unilateral build.

**Recommendation: BUILD NOW, pending operator go.** The downside is bounded at about two points per
question on the field's own forecasts, the upside is the largest single lever on the platform, and
waiting for live telemetry means paying the 1% tail on the first Series 2 questions to learn what
the Series 1 field already shows.

## 2. Correct the shipped out-of-range base-rate wording

**Failure mode.** `_MANTIC_OUT_OF_RANGE_RATE_QUANTITY` in `metaculus_bot/prompts.py` (line 1215)
tells the model "about one in seven past quantitative questions resolved outside the displayed
range (one in five of the discrete ones, one in fifteen of the continuous ones)". The corrected
corpus figures are one in five combined (51 of 274), one in four discrete (35 of 141) and one in
eight numeric (16 of 133). The date sentence ("101 of 188 with an open upper bound") is right as
written for its population.

**Evidence.** Verified: see the bucket reproduction in candidate 1. The undercount comes from seven
escapes stored as raw values, which a filter on the `above_upper_bound` / `below_lower_bound`
strings misses.

**Impact.** The clause exists to set a base rate; understating the continuous rate by nearly half
weakens the one sentence built to fix the platform's largest lever. Order of magnitude small.

**Mechanism.** Edit the constant and its receipt comment (`prompts.py` lines 1200-1221), the
presence pins in `tests/prompts/test_platform_and_mantic_clauses.py`, and the "22% of resolved
discrete questions, 7% of numeric" line in `docs/operations.md` ("Mantic-optimized forecasting").

**Cost and blast radius.** Wording only, Mantic-gated clause. **Recommendation: BUILD NOW**, in the
same prompt edit as candidates 3 and 4, because a stated base rate should be the measured one.

## 3. Scaled-unit conversion example is wrong when the base unit is billions

**Failure mode.** The numeric prompt's Units & Bounds bullet says "If your reasoning uses
billions/millions/thousands, convert to base unit numerically (e.g., 350B -> 350000000000)"
(`metaculus_bot/prompts.py` line 1323; the same sentence in the parse notes at
`metaculus_bot/forecaster_runners.py` line 126 and in the stacking prompt at `prompts.py` line
1881, which production does not use). When the question's base unit is itself "$ billions", the
correct output for 350 billion dollars is 350, and a model that follows the example emits
350000000000. The unit-mismatch guard in `metaculus_bot/numeric/validation.py` (lines 80-160)
fires only on declarations that are TINY relative to the range (thresholds 1e-5 and 1e-8), and the
edge review's accepted-no-change list records that a symmetric guard cannot be built because a
nearly-all-out-of-range declaration is the correct answer on a quarter of discrete and half of
date questions. So the too-large declaration publishes as about 98% above the ceiling. Hits
numeric and discrete questions with scaled units.

**Evidence.** Verified: 18 of 315 Mantic quantitative questions (5.7%) carry a million/billion unit
string (`$B`, `$ billions`, `Millions barrels of oil`, `billion JPY`, `millions of AUD`, ...);
`unit_of_measure` is populated from the API's `unit` field (forecasting-tools
`data_models/questions.py` line 178). The prompt already shows the base unit and the displayed
range next to the example. Assumed, not measured: how often a model follows the example against the
stated unit. No Mantic run has produced this failure; no Metaculus receipt for the example exists
in `docs/prompts.md` (it is a formatting rule the parser needs).

**Impact arithmetic.** One wrong member is mostly absorbed by the median of three; it takes two.
If each member misapplies the example with probability m on such a question, P(two or more of
three) is about 3m^2. At m = 0.1 that is 3% of the 5.7% share times a swing of about 260 points
(-220 at the floor versus about +40 for a sane forecast) = -0.4 per question averaged over the
tournament; at m = 0.3 it is about -4. A tail risk with a free fix.

**Mechanism.** Reword the sentence in all three places to name the base unit: "convert to the base
unit named above: 350B is 350000000000 when the base unit is dollars, and 350 when the base unit
is billions of dollars". Keep the date-prompt absence pin (`tests/prompts/test_date_prompt.py`
line 89 asserts "350b" is absent there) and add a presence pin for the numeric prompt. This
sentence is shared with the Metaculus numeric prompt (one template), so it is a shared wording
change of the kind the operator approved for Phase 2 inside the config-era window.

**Cost and blast radius.** Wording only; no branch, no flag. **Recommendation: BUILD NOW.** The
example is false as written on one Mantic question in eighteen, and the guard that would catch the
consequence cannot exist.

## 4. Scoring-grid sentence renders `precision` as a width on log-scaled grids

**Failure mode.** `_scoring_grid_clause` in `metaculus_bot/prompts.py` (lines 1284-1303) renders
"Scoring grid: N bins of width {precision} {unit}" whenever the question carries `precision`.
Mantic's OpenAPI spec defines `precision` as "additive bin width on a linear scale, or the ratio
between adjacent boundaries on a logarithmic scale". On a log-scaled quantitative question the
sentence would read, for example, "bins of width 1.05 USD" on a range of 1,000 to 10,000,000,
which is false and tells the model the grid is absurdly fine.

**Evidence.** Verified: the spec text (`mantic_research_2026-09-08/spec.diff`); the clause does not
read `zero_point`; 14 of 156 Series 1 numeric questions were log-scaled (`zero_point` 0.0), none
with `precision` set because the field postdates Series 1; Series 2 fuses numeric and discrete into
`quantitative` with a step always defined (rules doc section 6). Assumed: Series 2 log-scaled
questions will carry the ratio in `precision` as the spec says; there is no live example yet.
`FUTURE.md` already flags that the plateau cap's linear bin width is untested on a log-scaled
discrete grid; this is the prompt-side sibling of that note.

**Impact.** Hard to price; a wrong grid statement pushes toward over-sharpness or confusion on
about 9% of numeric questions if Series 2 keeps Series 1's share of log grids. Small.

**Mechanism.** Inside the same clause, when `view.zero_point is not None` render "Scoring grid: N
bins whose edges grow by a factor of {precision:g} (logarithmic grid)"; add one pin to
`TestScoringGridClause` in `tests/prompts/test_platform_and_mantic_clauses.py`. A wording variant
of an existing clause, not a new rule.

**Cost and blast radius.** Trivial, Mantic-gated (the clause renders nothing without `precision`).
**Recommendation: BUILD NOW, lowest rank.** It corrects a sentence that will be false on Series 2
log grids; there is nothing to wait for.

## 5. Day-bin edge convention (Wave C item C5, sharpened with corpus evidence)

**Failure mode.** Phase 2 maps a forecaster's date-only value ("2026-09-16") to 12:00 UTC so the
mass lands strictly inside that day's right-closed bin (`metaculus_bot/numeric/date_axis.py`,
`_DATE_ONLY_HOUR_UTC = 12`; pinned in `tests/test_numeric_date_axis.py` lines 179-190). The
platform's bucket formula (`unscaled_location_to_bucket_index`, cached at
`/tmp/mantic_probe/formulas.py`) maps a resolution at exactly 00:00 on day D to the bin ENDING at
D, that is, the previous day's bin, because it computes `int(u * N + 1 - 1e-10)`. So the bin a
date question scores depends on the time of day Mantic's LLM resolver stamps on the resolution.
Hits every day- or week-granularity date question (post 651 is the first).

**Evidence.**

- Verified: my reimplementation of that formula reproduced the platform's own scores on 465 of 466
  Series 1 continuous questions, so the edge semantics are the live ones.
- Verified: of the 90 Series 1 date questions that resolved in range, 29 resolutions are stamped
  12:00:00 UTC, 24 are stamped 00:00:00 UTC and 37 carry a specific time from the question. The two
  groups are not separated by phrasing: pure calendar-date events appear in both (a Trump-Putin
  call, a North Korean missile test and a Trump executive order at midnight; a bill becoming law, a
  Haitian election postponement and a memorandum release at noon). The resolver is inconsistent.
  On Series 1's 200-bin grids a midnight stamp almost never fell on a bin edge, so it cost nothing;
  on a daily grid every midnight stamp is an edge.
- Verified: under the midnight convention post 651's own labels break. The platform labels bin k by
  its left edge (nominal_max is 2026-09-19, "the last answer date", with range_max 2026-09-20).
  A midnight stamp for 2026-09-08 (u = 0) is special-cased to bin 1 and a midnight stamp for
  2026-09-09 (u = 1/12) also maps to bin 1, so two answer dates share a bin and the twelfth bin is
  reachable only by a 2026-09-20 stamp that is not an answer date. That incoherence is a strong
  argument that Mantic intends inside-day stamps for day-granularity questions, which is the noon
  convention Phase 2 chose. Whether the LLM resolver honours that intent is what question 651's
  resolution will show.
- Assumed: Mantic's Series 1.5 test tournament (rules doc section 8) does not appear in the
  corpus under any slug, so the daily-bin date path is untested by Mantic too.

**Impact.** If the midnight convention turned out to be live, every date-only forecast would be
one bin late: a concentrated forecast would score near the floor (about -220) and a diffuse one
would lose a bin's worth. Date questions are 38% of Mantic's supply. There is no single mapping
robust to both conventions (the two put day D in different bins), and an adjacent-bin hedge costs
50 ln(2) = 34.7 points on every date question once the convention is known, so it is rejected
below.

**Mechanism.** No code now. Two things: keep C5 (read 651's resolution string and score after
2026-09-20 12:00 UTC), and ask Mantic directly, which is free and fast. If the answer is the
midnight convention, the flip is one constant (`_DATE_ONLY_HOUR_UTC` to a value that lands in the
previous bin, or an offset of one bin) plus the two pins named above and the sentence "a date with
no time of day means that whole day" in `_scoring_grid_clause`.

**Recommendation: BUILD AFTER LIVE DATA (it is C5) plus the operator email below.** The corpus
says the resolver stamps midnight often enough that C5 is load-bearing, not a formality, and one
email can close it before Series 2 opens.

## 6. Starved outer tail: field evidence for Wave C item C6

**Failure mode.** A resolution that lands inside the range but outside the members' declared
p1..p99 interval scores the platform floor (about -220), because the CDF build assigns the
structural 1% to the out-of-range bucket and leaves every in-range bin beyond the outermost
declared percentile at the minimum step. Documented in `docs/performance_analysis.md`; Wave C
holds it as a watch item.

**Evidence (verified).** On the field's 4,082 Series 1 forecasts, the realised in-range bin held no
more than twice the server minimum step (my "starved" test) on 5.2% of date forecasts (38 of 732,
mean score -225), 1.0% of numeric (11 of 1,081, -215) and 0.6% of discrete (6 of 949, -218). The
starved outcomes are spread across the range (deciles 0.01 to 0.96), not confined to one end.
Separately, in-range outcomes skew toward the bottom of the displayed range: bottom-quintile share
33% (date), 40% (discrete), 30% (numeric) against a uniform 20%.

**Impact.** For the field, about -12 points per in-range date question from this cliff alone. Our
bot's exposure on Mantic is unmeasured; the archive's 16% fire rate on Metaculus open-bound sides
suggests it is not smaller.

**Recommendation: BUILD AFTER LIVE DATA (already C6).** Nothing new to build; the numbers raise
C6's priority and the `oor_low` / `oor_high` telemetry plus the existing `scan_outer_tails`
detector are the instruments.

## Rejected and why

- **Adjacent-bin hedge for the day-bin edge ambiguity.** Splitting each date's mass across bins D-1
  and D costs 34.7 points on every date question once the convention is known, for a case one
  observation (question 651) or one email settles before Series 2 opens. Mechanism for a
  temporary uncertainty.
- **Symmetric unit-mismatch guard, or a percent-versus-fraction guard.** A 100x-too-small
  declaration (0.05 for 5%) passes the guard too (span ratio about 1e-3 against a 1e-5 threshold),
  but any tightening blocks legitimate shapes, and the accepted-no-change ruling on the symmetric
  case stands. The prompt already shows the unit and the displayed range.
- **A time-zone prompt rule for date questions.** 15% of date questions name a local time or
  Eastern Time; the model outputs the calendar date the question itself defines, and the platform
  bins are UTC days. No measured failure; prompt bloat.
- **Auto-following the ongoing bots-only tournament when the configured slug ends.** It would
  spend on a tournament the operator never approved. The `MANTIC_TOURNAMENTS` warning and the red
  exit after the end date (Phase 2, B15) are the right proportion.
- **Reading `key_factors`, `last_validation_decision` or `curation_status` off the post.**
  `key_factors` are user-submitted news links on 12 of 556 posts; the validation decision is a
  string (`accepted` 185, `declined` 41, null 330) with no content. Nothing for research.
- **Per-writer priors.** Out-of-range rates run 0% to 65% by writer, but n is about 20 each,
  Series 2 writers are unknown, and conditioning a forecast on who wrote the question is gaming
  rather than forecasting.
- **Quoting the field's Series 1 tail behaviour in the prompt** (for example, "top bots put 26%
  beyond the upper bound on date questions"). The base-rate clause already does this job with the
  outcome rate; more numbers are bloat.
- **An "as-of count" prompt rule for API-count questions** (count-so-far scaled by the remaining
  window). The numeric template's extrapolation step covers it.
- **Retuning the fast-path threshold for 60-minute windows.** Timing code, strictly-safer-only
  territory, and Wave C item C3 already watches the alertability side.
- **Recording the +27.6 "25th-percentile fallback" bar in the docs.** `docs/operations.md` already
  states the fallback rule in substance; the number adds little.
- **Series 2 scoring-formula reading.** The rules doc's "same, unadjusted score for all question
  types" admits two readings (drop the /2, making a 1% tail -161; or 100 log_N(p) + 100 over N
  buckets, making a 1% tail about +25 at 452 buckets). Both keep every recommendation above
  direction-invariant; C5 reads the live formula after 2026-09-20.

## Operator decisions needed

1. **Flip the Mantic-only 5% out-of-range tail floor before merge, or keep the HOLD?** New
   evidence: applying a 5% per-open-side floor to the eleven Series 1 competitors' own published
   distributions (4,082 forecasts) costs at most 1.9 points per question on average for any
   question type and gains up to 9.1 per question for the thin-tailed bots that resemble ours; for
   a bot at our structural 1% the expected gain is about +11 per quantitative question and +42 per
   date question, and the floor is inert if the models already place 5% beyond. The per-member
   telemetry the HOLD was waiting for still measures the unfloored model behaviour.
   **Recommendation: flip now at 5%, Mantic-only, aggregate level (2% is the conservative
   alternative: a third of the cost, about half the gain).**

2. **Email Mantic (api-requests@mantic.com) asking how a day-granularity date resolution is
   bucketed: does a resolution of "2026-09-16" on post 651 score the bin labelled 2026-09-16, or
   the bin ending at 2026-09-16 00:00 UTC?** Their resolver stamped 24 of 90 in-range Series 1
   date resolutions at midnight, and under the platform's bucket formula a midnight stamp scores
   the previous day's bin. Our noon convention matches Mantic's own bin labels; the question is
   whether their resolver honours them. Free, and it can close Wave C item C5 before Series 2
   opens instead of after 2026-09-20. **Recommendation: send it; keep the noon convention until
   the answer arrives.**

3. **Approve one prompt wording edit before merge, covering three corrections:** the base-rate
   sentence (one in five quantitative, one in four discrete, one in eight numeric, replacing one in
   seven / one in five / one in fifteen); the unit-conversion example, which currently says 350B
   is 350000000000 even when the question's base unit is billions (18 of 315 Mantic quantitative
   questions carry such units) and which is shared with the Metaculus numeric prompt; and the
   scoring-grid sentence, which would call a log-grid ratio a width. **Recommendation: approve;
   wording only, no new mechanism, the unit example is the one line that also touches Metaculus.**

4. **Already open, restated only because it dominates the rest: the external dispatcher.** GitHub
   delivered about 22% of scheduled firings from 2026-08-27 to 2026-09-07, and under Series 1's
   60-minute windows that forfeits about half of all questions at the 25th-percentile fallback.
   Wave C owns the design; nothing here changes it. **Recommendation unchanged: a free Cloudflare
   Worker cron calling GitHub's workflow-dispatch API at :01.**

## Measurements recorded for the next round (no action)

- Mantic supply by type (556 posts): date 209, discrete 159, numeric 156, multiple choice 31,
  binary 1. Bounds: 192 of 209 date questions are closed-lower / open-upper; 106 of 156 numeric
  and 72 of 159 discrete are open on both sides; 17 discrete and 16 numeric are closed on both.
- Bin counts: every Series 1 date question used 200 bins; post 651 uses 12. Discrete grids run 3 to
  450 bins (modes 151, 200, 101, 51). Numeric is 200 bins on 148 of 156.
- Series 1 date questions' `range_min` sits at arbitrary hours of day (it equals the open time
  on only 57 of 200); `range_max` is 12:00 UTC on the tournament end date for 174 of 200. Post
  651's daily grid is midnight-aligned.
- Text patterns on date questions (criteria, fine print and description searched together):
  "annulled" appears in 62% (almost always "if the date
  is not verifiable to daily precision"); an explicit "resolves above the upper bound if not by
  the end" sentence in 35%; a specific time of day in 55%; UTC named in 76%.
- Multiple choice: 16 of 31 questions carry an "Other" / "None of the above" style option; option
  counts 4 to 51.
- A regex heuristic found 38 quantitative questions whose description states a current value; in
  27 of them that value sits outside the displayed range (18 above, 9 below). Weak evidence, noisy
  heuristic, consistent with ranges being set as hypotheses rather than containers.
- Preseason 2's `cp_reveal_time` equals the close time on all four questions, as on all 520 Series
  1 questions: no community prediction is visible while a question is open.
