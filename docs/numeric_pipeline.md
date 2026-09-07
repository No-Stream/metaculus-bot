# Numeric forecasting and aggregation

This is the reference for how the bot turns a numeric question into a probability
distribution. It covers the full path: each forecaster declares the standard percentile
set in plain text, the bot extracts and cleans them, builds a 201-point CDF that
satisfies Metaculus' server-side constraints, aggregates across the ensemble in CDF
space, and decides whether the ensemble disagrees enough to trigger conditional
stacking. It also documents the time-series anchor research provider, which grounds
numeric forecasts in the resolution series' own history.

Every model name, flag, and default here is verified against the code. Where a value
lives in a constants file this doc names the constant rather than restating its value,
so read the constant for the current magnitude.

## What a numeric question looks like

A Metaculus `NumericQuestion` has a lower and upper bound, and each bound is either
open or closed. A closed bound is a hard limit (the outcome cannot fall past it); an
open bound is just the edge of the displayed range, and the true outcome can land
beyond it. Some questions also carry a `zero_point` (log-scaled axis) and a `cdf_size`
that differs from 201 (a discrete question). All of these change how the CDF is built,
so they thread through every stage below.

## Step 1: forecasters declare the standard percentiles

Each forecaster is prompted to emit the standard percentile set for its distribution.
`STANDARD_PERCENTILES` in `numeric/config.py` owns the set, and
`EXPECTED_PERCENTILE_COUNT` is derived from it with `len()` so the count can never
disagree with the list, and `STANDARD_PERCENTILES_CSV` is the label string the prompts
interpolate. Never restate the percentiles anywhere else in code or prompts; derive them
from `numeric/config.py`. The listing below is illustrative, for readers of this doc
only, and `numeric/config.py` remains the authority. As shipped today the set is:

```
1, 2.5, 5, 10, 20, 40, 50, 60, 80, 90, 95, 97.5, 99
```

Stored internally as decimals in `[0, 1]` rather than as 1-100 labels. The tail anchors
P1 and P99 exist so a forecaster can express probability mass beyond an open bound: if
you believe the outcome very likely exceeds an open upper bound, you place P50 and above
outside the displayed range. The prompt (see `prompts.numeric_prompt` and the bound
helper text from `bound_messages` in `numeric/utils.py`) tells forecasters exactly this
and warns them not to pile percentiles against an open edge.

The forecaster writes its percentiles inside a fenced ```json STRUCTURED FORECAST
block, which the prompt requires to be the last thing in the response.

## Step 2: value extraction ladder

`extract_numeric` in `value_extraction.py` pulls the declared percentiles out of the
model's free-text rationale. It runs a four-rung ladder, most deterministic first, and
stops at the first rung that produces a valid result:

1. **block** — parse the fenced ```json block directly (`json.loads` plus Pydantic
   validation against `NumericStructured`). When the block carries a
   `declared_percentiles` dict, it is lifted straight into `Percentile` objects
   (`_numeric_from_block` in `value_extraction.py`). This is the normal path.
2. **repair** — if the block was malformed, run deterministic JSON repair
   (`json_repair`), or scan the rationale's last `_TAIL_SCAN_CHARS` characters for
   balanced braces. Skipped when rung 1 already produced a schema-valid model (repairing
   valid JSON is a no-op).
3. **llm** — as a last resort, call the parser LLM (`parse_structured`) over the full
   rationale. Logged loudly as a salvage.
4. **raise** — if every rung fails, raise `ValueExtractionError`; the caller drops or
   soft-fails that forecaster.

Post-rung validation is strict regardless of which rung produced the value
(`_validate_numeric` in `value_extraction.py`): it requires every label in
`STANDARD_PERCENTILES` and returns exactly that set, never padded. That is a SHAPE
check, and shape alone is not enough — an earlier version of this doc claimed the LLM
salvage rung "cannot smuggle in a fabricated set", which was false as written. A rung
decoding under a schema must emit numbers, so the validation is also a FIDELITY check:
values must be finite and ordered the way their labels claim. See
[value_extraction.md](value_extraction.md) for the fidelity rules on all three question
types and the incidents that motivated each one.

Every successful extraction logs one line:

```
EXTRACTION_RUNG: question=... model=... qtype=numeric rung=... block_present=...
```

Watch for `rung=llm` and `block_present=False` in prod logs. Those are the two signals
that a forecaster stopped emitting a clean block.

The same call also decides discrete vs. continuous. `run_numeric_forecast`
(`forecaster_runners.py`) reads `NumericStructured.outcome_type` from the block first,
and only falls back to a parser LLM call (`OutcomeTypeResult`) when the block does not
declare it. That vote is carried forward to the discrete-snap decision in Step 7.

## Step 3: sanitize_percentiles

`sanitize_percentiles` (`numeric/pipeline.py`) turns the raw declared percentiles into
a clean, strictly-increasing, in-bounds set. In order:

1. `filter_to_standard_percentiles` — keep only the standard labels, drop extras and
   duplicates.
2. `validate_percentile_count_and_values` — assert the count and label set match
   `EXPECTED_PERCENTILE_COUNT` / `STANDARD_PERCENTILES` exactly
   (`numeric/validation.py`).
3. `sort_by_percentile_level` — order by percentile level.
4. `_apply_jitter_and_clamp` (`numeric/pipeline.py`) — detect count-like (integer-
   adjacent) clusters and spread them, jitter exact duplicates, clamp values into the
   question bounds with a safety buffer, then enforce strictly increasing values.
   **Exception: a WHOLE-SET epsilon collapse is not spread.** A model declaring
   (near-)the same value at all 13 percentiles has declared no width, and spreading it
   manufactured a ±6-unit distribution — which was precisely what let it pass Step 8's
   span-ratio test, so the invented width was load-bearing for publishing a forecast the
   model never stated. A point mass now gets only the format minimum (the jitter /
   strict-ordering epsilon) and reaches the guard with its own honest span, which
   withholds the forecaster. Only PARTIAL clusters are spread. Each collapse logs
   `NUMERIC_DEGENERATE_DECLARATION: question=... model=... n_unique=... span=...
   value_eps=... spread_applied=false` (harvested as `numeric_degenerate_declaration`), so
   the per-model incidence is a query rather than a guess. Open item: a *confident*
   DISCRETE point mass ("the count will be exactly 3") is now withheld rather than
   published as a hedge. Building the spike CDF is expressible on a coarse discrete grid
   (`grid_step_constraints` relaxes the max step there) and is the faithful third option
   if the marker starts firing on real forecasts.
5. `_maybe_widen_tails` — optional tail widening (Step 4).

The schema deliberately accepts non-decreasing values: tied values are valid concentrated
declarations and the sanitizer can separate partial clusters. The prompt still asks for
strictly increasing values, while the schema rejects decreases before
`sort_by_percentile_level` can order the declarations. A whole-set collapse remains valid
schema input, but sanitization does not invent width for it; the unit-mismatch guard then
withholds the member. The archive audit found exact ties in 2 of 346 declarations and no
whole-set collapses, supporting this distinction without adding a separate distinct-value
requirement.

It also decides whether to force `zero_point=None`: discrete questions and questions
whose `zero_point` equals the lower bound fall back to a linear axis
(`check_discrete_question_properties` in `numeric/validation.py`).

## Step 4: tail widening

`widen_declared_percentiles` (`numeric/tail_widening.py`) can fatten the tails by
scaling each percentile's distance from the median in a transformed space (bounded
logit for closed-closed questions, log transforms for one-sided questions, identity for
open-open). The stretch ramps from zero near the center to a maximum `k_tail` at the
deepest tails.

In production this is an identity pass: `TAIL_WIDEN_K_TAIL` and
`TAIL_WIDEN_SPAN_FLOOR_GAMMA` (`numeric/config.py`) both default to their no-op
settings, so nothing is widened and no span floor is enforced. That is why: an empirical
calibration on 43 resolved numerics found `k_tail=1.0` gave the best-calibrated tails,
and `k_tail=1.25` moved away from ideal in every segment (see
`scratch_docs_and_planning/tail_widening_empirical_calibration.md`). Both knobs are
still per-call configurable, and the function raises `ValueError` if asked to narrow
rather than widen (narrowing is not implemented) or on a negative `span_floor_gamma`.

## Step 5: PCHIP 201-point CDF

`build_numeric_distribution` (`numeric/pipeline.py`) hands the sanitized percentiles
to `generate_pchip_cdf_with_smoothing`, which calls `generate_pchip_cdf`
(`numeric/pchip_cdf.py`). This produces the 201-point CDF the bot actually submits.

The construction:

- Build a value grid from lower to upper bound: linear normally, geometric when
  `zero_point` is set (`build_cdf_value_grid` in `numeric/pchip_cdf.py`, matches the
  Metaculus backend's non-linear spacing).
- Fit a monotone PCHIP interpolator through the declared percentiles (log-space for
  strictly-positive series), evaluate it on the grid, and clamp to `[0, 1]`.
- Blend in a uniform mixture so the minimum step is satisfied before any repair tier is
  reached. This is the primary min-step mechanism, inline in `generate_pchip_cdf`
  (`numeric/pchip_cdf.py`).
- Enforce the min-step and max-step constraints, then re-pin the bounds.

### Server-side constraints

Metaculus validates `continuous_cdf` submissions. The server formulas below are the
upstream contract; the constants in `constants.py` mirror them, and those are what to
read for the value we actually submit. `numeric/config.py` carries grid-scoped aliases
(`MIN_CDF_PROB_STEP`, `MAX_CDF_PROB_STEP`) — note that only the max-step alias imports
from `constants.py`; the min-step one restates the literal, so the two min-step
constants have to be changed together.

- **Length** = `cdf_size`, whose standard-continuous default is `PCHIP_CDF_POINTS`. On
  the platform side that is the question's `inbound_outcome_count + 1`, default 201.
- **Min step** per bin `NUM_MIN_PROB_STEP` — no flat segments allowed. The
  server formula is `round(0.01 / N, 9)` where `N = cdf_size - 1`, so 5e-5 at the
  default length.
- **Max step** per bin `NUM_MAX_STEP` — a spikiness cap. Server formula
  `0.2 * 200 / N`, so 0.2 at the default length.
- **Closed bounds** are pinned exactly: `cdf[0] == 0.0`, `cdf[-1] == 1.0`.
- **Open bounds**: `cdf[0] >= 0.001`, `cdf[-1] <= 0.999`.
- **Strictly increasing**, implied by min step > 0.

The upstream source for all of this is the open-source Metaculus backend,
<https://github.com/Metaculus/metaculus>, where the validation lives in
`questions/serializers/common.py`. The API itself is documented at
<https://www.metaculus.com/api/> (Swagger UI).

`grid_step_constraints` (`numeric/config.py`) is what applies those formulas to a
non-standard grid: the min step is floored at `MIN_CDF_PROB_STEP` so a fine grid never
demands a step below the historical constant, and the max step is clamped at `1.0`
(a probability step larger than that is vacuous). On the standard continuous grid it
returns exactly `(MIN_CDF_PROB_STEP, MAX_CDF_PROB_STEP)`, so continuous questions are
unaffected; a coarse discrete grid relaxes the max step upward, which is what lets a
small-count distribution keep its mass concentrated on the low integers.

`safe_cdf_bounds` (`numeric/pchip_cdf.py`) enforces the max-step rule by
redistributing excess mass while preserving the total, then re-enforces min-step after
the pin-and-cummax pass. `enforce_min_steps` (`numeric/pchip_cdf.py`) does a
forward-then-backward sweep to guarantee every adjacent pair is at least one min-step
apart. `_apply_ramp_smoothing` (`pchip_processing.py`) is a final tilt that adds a
tiny linear ramp when the raw CDF still has a sub-min-step bin.

**Where the clipped excess goes is our choice, and it is nearest-first.** The max-step
branch fires on any question where a forecaster declares more single-bin mass than the
platform's `0.2 * 200 / N` cap can hold, so unlike the min-step tiers it is a live path.
`_pack_excess_nearest_first` clips each over-cap bin and pours its excess into the
adjacent bins with headroom, walking outward one ring at a time and splitting the
remainder across a ring in proportion to each side's room. The retired policy handed the
excess out in proportion to every bin's *slack*, which on a fine grid where the other
bins are near-empty is a near-uniform spread: q45065 (2026-08-01) had all three
forecasters declare ~0.72 on the count that resolved, and published 47% of its mass above
35 deaths against their own ~2%, with no log line anywhere at the time. The cap itself is the platform's and is untouched — a 0.72 single-bin mass
is simply not expressible on a 201-point grid — so the honest repair is the legal shape
closest to the declaration. Every clip emits a `CDF_MAXSTEP_CLIP` WARN naming the
forecaster, the mass displaced, and how far it travelled (`scripts/telemetry/markers.py`
harvests it); it is deliberately **not** alertable, since a spike above a platform cap is
a forecaster's declaration rather than a bot defect. Its `bins_displaced` and
`max_offset_bins` fields are what make the repair's own footprint on a published forecast
a query rather than a reconstruction.

A THIRD numeric failure shape sits beside the min-step and max-step repairs and is
measured only offline: a STARVED OUTER TAIL, where the declared tail routes past the
displayed range and leaves every in-range bin above the members' p99 pinned at the
platform minimum step, so any resolution in that band earns the same floor score. It is a
cliff at a fixed location rather than a mis-sized band, which is why widening does not fix
it, and it has no publish-time WARN on purpose. Detail:
[performance_analysis.md](performance_analysis.md) and
`metaculus_bot/performance_analysis/outer_tail.py`.

`safe_cdf_bounds` holds the only implementation of this packing policy, and every path
that enforces the max-step rule reaches it: the per-model build (`generate_pchip_cdf`),
the per-model ramp pass, the ensemble CDF (`_postprocess_ensemble_cdf`, both branches),
the discrete integer snap, the forecasting-tools fallback builder on an OPEN bound, and
the offline pooling paths. It is not, however, a choke point every published CDF passes
through: `BoundSafeNumericDistribution.get_cdf` in `numeric/pchip_processing.py` returns
upstream's CDF unchanged when BOTH bounds are closed, so a closed-bound fallback
distribution gets no step or endpoint enforcement and can emit no `CDF_MAXSTEP_CLIP`
marker. That gap is accepted deliberately (it needs closed bounds and a PCHIP failure and
stacking enabled, and stacking is prod-disabled); see the sentinel-value entry in
FUTURE.md under "Sentinel-value sweep leftovers", sixth item.

### Open vs. closed bounds: a one-sided constraint, not a box

This is the subtle part. Bound pinning is **one-sided per tail**, not a clamp on
out-of-bound mass:

- **Closed lower bound** → `cdf[0]` is pinned to exactly `0.0`.
- **Closed upper bound** → `cdf[-1]` is pinned to exactly `1.0`.
- **Open lower bound** → `cdf[0]` is floored at a *minimum* of `0.001`. This is a
  required minimum positive mass, not a cap. The CDF can start well above 0.001.
- **Open upper bound** → `cdf[-1]` is ceilinged at a *maximum* of `0.999`. Again a
  required headroom, not a floor on how much mass sits below the ceiling.

There is no cap on out-of-bound mass. A distribution can legitimately place, say, 78%
of its mass below an open lower bound. That mass is expressed by placing declared
percentile *values* beyond the displayed range (values are not clamped on open bounds),
so that `F(bound)` interpolates to the intended fraction. The only ceiling on
out-of-bound mass comes from min/max-step feasibility (roughly 0.99, since 200 bins each
need at least a min-step). The `_pin_endpoints` helper (`numeric/utils.py`) and the
validation in `_validate_pchip_cdf` (`pchip_processing.py`) both apply this one-sided
logic.

If PCHIP construction fails outright, `create_fallback_numeric_distribution`
(`pchip_processing.py`) delegates the CDF build to forecasting-tools, but still
re-pins open-bound endpoints through `safe_cdf_bounds` (the native builder would anchor
an open lower bound at 0% once the standard set includes P1, which Metaculus rejects).

### Piling on an open edge (`OPEN_BOUND_PILING`)

A sibling WARN, `OPEN_BOUND_PILING: question=... model=... bound=... bin_mass=... declared_edge=... bound_value=...` (`numeric/diagnostics.py`, threshold `OPEN_BOUND_PILING_THRESHOLD` in `numeric/config.py`), fires when a model piles at least that fraction of mass on the terminal displayed bin of an *open*-bound numeric question without declaring any percentile beyond the edge — the "crammed the open ceiling" failure mode fixed 2026-07-12 by rendering nominal/displayed bounds in the numeric prompts (`nominal_bounds` in `numeric/utils.py`). It takes the pre-resample model-declared percentiles explicitly (the discrete resample overwrites `prediction.declared_percentiles` with a grid pinned to the raw bounds, which would defeat the above-edge exemption).

### The MIN-step repair-tier signals are dead code on real forecasts

`generate_pchip_cdf` logs five repair signals: `pchip_aggressive` (aggressive
enforcement), `clamp_frac` and `clamp_dist` (bounds-clamp corrections),
`violated_steps_frac`, and `ramp_smoothing_delta`. On real model output none of them
fires, because the uniform-mixture construction pre-enforces the min-step before any
repair tier is reached: 0 of 1182 archived numeric forecasts fired any of the five, and
the one genuinely degenerate case raises `pchip_failed` instead. Keep the guards (they
defend against pathological inputs), but their absence in logs carries no information,
and they must not be used as model-quality features. Verified 2026-07-15; receipts in
`scratch/coherence_2026-07-15/synthesis.md`.

Do NOT generalize that to the MAX-step repair, which is a live path — see the
nearest-first packing section below. That the 2026-07-15 audit did not see it is an
artifact of its old DEBUG level, not evidence it never fires.

## Step 6: PchipNumericDistribution

The result is wrapped in a `PchipNumericDistribution` (`pchip_processing.py`), a
subclass of forecasting-tools' `NumericDistribution` whose `get_cdf()` override returns
the pre-computed CDF on the canonical question grid instead of rebuilding it. `get_cdf()` is the real
override — the `.cdf` property is a deprecated shim that delegates to it, so overriding
`.cdf` alone would miss the publish and aggregate paths. The `_pchip_cdf_values` attribute
also acts as the marker that CDF validation should be skipped (the constraints were
already enforced) and that discrete snapping can read the CDF back out.

## Step 7: discrete integer snapping

Some questions are labeled continuous (`cdf_size=201`) but resolve on integers ("how
many X will happen?"). A smooth CDF wastes mass between integers. If a strict majority
of forecasters voted DISCRETE (`majority_votes_discrete`), `maybe_snap_to_integers`
(`post_processing.py`) snaps the ensemble distribution to a step function
concentrated on integer values.

`snap_cdf_to_integers` (`numeric/discrete_snap.py`) extracts an integer PMF by
half-integer interpolation, rebuilds a step CDF, mixes in a uniform component for
min-step compliance, then runs `safe_cdf_bounds`. It is skipped when the range holds
more than `DISCRETE_SNAP_MAX_INTEGERS` integers (`constants.py`), when there
are no integers in bounds, when bounds are non-finite, or when the question is already
natively discrete (`cdf_size != 201`). Snapping is decided at the ensemble level, after
aggregation.

## Step 8: unit-mismatch guard

Before a per-model prediction is accepted, `detect_unit_mismatch`
(`numeric/validation.py`) checks whether the declared values look off by orders of
magnitude relative to the question range. It flags a mismatch when any of three ratios
falls below its threshold — each threshold is a keyword argument on
`detect_unit_mismatch`, so the signature is where the values live:

- span between lowest and highest declared value, over the range
  (`span_ratio_threshold`);
- minimum adjacent step, over the range (`min_step_ratio_threshold`);
- maximum absolute value, over the range (`max_magnitude_ratio_threshold`).

On a flagged mismatch, `run_numeric_forecast` (`forecaster_runners.py`) raises
`UnitMismatchError` and withholds that forecaster's prediction rather than submitting a
distribution in the wrong units. No network or community stats are needed; it is a pure
sanity check on the numbers.

The guard **fails SHUT**: it used to wrap its arithmetic in a try/except that returned
"no mismatch" on any internal error, which is byte-identical to a passing check — so a
crash inside the guard silently published the order-of-magnitude error it exists to
block. Errors now propagate. Related: a point-mass declaration reaches this guard with
its real (zero) span rather than the cluster spreader's invented one, which is why
Step 3 no longer spreads whole-set collapses.

## Step 9: ensemble aggregation in CDF space

`aggregate_numeric` (`numeric/utils.py`) combines the per-model distributions
**pointwise in CDF space**, not by averaging percentiles:

1. Read each model's CDF heights in ORDER and align them POSITIONALLY: grid index `i` is
   Metaculus bucket `i/(n-1)`, so index `i` means the same thing for every model. A CDF
   that arrives on a different-length grid is resampled in cdf-LOCATION space (never
   value space — a log-scaled `zero_point` question's PCHIP CDF carries a linear value
   axis while forecasting-tools' fallback builder carries a geometric one, so their
   x-values disagree by construction even when bucket `i` matches) and logs
   `NUMERIC_AGGREGATE_GRID_MISMATCH`, which should read zero in prod.
2. Take the mean or median of the cumulative probabilities at each index.

   This replaced a group-by-VALUE aggregation (a pandas groupby on float-equal `value`).
   The PCHIP grid (`np.linspace`) and the ft-fallback grid (`min + span*i/(n-1)`) differ
   in float rounding, so a mixed-path ensemble medianed over a rotating SUBSET of its
   members at misaligned points — measured at 225 unique x-values from 3 models, 48 of
   them with fewer than 3 contributors — and nothing recorded the partial membership.
3. `_postprocess_ensemble_cdf` (`numeric/utils.py`) re-pins the endpoints (one-sided
   open/closed logic), enforces monotonicity, applies ramp smoothing if any bin is below
   min-step, and — for discrete questions whose `cdf_size != 201` — resamples the CDF to
   the target grid via PCHIP with the grid-scaled min-step.

Percentile-space averaging would blur multi-modal disagreement; CDF-space averaging
preserves it. In production the base-combine path uses **MEDIAN** of the raw per-model
CDFs (`base_combine` in `aggregation_pipeline.py`, because the default strategy is
`CONDITIONAL_STACKING` and stacking is disabled in prod). Backtests and the mean arm use
MEAN.

## Step 10: the numeric spread metric

`numeric_percentile_spread` (`spread_metrics.py`) is what decides whether the
ensemble disagrees enough to trigger conditional stacking. It reads each model's P10,
P50, and P90 values (by percentile label, so growing the standard set cannot silently
shift them), takes the max-minus-min spread at each of those three percentiles, and
normalizes:

- **Closed-bound questions** → divide by the question range (`upper - lower`).
- **Open-ended questions** → divide by the ensemble interquartile range (median P90
  minus median P10), since the range is unbounded.

The largest of the three normalized spreads is the reported value. If it exceeds
`CONDITIONAL_STACKING_NUMERIC_NORMALIZED_THRESHOLD` (`constants.py`), the
aggregator extracts the disagreement crux, runs a targeted search, and invokes the
stacker LLM; otherwise it returns the MEDIAN. (Stacking is disabled in all five bot
workflows, so in prod this metric is computed but the stacker branch does not fire; the
chain stays live in backtests and ablation.)

For discrete questions, the per-model `declared_percentiles` have already been
overwritten with a resampled CDF grid whose labels are cumulative probabilities, so
`_key_percentile_values` (`spread_metrics.py`) reads P10/P50/P90 by interpolating the
empirical CDF instead of looking up label nodes.

## The time-series anchor provider

`research/timeseries_anchor.py` is a research provider (not part of the CDF pipeline)
that grounds numeric forecasts whose resolution series is a fetchable FRED or yfinance
series. It renders a deterministic empirical anchor with no LLM: the latest value, a
multi-resolution history, a 52-week range, and a horizon-matched empirical band. Its
section header in the briefing is `## Time Series Anchor` (`TS_ANCHOR_SECTION_HEADER`,
rendered by `provider_header` in `research/section_format.py`). Gated by
`TS_ANCHOR_ENABLED` (`_select_research_providers` in `research/orchestrator.py`); on in
all five workflows.

### Routing (deterministic, no LLM)

`route_question` (`timeseries_anchor.py`) maps a question to a series two ways, URL
first:

1. **URL extraction** from resolution criteria and fine print — a cited FRED series or
   Yahoo ticker is the ground-truth resolving source and wins.
2. **A conservative curated keyword registry** (`_TEMPLATE_REGISTRY` in
   `timeseries_anchor.py`) — 10-year Treasury, VIX, CPI, unemployment, nonfarm
   payrolls, S&P 500, gold, and so on. Deliberately small and unambiguous.

Anything ambiguous (more than one series that is not a two-ticker spread, or more than
one keyword match) returns `""` and logs. Two Yahoo tickers become a relative-return
spread block.

### The empirical band

The band is the naive, model-free choice, deliberately. The Phase-A offline replay
(`scratch/ts_anchor_replay_2026-07-16/synthesis.md`) found that CV-gated statistical
model picks beat the naive out-of-sample only 43% of the time, while the naive empirical
h-step-change band was both sharper and better tail-calibrated. So the provider just
computes empirical quantiles (P10 / P50 / P90) of every overlapping h-step change in the
series' own history and applies them to the latest value (`_empirical_change_band` in
`timeseries_anchor.py`). Log-multiplicative for strictly-positive series, additive
otherwise. "Highest / peak / maximum" questions use a forward-window-max band instead
(`_empirical_max_band`). The horizon `h` is matched to the question's actual
forecast window, converted to native series steps by frequency (`horizon_steps`).

Derived-quantity questions (month-over-month change, MoM % inflation, monthly averages)
fit the band on the derived series, not the raw level (`_apply_derivation` in
`timeseries_anchor.py`).

### Point-in-time leakage safety

This is the only backtest-safe research provider that runs during benchmarks. Others
hard-disable under `is_benchmarking`; this one instead pins `as_of` to
`question.open_time` in benchmarks (live: `datetime.now(UTC)`) and fetches the series
point-in-time up to that date, so data known at forecast time is fair game without
leaking the resolution (`timeseries_anchor_provider` in `timeseries_anchor.py`).

The fetch layer (`research/ts_fetch.py`) enforces the invariant. Revising macro series
(CPI, payrolls, GDP) would leak if fetched from today's FRED, because today's data
contains revised historical values not known at forecast time. So those go through
**ALFRED point-in-time vintages** instead of plain FRED CSV. `fetch_series`
(`ts_fetch.py`) defaults every FRED series to ALFRED vintages *except* a small
non-revising allowlist (`FRED_NON_REVISING_SERIES` in `ts_fetch.py`: market prices and
survey levels like DGS10, Brent, gasoline). That default is fail-safe — an over-inclusive
ALFRED guess costs nothing for a non-revising series, but a revising series routed to
plain FRED would silently leak. A belt-and-suspenders check, `_assert_no_leakage`
(`ts_fetch.py`), raises `LeakageError` if any observation postdates the ceiling.

### Text anchor on, chart off

The provider always returns a text section. It can also render a small chart image
(matplotlib ribbon), stashed in a per-session side-channel for the forecaster's vision
message, but only for plain single-level questions. The chart is gated separately by
`TS_ANCHOR_CHART_ENABLED` (`_maybe_stash_single_chart` in `timeseries_anchor.py`), which
is **off** in all five workflows while the text anchor is **on**. Chart render failures
are swallowed so a plotting hiccup never breaks the text section.

## Clamps on the other question types

The numeric CDF bounds above are one of three publish-value clamps. The binary and MC
clamps live here too so that all three have one home.

- **Binary**: `[BINARY_PROB_MIN, BINARY_PROB_MAX]` (`constants.py`). Applied per-model in `forecaster_runners.py` and on stacker output in `stacking.py`. Median/mean of already-clamped values stays in-bounds, so no post-aggregation clip needed.
- **MC**: `[MC_PROB_MIN, MC_PROB_MAX]` (`constants.py`), set to match ft 0.2.92's `PredictedOptionList` validator, which clamps every option on construction — matching bounds makes it a no-op and removes publish-time `ValueError` risk on many-option ballots. Drift-free clamp-then-renormalize via `clamp_and_renormalize_probs` (`mc_processing.py`), applied BEFORE every `PredictedOptionList` construction and re-applied idempotently by `clamp_and_renormalize_mc` (`numeric/utils.py`); the repair pass keeps floored options from dividing back below the floor.
