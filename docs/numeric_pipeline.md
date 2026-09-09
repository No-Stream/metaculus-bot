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
   adjacent) clusters and spread them, jitter exact duplicates, clamp values just outside a
   closed bound back inside it (the tolerance and landing rules are below), then enforce
   strictly increasing values.
   **On the 201-point continuous grid a WHOLE-SET epsilon collapse is not spread.** A model
   declaring (near-)the same value at all 13 percentiles has declared no width, and spreading
   it manufactured a ±6-unit distribution — which was precisely what let it pass Step 8's
   span-ratio test, so the invented width was load-bearing for publishing a forecast the
   model never stated. There a point mass gets only the format minimum (the jitter /
   strict-ordering epsilon) and reaches the guard with its own honest span, which
   withholds the forecaster; only PARTIAL clusters are spread. **Where the published bins
   are the outcome space** (`grid_is_outcome_space`, below) the collapse IS spread, like any
   other plateau and under the same one-bin cap: "100% on 2026-09-16" is fully expressible
   on twelve one-day bins and the date prompt invites that shape, so the member publishes
   with its mass in that day (0.97 of it on post 651) instead of being dropped as a unit
   mismatch. The width the spread adds stays inside the bin the forecaster named, and a
   genuine scale error on such a grid sits several bins outside the bounds and still raises
   in the clamp. Each collapse logs `NUMERIC_DEGENERATE_DECLARATION: question=... model=...
   n_unique=... span=... value_eps=... spread_applied=true|false` (harvested as
   `numeric_degenerate_declaration`; `spread_applied` reports which branch it took), so the
   per-model incidence is a query rather than a guess.
5. `_maybe_widen_tails` — optional tail widening (Step 4).

The schema deliberately accepts non-decreasing values: tied values are valid concentrated
declarations and the sanitizer can separate partial clusters. The prompt still asks for
strictly increasing values, while the schema rejects decreases before
`sort_by_percentile_level` can order the declarations. A whole-set collapse remains valid
schema input; on the continuous grid sanitization does not invent width for it and the
unit-mismatch guard withholds the member. The archive audit found exact ties in 2 of 346
declarations and no whole-set collapses, supporting this distinction without adding a
separate distinct-value requirement.

The closed-bound clamp (`numeric/bounds_clamping.py`) is governed by two different numbers.
`calculate_bounds_buffer` is the accept-or-raise TOLERANCE: a declared value may sit outside
a closed bound by at most the larger of the range-based tolerance (1% of the range, a flat
1.0 once the range exceeds 100) and one grid bin (`grid_bin_width`), and is clamped in;
anything further out raises and the member is dropped as a scale error. The bin floor holds
on EVERY grid, the 201-point continuous one included, because a value within one bin of the
edge is indistinguishable from the edge once bucketed: on a Metaculus question with range
> 200 the drop-versus-clamp threshold is therefore `range / 200` rather than the old flat
1.0, which removes the discontinuity where 1.0 was 1% of a 100-wide range and 0.005% of a
20,000-wide one. In plain words: on a 201-point question whose range exceeds 200, a
percentile that sits between 1.0 and `range / 200` outside a closed bound is now clamped to
the bound instead of the member being dropped (an intentional, documented change of
2026-09; inside 1.0 it always clamped, beyond `range / 200` it still drops). On a date question the axis is epoch seconds, so the flat 1.0 was a
one-second tolerance and a date named one day outside a closed bound used to drop the
member. The tolerance changes only drop-versus-clamp, never where a clamped value lands:
that is `minimum_separation` (`numeric/config.py`), the same standoff the spreader and the
strict-ordering passes keep just inside a closed bound. Landing the value one tolerance
inside instead moved it a whole bin on a coarse grid, and the left-to-right strict-ordering
pass then dragged every percentile declared inside that first bin up behind it (post 651:
6.3 points of mass left the day the member declared; a [0, 20000] Metaculus question
published four percentiles at 100.0 when three were in range). The heavy-clamping WARNING
counts values at that same standoff, so an in-range forecast concentrated near a bound no
longer reads as clamped.

Wherever the published bins are the outcome space, the spreader keeps a plateau inside the
bin it names: the plateau's TOTAL spread is capped at one bin width, `grid_bin_width`
(`numeric/config.py`), so the per-position spread is the smaller of the count-like unit and
`bin_width / (plateau_size - 1)` (`_spread_cluster_values`, `numeric/cluster_processing.py`).
The grid points of a discrete question are its bin edges, so a plateau at integer k spread
past k +- 0.5 handed the mass the forecaster put on k to the neighbouring bins: on the
three-bin Mantic post 253 a declared 90% on the first bin published 0.558 (Mantic edge-case
review 2026-09, rank 7; the corpus reproductions are
`tests/test_numeric_discrete_grid_plateaus.py`). What a plateau pins is an interval
(`P90 = 0` and `P95 = 1` say `F(0.5)` lies in [0.90, 0.95)), and the published mass is
that interval's lower edge less the uniform mixture. The predicate is
`grid_is_outcome_space` (`numeric/config.py`): a natively discrete question
(`DiscreteQuestion`, which every Metaculus discrete and every Mantic quantitative question
parses as) or any non-201 grid, the same predicate on which Step 7's discrete snap skips. On
those shapes the spread is the final word; on the 201-point continuous grid of a
`NumericQuestion` the unit spread is pre-processing the vote-gated snap can re-concentrate,
so that grid is byte-identical. The ablation harness rehydrates questions as
`NumericQuestion`, which diverges from prod only on Mantic 200-bin discrete questions, absent
from the Metaculus archives it replays.

It also resolves the `zero_point` every CDF for the question is built with
(`resolve_zero_point` in `numeric/validation.py`): the question's own, or `None` when it
equals the lower bound, where the geometric axis divides by zero. A non-201 grid is not a
reason to drop the log scale. The platform declares geometric bins on a `zero_point`
question at any bin count and maps the submitted probabilities onto them positionally, so
the linear axis the bot forced on every non-201 grid until 2026-09 would have published
probabilities computed at linear positions against geometric bins (a `[1, 1e6]` question
forecast at 1,000 read back as a median near 1.2). Metaculus never reached that branch,
because its log-scaled questions are always 201 points; Mantic's fine grids do.

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

Metaculus validates `continuous_cdf` submissions (Mantic runs a fork of the same backend
with the same constants). The server formulas below are the upstream contract; the
constants `NUM_MIN_PROB_STEP` and `NUM_MAX_STEP` in `constants.py` mirror them at the
standard 201-point grid and are the defaults of the 201-point builders. Every grid,
including the standard one, derives its live limits from `grid_step_constraints`, below;
`numeric/config.py` keeps `MAX_CDF_PROB_STEP` as the named 201-grid cap: the reference the
open-bound piling threshold is calibrated against (`numeric/diagnostics.py`) and the
pre-scaling cap the residual analysis reads back (`performance_analysis/analysis.py`).

- **Length** = `cdf_size`, whose standard-continuous default is `PCHIP_CDF_POINTS`. On
  the platform side that is the question's `inbound_outcome_count + 1`, default 201.
- **Min step** per bin `NUM_MIN_PROB_STEP` — no flat segments allowed. The
  server formula is `round(0.01 / N, 9)` where `N = cdf_size - 1`, so 5e-5 at the
  default length.
- **Max step** per bin `NUM_MAX_STEP` — a spikiness cap. Server formula
  `0.2 * 200 / N`, so 0.2 at the default length. The server compares its 9-decimal-rounded
  PMF against this UNROUNDED cap, so on a grid whose cap is not 9-decimal exact a bin sitting
  exactly at the cap rounds above it and is rejected (450 bins: 0.0888... rounds to
  0.088888889).
- **Closed bounds** are pinned exactly: `cdf[0] == 0.0`, `cdf[-1] == 1.0`.
- **Open bounds**: `cdf[0] >= 0.001`, `cdf[-1] <= 0.999`.
- **Strictly increasing**, implied by min step > 0.

The upstream source for all of this is the open-source Metaculus backend,
<https://github.com/Metaculus/metaculus>, where the validation lives in
`questions/serializers/common.py`. The API itself is documented at
<https://www.metaculus.com/api/> (Swagger UI).

`grid_step_constraints` (`numeric/config.py`) applies those formulas to any grid as the
9-decimal values that survive the server's rounding: the min step is `round(0.01 / N, 9)`,
the server's own rounded floor, and the max step is the largest 9-decimal value not
exceeding `0.2 * 200 / N`, clamped at `1.0` (a probability step larger than that is
vacuous). The max-step floor exists because `safe_cdf_bounds` clips over-cap bins to
exactly the max step: with the raw cap, a clipped bin on the live 451-point Mantic question
rounded up to 0.088888889 and the whole submission was rejected with HTTP 400 (found in
review 2026-09, before any live run). The floor is a no-op wherever the cap is 9-decimal
exact, which covers every grid of 41 points or fewer and the 51, 101, 201 and 2,001-point
grids. There is no floor on the min step: an earlier version floored it at the 201-grid
value, which was a no-op on every Metaculus grid but 2.25x stricter than the server at 450
bins and 10x at 2,000, forcing that much extra uniform mixture into fine-grid tails. Mantic
quantitative questions run up to 2,000 bins, so that floor was removed in 2026-09
(`tests/test_numeric_fine_grids.py` pins the formulas and the 451- and 2,001-point publish
path, broad and cap-binding). On the standard continuous grid the function returns exactly
`(NUM_MIN_PROB_STEP, NUM_MAX_STEP)`, so continuous questions are unaffected; a coarse
discrete grid relaxes the max step upward, which is what lets a small-count distribution
keep its mass concentrated on the low integers.

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
the per-model ramp pass, the ensemble CDF (`_postprocess_ensemble_cdf`),
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

A sibling WARN, `OPEN_BOUND_PILING: question=... model=... bound=... bin_mass=... declared_edge=... bound_value=...` (`numeric/diagnostics.py`, threshold `OPEN_BOUND_PILING_THRESHOLD` in `numeric/config.py`), fires when a model piles at least that fraction of mass on the terminal displayed bin of an *open*-bound numeric question without declaring any percentile beyond the edge — the "crammed the open ceiling" failure mode fixed 2026-07-12 by rendering nominal/displayed bounds in the numeric prompts (`nominal_bounds` in `numeric/utils.py`). It takes the pre-resample model-declared percentiles explicitly (the discrete resample overwrites `prediction.declared_percentiles` with a grid pinned to the raw bounds, which would defeat the above-edge exemption). The threshold is calibrated against the 201-grid per-bin cap of 0.2 and scales down with the grid's own cap on finer grids (0.044 at 451 points, 0.01 at 2,001), because the max-step repair clips a crammed terminal bin to that cap, which on a fine grid is below the fixed 0.10; the 201-point and every coarser grid keep 0.10 exactly.

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

The rebuild trigger and the rebuild's own range check carry the same `_MIN_STEP_TOLERANCE`
(1e-10) as the post-check and the final assertion (`numeric/pchip_cdf.py`). A forecast that
puts essentially all of its mass beyond an open bound leaves the in-range CDF as the bare
min-step ramp, whose required range equals its available range to within float epsilon;
untoleranced, the trigger fired on a step 1e-18 short and the range check then refused a range
1e-16 short, so the member was dropped on every non-201 grid and the 201-point grid fell
through to a forecasting-tools fallback that failed on the same input (Mantic edge-case
review 2026-09, rank 3; `TestAllMassBeyondABoundStillBuilds` in
`tests/test_numeric_fine_grids.py` pins 15, 201, 451 and 2,001 points, both bounds). With the
tolerance the rebuild can only be reached by a genuine shortfall, where it raises.

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
are no integers in bounds, when bounds are non-finite, when the question is natively
discrete (a `DiscreteQuestion`: every Metaculus discrete question and every Mantic
quantitative question), or when the grid is not the 201-point one the snap's step limits
belong to. The type is the signal, not `cdf_size`: a 200-bin Mantic discrete question has
`cdf_size == 201` (thirteen in the Series 1 corpus, steps 1.0, 1.005 and 251.25) and the old
`cdf_size != 201` guard let them through. A natively discrete grid is already its outcome
space, so a 0.1-step grid resolves in tenths and an integer vote is simply wrong there, while
on a 1.0-step integer-centred grid the snap is a no-op (Mantic edge-case review 2026-09,
rank 14). Metaculus continuous count questions on [0, 10] or [0, 50] at 201 points, the
snap's designed target, are unaffected. Snapping is decided at the ensemble level, after
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
block. Errors now propagate. Related: on the 201-point continuous grid a point-mass
declaration reaches this guard with its real (zero) span rather than the cluster spreader's
invented one, which is why Step 3 does not spread whole-set collapses there; where the bins
are the outcome space Step 3 spreads the collapse inside its bin and the guard passes it.

## Step 9: ensemble aggregation in CDF space

`aggregate_numeric` (`numeric/utils.py`) combines the per-model distributions
**pointwise in CDF space**, not by averaging percentiles:

1. Read each model's CDF heights in ORDER and align them POSITIONALLY: grid index `i` is
   Metaculus bucket `i/(n-1)`, so index `i` means the same thing for every model. A CDF
   that arrives on a different-length grid is resampled in cdf-LOCATION space (never
   value space: the PCHIP grid and forecasting-tools' builder compute the same value axis
   by different formulas, equal in exact arithmetic but not in the last float bits, so
   only the bucket index is shared by construction) and logs
   `NUMERIC_AGGREGATE_GRID_MISMATCH`, which should read zero in prod.
2. Take the mean or median of the cumulative probabilities at each index.

   This replaced a group-by-VALUE aggregation (a pandas groupby on float-equal `value`).
   The PCHIP grid (`np.linspace`) and the ft-fallback grid (`min + span*i/(n-1)`) differ
   in float rounding, so a mixed-path ensemble medianed over a rotating SUBSET of its
   members at misaligned points — measured at 225 unique x-values from 3 models, 48 of
   them with fewer than 3 contributors — and nothing recorded the partial membership.
3. `_postprocess_ensemble_cdf` (`numeric/utils.py`) re-pins the endpoints (one-sided
   open/closed logic), enforces monotonicity, applies ramp smoothing if any bin is below
   min-step, and runs `safe_cdf_bounds` with the step limits of the grid the CDF is on
   (`grid_step_constraints`). Nothing is resampled at this stage: step 1 already put the
   ensemble on the question's `cdf_size` grid, so a discrete question's aggregate gets
   the coarse grid's limits by construction. The result is labelled with the question's
   own value axis (`build_cdf_value_grid` on the `zero_point` that `resolve_zero_point`
   picked, the same one every member was built with), so the aggregate's
   `declared_percentiles`, its `get_cdf()` and the members all agree on where each
   probability sits. Until 2026-09 the aggregate's `declared_percentiles` were labelled
   with a linear axis even on a `zero_point` question, so the comment and the spread
   metric read a different distribution from the one published.

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

## Step 11: the Mantic out-of-range tail floor

`floor_published_tails` (`numeric/out_of_range_floor.py`) is the last touch on a numeric,
discrete or date aggregate before it is published, applied in
`TemplateForecaster._aggregate_predictions` (`forecaster.py`), the one seam every aggregation
path (median, mean, base-combine, stacker, single survivor) returns through, so the publish
gate, the comment and the `NUMERIC_AGGREGATE` marker all read the floored distribution.

**The rule.** On a question whose platform is Mantic (`question_platform` reads it off
`page_url`; `PLATFORM_MANTIC` in `constants.py`), the published CDF carries at least
`MANTIC_OUT_OF_RANGE_TAIL_FLOOR` (`constants.py`, 0.05) beyond each OPEN bound: an open tail
under the floor is raised to exactly the floor (`cdf[0] = 0.05`, or `cdf[-1] = 0.95`), an open
tail already at or above it is left alone, and a closed bound stays at its exact `0.0` / `1.0`.
Both sides may move at once. The interior is rescaled affinely between the new endpoints, which
keeps it monotone and can only shrink steps, so the platform's max step cannot be newly
violated; the bins that sat exactly on the min step (the uniform-mixture tails of a
concentrated PCHIP build) do land below it, so `enforce_min_steps` (`numeric/pchip_cdf.py`)
runs again with the new endpoints as its caps. When an endpoint moves, the distribution is
rebuilt through `create_pchip_numeric_distribution` on the question's numeric view (the epoch
adapter for a date question, so `is_date` and the date rendering survive) with the floored
heights as both its CDF and its `declared_percentiles`, on the value axis the aggregate already
carried. `tests/test_out_of_range_floor.py` checks the server's `continuous_cdf` rules on the
result for every distinct grid in the recorded 2026-09-08 Mantic corpus
(`tests/data/mantic_cdf_grids_2026_09_08.json`: 88 grids, 3 to 450 bins, every bound
combination), on six aggregate shapes each.

**Why.** Mantic scores an out-of-range resolution against a fixed 0.05 reference,
`50 * ln(mass / 0.05)`: 5% there scores 0 and the structural 1% the pipeline builds whenever every
declared percentile sits inside the range scores -80.5. Mantic's writers set the ranges and are
paid for bot disagreement; in Series 1 half the date questions, a quarter of the discrete and an
eighth of the numeric ones resolved outside the displayed range. Applying the same floor to every
Series 1 competitor's own 4,082 published distributions cost at most 1.9 points per question on
average for any type and gained up to 9.1 for thin-tailed bots
(`scratch_docs_and_planning/mantic_adversarial_candidates_2026-09-08.md`, section 1).
Operator-approved 2026-09-08.

**The platform gate.** A Metaculus aggregate comes back as the very same object, byte-identical;
per-member forecasts are never touched (their `MEMBER_FORECAST` tails keep measuring what the
models declared); the backtest, ablation and benchmark harnesses replay Metaculus questions, so
the gate covers them without a flag.

**Telemetry.** `NUMERIC_AGGREGATE` keeps `oor_low` / `oor_high` as the PUBLISHED tails and adds
three trailing fields: `oor_low_raw` / `oor_high_raw`, the aggregate's own tails before the
floor, and `tail_floor`, the floor that moved an endpoint (`0.000000` on Metaculus, on a
closed-bound question, or when the tails already met it). Raw against published is how the floor
gets benchmarked on this bot's own forecasts once live Mantic telemetry accumulates.

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
