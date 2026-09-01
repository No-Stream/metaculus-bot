"""Era-bucketed numeric-width / calibration monitor (READ-ONLY, free).

Tracks how wide the bot's published numeric distributions are, and how well
that width is calibrated, split by config era. Era boundaries are
**merge-to-main timestamps**, not authoring dates — see the constants below.
The bot has historically oscillated between too-wide and too-narrow numeric
forecasts:

  * Until 2026-05-18 the pipeline INTENTIONALLY widened tails (``k_tail=1.25``
    in the tail-widening pass).
  * On 2026-05-18 widening was turned off (``k_tail=1.0``, identity) after a
    calibration study showed the widened tails were too fat.
  * On 2026-07-21 the july15 bundle landed, whose width-relevant piece is the
    Time-Series-Anchor prompt clause. It pushes "sharpen, don't widen"
    (published low-tail coverage was ~0.03 vs a 0.10 target — badly too wide),
    so the forward risk flips toward over-sharpening. This monitor is the
    loop-closer for that transition. The same merge dropped the forecaster
    roster from six models to the latest-per-vendor triple and lowered
    ``MIN_FORECASTERS_TO_PUBLISH``, so a width shift across this boundary
    cannot be attributed to the anchor alone. The bucket stays empty until a
    post-bundle numeric question resolves and is pulled.

Per era it reports, on the bot's PUBLISHED 201-point CDF:

  * central-80% coverage  = fraction of PIT in [0.10, 0.90]  (calibrated 0.80)
  * central-50% coverage  = fraction of PIT in [0.25, 0.75]  (calibrated 0.50)
    both with Beta-Binomial / Jeffreys-prior 95% CIs.
  * cov@10 = P(PIT <= 0.10)  (calibrated 0.10; low-tail coverage)
  * cov@50 = P(PIT <= 0.50)  (calibrated 0.50; directional bias / below-median)
  * cov@90 = P(PIT <= 0.90)  (calibrated 0.90; high-tail coverage)
  * PIT std (calibrated Uniform(0,1) std = 1/sqrt(12) ~= 0.289; smaller => PIT
    piled in the center => distributions too WIDE; larger => piled at the
    extremes => too NARROW).
  * median relative band width = median over questions of (P90 - P10) / |P50|,
    read off the published CDF. This is the RAW sharpness metric and does not
    depend on resolutions — it answers "how wide are we, in absolute terms",
    complementing the coverage metrics which answer "is that width calibrated".
  * band_miss, split into its low and high tails. ``band_miss`` is the
    out-of-band rate (P(PIT < 0.10) + P(PIT > 0.90), i.e. 1 - raw cov80); the
    split is what separates a band that is too TIGHT (both tails high) from one
    that is the right width but MIS-CENTERED (one tail carries the misses).
    ``cov80`` alone cannot express that distinction, and the two call for
    opposite corrections.

PIT is F_bot(resolution) evaluated on the canonical Metaculus value grid
(``build_cdf_value_grid``). Two out-of-range cases, and they differ by what the
platform told us (see ``compute_pit_reading``): a STRING marker
(``below_lower_bound`` / ``above_upper_bound``) gives no value, so the reading is
the INTERVAL our own tail mass pins F to and every coverage column counts it on
band INTERSECTION while PIT std / mean PIT exclude it; a NUMERIC resolution
beyond the grid keeps a point PIT, scored off the members' declared-percentile
curves rather than the grid clamp. Method mirrors
``scratch/calibration_audit_2026-07-16/mc_numeric_calibration.py``.

Alongside the era table it renders a per-QUESTION section, the STARVED OUTER TAIL
scan: open bounds whose published mass beyond the members' declared outer anchor
is pinned at the platform's structural minimum step, so every resolution out
there earns the same floor score. That is a different failure from a
mis-calibrated width — it is a cliff at a fixed location rather than a band of
the wrong size — which is why it reads per question and not per era. See
``STARVED_OUTER_TAIL_FLOOR_MULTIPLE``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

import numpy as np
from scipy import stats

from metaculus_bot.api_preflight import verify_metaculus_api_identity
from metaculus_bot.numeric.pchip_cdf import build_cdf_value_grid
from metaculus_bot.performance_analysis.analysis import (
    B4E9DF0_MERGED_AT,
    PitReading,
    out_of_range_pit_reading,
    pit_band_count,
    pit_on_grid,
    pit_point_values,
)
from metaculus_bot.performance_analysis.cohorts import (
    EXCLUSION_COHORTS,
    KNOWN_BUG_SHORTHAND,
    parse_exclude_qids,
)
from metaculus_bot.performance_analysis.collector import build_performance_dataset, load_dataset
from metaculus_bot.performance_analysis.parsing import declared_anchors, is_anonymous_model_key
from metaculus_bot.performance_analysis.scaling import grid_zero_point as _grid_zero_point
from metaculus_bot.performance_analysis.scoring import numeric_log_score
from metaculus_bot.time_utils import parse_iso_utc

logger: logging.Logger = logging.getLogger(__name__)

NUMERIC_TYPES: tuple[str, ...] = ("numeric", "discrete")

# Calibrated reference values, surfaced in the legend so a reader knows which
# direction "off" points.
UNIFORM_PIT_STD: float = 1.0 / np.sqrt(12.0)  # ~0.2887

# Below this many PITs, a row's point metrics (cov@10/50/90, PIT std, mean PIT,
# band_miss) are not estimates: their resolution is 1/n, coarser than the finest
# calibrated target they are compared against (cov@10 = 0.10), so the value can
# only land on a grid whose spacing exceeds the quantity being measured. At n=1
# pit_std is exactly 0.0, which reads as "maximally too wide" while carrying no
# information. Those cells render ``n/a`` in the markdown; the JSON keeps the raw
# values alongside an ``underpowered`` flag, since a script can decide for itself
# but a reader cannot un-see a number. cov80/cov50 are exempt — they carry CIs
# that widen honestly at small n, which is exactly the disclosure the point
# metrics lack.
MIN_N_FOR_POINT_METRICS: int = 10

# --- Starved outer tails (the open-bound p99 cliff) --------------------------------------
#
# Distinct from the max-step smear ``CDF_MAXSTEP_CLIP`` already covers. On an OPEN bound the
# declared outer tail can end up routed past the displayed range entirely, leaving every
# in-range bin above the declared p99 pinned at the platform's structural minimum step
# (``0.01 / N`` per bin). Every resolution in that band then earns the same floor score —
# ``50 * ln(min_step / baseline)``, about -219 on ANY grid size — so the band is a cliff
# rather than a gradient, and no modest widening walks out of it: the defect is a step
# function in where the declared p99 lands. q45218 published its WINNING rig-count forecast
# with 27 such bins starting one rig above its declared p99 (a flat -219.5 zone 16 rigs from
# the resolution), and the same shape is what made q44182 (-219.0) the worst record on the
# board. Detector only: any width response stays gated on the standing ``k_tail`` hold.
#
# The first thing it measured is that the shape is SYSTEMATIC, not a rare accident: 68 of the
# 417 measurable open-bound sides in the archived cohort (16%, across 49 distinct questions,
# 19 of them starved on both sides) sit at the floor. Read a fire as "this question carries a
# cliff", not as "something went wrong on this question".
#
# There is deliberately NO publish-time twin of this detector (a ``STARVED_OUTER_TAIL`` WARN
# beside ``OPEN_BOUND_PILING``), and the reason is worth keeping: on DISCRETE questions —
# exactly where both motivating records live — ``numeric.pipeline._build_discrete_distribution``
# overwrites ``declared_percentiles`` with a resample grid pinned to the raw bounds, so a
# detector reading that field at publish time would put the anchor AT the bound and quietly
# never fire on the cohort it exists for. That is the same trap
# ``log_open_bound_piling_diagnostics`` documents and dodges by taking the sanitized
# declarations as an argument. Firing correctly on the published aggregate needs each member's
# sanitized declarations threaded from ``forecaster_runners`` to the aggregation site, i.e. new
# plumbing on the publish path. The alternative that needs no plumbing is to locate the band
# WITHOUT the declaration — the terminal run of bins sitting at the min step — which is a
# second trigger definition to calibrate, so it is a decision rather than an oversight.

# A band whose MEAN per-bin mass is below this multiple of the platform's per-bin minimum
# step (``0.01 / N``, the server's ``round(0.01 / N, 9)`` rule) holds no declared shape — it
# is carrying the structural minimum and nothing else. Scale-free on purpose: an absolute
# mass threshold cannot tell a 2-bin band holding 0.003 (real density, harmless) from a
# 27-bin band holding 0.004 (the cliff), and the pipeline's own applied floor is ~1.1x the
# platform's, so a multiple is the quantity that means "flat".
#
# Calibrated against the archived performance dataset (271 numeric/discrete records, 417
# measurable open-bound sides): q45218 reads 1.12 on both sides and q44182 1.46 high / 1.13
# low, so both fire, and their measured flat-zone scores reproduce the published ones exactly
# (-219.53 and -219.02). q44842 — which deliberately declared its p99 past the displayed
# ceiling and won spot peer +24.4 — is not measurable on either side, so it cannot fire. The
# measured distribution is bimodal: 44 sides sit in [1.00, 1.25), right at the pipeline's own
# applied floor, then roughly 8 per 0.25-wide bucket up to 3.0, then the bulk above (median
# 10.3). So the exact cut is not load-bearing — 1.5 fires 52 sides, 2.0 fires 68, 2.5 fires 79
# — and 2.0 keeps margin over q44182's 1.46 without reaching into bands that carry shape.
# Receipts: scratch/next_season_bundle_2026-09/item19/.
STARVED_OUTER_TAIL_FLOOR_MULTIPLE: float = 2.0

# The band is measured from the most extreme percentile EVERY member declared, and that
# anchor has to actually be a tail anchor. A trimmed comment can leave the members sharing
# only their p50, and reading the band as "everything above the median" would call almost
# every record starved. Applied symmetrically: the low side needs an anchor at or below
# ``100 - this``. The canonical sets both clear it (p99 today, p97.5 in the 11-point era).
STARVED_OUTER_TAIL_MIN_ANCHOR_PERCENTILE: float = 90.0

# The platform's per-bin minimum step is ``round(0.01 / N, 9)`` where N = cdf_size - 1 (see
# the Metaculus CDF constraints in the repo instructions); the rounding is irrelevant at the
# ratios this detector reads.
_PLATFORM_MIN_STEP_NUMERATOR: float = 0.01


@dataclass(frozen=True)
class Era:
    """A config era: records whose bot_comment_created_at falls in
    ``[start, end)`` belong to this era. ``None`` bounds are open (-inf / +inf).
    """

    label: str
    start: datetime | None
    end: datetime | None

    def contains(self, dt: datetime) -> bool:
        return (self.start is None or dt >= self.start) and (self.end is None or dt < self.end)


# Config-flip boundaries that plausibly shift the numeric width distribution.
# These are the ONLY width-relevant flips (per CLAUDE.md era-bucketing guidance:
# bucket by pipeline-behavior changes, not every git hash).
#
# Each value is the committer timestamp of the MERGE COMMIT that carried the
# change onto `main`, never the authoring date of the commit on its branch:
# prod runs from `main`, so a change is live only from the moment it lands
# there. A branch can sit for days, and keying on the authoring date files every
# run in that gap under the wrong config. Re-derive with
# `TZ=UTC git log -1 --date=iso-local --format='%h %cd' <merge-sha>`.
WIDENING_FLIP = datetime(2026, 5, 18, 17, 21, 19, tzinfo=UTC)  # 0e85e1b: k_tail 1.25 -> 1.0
# b4e9df0 (july15 bundle) — aliased so this boundary and the max-step clamp screen's
# era gate can never disagree; the timestamp's single home is analysis.py.
TS_ANCHOR_ENABLE = B4E9DF0_MERGED_AT


def default_eras() -> list[Era]:
    """The three width-relevant config eras, oldest first.

    ``ts_anchor`` is the active era from 2026-07-21T17:07Z (``b4e9df0``) onward.
    That merge landed the timeseries-anchor "sharpen, don't widen" clause
    alongside the 6-model-to-triple roster drop and gap-fill v2, so it is the
    july15 bundle's boundary rather than the anchor's alone. It stays empty for
    as long as no post-bundle numeric question has resolved and been pulled, and
    ``compute_all_eras`` omits empty eras — so until then the table carries the
    two populated eras only, and the ``ts_anchor`` row is absent rather than
    present-and-empty. (Alongside those it still emits ``no_timestamp`` when any
    record lacks a comment timestamp, plus the spanning ``all`` row.)
    """
    return [
        Era("widening_on (k_tail=1.25)", None, WIDENING_FLIP),
        Era("widening_off (k_tail=1.0)", WIDENING_FLIP, TS_ANCHOR_ENABLE),
        Era("ts_anchor (sharpen)", TS_ANCHOR_ENABLE, None),
    ]


NO_TIMESTAMP_LABEL = "no_timestamp"


def assign_era(record: dict, eras: list[Era]) -> str:
    """Return the era label for a record, or ``NO_TIMESTAMP_LABEL`` when the
    bot-comment timestamp is missing/unparseable (can't be era-attributed)."""
    dt = parse_iso_utc(record.get("bot_comment_created_at"))
    if dt is None:
        return NO_TIMESTAMP_LABEL
    for era in eras:
        if era.contains(dt):
            return era.label
    return NO_TIMESTAMP_LABEL


def jeffreys_ci(k: int, n: int, cl: float = 0.95) -> tuple[float, float, float]:
    """Beta-Binomial posterior mean + equal-tailed CI under a Jeffreys(0.5, 0.5)
    prior. Mirrors ``bb`` in mc_numeric_calibration.py."""
    a = 0.5 + k
    b = 0.5 + (n - k)
    mean = a / (a + b)
    lo = float(stats.beta.ppf((1 - cl) / 2, a, b))
    hi = float(stats.beta.ppf(1 - (1 - cl) / 2, a, b))
    return mean, lo, hi


def _cdf_and_grid(record: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (cdf, value_grid) for a numeric/discrete record, or None if the
    record lacks the bounds / CDF needed to build the grid.

    Prefers the API's grid-exact ``scaling.continuous_range`` when present (it
    already encodes log-vs-linear spacing, so no scale has to be re-derived from
    ``zero_point``); falls back to reconstructing via ``build_cdf_value_grid``
    with a zero_point interpretation that handles the ``zero_point == 0`` log
    case (see ``_grid_zero_point``).
    """
    cdf = record.get("our_forecast_values")
    scaling = record.get("scaling") or {}
    lo, hi = scaling.get("range_min"), scaling.get("range_max")
    if cdf is None or lo is None or hi is None or len(cdf) < 3:
        return None
    cdf_arr = np.asarray(cdf, dtype=float)
    api_grid = scaling.get("continuous_range")
    if api_grid is not None and len(api_grid) == len(cdf):
        return cdf_arr, np.asarray(api_grid, dtype=float)
    zp = _grid_zero_point(scaling.get("zero_point"), float(lo))
    grid = build_cdf_value_grid(float(lo), float(hi), zp, len(cdf))
    return cdf_arr, grid


def compute_pit(record: dict) -> float | None:
    """The record's POINT PIT = F_bot(resolution) on the canonical value grid.

    None when the record can't be scored AND when the reading is set-valued (a STRING
    out-of-range resolution, whose PIT is an interval — see :func:`compute_pit_reading`).
    A NUMERIC resolution beyond the grid still has a point PIT, read off the members'
    declared-percentile curves rather than the grid clamp."""
    reading = compute_pit_reading(record)
    return reading.point if reading is not None else None


def compute_pit_reading(record: dict) -> PitReading | None:
    """The record's :class:`PitReading`, or None when the record can't be scored.

    Two cases, and the difference is what the platform told us:

    * A STRING out-of-range resolution (``below_lower_bound`` / ``above_upper_bound``)
      gives no value, so the reading is the INTERVAL our own published tail mass pins
      ``F(resolution)`` to — ``[cdf[-1], 1]`` or ``[0, cdf[0]]``. The convention lives in
      ``analysis.out_of_range_pit_reading``.
    * A NUMERIC resolution gives a point PIT through the shared :func:`pit_on_grid`, whose
      docstring holds the out-of-grid rule (declared-percentile fallback beyond the grid,
      endpoint clamp only when no member curve is usable).

    ``PitReading.oob_side`` reports the beyond-grid side in both cases, which is what the
    ``n_oob_*`` counters read.
    """
    built = _cdf_and_grid(record)
    if built is None:
        return None
    cdf, grid = built
    res = record.get("resolution_parsed")
    out_of_range = out_of_range_pit_reading(res, cdf)
    if out_of_range is not None:
        return out_of_range
    if isinstance(res, (int, float)) and not isinstance(res, bool):
        pit, oob_side = pit_on_grid(float(res), grid, cdf, record.get("per_model_numeric_percentiles"))
        return PitReading.from_point(pit, oob_side=oob_side)
    return None


def relative_band_width(record: dict, *, median_floor: float = 1e-9) -> float | None:
    """(P90 - P10) / |P50| read off the published CDF (resolution-independent).

    Returns None when the record lacks a usable CDF, or when |P50| is below
    ``median_floor`` (the ratio blows up for questions centred on ~0, e.g. a
    signed-change quantity; those are excluded and counted rather than
    poisoning the median)."""
    built = _cdf_and_grid(record)
    if built is None:
        return None
    cdf, grid = built
    # Invert the (monotone) CDF: value at quantile q = interp of grid over cdf.
    p10, p50, p90 = (float(np.interp(q, cdf, grid)) for q in (0.10, 0.50, 0.90))
    if abs(p50) < median_floor:
        return None
    return (p90 - p10) / abs(p50)


@dataclass
class EraWidthMetrics:
    label: str
    n_pit: int
    """PIT READINGS in this row — the coverage denominator, points and intervals together."""
    n_point: int
    """Readings carrying a point value: the denominator of pit_std / mean_pit."""
    n_oob_interval: int
    """``n_pit - n_point``: out-of-range resolutions whose PIT is a set (see ``PitReading``).

    A subset of ``n_oob_low + n_oob_high``, which also counts NUMERIC beyond-grid
    resolutions — those keep a point PIT off the members' declared curves.
    """
    n_eff: int
    n_width: int
    n_excluded: int
    n_oob_low: int
    n_oob_high: int
    cov80: tuple[float, float, float]
    cov50: tuple[float, float, float]
    cov_at_10: float
    cov_at_50: float
    cov_at_90: float
    pit_std: float | None
    mean_pit: float | None
    median_rel_width: float | None
    band_miss: float
    band_lo: float
    band_hi: float

    @property
    def ci_clustered(self) -> bool:
        """True when clustering actually widened this row's CIs.

        ``n_eff < n`` means at least one post carried more than one record. On
        every archived pull that has never happened (see
        ``_n_effective_clusters``), so the correction is normally inert and the
        rendered CI is the naive one — which the table has to say, or the legend's
        cluster-widening claim describes something that did not occur.
        """
        return self.n_eff < self.n_pit

    @property
    def underpowered(self) -> bool:
        """True when this row has too few PIT readings for its point metrics to be read."""
        return self.n_pit < MIN_N_FOR_POINT_METRICS

    @property
    def point_metrics_underpowered(self) -> bool:
        """Same floor applied to the point-only denominator, which set-valued readings shrink.

        pit_std and mean_pit are computed over ``n_point``, so a row can clear the floor on
        readings and still be under it on point values — that row's std is as uninformative
        as any other under-floor one.
        """
        return self.n_point < MIN_N_FOR_POINT_METRICS

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "n_pit": self.n_pit,
            "n_point": self.n_point,
            "n_oob_interval": self.n_oob_interval,
            "point_metrics_underpowered": self.point_metrics_underpowered,
            "n_eff": self.n_eff,
            "ci_clustered": self.ci_clustered,
            "underpowered": self.underpowered,
            "n_width": self.n_width,
            "n_excluded": self.n_excluded,
            "n_oob_low": self.n_oob_low,
            "n_oob_high": self.n_oob_high,
            "cov80": {"mean": self.cov80[0], "lo": self.cov80[1], "hi": self.cov80[2]},
            "cov50": {"mean": self.cov50[0], "lo": self.cov50[1], "hi": self.cov50[2]},
            "cov_at_10": self.cov_at_10,
            "cov_at_50": self.cov_at_50,
            "cov_at_90": self.cov_at_90,
            "pit_std": self.pit_std,
            "mean_pit": self.mean_pit,
            "median_rel_width": self.median_rel_width,
            "band_miss": self.band_miss,
            "band_lo": self.band_lo,
            "band_hi": self.band_hi,
        }


def _n_effective_clusters(post_ids: list[object]) -> int:
    """Count distinct question families for the CI's effective sample size.

    Records sharing a ``post_id`` are one correlated family: the collector expands
    a ``group_of_questions`` post into one record per sub-question, and those share
    a series, a window and a resolution source. A record with no ``post_id``
    (``None``) is treated as its own family — assigned a unique sentinel by
    position so it is never merged with another None-post record — since we can't
    prove it shares a family with anything else.

    **The correction is currently inert, and the table says so per row.** Measured
    2026-08-25 across all archived pulls (residual_2026-06-15 through
    residual_2026-08-24, plus coherence_2026-07-15): every post carried exactly one
    resolved record, so ``n_eff == n`` everywhere and the clustered CI equals the
    naive one. The mechanism is kept because a group post resolving into the
    tournament is a matter of question supply, not of code — but nothing may claim
    the CIs have been widened unless ``EraWidthMetrics.ci_clustered`` says they
    were. (An earlier version of this comment asserted "~62% of records share a
    post"; no archived dataset supports that figure.)
    """
    clusters: set[object] = set()
    for i, pid in enumerate(post_ids):
        clusters.add(pid if pid is not None else f"__no_post_{i}")
    return len(clusters)


@dataclass(frozen=True, slots=True)
class _EraSamples:
    """The per-record readings one era contributes: PITs, their posts, and band widths."""

    readings: list[PitReading]
    pit_post_ids: list[object]
    widths: list[float]
    n_oob_low: int
    n_oob_high: int


def _collect_era_samples(records: list[dict]) -> _EraSamples:
    """Read every numeric/discrete record's PIT reading and relative band width.

    OOB is a property of the RESOLUTION (beyond the grid), not of the PIT value: an
    out-of-grid resolution scored off the declared-percentile curves rarely lands at
    exactly 0.0/1.0, and an in-grid PIT of exactly 0.0 (closed bound, resolution at the
    minimum) is not OOB — which is why the side comes from the reading rather than from
    comparing the PIT against 0 or 1.
    """
    readings: list[PitReading] = []
    pit_post_ids: list[object] = []
    widths: list[float] = []
    n_oob_low = 0
    n_oob_high = 0

    for r in records:
        if r.get("type") not in NUMERIC_TYPES:
            continue
        reading = compute_pit_reading(r)
        if reading is not None:
            readings.append(reading)
            pit_post_ids.append(r.get("post_id"))
            if reading.oob_side == "low":
                n_oob_low += 1
            elif reading.oob_side == "high":
                n_oob_high += 1
        w = relative_band_width(r)
        if w is not None:
            widths.append(w)

    return _EraSamples(
        readings=readings,
        pit_post_ids=pit_post_ids,
        widths=widths,
        n_oob_low=n_oob_low,
        n_oob_high=n_oob_high,
    )


def _fraction(readings: list[PitReading], predicate: Callable[[PitReading], bool]) -> float:
    """Fraction of readings satisfying ``predicate`` (readings is never empty here)."""
    return sum(1 for reading in readings if predicate(reading)) / len(readings)


def compute_era_metrics(label: str, records: list[dict], n_excluded: int = 0) -> EraWidthMetrics | None:
    """Compute width/calibration metrics for one era's records. Returns None if
    no numeric/discrete records in the era yield a PIT.

    ``n_excluded`` is carried through for reporting only — the caller has
    already filtered those records out. It exists so the rendered table can say
    that rows were dropped rather than silently reporting a smaller n.
    """
    samples = _collect_era_samples(records)
    if not samples.readings:
        return None

    readings = samples.readings
    n = len(readings)
    # Point statistics run on the point readings only: a set-valued (out-of-range)
    # reading has no value to average, and imputing its midpoint would manufacture one.
    points = np.asarray(pit_point_values(readings), dtype=float)
    cov80_k = pit_band_count(readings, 0.10, 0.90)
    cov50_k = pit_band_count(readings, 0.25, 0.75)

    # Coverage CIs are computed at n_eff (distinct post_ids) rather than the raw
    # question count, so that a post carrying several correlated sub-questions
    # cannot narrow the CI as if they were independent. Cluster on post_id only —
    # the one grouping key already on every record; a record missing a post_id
    # counts as its own cluster (via a unique sentinel) so it is never merged with
    # another. The point estimate is unchanged (cov_k / n); only the CI width
    # reflects n_eff, via jeffreys_ci(round(cov_k * n_eff / n), n_eff). On every
    # archived pull n_eff == n, so this is a no-op there — see
    # ``_n_effective_clusters`` and the per-row ``ci_clustered`` marker.
    n_eff = _n_effective_clusters(samples.pit_post_ids)
    cov80 = jeffreys_ci(round(cov80_k * n_eff / n), n_eff)
    cov50 = jeffreys_ci(round(cov50_k * n_eff / n), n_eff)

    # Out-of-band rate, split by tail. band_miss == 1 - raw cov80, so it adds no
    # information on its own; the low/high split is the point — it distinguishes
    # a band that is too tight (both tails elevated) from one of roughly the
    # right width that is mis-centered (misses piled in one tail), which cov80
    # cannot, and which call for opposite corrections.
    # A set-valued reading misses a tail only when the WHOLE interval lies outside it,
    # which keeps the band_miss == 1 - cov80 identity exact (an interval that fails to
    # intersect [0.10, 0.90] lies entirely on one side of it).
    band_lo = _fraction(readings, lambda reading: reading.entirely_below(0.10))
    band_hi = _fraction(readings, lambda reading: reading.entirely_above(0.90))

    return EraWidthMetrics(
        label=label,
        n_pit=n,
        n_point=len(points),
        n_oob_interval=n - len(points),
        n_eff=n_eff,
        n_width=len(samples.widths),
        n_excluded=n_excluded,
        n_oob_low=samples.n_oob_low,
        n_oob_high=samples.n_oob_high,
        cov80=cov80,
        cov50=cov50,
        cov_at_10=_fraction(readings, lambda reading: reading.at_or_below(0.10)),
        cov_at_50=_fraction(readings, lambda reading: reading.at_or_below(0.50)),
        cov_at_90=_fraction(readings, lambda reading: reading.at_or_below(0.90)),
        pit_std=(float(points.std()) if len(points) else None),
        mean_pit=(float(points.mean()) if len(points) else None),
        median_rel_width=(float(np.median(samples.widths)) if samples.widths else None),
        band_miss=band_lo + band_hi,
        band_lo=band_lo,
        band_hi=band_hi,
    )


def compute_all_eras(
    data: list[dict],
    eras: list[Era] | None = None,
    exclude_qids: frozenset[str] | None = None,
) -> list[EraWidthMetrics]:
    """Bucket records by era and compute per-era metrics. Eras with no scorable
    numeric records are omitted. Emits an ``all`` row spanning every era.

    ``exclude_qids`` drops the named questions from every row and reports the
    dropped count per row (``EraWidthMetrics.n_excluded``, rendered in the
    table), so an exclusion is never silent. Pass one of the documented cohorts in
    ``EXCLUSION_COHORTS`` — ``KNOWN_BUG_QIDS`` (known pipeline bugs),
    ``DEGRADED_RUN_QIDS`` (dry-key 1-of-3 publishes) or ``PARTIAL_DEGRADED_QIDS``
    (2-of-3) — rather than re-hardcoding ids: the known-bug set's private copies have
    already drifted, and three rounds retyped the degraded ids before they had a home.
    """
    if eras is None:
        eras = default_eras()
    excluded = exclude_qids or frozenset()
    order = [e.label for e in eras] + [NO_TIMESTAMP_LABEL]
    buckets: dict[str, list[dict]] = {lbl: [] for lbl in order}
    excluded_counts: dict[str, int] = dict.fromkeys(order, 0)
    numeric_records: list[dict] = []
    n_excluded_total = 0
    for r in data:
        if r.get("type") not in NUMERIC_TYPES:
            continue
        label = assign_era(r, eras)
        # The collector writes question_id straight from the API (an int), so
        # coerce rather than compare an int against a string set and no-op.
        if str(r.get("question_id")) in excluded:
            excluded_counts[label] += 1
            n_excluded_total += 1
            continue
        numeric_records.append(r)
        buckets[label].append(r)

    results: list[EraWidthMetrics] = []
    for lbl in order:
        m = compute_era_metrics(lbl, buckets[lbl], n_excluded=excluded_counts[lbl])
        if m is not None:
            results.append(m)
    overall = compute_era_metrics("all", numeric_records, n_excluded=n_excluded_total)
    if overall is not None:
        results.append(overall)
    return results


class OuterTailVerdict(StrEnum):
    """What one open-bound side of one published CDF was found to be.

    Only ``STARVED`` and ``HEALTHY`` are measurements; the rest say why no measurement was
    possible, and are counted and rendered rather than folded into ``HEALTHY`` — an absent
    declaration must never read as "this tail is fine".
    """

    STARVED = "starved"
    HEALTHY = "healthy"
    NO_USABLE_CDF = "no_usable_cdf"
    NO_MEMBER_CURVE = "no_member_curve"
    NO_SHARED_ANCHOR = "no_shared_anchor"
    ANCHOR_NOT_EXTREME = "anchor_not_extreme"
    DECLARED_BEYOND_BOUND = "declared_beyond_bound"
    EMPTY_BAND = "empty_band"


@dataclass(frozen=True, slots=True)
class OuterTailReading:
    """One open-bound side of one record, measured for outer-tail starvation.

    The measured fields are None exactly when ``verdict`` is one of the unmeasurable ones.
    ``DECLARED_BEYOND_BOUND`` is the common and legitimate case: the members put their outer
    anchor past the displayed bound, so the tail is expressed as out-of-range mass and there
    is no in-range band to starve.
    """

    question_id: object
    title: str
    side: str
    verdict: OuterTailVerdict
    declared_percentile: float | None = None
    declared_value: float | None = None
    bound_value: float | None = None
    tail_mass: float | None = None
    """Published mass in the bins lying fully beyond the declared anchor."""
    beyond_bound_mass: float | None = None
    """Published mass beyond the DISPLAYED bound (``1 - cdf[-1]`` / ``cdf[0]``).

    Reported beside ``tail_mass`` because the inversion is the defect's signature: q45218 held
    0.0042 across the 27 in-range bins above its declared p99 and 0.0100 past the ceiling.
    """
    band_bins: int | None = None
    mean_bin_mass: float | None = None
    platform_min_step: float | None = None
    floor_multiple: float | None = None
    """``mean_bin_mass / platform_min_step`` — the trigger. 1.0 is exactly at the floor."""
    flat_zone_log_score: float | None = None
    """Bot-side Metaculus log score a resolution in the band's THINNEST bin would earn."""

    @property
    def starved(self) -> bool:
        return self.verdict is OuterTailVerdict.STARVED

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "title": self.title,
            "side": self.side,
            "verdict": self.verdict,
            "declared_percentile": self.declared_percentile,
            "declared_value": self.declared_value,
            "bound_value": self.bound_value,
            "tail_mass": self.tail_mass,
            "beyond_bound_mass": self.beyond_bound_mass,
            "band_bins": self.band_bins,
            "mean_bin_mass": self.mean_bin_mass,
            "platform_min_step": self.platform_min_step,
            "floor_multiple": self.floor_multiple,
            "flat_zone_log_score": self.flat_zone_log_score,
        }


@dataclass(frozen=True, slots=True)
class OuterTailScan:
    """Every open-bound side scanned in one dataset, plus the exclusion count."""

    readings: list[OuterTailReading]
    n_excluded: int = 0

    @property
    def n_scanned(self) -> int:
        return len(self.readings)

    @property
    def starved(self) -> list[OuterTailReading]:
        """The flagged sides, worst flat-zone score first."""
        return sorted((r for r in self.readings if r.starved), key=_flat_zone_sort_key)

    @property
    def verdict_counts(self) -> Counter[OuterTailVerdict]:
        return Counter(r.verdict for r in self.readings)

    def to_dict(self) -> dict:
        return {
            "n_scanned": self.n_scanned,
            "n_excluded": self.n_excluded,
            "verdict_counts": dict(self.verdict_counts),
            "readings": [r.to_dict() for r in self.readings],
        }


def _flat_zone_sort_key(reading: OuterTailReading) -> float:
    """Worst-first key. Only measured readings are ranked, so the fallback never applies."""
    return reading.flat_zone_log_score if reading.flat_zone_log_score is not None else 0.0


def _member_declared_anchors(record: dict) -> dict[str, dict[float, float]]:
    """``{model: {percentile label: value}}`` for the members whose curve is usable.

    Anonymous ``Forecaster N`` keys are EXCLUDED, matching ``declared_percentile_pit`` and
    ``max_step_clamp_screen``: on a stacker-fired record that positional bucket holds the
    stacker's AGGREGATE, so pooling it into a median-of-members moves the anchor.
    """
    curves: dict[str, dict[float, float]] = {}
    for model, pairs in (record.get("per_model_numeric_percentiles") or {}).items():
        if is_anonymous_model_key(str(model)):
            continue
        try:
            anchors, _conflicts = declared_anchors(pairs)
        except (TypeError, ValueError, IndexError):
            # Archived per-model pairs are parsed from comment text and can be malformed;
            # an unusable curve reads as no-curve rather than raising (same rule as
            # ``analysis._single_curve_pit``).
            continue
        if len(anchors) >= 2:
            curves[str(model)] = anchors
    return curves


def _shared_extreme_anchor(record: dict, side: str) -> tuple[float, float] | OuterTailVerdict:
    """``(percentile label, median member value)`` at the most extreme SHARED anchor.

    Shared across members on purpose: medianing p99 against p97.5 would mix two different
    quantities, and the canonical percentile set changed mid-season (11-point p2.5..p97.5 ->
    13-point p1..p99), so the label cannot be hardcoded. Returns the verdict that explains
    itself when no anchor qualifies.
    """
    curves = _member_declared_anchors(record)
    if not curves:
        return OuterTailVerdict.NO_MEMBER_CURVE
    shared_labels = set.intersection(*(set(anchors) for anchors in curves.values()))
    if not shared_labels:
        return OuterTailVerdict.NO_SHARED_ANCHOR
    label = max(shared_labels) if side == "high" else min(shared_labels)
    extreme_enough = (
        label >= STARVED_OUTER_TAIL_MIN_ANCHOR_PERCENTILE
        if side == "high"
        else label <= 100.0 - STARVED_OUTER_TAIL_MIN_ANCHOR_PERCENTILE
    )
    if not extreme_enough:
        return OuterTailVerdict.ANCHOR_NOT_EXTREME
    return label, float(np.median([anchors[label] for anchors in curves.values()]))


def measure_outer_tails(
    record: dict,
    *,
    floor_multiple: float = STARVED_OUTER_TAIL_FLOOR_MULTIPLE,
) -> list[OuterTailReading]:
    """One :class:`OuterTailReading` per OPEN bound of a numeric/discrete record.

    A closed bound is never scanned: the CDF is pinned to 0.0 / 1.0 there, so a thin terminal
    band is the question's own edge rather than a tail our declaration put out of reach.
    """
    built = _cdf_and_grid(record)
    readings: list[OuterTailReading] = []
    for side, is_open in (("low", record.get("open_lower_bound")), ("high", record.get("open_upper_bound"))):
        if not is_open:
            continue
        if built is None:
            readings.append(_unmeasurable(record, side, OuterTailVerdict.NO_USABLE_CDF))
            continue
        cdf, grid = built
        readings.append(_measure_one_outer_tail(record, side, cdf, grid, floor_multiple=floor_multiple))
    return readings


def _unmeasurable(record: dict, side: str, verdict: OuterTailVerdict) -> OuterTailReading:
    """A reading that carries only the reason no measurement was possible."""
    return OuterTailReading(
        question_id=record.get("question_id"),
        title=str(record.get("title") or ""),
        side=side,
        verdict=verdict,
    )


@dataclass(frozen=True, slots=True)
class _OuterBand:
    """The published bins lying FULLY beyond the declared anchor: which ones, and their mass.

    Fully beyond on purpose, so the mass and the bin count describe the same segment — a
    partially-covered bin would inflate one and not the other, and the ratio between them is
    the whole measurement.
    """

    bin_indices: range
    mass: float


def _outer_band(cdf: np.ndarray, grid: np.ndarray, declared_value: float, side: str) -> _OuterBand | OuterTailVerdict:
    """The band beyond ``declared_value``, or the verdict explaining why there isn't one."""
    if side == "high":
        if declared_value >= float(grid[-1]):
            return OuterTailVerdict.DECLARED_BEYOND_BOUND
        first_band_bin = int(np.searchsorted(grid, declared_value, side="left"))
        band = _OuterBand(range(first_band_bin, len(cdf) - 1), float(cdf[-1] - cdf[first_band_bin]))
    else:
        if declared_value <= float(grid[0]):
            return OuterTailVerdict.DECLARED_BEYOND_BOUND
        last_band_edge = int(np.searchsorted(grid, declared_value, side="right")) - 1
        band = _OuterBand(range(last_band_edge), float(cdf[last_band_edge] - cdf[0]))
    return band if len(band.bin_indices) else OuterTailVerdict.EMPTY_BAND


def _flat_zone_log_score(record: dict, cdf: np.ndarray, grid: np.ndarray, band: _OuterBand) -> float:
    """The bot-side log score a resolution in the band's THINNEST bin would earn.

    The thinnest bin rather than the mean one: on a starved band every bin sits at the same
    floor, so this reads the cliff's depth, and on a band that still carries shape it reads
    the worst landing rather than an average nobody experiences.
    """
    scaling = record["scaling"]
    lower = float(scaling["range_min"])
    thinnest_bin = min(band.bin_indices, key=lambda j: float(cdf[j + 1] - cdf[j]))
    return numeric_log_score(
        [float(v) for v in cdf],
        float((grid[thinnest_bin] + grid[thinnest_bin + 1]) / 2.0),
        lower,
        float(scaling["range_max"]),
        open_lower_bound=bool(record.get("open_lower_bound")),
        open_upper_bound=bool(record.get("open_upper_bound")),
        zero_point=_grid_zero_point(scaling.get("zero_point"), lower),
    )


def _measure_one_outer_tail(
    record: dict,
    side: str,
    cdf: np.ndarray,
    grid: np.ndarray,
    *,
    floor_multiple: float,
) -> OuterTailReading:
    scaling = record.get("scaling") or {}
    if float(scaling["range_max"]) <= float(scaling["range_min"]):
        # A degenerate range has no bins to score; the same guard numeric_pit_analysis applies.
        return _unmeasurable(record, side, OuterTailVerdict.NO_USABLE_CDF)

    anchor = _shared_extreme_anchor(record, side)
    if isinstance(anchor, OuterTailVerdict):
        return _unmeasurable(record, side, anchor)
    percentile, declared_value = anchor

    band = _outer_band(cdf, grid, declared_value, side)
    if isinstance(band, OuterTailVerdict):
        return _unmeasurable(record, side, band)

    platform_min_step = _PLATFORM_MIN_STEP_NUMERATOR / (len(cdf) - 1)
    mean_bin_mass = band.mass / len(band.bin_indices)
    measured_floor_multiple = mean_bin_mass / platform_min_step

    return OuterTailReading(
        question_id=record.get("question_id"),
        title=str(record.get("title") or ""),
        side=side,
        verdict=(OuterTailVerdict.STARVED if measured_floor_multiple < floor_multiple else OuterTailVerdict.HEALTHY),
        declared_percentile=percentile,
        declared_value=declared_value,
        bound_value=float(grid[-1] if side == "high" else grid[0]),
        tail_mass=band.mass,
        beyond_bound_mass=float(1.0 - cdf[-1]) if side == "high" else float(cdf[0]),
        band_bins=len(band.bin_indices),
        mean_bin_mass=mean_bin_mass,
        platform_min_step=platform_min_step,
        floor_multiple=measured_floor_multiple,
        flat_zone_log_score=_flat_zone_log_score(record, cdf, grid, band),
    )


def scan_outer_tails(
    data: list[dict],
    *,
    floor_multiple: float = STARVED_OUTER_TAIL_FLOOR_MULTIPLE,
    exclude_qids: frozenset[str] | None = None,
) -> OuterTailScan:
    """Measure every open-bound side of every numeric/discrete record in ``data``.

    ``exclude_qids`` takes the same cohorts as ``compute_all_eras`` and the count is reported,
    so an exclusion is a visible choice: a known-bug record's starved tail measures a retired
    defect, not the current pipeline.
    """
    excluded = exclude_qids or frozenset()
    readings: list[OuterTailReading] = []
    n_excluded = 0
    for record in data:
        if record.get("type") not in NUMERIC_TYPES:
            continue
        if str(record.get("question_id")) in excluded:
            n_excluded += 1
            continue
        readings.extend(measure_outer_tails(record, floor_multiple=floor_multiple))
    return OuterTailScan(readings=readings, n_excluded=n_excluded)


def _fmt_measured(value: float | None, spec: str) -> str:
    """Format a measured field; only starved rows are rendered, so None never reaches here."""
    return "n/a" if value is None else format(value, spec)


def render_starved_outer_tails(
    scan: OuterTailScan, *, floor_multiple: float = STARVED_OUTER_TAIL_FLOOR_MULTIPLE
) -> str:
    """Markdown section listing the flagged sides, worst flat-zone score first."""
    counts = scan.verdict_counts
    unmeasurable = {
        verdict.value: counts[verdict]
        for verdict in OuterTailVerdict
        if verdict not in (OuterTailVerdict.STARVED, OuterTailVerdict.HEALTHY) and counts[verdict]
    }
    lines = [
        "## Starved outer tails (open-bound p99 cliff)",
        "",
        "An OPEN bound's outer band is STARVED when the published bins lying beyond the most "
        "extreme percentile every member declared carry no more than "
        f"{floor_multiple:g}x the platform's per-bin minimum step (0.01/N) on average: the band "
        "holds the structural minimum and nothing else, so every resolution in it earns the same "
        "floor score (~-219 on any grid) — a cliff nobody declared. Compare `tail mass` against "
        "`beyond bound`: the signature is the inversion, the declared outer mass sitting past the "
        "displayed bound instead of spread across the band the declaration pointed at. DETECTOR "
        "ONLY; any width response stays gated on the standing k_tail hold.",
        "",
        (
            f"Scanned {scan.n_scanned} open-bound side(s); starved "
            f"{counts[OuterTailVerdict.STARVED]}, healthy {counts[OuterTailVerdict.HEALTHY]}; "
            f"excluded records {scan.n_excluded}. Unmeasurable: "
            + (", ".join(f"{name}: {n}" for name, n in unmeasurable.items()) if unmeasurable else "none")
            + ". `declared_beyond_bound` is not a defect — the members put their outer anchor past "
            "the displayed bound, so the tail is out-of-range mass and there is no in-range band "
            "to starve."
        ),
        "",
    ]
    if not scan.starved:
        lines.append("Starved sides: none.")
        lines.append("")
        return "\n".join(lines)

    header = (
        "| question | side | declared p | declared value | displayed bound | tail mass | bins "
        "| mean bin / min step | beyond bound | flat-zone log score | title |"
    )
    lines.append(header)
    lines.append("|" + "|".join(["---"] * (header.count("|") - 1)) + "|")
    for r in scan.starved:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r.question_id),
                    r.side,
                    _fmt_measured(r.declared_percentile, "g"),
                    _fmt_measured(r.declared_value, ".6g"),
                    _fmt_measured(r.bound_value, ".6g"),
                    _fmt_measured(r.tail_mass, ".4f"),
                    str(r.band_bins),
                    _fmt_measured(r.floor_multiple, ".2f"),
                    _fmt_measured(r.beyond_bound_mass, ".4f"),
                    _fmt_measured(r.flat_zone_log_score, ".1f"),
                    r.title[:60],
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _fmt_ci(ci: tuple[float, float, float]) -> str:
    m, lo, hi = ci
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def render_markdown(metrics: list[EraWidthMetrics]) -> str:
    """Compact markdown table of the per-era width/calibration metrics.

    Two honesty rules are enforced here rather than left to the reader: the n_eff
    cell says whether the cluster correction actually fired on that row, and a row
    below ``MIN_N_FOR_POINT_METRICS`` renders its point metrics as ``n/a`` instead
    of printing a number whose resolution is coarser than the target it is being
    compared to.
    """
    lines: list[str] = []
    lines.append("## Numeric width / calibration monitor (per config era)")
    lines.append("")
    lines.append(
        "Calibrated targets: cov80=0.80, cov50=0.50, cov@10=0.10, cov@50=0.50, "
        f"cov@90=0.90, PIT std={UNIFORM_PIT_STD:.3f}. "
        "PIT std below target => too WIDE; above => too NARROW. "
        "cov@10 below 0.10 => low tail too wide; median rel width = (P90-P10)/|P50| (raw sharpness)."
    )
    lines.append("")
    lines.append(
        "cov80/cov50 CIs are computed at n_eff = distinct post_ids, so several correlated "
        "sub-questions on one post cannot narrow the CI as though they were independent. The n_eff "
        "cell states whether that correction did anything on the row: `(widened)` when a post "
        "carried more than one record, `(=n)` when none did and the CI is therefore the naive "
        "n-based one. Every archived pull to date is `(=n)`."
    )
    lines.append("")
    lines.append(
        f"Rows with fewer than {MIN_N_FOR_POINT_METRICS} PITs render cov@10/cov@50/cov@90, PIT std, "
        "mean PIT and band_miss as `n/a`: at that n the metric's resolution (1/n) is coarser than "
        "the target it is compared against, so the number is not an estimate (at n=1, PIT std is "
        "0.0, which reads as maximally too WIDE). cov80/cov50 still render — their CIs widen "
        "honestly. The JSON output keeps the raw values under an `underpowered` flag."
    )
    lines.append("")
    lines.append(
        "band_miss = P(PIT<0.10) + P(PIT>0.90), target 0.20 with lo ~= hi ~= 0.10. Well above 0.20 => "
        "band too TIGHT; a lo/hi skew at roughly the target => band roughly the right width but "
        "MIS-CENTERED (misses piled in one tail), which calls for shifting the band rather than "
        "widening it. Distinct from OOB lo/hi, which counts resolutions that fell beyond the "
        "QUESTION's own value grid (string marker or numeric; their PIT is read off the members' "
        "declared-percentile curves). excl = records dropped by --exclude-qids."
    )
    lines.append("")
    lines.append(
        "set-valued (pt n) = out-of-range resolutions whose PIT is an INTERVAL rather than a value "
        "(`above_upper_bound` -> [cdf[-1], 1], `below_lower_bound` -> [0, cdf[0]]: the platform gives "
        "no value, and on an open bound our own CDF says how much mass we put out there), with the "
        "point-metric denominator beside it. Those readings count in every coverage column when the "
        "interval INTERSECTS the band, and are EXCLUDED from PIT std / mean PIT, which is why the two "
        "denominators can differ. No midpoint is imputed."
    )
    lines.append("")
    header = (
        "| era | n | excl | n_eff | cov80 [95% CI] | cov50 [95% CI] | cov@10 | cov@50 | cov@90 "
        "| PIT std | mean PIT | med rel width (n) | band_miss (lo/hi) | OOB lo/hi | set-valued (pt n) |"
    )
    sep = "|" + "|".join(["---"] * (header.count("|") - 1)) + "|"
    lines.append(header)
    lines.append(sep)
    for m in metrics:
        rel = f"{m.median_rel_width:.3f} ({m.n_width})" if m.median_rel_width is not None else f"n/a ({m.n_width})"

        def _point(value: float | None, *, underpowered: bool = m.underpowered) -> str:
            return "n/a" if underpowered or value is None else f"{value:.3f}"

        band = "n/a" if m.underpowered else f"{m.band_miss:.3f} ({m.band_lo:.3f}/{m.band_hi:.3f})"
        cells = [
            m.label,
            str(m.n_pit),
            str(m.n_excluded),
            f"{m.n_eff} ({'widened' if m.ci_clustered else '=n'})",
            _fmt_ci(m.cov80),
            _fmt_ci(m.cov50),
            _point(m.cov_at_10),
            _point(m.cov_at_50),
            _point(m.cov_at_90),
            _point(m.pit_std, underpowered=m.point_metrics_underpowered),
            _point(m.mean_pit, underpowered=m.point_metrics_underpowered),
            rel,
            band,
            f"{m.n_oob_low}/{m.n_oob_high}",
            f"{m.n_oob_interval} ({m.n_point})",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Era-bucketed numeric width / calibration monitor (read-only)")
    parser.add_argument(
        "--cached",
        default="scratch/coherence_2026-07-15/perf_all_tagged.json",
        help="Path to a cached performance dataset JSON (list of records). Default: %(default)s",
    )
    parser.add_argument(
        "--tournament",
        default=None,
        help="Instead of --cached, pull a tournament live (read-only, free). Overrides --cached when set.",
    )
    parser.add_argument("--output-json", default=None, help="Optional path to also write the metrics as JSON.")
    parser.add_argument(
        "--output-starved-json",
        default=None,
        help=(
            "Optional path to write the starved-outer-tail scan as JSON (every scanned side with "
            "its verdict, not just the flagged ones). The markdown section is always printed."
        ),
    )
    parser.add_argument(
        "--exclude-qids",
        default="",
        help=(
            "Comma-separated question ids to drop from every row (the count is rendered in the table "
            "so the exclusion is visible). Each cohort shorthand below composes with explicit ids "
            "and is recognized anywhere in the list: "
            + "; ".join(f"'{name}' = {','.join(sorted(ids))}" for name, ids in sorted(EXCLUSION_COHORTS.items()))
            + f". So '{KNOWN_BUG_SHORTHAND},43800' excludes that cohort AND 43800. An unrecognized "
            "non-numeric token is an error rather than a silent no-op. Default: exclude nothing."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stderr)

    exclude_qids = parse_exclude_qids(args.exclude_qids)

    if args.tournament:
        # Confirm the host is the real Metaculus before the token-sending pull.
        verify_metaculus_api_identity()
        data = build_performance_dataset(tournament=args.tournament)
    else:
        data = load_dataset(args.cached)

    metrics = compute_all_eras(data, exclude_qids=exclude_qids)
    if exclude_qids:
        numeric_qids = {str(r.get("question_id")) for r in data if r.get("type") in NUMERIC_TYPES}
        matched = len(exclude_qids & numeric_qids)
        logger.info(
            f"--exclude-qids: {len(exclude_qids)} requested id(s), {matched} matched a "
            "numeric/discrete record in this pull"
        )
        # A cohort is defined by an incident, not by what resolved into a pull, so an
        # absent cohort id is normal and gets no alarm. The id-space trap — question and
        # post ids share one integer namespace — is only worth a WARN on ids the operator
        # typed explicitly.
        explicit_ids = {
            token.strip()
            for token in args.exclude_qids.split(",")
            if token.strip() and token.strip() not in EXCLUSION_COHORTS
        }
        all_qids = {str(r.get("question_id")) for r in data}
        post_ids = {str(r.get("post_id")) for r in data}
        id_space_confused = sorted((explicit_ids - all_qids) & post_ids)
        if id_space_confused:
            logger.warning(
                f"--exclude-qids: {id_space_confused} matched no question_id but IS a post_id in "
                "this pull — question and post ids share one integer space; translate through "
                "performance_analysis.id_mapping"
            )
    # The rendered markdown IS this CLI's product and belongs on stdout; logging above
    # is deliberately pinned to stderr so the report can be piped on its own.
    print(render_markdown(metrics))  # noqa: T201

    # The starved-outer-tail scan reads the same records and the same exclusions. It is a
    # per-QUESTION report rather than a per-era one, so it renders as its own section instead
    # of a column, and it is printed unconditionally — a monitor nobody has to ask for.
    scan = scan_outer_tails(data, exclude_qids=exclude_qids)
    print()  # noqa: T201
    print(render_starved_outer_tails(scan))  # noqa: T201

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump([m.to_dict() for m in metrics], f, indent=2)
        logger.info(f"Wrote {len(metrics)} era rows to {args.output_json}")

    if args.output_starved_json:
        with open(args.output_starved_json, "w") as f:
            json.dump(scan.to_dict(), f, indent=2)
        logger.info(f"Wrote {scan.n_scanned} outer-tail side readings to {args.output_starved_json}")


if __name__ == "__main__":
    main()
