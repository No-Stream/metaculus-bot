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
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

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
from metaculus_bot.performance_analysis.scaling import grid_zero_point as _grid_zero_point
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

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump([m.to_dict() for m in metrics], f, indent=2)
        logger.info(f"Wrote {len(metrics)} era rows to {args.output_json}")


if __name__ == "__main__":
    main()
