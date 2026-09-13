"""Era-bucketed numeric-width / calibration monitor (READ-ONLY, free).

Tracks how wide the bot's published numeric distributions are, and how well that width is
calibrated, split by config era. Per era it reports, on the bot's PUBLISHED 201-point CDF:
central-80% and central-50% coverage with Beta-Binomial / Jeffreys-prior 95% CIs, tail
coverage (cov@10 / cov@50 / cov@90), PIT std, mean PIT, median relative band width, and
``band_miss`` split into its low and high tails. PIT is F_bot(resolution) on the canonical
Metaculus value grid (``build_cdf_value_grid``); ``compute_pit_reading`` holds the two
out-of-range conventions.

Every column's definition and calibrated target, which direction "off" points, the width
history the eras bucket on, the underpowered floor and the clustered CI are in
``docs/performance_analysis.md`` "Reading the width monitor's era table". Era boundaries
are **merge-to-main timestamps**, not authoring dates.

Alongside the era table this CLI prints a per-QUESTION section, the starved-outer-tail
scan, which lives in ``outer_tail.py``: that failure is a cliff at a fixed location
rather than a band of the wrong size, so it reads per question rather than per era.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from metaculus_bot.api_preflight import verify_metaculus_api_identity
from metaculus_bot.performance_analysis.analysis import (
    B4E9DF0_MERGED_AT,
    WIDENING_FLIP_MERGED_AT,
    PitReading,
    jeffreys_ci,
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
from metaculus_bot.performance_analysis.markdown import markdown_table
from metaculus_bot.performance_analysis.outer_tail import render_starved_outer_tails, scan_outer_tails
from metaculus_bot.performance_analysis.scaling import NUMERIC_TYPES, cdf_and_grid
from metaculus_bot.time_utils import parse_iso_utc

logger: logging.Logger = logging.getLogger(__name__)

# Surfaced in the legend so a reader knows which direction "off" points.
UNIFORM_PIT_STD: float = 1.0 / np.sqrt(12.0)  # ~0.2887

# Why this floor: docs/performance_analysis.md "Reading the width monitor's era table"
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


# Aliases of analysis.py's merge timestamps: docs/performance_analysis.md "Reading the width monitor's era table"
WIDENING_FLIP = WIDENING_FLIP_MERGED_AT
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
    built = cdf_and_grid(record)
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
    built = cdf_and_grid(record)
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

    Records sharing a ``post_id`` are one correlated family; a record with no ``post_id``
    is its own family, assigned a unique sentinel by position so it is never merged with
    another such record. The correction is currently inert on every archived pull, which
    the table states per row via ``EraWidthMetrics.ci_clustered``. The measurement behind
    that, and the retracted claim it replaced, are in ``docs/performance_analysis.md``
    "Reading the width monitor's era table".
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
    # A set-valued reading has no value to average, and imputing its midpoint would manufacture one.
    points = np.asarray(pit_point_values(readings), dtype=float)
    cov80_k = pit_band_count(readings, 0.10, 0.90)
    cov50_k = pit_band_count(readings, 0.25, 0.75)

    # Why the CIs run at n_eff: docs/performance_analysis.md "Reading the width monitor's era table"
    n_eff = _n_effective_clusters(samples.pit_post_ids)
    cov80 = jeffreys_ci(round(cov80_k * n_eff / n), n_eff)
    cov50 = jeffreys_ci(round(cov50_k * n_eff / n), n_eff)

    # Why the lo/hi split: docs/performance_analysis.md "Reading the width monitor's era table"
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
        # question_id arrives from the API as an int, so coerce before comparing against the string set.
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
    header = [
        "era",
        "n",
        "excl",
        "n_eff",
        "cov80 [95% CI]",
        "cov50 [95% CI]",
        "cov@10",
        "cov@50",
        "cov@90",
        "PIT std",
        "mean PIT",
        "med rel width (n)",
        "band_miss (lo/hi)",
        "OOB lo/hi",
        "set-valued (pt n)",
    ]
    rows: list[list[str]] = []
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
        rows.append(cells)
    lines += markdown_table(header, rows)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Era-bucketed numeric width / calibration monitor (read-only)")
    parser.add_argument(
        "--cached",
        default=None,
        help=(
            "Path to a cached performance dataset JSON (list of records), normally the current "
            "round's perf_all_tagged.json. Required unless --tournament is given: a default "
            "naming one round keeps resolving after that round is superseded, so the monitor "
            "would silently read a stale dataset."
        ),
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
    elif args.cached is not None:
        data = load_dataset(args.cached)
    else:
        parser.error("pass --cached <dataset> or --tournament <slug>: there is no dataset to read otherwise")

    metrics = compute_all_eras(data, exclude_qids=exclude_qids)
    if exclude_qids:
        numeric_qids = {str(r.get("question_id")) for r in data if r.get("type") in NUMERIC_TYPES}
        matched = len(exclude_qids & numeric_qids)
        logger.info(
            f"--exclude-qids: {len(exclude_qids)} requested id(s), {matched} matched a "
            "numeric/discrete record in this pull"
        )
        # A cohort id absent from a pull is normal, so only explicitly typed ids earn the id-space WARN.
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
    # The report is this CLI's product, so it goes to stdout while logging above stays on stderr.
    print(render_markdown(metrics))  # noqa: T201

    # A per-QUESTION report, so it renders as its own section rather than a column, and is unconditional.
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
