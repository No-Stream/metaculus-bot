"""Per-QUESTION scan for STARVED OUTER TAILS (the open-bound p99 cliff).

An open bound whose published mass beyond the members' declared outer anchor sits at the
platform's structural minimum step gives every resolution out there the same floor score.
That is a cliff at a fixed location rather than a band of the wrong size, which is why it
reads per QUESTION while the width monitor next door reads per ERA, and why widening does not
address it. The mechanism, the trigger's calibration, and why there is deliberately no
publish-time twin are recorded in the comment block below.

Measured and rendered here; ``width_monitor``'s CLI owns the wiring, printing this section
after its era table (``--output-starved-json`` writes every scanned side, flagged or not).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from metaculus_bot.performance_analysis.parsing import declared_anchors, is_anonymous_model_key
from metaculus_bot.performance_analysis.scaling import NUMERIC_TYPES, cdf_and_grid, grid_zero_point
from metaculus_bot.performance_analysis.scoring import numeric_log_score

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

# The band is measured from the most extreme percentile EVERY member declared, and that anchor
# has to actually be a tail anchor. A trimmed comment can leave the members sharing only their
# p50, and a band read as "everything above the median" spans about half the grid, so its mean
# per-bin mass sits tens of times above the platform minimum and the side reads HEALTHY however
# starved its terminal bins are: measured 61.5x on the q45218 geometry, whose top 27 bins do sit
# at the floor, and 98.0x on a plain ramp, against a trigger of 2.0x. So lowering this constant
# does not over-flag — it turns a disclosed ``ANCHOR_NOT_EXTREME`` into an affirmative clean
# bill of health, which is the one thing the verdict enum exists to prevent. Applied
# symmetrically: the low side needs an anchor at or below ``100 - this``. The canonical sets
# both clear it (p99 today, p97.5 in the 11-point era).
STARVED_OUTER_TAIL_MIN_ANCHOR_PERCENTILE: float = 90.0

# The platform's per-bin minimum step is ``round(0.01 / N, 9)`` where N = cdf_size - 1 (see
# the Metaculus CDF constraints in the repo instructions); the rounding is irrelevant at the
# ratios this detector reads.
_PLATFORM_MIN_STEP_NUMERATOR: float = 0.01


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
    members_used: int | None = None
    """Member curves that set the anchor label and its median VALUE.

    Disclosed because the survivors move the band boundary the verdict is measured against: an
    anchor medianed over 5 of 6 members sits somewhere else than one medianed over all 6, so a
    row reading HEALTHY on a partial set is a weaker claim than the same row on a whole one.
    """
    members_dropped: int | None = None
    """Members present on the record that did not reach the anchor median.

    Three causes, counted together because each has the same effect on the boundary: an
    anonymous positional ``Forecaster N`` key (deliberately excluded — on a stacker-fired
    record it holds the AGGREGATE), a curve too malformed to parse, and a curve carrying fewer
    than two distinct percentile labels. ``members_used + members_dropped`` is the record's
    member count, so the pair is checkable rather than indicative. Both are None on a side
    whose members were never inspected (``NO_USABLE_CDF``), which is not the same as a side
    that inspected them and found none.
    """

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
            "members_used": self.members_used,
            "members_dropped": self.members_dropped,
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
    """Worst-first key over STARVED readings, every one of which carries a measurement.

    Asserted rather than defaulted: 0.0 is the most FAVOURABLE key this table can hold, so a
    broken invariant would sort an unmeasured side to the top of a worst-first list instead of
    saying anything. The assert also narrows the type for the checker.
    """
    assert reading.flat_zone_log_score is not None, f"unmeasured reading ranked: {reading!r}"
    return reading.flat_zone_log_score


@dataclass(frozen=True, slots=True)
class _MemberCurves:
    """The usable member curves on one record, and how many members did not make it.

    The drop count travels with the curves rather than being recomputed downstream: the
    survivors set both the shared anchor LABEL and its median VALUE, so a reading that does not
    carry the count cannot say what its band boundary was measured over.
    """

    curves: dict[str, dict[float, float]]
    n_dropped: int

    @property
    def n_used(self) -> int:
        return len(self.curves)


def _member_declared_anchors(record: dict) -> _MemberCurves:
    """``{model: {percentile label: value}}`` for the members whose curve is usable, plus drops.

    Anonymous ``Forecaster N`` keys are EXCLUDED, matching ``declared_percentile_pit`` and
    ``max_step_clamp_screen``: on a stacker-fired record that positional bucket holds the
    stacker's AGGREGATE, so pooling it into a median-of-members moves the anchor. That
    exclusion is counted as a drop alongside the failures below, because the anchor moves the
    same way whichever reason removed the member. On the archived cohort it is also the only
    cause that fires at all — 12 members over 12 records, zero parse failures and zero short
    curves — and the disclosure is not academic: 6 of those records are PARTIAL losses that
    left 5 or 6 members standing, which is 12 of the 487 scanned sides, and 8 of them are
    flagged rows. So roughly one flagged row in nine measured its band from an anchor a dropped
    member had already moved.
    """
    curves: dict[str, dict[float, float]] = {}
    n_dropped = 0
    for model, pairs in (record.get("per_model_numeric_percentiles") or {}).items():
        if is_anonymous_model_key(str(model)):
            n_dropped += 1
            continue
        try:
            anchors, _conflicts = declared_anchors(pairs)
        except (TypeError, ValueError, IndexError):
            # Archived per-model pairs are parsed from comment text and can be malformed;
            # an unusable curve reads as no-curve rather than raising (same rule as
            # ``analysis._single_curve_pit``).
            n_dropped += 1
            continue
        if len(anchors) >= 2:
            curves[str(model)] = anchors
        else:
            n_dropped += 1
    return _MemberCurves(curves=curves, n_dropped=n_dropped)


def _shared_extreme_anchor(members: _MemberCurves, side: str) -> tuple[float, float] | OuterTailVerdict:
    """``(percentile label, median member value)`` at the most extreme SHARED anchor.

    Shared across members on purpose: medianing p99 against p97.5 would mix two different
    quantities, and the canonical percentile set changed mid-season (11-point p2.5..p97.5 ->
    13-point p1..p99), so the label cannot be hardcoded. Returns the verdict that explains
    itself when no anchor qualifies.
    """
    curves = members.curves
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
    built = cdf_and_grid(record)
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


def _unmeasurable(
    record: dict, side: str, verdict: OuterTailVerdict, *, members: _MemberCurves | None = None
) -> OuterTailReading:
    """A reading that carries the reason no measurement was possible, and the member census.

    ``members`` is None only where the failure preceded any member inspection
    (``NO_USABLE_CDF``), which the reading then reports as None rather than as an empty census.
    """
    return OuterTailReading(
        question_id=record.get("question_id"),
        title=str(record.get("title") or ""),
        side=side,
        verdict=verdict,
        members_used=None if members is None else members.n_used,
        members_dropped=None if members is None else members.n_dropped,
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
        zero_point=grid_zero_point(scaling.get("zero_point"), lower),
    )


def _measure_one_outer_tail(
    record: dict,
    side: str,
    cdf: np.ndarray,
    grid: np.ndarray,
    *,
    floor_multiple: float,
) -> OuterTailReading:
    # Reached only when cdf_and_grid succeeded, which required both bounds, so the sub-dict is
    # there — the sibling _flat_zone_log_score reads it the same way.
    scaling = record["scaling"]
    if float(scaling["range_max"]) <= float(scaling["range_min"]):
        # A degenerate range has no bins to score; the same guard numeric_pit_analysis applies.
        return _unmeasurable(record, side, OuterTailVerdict.NO_USABLE_CDF)

    members = _member_declared_anchors(record)
    anchor = _shared_extreme_anchor(members, side)
    if isinstance(anchor, OuterTailVerdict):
        return _unmeasurable(record, side, anchor, members=members)
    percentile, declared_value = anchor

    band = _outer_band(cdf, grid, declared_value, side)
    if isinstance(band, OuterTailVerdict):
        return _unmeasurable(record, side, band, members=members)

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
        members_used=members.n_used,
        members_dropped=members.n_dropped,
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


def _fmt_members(reading: OuterTailReading) -> str:
    """``used/dropped`` for the members the anchor was medianed over."""
    if reading.members_used is None or reading.members_dropped is None:
        return "n/a"
    return f"{reading.members_used}/{reading.members_dropped}"


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
    dropped_total = sum(r.members_dropped or 0 for r in scan.readings)
    dropped_sides = sum(1 for r in scan.readings if r.members_dropped)
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
        (
            f"Dropped an anchor member on {dropped_sides} of {scan.n_scanned} scanned side(s) "
            f"({dropped_total} member-drop(s) in total, counted per side, so a record with two "
            "open bounds contributes its drop twice)."
            if dropped_total
            else "Dropped an anchor member on no side."
        )
        + " The anchor is medianed over the members whose declared curve is usable, so a dropped "
        "member (an anonymous positional `Forecaster N` bucket, an unparseable curve, or one "
        "carrying fewer than two distinct labels) moves the boundary the verdict is measured "
        "against. The `members` column states used/dropped per flagged row; every scanned side "
        "carries the pair in `--output-starved-json`.",
        "",
    ]
    if not scan.starved:
        lines.append("Starved sides: none.")
        lines.append("")
        return "\n".join(lines)

    header = (
        "| question | side | declared p | declared value | members (used/dropped) | displayed "
        "bound | tail mass | bins | mean bin / min step | beyond bound | flat-zone log score "
        "| title |"
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
                    _fmt_members(r),
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
