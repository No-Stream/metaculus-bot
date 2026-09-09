"""The Mantic-only floor on the mass a PUBLISHED aggregate places beyond each open bound.

Mantic (the Crucible competition, a fork of the Metaculus platform) scores a resolution beyond an
OPEN bound against a fixed 0.05 reference, ``50 * ln(mass / 0.05)``: 5% in that bucket scores 0 and
the structural 1% this pipeline publishes whenever every declared percentile sits inside the range
scores -80.5. Its question writers set the ranges themselves and are paid for bot disagreement; in
Series 1 half the date questions, a quarter of the discrete and an eighth of the numeric ones
resolved outside the displayed range. So on a Mantic question the published aggregate carries at
least ``MANTIC_OUT_OF_RANGE_TAIL_FLOOR`` beyond each open bound. Applied to every Series 1
competitor's own published distributions the same floor cost at most 1.9 points per question on
average and gained up to 9.1 for thin-tailed bots (``scratch_docs_and_planning/
mantic_adversarial_candidates_2026-09-08.md``, section 1).

Where it applies: ``forecaster._aggregate_predictions``, the one seam every aggregation path (median,
mean, base-combine, stacker, single survivor) returns through, so it is the last touch before the
publish gate, the comment and the ``NUMERIC_AGGREGATE`` marker read the distribution. Nowhere else:
a Metaculus aggregate comes back as the very same object, per-member forecasts are never touched
(their ``MEMBER_FORECAST`` tails keep measuring what the models declared), and the backtest and
ablation harnesses replay Metaculus questions, so the platform gate covers them.

What it does, per side: an open tail under the floor is raised to exactly the floor, an open tail
already at or above it is left alone, and a closed bound stays at its exact 0.0 or 1.0 (the server
requires that). The interior is rescaled affinely between the new endpoints, which keeps it monotone,
fixes the median of a symmetric shape in place and can only shrink steps, so the platform's max step
cannot be newly violated; it CAN push the bins that sat exactly on the min step (the uniform-mixture
tails of a concentrated PCHIP build) below it, so the pipeline's min-step sweep runs again with the
new endpoints as its caps. ``tests/test_out_of_range_floor.py`` proves the server's rules on the
result for every distinct grid Mantic has published.

The raw (pre-floor) tails are kept beside the published ones so the floor can be benchmarked on
this bot's own forecasts once live telemetry accumulates: the marker records both.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from forecasting_tools import MetaculusQuestion, NumericDistribution, Percentile

from metaculus_bot.constants import MANTIC_OUT_OF_RANGE_TAIL_FLOOR, PLATFORM_MANTIC
from metaculus_bot.numeric.config import grid_step_constraints
from metaculus_bot.numeric.date_axis import numeric_view
from metaculus_bot.numeric.pchip_cdf import enforce_min_steps
from metaculus_bot.numeric.pchip_processing import create_pchip_numeric_distribution
from metaculus_bot.question_platform import question_platform

__all__ = [
    "FlooredCdf",
    "TailFloorOutcome",
    "floor_out_of_range_tails",
    "floor_published_tails",
    "tail_floor_for_platform",
]


@dataclass(frozen=True)
class FlooredCdf:
    """A CDF's heights after the floor, with the mass beyond each bound before and after.

    ``raw`` and ``published`` are ``(below the lower bound, above the upper bound)``, i.e.
    ``cdf[0]`` and ``1 - cdf[-1]``; ``floored`` says whether either endpoint moved.
    """

    cdf: np.ndarray
    raw: tuple[float, float]
    published: tuple[float, float]
    floored: bool


@dataclass(frozen=True)
class TailFloorOutcome:
    """What the seam publishes and logs: the distribution to publish and the marker's tail fields.

    ``distribution`` is the input object itself when nothing was floored. ``floor`` is the floor that
    moved an endpoint, ``0.0`` when none did (a Metaculus question, a closed-bound question, or
    tails already at or above the floor).
    """

    distribution: NumericDistribution
    cdf_size: int
    raw: tuple[float, float]
    published: tuple[float, float]
    floor: float


def tail_floor_for_platform(question: MetaculusQuestion) -> float:
    """``MANTIC_OUT_OF_RANGE_TAIL_FLOOR`` on a Mantic question, ``0.0`` on a Metaculus one."""
    return MANTIC_OUT_OF_RANGE_TAIL_FLOOR if question_platform(question) == PLATFORM_MANTIC else 0.0


def floor_out_of_range_tails(cdf: np.ndarray, *, open_lower: bool, open_upper: bool, floor: float) -> FlooredCdf:
    """Raise each OPEN tail of a server-legal CDF to at least ``floor``; pure, input untouched.

    ``cdf`` is the heights the server would receive (length ``inbound bins + 1``, monotone, endpoints
    already pinned for the bound flags). A floor of ``0.0`` is a no-op that hands the input back.
    """
    low, high = float(cdf[0]), float(cdf[-1])
    target_low = max(low, floor) if open_lower else low
    target_high = min(high, 1.0 - floor) if open_upper else high
    raw = (low, 1.0 - high)
    if target_low == low and target_high == high:
        return FlooredCdf(cdf=cdf, raw=raw, published=raw, floored=False)

    rescaled = target_low + (cdf - low) * ((target_high - target_low) / (high - low))
    rescaled[0] = target_low
    rescaled[-1] = target_high
    min_step, _ = grid_step_constraints(cdf.size)
    legal = enforce_min_steps(rescaled, min_step, upper_cap=target_high, lower_cap=target_low)
    return FlooredCdf(cdf=legal, raw=raw, published=(float(legal[0]), 1.0 - float(legal[-1])), floored=True)


def floor_published_tails(aggregated: NumericDistribution, question: MetaculusQuestion) -> TailFloorOutcome:
    """Apply the platform's tail floor to the aggregate about to be published.

    Materialises the CDF once (the same read the marker's ``cdf_size`` and tails come from). When an
    endpoint moves, the distribution is rebuilt on the question's numeric view with the floored
    heights as both its CDF and its ``declared_percentiles``, on the value axis the aggregate already
    carried, so the comment, the spread readers and the submission all see one distribution; the
    view is the epoch adapter on a date question, which keeps ``is_date`` and the date rendering.
    """
    heights = aggregated.get_cdf()
    cdf = np.fromiter((point.percentile for point in heights), dtype=float, count=len(heights))
    result = floor_out_of_range_tails(
        cdf,
        open_lower=aggregated.open_lower_bound,
        open_upper=aggregated.open_upper_bound,
        floor=tail_floor_for_platform(question),
    )
    if not result.floored:
        return TailFloorOutcome(
            distribution=aggregated, cdf_size=cdf.size, raw=result.raw, published=result.raw, floor=0.0
        )

    declared = [
        Percentile(percentile=float(probability), value=float(point.value))
        for point, probability in zip(heights, result.cdf, strict=True)
    ]
    rebuilt = create_pchip_numeric_distribution(
        pchip_cdf=[float(probability) for probability in result.cdf],
        percentile_list=declared,
        question=numeric_view(question),
        zero_point=aggregated.zero_point,
    )
    return TailFloorOutcome(
        distribution=rebuilt,
        cdf_size=cdf.size,
        raw=result.raw,
        published=result.published,
        floor=tail_floor_for_platform(question),
    )
