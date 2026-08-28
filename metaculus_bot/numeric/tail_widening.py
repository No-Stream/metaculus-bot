"""
Tail widening utilities for numeric declared percentiles.

Implements transform-space scaling around the median to fatten tails while preserving
bounds and monotonicity. Optionally enforces a span floor at the extreme tail percentiles.
"""

from __future__ import annotations

import logging
import math
from typing import NamedTuple

import numpy as np
from forecasting_tools.data_models.numeric_report import Percentile
from forecasting_tools.data_models.questions import NumericQuestion

from metaculus_bot.numeric.cluster_processing import ensure_strictly_increasing_bounded
from metaculus_bot.prob_math_utils import logit, sigmoid

logger = logging.getLogger(__name__)


def _choose_transform(
    question: NumericQuestion,
    eps: float,
):
    """
    Choose forward/inverse transforms based on bound semantics.

    - Both bounds closed: use bounded logit on normalized u.
    - Lower closed, upper open: lower-bounded → log(x - L + eps).
    - Lower open, upper closed: upper-bounded → -log(U - x + eps) (monotone in x).
    - Both open: identity.
    """

    L = float(question.lower_bound)
    U = float(question.upper_bound)
    open_low = bool(question.open_lower_bound)
    open_up = bool(question.open_upper_bound)

    if not open_low and not open_up:
        # Bounded both sides → normalize and use logit
        rng = max(U - L, eps)

        def fwd(x: float) -> float:
            u = (x - L) / rng
            # Use a tight clamp here (coordinate transform, not probability scoring)
            # so percentiles within the outer 0.01% of the range remain distinguishable.
            return logit(u, eps=1e-12)

        def inv(y: float) -> float:
            u = sigmoid(y)
            return L + rng * u

        return fwd, inv

    if not open_low and open_up:
        # Lower-bounded → y = log(x - L + eps)
        def fwd(x: float) -> float:
            return math.log(max(x - L + eps, 1e-18))

        def inv(y: float) -> float:
            return L - eps + math.exp(y)

        return fwd, inv

    if open_low and not open_up:
        # Upper-bounded → y = -log(U - x + eps), monotone increasing in x
        def fwd(x: float) -> float:
            return -math.log(max(U - x + eps, 1e-18))

        def inv(y: float) -> float:
            return U + eps - math.exp(-y)

        return fwd, inv

    # Both open → identity
    return (lambda x: x), (lambda y: y)


def _tail_weight(p: float, tail_start: float) -> float:
    """
    Linear tail ramp weight in [0,1]: 0 at center, 1 at deepest tails.
    No widening for p in [0.5 - (0.5 - tail_start), 0.5 + (0.5 - tail_start)].
    """
    t = abs(p - 0.5)
    no_widen_zone = 0.5 - tail_start
    if t <= no_widen_zone:
        return 0.0
    # Linear ramp from no_widen_zone → 0.5
    return min(1.0, (t - no_widen_zone) / tail_start)


class _TailIndices(NamedTuple):
    """Positions of the six percentiles the span-floor and outer-span rules key on.

    ``None`` where the declaration does not carry that percentile — both rules are
    skipped rather than approximated when a member of their triple is missing.
    """

    p025: int | None
    p05: int | None
    p10: int | None
    p90: int | None
    p95: int | None
    p975: int | None


def _validate_widening_knobs(k_tail: float, span_floor_gamma: float) -> None:
    if k_tail < 1.0:
        raise ValueError(
            f"k_tail={k_tail} is invalid: narrowing is not implemented (k_tail < 1.0). "
            "Only widening (k_tail >= 1.0) is supported; k_tail=1.0 is the identity "
            "pass. See scratch_docs_and_planning/tail_widening_empirical_calibration.md."
        )
    if span_floor_gamma < 0.0:
        raise ValueError(
            f"span_floor_gamma={span_floor_gamma} is invalid: must be >= 0.0. "
            "Use 0.0 to disable the floor check; any positive value enables it."
        )


def _stretch_tails(
    p_vals: np.ndarray,
    x_vals: np.ndarray,
    question: NumericQuestion,
    *,
    k_tail: float,
    tail_start: float,
) -> np.ndarray:
    """Scale each percentile's distance from the median in a bound-aware transform space.

    Returns the widened values clamped back into ``[lower_bound, upper_bound]``. With
    ``k_tail == 1.0`` the transform round-trip is skipped entirely and the declared
    values pass through, so the identity case cannot drift on floating point.
    """
    lower = float(question.lower_bound)
    upper = float(question.upper_bound)
    eps = max(1e-12, max(upper - lower, 1e-12) * 1e-12)
    fwd, inv = _choose_transform(question, eps)

    y_vals = np.array([fwd(x) for x in x_vals], dtype=float)

    # Locate the median in transformed space; interpolate in p-space when p=0.5 is absent.
    if any(abs(p - 0.5) < 1e-12 for p in p_vals):
        y_m = float(y_vals[np.argmin(np.abs(p_vals - 0.5))])
    else:
        y_m = float(np.interp(0.5, p_vals, y_vals))

    k_delta = max(0.0, k_tail - 1.0)
    widened_y = []
    for p, y in zip(p_vals, y_vals, strict=True):
        k_eff = 1.0 + k_delta * _tail_weight(p, tail_start)
        widened_y.append(y_m + k_eff * (y - y_m))

    widened_x = np.array([inv(y) for y in widened_y], dtype=float) if k_tail > 1.0 else x_vals.copy()
    return np.clip(widened_x, lower, upper)


def _locate_tail_indices(p_vals: np.ndarray) -> _TailIndices:
    def _find_index(target: float) -> int | None:
        idxs = np.where(np.isclose(p_vals, target, atol=5e-6))[0]
        return int(idxs[0]) if len(idxs) else None

    return _TailIndices(
        p025=_find_index(0.025),
        p05=_find_index(0.05),
        p10=_find_index(0.10),
        p90=_find_index(0.90),
        p95=_find_index(0.95),
        p975=_find_index(0.975),
    )


def _apply_span_floors(
    widened_x: np.ndarray,
    tail_idx: _TailIndices,
    span_floor_gamma: float,
    question: NumericQuestion,
) -> None:
    """Enforce ``(p05 - p025) >= gamma * (p10 - p05)`` and its upper mirror, in place."""
    if span_floor_gamma <= 0:
        return

    if None not in (tail_idx.p025, tail_idx.p05, tail_idx.p10):
        inner = max(0.0, widened_x[tail_idx.p10] - widened_x[tail_idx.p05])
        target_span = span_floor_gamma * inner
        current = widened_x[tail_idx.p05] - widened_x[tail_idx.p025]
        if target_span > current + 1e-15:
            widened_x[tail_idx.p025] = max(float(question.lower_bound), widened_x[tail_idx.p05] - target_span)

    if None not in (tail_idx.p90, tail_idx.p95, tail_idx.p975):
        inner = max(0.0, widened_x[tail_idx.p95] - widened_x[tail_idx.p90])
        target_span = span_floor_gamma * inner
        current = widened_x[tail_idx.p975] - widened_x[tail_idx.p95]
        if target_span > current + 1e-15:
            widened_x[tail_idx.p975] = min(float(question.upper_bound), widened_x[tail_idx.p95] + target_span)


def _preserve_outer_spans(
    widened_x: np.ndarray,
    x_vals: np.ndarray,
    tail_idx: _TailIndices,
    question: NumericQuestion,
) -> None:
    """Keep the outer tail spans from SHRINKING relative to the declaration, in place.

    Widening is monotone in the transform space but not necessarily in value space once
    the bound clamp and span floors have run, so this guards the case where the caller
    asked for wider tails and a pass above narrowed one.
    """
    if None not in (tail_idx.p025, tail_idx.p05):
        base_low_outer = x_vals[tail_idx.p05] - x_vals[tail_idx.p025]
        new_low_outer = widened_x[tail_idx.p05] - widened_x[tail_idx.p025]
        if new_low_outer + 1e-15 < base_low_outer:
            widened_x[tail_idx.p025] = max(float(question.lower_bound), widened_x[tail_idx.p05] - base_low_outer)

    if None not in (tail_idx.p95, tail_idx.p975):
        base_up_outer = x_vals[tail_idx.p975] - x_vals[tail_idx.p95]
        new_up_outer = widened_x[tail_idx.p975] - widened_x[tail_idx.p95]
        if new_up_outer + 1e-15 < base_up_outer:
            widened_x[tail_idx.p975] = min(float(question.upper_bound), widened_x[tail_idx.p95] + base_up_outer)


def _nudge_off_open_bounds(updated: list[float], question: NumericQuestion, value_floor: float) -> None:
    """Pull the extreme tails off an OPEN bound so they aren't near-duplicates of the edge."""
    if question.open_lower_bound:
        updated[0] = max(updated[0], float(question.lower_bound) + value_floor)
        if len(updated) >= 2:
            updated[1] = max(updated[1], updated[0] + value_floor)
    if question.open_upper_bound:
        updated[-1] = min(updated[-1], float(question.upper_bound) - value_floor)
        if len(updated) >= 2:
            updated[-2] = min(updated[-2], updated[-1] - value_floor)


def _apply_spacing_schedule(updated: list[float], question: NumericQuestion, value_floor: float) -> None:
    """Keep ``value_floor`` between neighbours while reserving room for the points above.

    The forward pass leaves each point at least ``value_floor`` above its predecessor; on
    an open upper bound its ceiling shrinks by one floor per remaining point so the sweep
    can never run the tail into ``U``. The backward pass mirrors it.
    """
    if len(updated) < 2:
        return

    lower = float(question.lower_bound)
    upper = float(question.upper_bound)
    open_low = bool(question.open_lower_bound)
    open_up = bool(question.open_upper_bound)

    for i in range(1, len(updated)):
        min_allowed = updated[i - 1] + value_floor
        max_allowed = upper - value_floor * (len(updated) - 1 - i) if open_up else upper
        if updated[i] < min_allowed:
            updated[i] = min(max_allowed, min_allowed)

    for i in range(len(updated) - 2, -1, -1):
        max_allowed = updated[i + 1] - value_floor
        min_allowed = lower + value_floor * i if open_low else lower
        if updated[i] > max_allowed:
            updated[i] = max(min_allowed, max_allowed)


def widen_declared_percentiles(
    percentile_list: list[Percentile],
    question: NumericQuestion,
    *,
    k_tail: float = 1.0,
    tail_start: float = 0.2,
    span_floor_gamma: float = 0.0,
) -> list[Percentile]:
    """
    Widen tails by scaling distances from the median in a transformed space.

    Parameters
    ----------
    percentile_list: list of Percentile (assumed sorted by percentile and strictly increasing values)
    question: NumericQuestion with bound semantics
    k_tail: maximum stretch factor at the deepest tails in transformed space.
        `k_tail=1.0` disables widening (identity pass). Values `< 1.0` raise
        `ValueError` because narrowing is not implemented — the ramp weight in
        `_tail_weight` never goes below zero and the inverse-transform branch
        only runs when `k_tail > 1.0`, so a caller passing `0.8` would silently
        get the identity pass. See
        `scratch_docs_and_planning/tail_widening_empirical_calibration.md` for
        the empirical rationale behind the `1.0` default.
    tail_start: tail ramp start (fraction of percentile distance from the median).
    span_floor_gamma: enforces `(p05 - p02.5) >= gamma * (p10 - p05)` and the
        upper mirror `(p97.5 - p95) >= gamma * (p95 - p90)`. `0.0` disables the
        check (the default — on all 2026 data the floor never bound). Any
        positive value re-enables the existing floor enforcement; negative
        values raise `ValueError`. Kept configurable for forecasters that
        declare unusually sharp tails.
    """

    _validate_widening_knobs(k_tail, span_floor_gamma)

    # If no percentiles, or both widening and span-floor disabled, bail out
    if not percentile_list or (k_tail <= 1.0 and span_floor_gamma <= 0.0):
        return percentile_list

    range_size = max(float(question.upper_bound) - float(question.lower_bound), 1e-12)

    # Build arrays in percentile order
    p_vals = np.array([float(p.percentile) for p in percentile_list], dtype=float)
    x_vals = np.array([float(p.value) for p in percentile_list], dtype=float)

    widened_x = _stretch_tails(p_vals, x_vals, question, k_tail=k_tail, tail_start=tail_start)

    tail_idx = _locate_tail_indices(p_vals)
    _apply_span_floors(widened_x, tail_idx, span_floor_gamma, question)
    if k_tail > 1.0:
        _preserve_outer_spans(widened_x, x_vals, tail_idx, question)

    # A final gentle pass to guarantee strict monotonicity and bound proximity
    updated = ensure_strictly_increasing_bounded(widened_x.tolist(), question, range_size)

    # Modest floor relative to the range, so the unit-mismatch detector isn't tripped.
    value_floor = max(range_size * 1e-6, 1e-8)
    _nudge_off_open_bounds(updated, question, value_floor)
    updated = ensure_strictly_increasing_bounded(updated, question, range_size)

    # Final safety: clamp into [L, U], then re-space so ordering survives the clamp.
    updated = np.clip(updated, float(question.lower_bound), float(question.upper_bound)).tolist()
    _apply_spacing_schedule(updated, question, value_floor)

    # Rebuild Percentile objects preserving original percentiles
    result: list[Percentile] = [
        Percentile(value=float(v), percentile=float(p)) for v, p in zip(updated, p_vals, strict=True)
    ]

    if not np.all(np.diff([pp.value for pp in result]) > -1e-12):
        logger.warning(
            "Tail widening produced non-monotone sequence; enforced correction applied | Q=%s",
            getattr(question, "id_of_question", None),
        )

    return result
