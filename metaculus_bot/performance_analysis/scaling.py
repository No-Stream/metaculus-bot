"""Shared interpretation of the Metaculus ``scaling`` sub-dict for grid work.

A single home for the ``zero_point`` sentinel coercion so the record-scoring,
PIT-interpolation, and width-monitor paths all read log-vs-linear scale the same
way (previously three copies of the same idiom drifted, one of them wrong); for
the value-grid reconstruction built on top of it (:func:`cdf_and_grid`); and for
the two question types that carry a ``scaling`` sub-dict at all
(:data:`NUMERIC_TYPES`) — a binary or multiple-choice record has no value grid, so
every grid consumer filters on the same tuple.

This module deliberately sits BELOW the rest of the package (it imports nothing
from it), so the era report in ``width_monitor`` and the per-question scan in
``outer_tail`` can both read the grid without either importing the other.
"""

from __future__ import annotations

import numpy as np

from metaculus_bot.numeric.pchip_cdf import build_cdf_value_grid

NUMERIC_TYPES: tuple[str, ...] = ("numeric", "discrete")


def grid_zero_point(zero_point_raw: float | int | None, range_min: float) -> float | None:
    """Interpret ``scaling.zero_point`` for grid reconstruction / bucket mapping.

    A linear-scaled question serializes ``zero_point`` as ``null``; a log-scaled
    question with a positive floor legitimately carries ``zero_point == 0`` (the
    geometric grid then has ``ratio = range_max / range_min``). Treating that 0
    as the linear sentinel builds a linear grid where the API grid is geometric
    and corrupts the reconstructed value grid (up to ~0.55 span-normalized error,
    observed on 9 log-scale questions in the 2026-07-18 width audit). So only
    drop ``zero_point`` when it is genuinely absent, or when a non-positive
    ``range_min`` rules out a log transform.
    """
    if zero_point_raw is None:
        return None
    zp = float(zero_point_raw)
    if zp == 0.0:
        return 0.0 if range_min > 0 else None
    return zp


def cdf_and_grid(record: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (cdf, value_grid) for a numeric/discrete record, or None if the
    record lacks the bounds / CDF needed to build the grid.

    Prefers the API's grid-exact ``scaling.continuous_range`` when present (it
    already encodes log-vs-linear spacing, so no scale has to be re-derived from
    ``zero_point``); falls back to reconstructing via ``build_cdf_value_grid``
    with a zero_point interpretation that handles the ``zero_point == 0`` log
    case (see :func:`grid_zero_point`).
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
    zp = grid_zero_point(scaling.get("zero_point"), float(lo))
    grid = build_cdf_value_grid(float(lo), float(hi), zp, len(cdf))
    return cdf_arr, grid
