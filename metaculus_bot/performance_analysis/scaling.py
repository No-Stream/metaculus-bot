"""Shared interpretation of the Metaculus ``scaling`` sub-dict for grid work.

A single home for the ``zero_point`` sentinel coercion so the record-scoring,
PIT-interpolation, and width-monitor paths all read log-vs-linear scale the same
way (previously three copies of the same idiom drifted, one of them wrong).
"""

from __future__ import annotations


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
