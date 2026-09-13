"""Synthetic record builders and the rendered-row reader shared by the width-monitor tests.

The coverage math is verified against hand-computed values on records with linear CDFs (so
PIT = (resolution - lower) / (upper - lower)), which is what makes every expected number in the
test modules readable off the ramp by hand.

Not named ``test_*`` on purpose: pytest imports it without collecting it, as with
``tests/supply_probe_fakes.py``. Every record here is a plain dict, so nothing opens a socket.
"""

import numpy as np

from metaculus_bot.performance_analysis.width_monitor import compute_pit_reading

GRID_N = 201


def _linear_cdf_record(
    *,
    resolution,
    lower: float = 0.0,
    upper: float = 100.0,
    created_at: str | None = "2026-01-01T00:00:00Z",
    q_type: str = "numeric",
    question_id: object = None,
) -> dict:
    """A numeric record whose published CDF is the identity ramp on a linear
    grid over ``[lower, upper]``. For such a CDF, F(x) = (x - lower) / (upper -
    lower), so PIT of a resolution ``r`` is exactly ``(r - lower) / (upper -
    lower)`` and the p-quantile value is ``lower + p * (upper - lower)``.
    """
    cdf = np.linspace(0.0, 1.0, GRID_N).tolist()
    return {
        "type": q_type,
        "our_forecast_values": cdf,
        "resolution_parsed": resolution,
        "scaling": {"range_min": lower, "range_max": upper, "zero_point": None},
        "open_lower_bound": True,
        "open_upper_bound": True,
        "bot_comment_created_at": created_at,
        "question_id": question_id,
    }


def _record_with_pit(pit: float, **kwargs) -> dict:
    """A record whose PIT is exactly ``pit`` (identity ramp over [0, 100])."""
    return _linear_cdf_record(resolution=pit * 100.0, **kwargs)


def _out_of_range_mass_record(*, resolution, cdf_start: float = 0.0, cdf_end: float = 1.0, **kwargs) -> dict:
    """A record whose published CDF ramps ``cdf_start -> cdf_end`` over the displayed range.

    ``1 - cdf_end`` is the mass declared ABOVE the displayed ceiling and ``cdf_start`` the
    mass below the floor, which is what an out-of-range resolution's PIT interval is read
    off.
    """
    rec = _linear_cdf_record(resolution=resolution, **kwargs)
    rec["our_forecast_values"] = np.linspace(cdf_start, cdf_end, GRID_N).tolist()
    return rec


def _below_bound_mass_record(*, resolution, per_model_percentiles=None, **kwargs) -> dict:
    """The q44218 shape: open lower bound with most of the mass declared BELOW it.

    Published CDF ramps 0.90 -> 0.975 over [100, 200], i.e. F(100) = 0.90: 90% of the
    mass sits below the displayed lower bound. A resolution under 100 is therefore a
    LOW-tail event, but grid interpolation clamps it to cdf[0] = 0.90 — the sign flip
    the declared-percentile fallback exists to prevent.
    """
    rec = _linear_cdf_record(resolution=resolution, lower=100.0, upper=200.0, **kwargs)
    rec["our_forecast_values"] = np.linspace(0.90, 0.975, GRID_N).tolist()
    if per_model_percentiles is not None:
        rec["per_model_numeric_percentiles"] = per_model_percentiles
    return rec


def _row_cells(md: str, label: str) -> list[str]:
    """The stripped cells of one rendered table row, indexed as in the header.

    1=era, 2=n, 3=excl, 4=n_eff, 5=cov80, 6=cov50, 7=cov@10, 8=cov@50, 9=cov@90,
    10=PIT std, 11=mean PIT, 12=med rel width, 13=band_miss, 14=OOB, 15=set-valued (pt n).
    """
    [row] = [line for line in md.splitlines() if line.startswith(f"| {label} |")]
    return [cell.strip() for cell in row.split("|")]


def _pit_and_side(record: dict) -> tuple[float | None, str | None]:
    """``(point PIT, oob_side)`` for a record, for the point-valued cases in the PIT tests."""
    reading = compute_pit_reading(record)
    return (None, None) if reading is None else (reading.point, reading.oob_side)
