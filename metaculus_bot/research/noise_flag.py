"""The vendor-noise screen shared by both rendered volatilities, and its telemetry line.

Two surfaces annualize ONE-day returns for a forecaster — ``financial_data``'s yfinance block
and ``ts_render``'s time-series anchor — and both are wrong in the same way on a thin quote:
independent quote noise inflates every volatility computed from one-day returns. The screen
is the Lo-MacKinlay variance ratio (``ts_estimators.variance_ratio``) against
``FINANCIAL_VARIANCE_RATIO_FLOOR``, and the remedy is the same statistic's own answer, the
volatility measured on overlapping ``FINANCIAL_VARIANCE_RATIO_LAG``-step returns.

This module exists because the screen and its ``FINANCIAL_NOISE_FLAG`` format string were
duplicated across the two renderers, which is how the two copies of the vol estimator drifted
before (the q44882 sqrt(252)-on-a-24/7-series defect was fixed in one copy weeks before the
other). Only the forecaster-facing PROSE is local to each renderer — the two say different
things about what else in their section the noise does or does not affect.

The line is RETURNED rather than logged here so each surface logs it under its own module
logger, which is what the marker spec and docs/operations.md describe ("``surface=ts_anchor``
is ts_render.py's ``_realized_vol_lines``"); the format string still has exactly one owner.
"""

from dataclasses import dataclass

import pandas as pd

from metaculus_bot.constants import (
    FINANCIAL_VARIANCE_RATIO_FLOOR,
    FINANCIAL_VARIANCE_RATIO_LAG,
    FINANCIAL_VARIANCE_RATIO_MIN_RETURNS,
)
from metaculus_bot.research.ts_estimators import multi_period_annualized_vol_pct, variance_ratio


@dataclass(frozen=True)
class NoiseScreen:
    """A screen that FIRED: the ratio below the floor, and the volatility that survives it.

    ``robust_vol_pct`` is the annualized volatility measured on overlapping
    ``FINANCIAL_VARIANCE_RATIO_LAG``-step returns — the figure a forecaster should size an
    interval from. It is typed optional because the estimator can refuse a sample, though the
    two estimators share one refusal policy (``_overlapping_log_return_sample``), so a fired
    screen with no remedy is not currently reachable.
    """

    ratio: float
    robust_vol_pct: float | None


def screen_for_quote_noise(series: pd.Series, *, periods_per_year: int) -> NoiseScreen | None:
    """The fired screen, or ``None`` when the series is not noise-dominated.

    ``None`` covers both "the ratio sits at or above the floor" and "the sample cannot carry
    the statistic" — neither is a flag, and the caller renders its unqualified volatility line
    in both cases.
    """
    ratio = variance_ratio(series, lag=FINANCIAL_VARIANCE_RATIO_LAG, min_returns=FINANCIAL_VARIANCE_RATIO_MIN_RETURNS)
    if ratio is None or ratio >= FINANCIAL_VARIANCE_RATIO_FLOOR:
        return None
    robust_vol = multi_period_annualized_vol_pct(
        series,
        lag=FINANCIAL_VARIANCE_RATIO_LAG,
        periods_per_year=periods_per_year,
        min_returns=FINANCIAL_VARIANCE_RATIO_MIN_RETURNS,
    )
    return NoiseScreen(ratio=ratio, robust_vol_pct=robust_vol)


def noise_flag_line(
    screen: NoiseScreen,
    *,
    surface: str,
    symbol: str,
    short_vol: float,
    long_vol: float | None,
) -> str:
    """The ``FINANCIAL_NOISE_FLAG`` telemetry line, in the shape ``markers.py`` harvests.

    One shape for both surfaces, every field always present: ``long_vol`` reads ``None`` on
    the anchor surface, which computes no long-horizon window at all, exactly as it does on a
    yfinance series too short to hold one — ``surface`` is what tells those apart, and the
    harvested record was already null in both cases. A uniform shape is why the spec's group
    is required rather than optional, so a future field reorder harvests as a clean zero
    instead of recording ``None`` for a value that WAS emitted.

    ``symbol`` is the ticker or FRED series id. It is not derivable at either call site (the
    close column's ``.name`` is "Close"), and without it the fan-out's concurrent per-ticker
    lines are indistinguishable.
    """
    robust = screen.robust_vol_pct
    return (
        f"FINANCIAL_NOISE_FLAG: surface={surface} symbol={symbol} vr_lag={FINANCIAL_VARIANCE_RATIO_LAG} "
        f"vr={screen.ratio:.3f} floor={FINANCIAL_VARIANCE_RATIO_FLOOR} short_vol={short_vol:.1f} "
        f"long_vol={long_vol if long_vol is None else round(long_vol, 1)} "
        f"robust_vol={robust if robust is None else round(robust, 1)}"
    )
