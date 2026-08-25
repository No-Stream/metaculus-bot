"""Deterministic band estimators for the time-series-anchor provider (pure numpy).

Ported from ``estimators.py`` in the Phase-A offline replay
(``scratch/ts_anchor_replay_2026-07-16/synthesis.md``), simplified: no registry, no CV
policy, no model selection. The replay found CV-gated model picks beat naive out-of-sample
only 43% of the time, while the naive empirical h-step-change band is sharper AND better
tail-calibrated than what we publish.

Two families:
  - level / spread → empirical h-step-change quantiles (log for strictly-positive series,
    absolute otherwise) applied to the last value;
  - max-over-window → empirical h-window-max distribution.

A leaf module: it depends on nothing else in the anchor stack, so ``ts_render`` and
``timeseries_anchor`` can both build on it without a cycle.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

Freq = Literal["daily", "weekly", "monthly"]

# Annualization / horizon-conversion bases (ported from the replay's run_replay.py).
# THE package's single definition of these two facts — `financial_data` imports them for its
# period-return row offsets and vol annualization, so they are ints: the row-count consumers
# index and slice with them (`_PERIOD_ROW_OFFSETS[...]`, `close.iloc[-n:]`), and the float
# arithmetic here is unchanged by int operands (true division and `round` don't care).
TRADING_DAYS_PER_YEAR = 252
CALENDAR_DAYS_PER_YEAR = 365
CALENDAR_DAYS_PER_MONTH = 30.4375

QUANTILE_LEVELS = (0.10, 0.50, 0.90)


def _detect_freq(index: pd.DatetimeIndex) -> Freq:
    """Infer native frequency from the median day-gap between observations."""
    if len(index) < 3:
        return "daily"
    diffs = np.diff(index.values).astype("timedelta64[D]").astype("float64")
    median_gap = float(np.median(diffs))
    if median_gap <= 4.0:
        return "daily"
    if median_gap <= 10.0:
        return "weekly"
    return "monthly"


def horizon_steps(freq: Freq, calendar_days: int) -> int:
    """Native-step horizon for a calendar-day window, by series frequency (>=1)."""
    if freq == "daily":
        h = round(calendar_days * TRADING_DAYS_PER_YEAR / CALENDAR_DAYS_PER_YEAR)
    elif freq == "weekly":
        h = round(calendar_days / 7.0)
    else:  # monthly
        h = round(calendar_days / CALENDAR_DAYS_PER_MONTH)
    return max(1, h)


def _horizon_end_date(as_of: pd.Timestamp, freq: Freq, h: int) -> pd.Timestamp:
    """Approximate calendar date of the horizon end, for placing the projected
    band ribbon on the chart (mirrors the replay's make_charts._horizon_dates)."""
    if freq == "daily":
        return as_of + pd.Timedelta(days=round(h * CALENDAR_DAYS_PER_YEAR / TRADING_DAYS_PER_YEAR))
    if freq == "weekly":
        return as_of + pd.Timedelta(weeks=h)
    return as_of + pd.DateOffset(months=h)


def _empirical_change_band(y: np.ndarray, h: int, *, use_log: bool, anchor: float) -> tuple[float, float, float]:
    """P10/P50/P90 of the h-step-ahead value: empirical quantiles of all overlapping
    h-step changes applied to ``anchor``. Log-multiplicative for positive series,
    additive otherwise. Overlap induces autocorrelation (harmless for quantiles)."""
    base, fwd = y[:-h], y[h:]
    changes = (np.log(fwd) - np.log(base)) if use_log else (fwd - base)
    q10, q50, q90 = (float(v) for v in np.quantile(changes, QUANTILE_LEVELS, method="linear"))
    if use_log:
        return anchor * np.exp(q10), anchor * np.exp(q50), anchor * np.exp(q90)
    return anchor + q10, anchor + q50, anchor + q90


def _empirical_max_band(y: np.ndarray, h: int, *, use_log: bool, last: float) -> tuple[float, float, float]:
    """P10/P50/P90 of the MAX over the forward h-window: empirical quantiles of the
    window-max / window-anchor ratio (or difference) applied to the last value.

    Each window spans y[i..i+h] (h+1 points = an h-step horizon), matching the change
    band's y[i+h]-vs-y[i] span. Length-h windows would cover only y[i..i+h-1] and
    understate the h-step-ahead max (and collapse to ``last`` at h=1). The caller only
    invokes this when y.size > h, so the length-(h+1) view is always non-empty."""
    windows = np.lib.stride_tricks.sliding_window_view(y, h + 1)  # (n-h, h+1)
    window_max = windows.max(axis=1)
    win_anchor = y[: window_max.size]
    if use_log:
        ratios = np.log(window_max) - np.log(win_anchor)  # >= 0 by construction
        r10, r50, r90 = (float(v) for v in np.quantile(ratios, QUANTILE_LEVELS, method="linear"))
        return last * np.exp(r10), last * np.exp(r50), last * np.exp(r90)
    diffs = window_max - win_anchor  # >= 0
    d10, d50, d90 = (float(v) for v in np.quantile(diffs, QUANTILE_LEVELS, method="linear"))
    return last + d10, last + d50, last + d90


def _build_spread_series(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    """Cumulative-from-start relative-return spread (pp): 100·[(logA−logA₀)−(logB−logB₀)].

    Inner-joined on date. The h-step change of this cumulative series equals the
    forward-window relative return, so the band machinery reads it directly. Both
    legs must be strictly positive (log-returns); a non-positive value raises."""
    joined = pd.concat([series_a.rename("a"), series_b.rename("b")], axis=1, join="inner").dropna()
    if joined.empty:
        raise ValueError("spread legs have no overlapping dates")
    a = joined["a"].to_numpy(dtype="float64")
    b = joined["b"].to_numpy(dtype="float64")
    if np.any(a <= 0.0) or np.any(b <= 0.0):
        raise ValueError("spread relative-return needs strictly-positive price series")
    rel = 100.0 * ((np.log(a) - np.log(a[0])) - (np.log(b) - np.log(b[0])))
    return pd.Series(rel, index=joined.index, name="spread_relret")


def _n_eff(n_obs: int, h: int) -> int:
    """Rough count of statistically-INDEPENDENT h-step windows in ``n_obs`` observations.

    Overlapping windows share observations, so the effective independent sample size is
    ~``n_obs // h`` — far below the raw overlapping-window count at long horizons (a
    1-year daily horizon over 15 years is ~15 independent windows, not thousands).
    Floored at 1 (the band is only rendered when ``n_obs > h``, so this never bites in
    practice, but it keeps the reported number honest for degenerate inputs)."""
    return max(1, n_obs // h)
