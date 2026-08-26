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

from dataclasses import dataclass
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd

Freq = Literal["daily", "weekly", "monthly", "quarterly", "annual"]

# Annualization / horizon-conversion bases (ported from the replay's run_replay.py).
# THE package's single definition of these two facts — `financial_data` imports them for its
# period-return slip grace and vol annualization, so they are ints: consumers key, count,
# and slice with them (`_PERIOD_SLIP_GRACE_DAYS[...]`, `pd.Timedelta(days=n)`), and the float
# arithmetic here is unchanged by int operands (true division and `round` don't care).
TRADING_DAYS_PER_YEAR = 252
CALENDAR_DAYS_PER_YEAR = 365
CALENDAR_DAYS_PER_MONTH = 30.4375
# Mean calendar quarter/year for the coarse-cadence step conversions (Julian year / 4).
# Distinct from the int CALENDAR_DAYS_PER_YEAR above, which is a daily-bar ROW basis.
CALENDAR_DAYS_PER_QUARTER = 91.3125
CALENDAR_DAYS_PER_ANNUM = 365.25

QUANTILE_LEVELS = (0.10, 0.50, 0.90)

# Below this many rows the observed-density read is too noisy to overrule the trading-day
# default (a fortnight of bars can't distinguish 5/7 from 7/7).
_MIN_ROWS_FOR_DENSITY_READ = 14

# Rows per calendar day above which a daily-bar series is read as a 24/7 market. 5/7 = 0.714
# for exchange-traded assets (0.690 with holidays), 1.0 for 24/7; 6/7 splits them with room
# on both sides.
_TWENTY_FOUR_SEVEN_ROWS_PER_DAY = 6.0 / 7.0


def _detect_freq(index: pd.DatetimeIndex) -> Freq:
    """Infer native frequency from the median day-gap between observations.

    Median, so a business-day series' weekend gaps don't drag it out of "daily" — which also
    means it CANNOT tell 5-bars-a-week from 7 (both have a 1.0-day median gap). That
    distinction is ``observed_periods_per_year``'s job; use ``series_clock`` to get both.

    The coarse buckets exist because "monthly" used to be the ceiling: a quarterly series
    (GDPC1, ~92-day gaps) classified monthly and every horizon converted at 30.4375 days per
    step — a 3x-too-wide band under a false "3-month" label, the same row-count-as-calendar
    defect class the daily 252/365 fix removed, reachable through any coarse FRED series
    cited by URL. Cadences the buckets still misdescribe (a ~183-day semiannual series lands
    in "annual") are refused a band by the cadence guard (``clock_matches_cadence``) rather
    than converted wrong.
    """
    if len(index) < 3:
        return "daily"
    diffs = np.diff(index.values).astype("timedelta64[D]").astype("float64")
    median_gap = float(np.median(diffs))
    if median_gap <= 4.0:
        return "daily"
    if median_gap <= 10.0:
        return "weekly"
    if median_gap <= 45.0:
        return "monthly"
    if median_gap <= 135.0:
        return "quarterly"
    return "annual"


def observed_periods_per_year(index: pd.Index) -> int:
    """Observed-density annualization basis for a DAILY-bar series: 252 or 365.

    Reads rows per calendar day over the series' own span: ~5/7 (0.71) for exchange-traded
    assets -> 252, ~1.0 for 24/7 markets (crypto) -> 365. Deliberately reads the SERIES, not
    the ticker: any heuristic keyed on ticker names goes stale the first time a new 24/7
    symbol is classified. Anything unmeasurable — short series, non-datetime index, zero
    span — degrades to the trading-day basis, the historical behavior.

    THE package's one definition of this read (``financial_data`` and ``series_clock`` both
    call it), because the whole defect class is a row count standing in for calendar time and
    a second copy is where a correction goes missing.

    Known limitation: a series sampled every 2-4 days reads as "daily" in ``_detect_freq`` and
    lands on the 252 basis here, which OVERSTATES its horizon in steps. No registry series has
    that cadence, and an overstated horizon fails safe — ``_render_single`` withholds the band
    (and the caller then drops the section) once ``y.size <= h``.
    """
    if len(index) < _MIN_ROWS_FOR_DENSITY_READ:
        return TRADING_DAYS_PER_YEAR
    try:
        span_days = (index[-1] - index[0]).days
    except (TypeError, AttributeError):
        # yfinance/FRED normally hand us a DatetimeIndex, but a caller passing a plain
        # RangeIndex has no span to measure; treat it as unmeasurable rather than crashing
        # a research provider on a cosmetic stat.
        return TRADING_DAYS_PER_YEAR
    if span_days <= 0:
        return TRADING_DAYS_PER_YEAR
    rows_per_day = (len(index) - 1) / span_days
    return CALENDAR_DAYS_PER_YEAR if rows_per_day > _TWENTY_FOUR_SEVEN_ROWS_PER_DAY else TRADING_DAYS_PER_YEAR


def daily_step_unit(periods_per_year: int) -> str:
    """Label noun for one daily-bar observation on the given annualization basis.

    ONE definition of the ternary (``SeriesClock.step_unit`` and ``financial_data``'s
    vol label both call it): the unit name and the basis must flip together, and a
    second copy of the rule is where a correction goes missing.
    """
    return "trading-day" if periods_per_year == TRADING_DAYS_PER_YEAR else "calendar-day"


# Calendar-day age a daily-bar series' latest observation may reach before it reads as
# STALE — older than the series' own observed cadence explains. 365 basis: one nominal
# 1-day step plus one grace day. 252 basis: a weekend plus one holiday (a Friday close
# read on a Tuesday is routine; a fifth day is not).
_STALE_AGE_ALLOWED_DAYS: dict[int, int] = {CALENDAR_DAYS_PER_YEAR: 2, TRADING_DAYS_PER_YEAR: 4}


def stale_latest_age_days(last_obs: date, as_of: date, periods_per_year: int) -> int | None:
    """Age in days of a daily-bar series' latest observation, when staler than its cadence allows.

    Returns the age when it exceeds the basis's allowance, else ``None``. THE package's one
    definition of "the latest value is stale" — ``financial_data``'s price line and
    ``ts_render``'s anchor header both consume it, the same single-copy rule the vol
    estimator above earned the hard way. Daily bases only: the keys are the two daily-bar
    annualization bases, and a weekly/monthly caller has no business here (KeyError is the
    correct failure, not a silent default).
    """
    age = (as_of - last_obs).days
    if age > _STALE_AGE_ALLOWED_DAYS[periods_per_year]:
        return age
    return None


def annualized_realized_vol_pct(series: pd.Series, *, window: int, periods_per_year: int) -> float | None:
    """Annualized realized volatility (%) over the last ``window`` simple returns.

    THE package's one vol estimator — ``ts_render._realized_vol_line`` and
    ``financial_data``'s yfinance block previously carried byte-identical copies, and the
    q44882 sqrt(252)-on-a-24/7-series defect was fixed in one copy weeks before the other
    (e6ae276 vs c577231): same window, same formula, same basis, corrected separately.
    ``None`` when fewer than ``window`` returns exist — a vol computed on a shorter sample
    would wear the window's label without its sample size.
    """
    returns = series.pct_change().dropna()
    if len(returns) < window:
        return None
    return float(returns.tail(window).std() * np.sqrt(periods_per_year) * 100.0)


@dataclass(frozen=True)
class SeriesClock:
    """A fetched series' observation clock: native resolution AND observed sampling density.

    Both facts are needed to convert between calendar time (what a question asks about: "by
    2026-09-01", "annualized") and observation index (rows the estimators slide over), and
    ``freq`` alone cannot do it: ``_detect_freq`` reads the MEDIAN gap, which is 1.0 day for a
    business-day series (gaps 1,1,1,1,3) AND 1.0 for a 24/7 series, so both land in "daily"
    while their rows-per-year differ by 365/252 = 1.45x. Bundling the pair in one object is
    what stops a caller from converting with the default basis on a 24/7 series — the
    q44882 (ETH-USD) defect class.
    """

    freq: Freq
    periods_per_year: int

    @property
    def step_unit(self) -> str:
        """Noun for one observation, for rendered horizon labels ("21-trading-day windows")."""
        if self.freq == "weekly":
            return "week"
        if self.freq == "monthly":
            return "month"
        if self.freq == "quarterly":
            return "quarter"
        if self.freq == "annual":
            return "year"
        return "trading-day" if self.periods_per_year == TRADING_DAYS_PER_YEAR else "calendar-day"

    @property
    def nominal_step_days(self) -> float:
        """Calendar days one observation step is ASSUMED to span, per the freq bucket.

        The other half of the cadence guard: ``horizon_steps`` converts on this assumption,
        so a series whose real gap disagrees with it gets a band whose true span disagrees
        by the same factor. Compared against the observed median gap in
        ``clock_matches_cadence``.
        """
        if self.freq == "weekly":
            return 7.0
        if self.freq == "monthly":
            return CALENDAR_DAYS_PER_MONTH
        if self.freq == "quarterly":
            return CALENDAR_DAYS_PER_QUARTER
        if self.freq == "annual":
            return CALENDAR_DAYS_PER_ANNUM
        return CALENDAR_DAYS_PER_YEAR / self.periods_per_year


# The clock of a DERIVED monthly target (MoM change / MoM % / monthly average). One step is
# one month, so the daily-only density field is never read; it carries the nominal basis.
MONTHLY_CLOCK = SeriesClock(freq="monthly", periods_per_year=TRADING_DAYS_PER_YEAR)


def series_clock(index: pd.DatetimeIndex) -> SeriesClock:
    """Read both clock facts off a series' index. The density read only means anything for a
    daily-bar series, so the coarser buckets carry the nominal trading-day basis unread (their
    own horizon conversions are already calendar-honest: /7, /30.4375, /91.3125, /365.25)."""
    freq = _detect_freq(index)
    periods_per_year = observed_periods_per_year(index) if freq == "daily" else TRADING_DAYS_PER_YEAR
    return SeriesClock(freq=freq, periods_per_year=periods_per_year)


def clock_matches_cadence(clock: SeriesClock, index: pd.DatetimeIndex) -> bool:
    """True when the series' observed cadence agrees with the clock's assumed step (within 1.5x).

    The fail-safe for every cadence the freq buckets misdescribe: a ~183-day semiannual series
    lands in "annual" and one 365.25-day step then spans two real observations (band too
    narrow); a 2-4-day series lands in "daily" on the 252 basis (band too wide). Either
    direction is a wrong quantity under a confident label, so a mismatched series gets NO band
    — the render path's band-withheld → section-dropped guard — rather than a mis-converted
    one. Checked in BOTH directions because the two failure modes sit on opposite sides.

    Too-short-to-measure indexes pass: with fewer than 3 observations ``_detect_freq``
    defaulted the bucket rather than reading the cadence, and the history-vs-horizon guard is
    the one that fires on those.
    """
    if len(index) < 3:
        return True
    diffs = np.diff(index.values).astype("timedelta64[D]").astype("float64")
    median_gap = float(np.median(diffs))
    if median_gap <= 0.0:
        return True  # intraday duplicates measure 0 days; the daily bucket is honest for them
    step = clock.nominal_step_days
    return step / 1.5 <= median_gap <= step * 1.5


def horizon_steps(clock: SeriesClock, calendar_days: int) -> int:
    """Native-step horizon for a calendar-day window, on the series' own clock (>=1).

    Takes the whole ``SeriesClock`` rather than a ``Freq`` plus an optional density, and has no
    default basis, so a new call site physically cannot convert a 24/7 series on the
    trading-day factor. On the 252 basis this is byte-identical to the historical formula; on
    the 365 basis one step IS one calendar day, so h == calendar_days.
    """
    if clock.freq == "daily":
        h = round(calendar_days * clock.periods_per_year / CALENDAR_DAYS_PER_YEAR)
    elif clock.freq == "weekly":
        h = round(calendar_days / 7.0)
    elif clock.freq == "monthly":
        h = round(calendar_days / CALENDAR_DAYS_PER_MONTH)
    elif clock.freq == "quarterly":
        h = round(calendar_days / CALENDAR_DAYS_PER_QUARTER)
    else:  # annual
        h = round(calendar_days / CALENDAR_DAYS_PER_ANNUM)
    return max(1, h)


def _horizon_end_date(as_of: pd.Timestamp, clock: SeriesClock, h: int) -> pd.Timestamp:
    """Approximate calendar date of the horizon end, for placing the projected band ribbon on
    the chart (mirrors the replay's make_charts._horizon_dates).

    The exact inverse of ``horizon_steps`` on the same clock, and that pairing is load-bearing:
    the two conversions used to be wrong in OPPOSITE directions by the same 365/252, so on a
    24/7 series they cancelled and the chart's ribbon happened to end at the right date while
    the band it drew was a 62-day band labelled 90 days. Fixing either alone would have broken
    the x-extent that was accidentally right.
    """
    if clock.freq == "daily":
        return as_of + pd.Timedelta(days=round(h * CALENDAR_DAYS_PER_YEAR / clock.periods_per_year))
    if clock.freq == "weekly":
        return as_of + pd.Timedelta(weeks=h)
    if clock.freq == "quarterly":
        return as_of + pd.DateOffset(months=3 * h)
    if clock.freq == "annual":
        return as_of + pd.DateOffset(years=h)
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
