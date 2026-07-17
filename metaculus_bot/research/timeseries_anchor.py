"""Time-series anchor research provider (Phase B).

For a numeric Metaculus question whose resolution series is a fetchable
FRED/yfinance series, render a deterministic empirical anchor: the latest value,
a multi-resolution history, a 52-week range, and a horizon-matched empirical band
built ONLY from the series' own past. No LLM, no statsforecast, no model selection
— the Phase-A offline replay (``scratch/ts_anchor_replay_2026-07-16/synthesis.md``)
found CV-gated model picks beat naive out-of-sample only 43% of the time, while the
naive empirical h-step-change band is sharper AND better tail-calibrated than what
we publish (published cov@10 was 0.03 vs a 0.10 target; the anchor's was 0.18). The
band's value is grounding the forecaster and pulling in our over-wide low tail.

Backtest-safe — the FIRST research provider that is. Unlike prediction_market /
resolution_source (which hard-disable under ``is_benchmarking``), this provider
runs in benchmarks by pinning ``as_of`` to ``question.open_time`` and fetching the
series point-in-time up to that date (ALFRED vintages for revising macro series),
so series data known at forecast time IS the answer without leaking the resolution.

Routing is deterministic (no LLM):
  (a) URL extraction from resolution_criteria + fine_print (FRED series / Yahoo
      ticker URLs — the resolving source, highest precedence);
  (b) a conservative curated title-keyword registry.
Ambiguous → "" + log. Two Yahoo tickers → a relative-return spread block.

Estimators are pure-numpy ports of ``estimators.py`` (simplified, no registry or
CV policy): level/spread → empirical h-step-change quantiles (log for strictly-
positive series, absolute otherwise) applied to the last value; max-over-window →
empirical h-window-max distribution.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Literal
from urllib.parse import unquote

import numpy as np
import pandas as pd
from forecasting_tools.data_models.questions import MetaculusQuestion, NumericQuestion

from metaculus_bot.constants import (
    TS_ANCHOR_CHART_ENABLED_ENV,
    TS_ANCHOR_ENABLED_ENV,
    TS_ANCHOR_LOOKBACK_YEARS,
    TS_ANCHOR_MONTHLY_TABLE_ROWS,
    TS_ANCHOR_NATIVE_TABLE_ROWS,
    TS_ANCHOR_SECTION_MAX_CHARS,
    TS_ANCHOR_SPREAD_LOOKBACK_YEARS,
    TS_ANCHOR_TIMEOUT,
    TS_ANCHOR_WEEKLY_TABLE_ROWS,
    env_flag_enabled,
)
from metaculus_bot.research.providers import ResearchCallable
from metaculus_bot.research.ts_fetch import (
    FRED_NON_REVISING_SERIES,
    FetchError,
    SeriesSpec,
    _reset_series_cache,
    fetch_series,
)

logger = logging.getLogger(__name__)

Freq = Literal["daily", "weekly", "monthly"]
YfColumn = Literal["Close", "High", "Low", "Open"]

# Horizon conversion constants (ported from the replay's run_replay.py).
TRADING_DAYS_PER_YEAR = 252.0
CALENDAR_DAYS_PER_YEAR = 365.0
CALENDAR_DAYS_PER_MONTH = 30.4375

QUANTILE_LEVELS = (0.10, 0.50, 0.90)
REALIZED_VOL_WINDOW = 30  # trailing daily returns for the annualized-vol note

PROVENANCE_FOOTER = (
    "Statistical extrapolation of the resolution series' own history; blind to news, "
    "events, and policy — weigh against the rest of the research."
)

# ---------------------------------------------------------------------------
# Deterministic URL/keyword routing
# ---------------------------------------------------------------------------

# Mirror financial_data's URL extraction regexes (kept local to avoid importing
# financial_data's yfinance/fredapi module chain just for two patterns).
_FRED_SERIES_URL_RE = re.compile(r"fred\.stlouisfed\.org/series/([A-Za-z0-9_]+)")
_YAHOO_TICKER_URL_RE = re.compile(r"finance\.yahoo\.com/quote/([A-Za-z0-9.^=\-]+)")
# Metaculus injects markdown backslash escapes into rendered URLs (à la
# resolution_source.strip_markdown_escapes); undo them before matching.
_MARKDOWN_ESCAPE_RE = re.compile(r"\\([_&.\-#()])")
# "Highest / peak / maximum" framings resolve on the forward-window max, not the
# period-end level.
_MAX_KEYWORD_RE = re.compile(r"\b(highest|peak|maximum|max)\b", re.IGNORECASE)

# The non-revising FRED allowlist lives in ts_fetch (fetch-layer knowledge shared with
# financial_data); imported above as FRED_NON_REVISING_SERIES.


@dataclass(frozen=True)
class _TemplateEntry:
    keywords: tuple[str, ...]
    series_id: str
    source: Literal["fred", "yfinance"]
    label: str
    # PAYEMS-style series: the question resolves on the monthly CHANGE, not the
    # level. v1 renders the level history + a note and skips the band (no derived-
    # target modeling). model_target=False marks that.
    model_target: bool = True
    note: str = ""
    # NOTE: whether a fred series revises is NOT stored here — it is derived from the
    # single source of truth `_FRED_NON_REVISING_SERIES` in `_single_spec` (default:
    # revises=True unless allowlisted), so URL-routed and registry-routed series share
    # one revises decision. A per-entry flag here would silently do nothing (and mask a
    # leakage bug if it disagreed with the allowlist).


# Conservative curated registry: title keyword(s) → resolving series. Kept small
# and unambiguous on purpose; anything not here (or matching >1 entry) yields "".
_TEMPLATE_REGISTRY: tuple[_TemplateEntry, ...] = (
    _TemplateEntry(
        (
            "10-year treasury",
            "10 year treasury",
            "10-yr treasury",
            "10y treasury",
            "ten-year treasury",
            "10-year yield",
        ),
        "DGS10",
        "fred",
        "10-Year Treasury constant-maturity yield (%)",
    ),
    _TemplateEntry(
        ("high yield oas", "high-yield oas", "hy oas", "ice bofa", "high-yield spread", "high yield spread"),
        "BAMLH0A0HYM2",
        "fred",
        "ICE BofA US High Yield OAS (%)",
    ),
    _TemplateEntry(("vix",), "^VIX", "yfinance", "CBOE Volatility Index (VIX)"),
    _TemplateEntry(
        ("regular gasoline", "regular gas price", "gasoline price"),
        "GASREGW",
        "fred",
        "US regular all-formulations gasoline price ($/gal)",
    ),
    _TemplateEntry(("brent",), "DCOILBRENTEU", "fred", "Brent crude spot ($/bbl)"),
    _TemplateEntry(("wti", "west texas intermediate"), "CL=F", "yfinance", "WTI crude front-month ($/bbl)"),
    _TemplateEntry(
        ("cpi", "consumer price index"),
        "CPIAUCSL",
        "fred",
        "CPI-U, all items (SA index)",
    ),
    _TemplateEntry(("unemployment rate",), "UNRATE", "fred", "US unemployment rate (%)"),
    _TemplateEntry(
        ("nonfarm payroll", "non-farm payroll", "payroll employment", "payrolls"),
        "PAYEMS",
        "fred",
        "Total nonfarm payrolls (thousands of persons, LEVEL)",
        model_target=False,
        note=(
            "This is the payrolls LEVEL series. The question most likely resolves on the monthly "
            "CHANGE (jobs added); v1 shows the level history only and does NOT extrapolate a change "
            "band — do not read a level anchor as a jobs-added forecast."
        ),
    ),
    _TemplateEntry(
        ("consumer sentiment", "michigan sentiment", "umich sentiment", "consumer confidence"),
        "UMCSENT",
        "fred",
        "U. Michigan consumer sentiment index",
    ),
)


@dataclass(frozen=True)
class _Route:
    """A resolved routing decision: what series to fetch and how to render it."""

    kind: Literal["single", "spread"]
    spec: SeriesSpec
    label: str
    spec_b: SeriesSpec | None = None  # second leg for spreads
    label_b: str = ""
    model_target: bool = True
    is_max: bool = False  # forward-window-max question (VIX/Brent "highest") vs period-end level
    note: str = ""


def _extract_url_identifiers(criteria: str, fine_print: str) -> tuple[list[str], list[str]]:
    """Return (fred_series, yahoo_tickers) cited in the question's resolution text.

    URL-decodes (``%5E`` → ``^``) and unescapes Metaculus markdown backslashes
    before matching, matching financial_data / resolution_source handling.
    Order-preserving dedup.
    """
    text = _MARKDOWN_ESCAPE_RE.sub(r"\1", unquote(f"{criteria}\n{fine_print}"))
    fred = list(dict.fromkeys(_FRED_SERIES_URL_RE.findall(text)))
    # The Yahoo char class swallows a sentence-final '.'; strip trailing dots only
    # (internal dots like DX-Y.NYB are preserved).
    tickers = list(dict.fromkeys(t.rstrip(".") for t in _YAHOO_TICKER_URL_RE.findall(text)))
    return fred, tickers


def _wants_max(text: str) -> bool:
    return bool(_MAX_KEYWORD_RE.search(text))


def _single_spec(series_id: str, source: Literal["fred", "yfinance"], text: str) -> SeriesSpec:
    """Build the SeriesSpec for one leg, honoring ALFRED-revising and daily-High."""
    if source == "fred":
        # Default to ALFRED vintages (revises=True) for every fred series except the
        # curated non-revising allowlist. Unknown / URL-cited series therefore fetch
        # point-in-time, which is leakage-safe (see FRED_NON_REVISING_SERIES).
        revises = series_id.upper() not in FRED_NON_REVISING_SERIES
        return SeriesSpec(source="fred", series_id=series_id, revises=revises)
    column: YfColumn = "High" if _wants_max(text) else "Close"
    return SeriesSpec(source="yfinance", series_id=series_id, column=column)


def route_question(question: MetaculusQuestion) -> _Route | None:
    """Deterministically route a question to a series, or None if it doesn't map.

    URL extraction wins over the keyword registry (the cited source is the ground
    truth). Two Yahoo tickers → a relative-return spread. A single cited series →
    a level/max anchor. Anything ambiguous (>1 series that isn't the 2-ticker
    spread, or >1 registry keyword match) returns None with a log line.
    """
    text = f"{question.question_text or ''}\n{question.resolution_criteria or ''}\n{question.fine_print or ''}"
    is_max = _wants_max(text)
    fred, tickers = _extract_url_identifiers(question.resolution_criteria or "", question.fine_print or "")

    total_ids = len(fred) + len(tickers)
    if total_ids >= 1:
        if len(tickers) == 2 and not fred:
            return _Route(
                kind="spread",
                spec=SeriesSpec(source="yfinance", series_id=tickers[0], column="Close"),
                label=tickers[0],
                spec_b=SeriesSpec(source="yfinance", series_id=tickers[1], column="Close"),
                label_b=tickers[1],
            )
        if total_ids == 1:
            if fred:
                spec = _single_spec(fred[0], "fred", text)
                return _Route(kind="single", spec=spec, label=fred[0], is_max=is_max)
            spec = _single_spec(tickers[0], "yfinance", text)
            return _Route(kind="single", spec=spec, label=tickers[0], is_max=is_max)
        logger.info(
            "ts_anchor: ambiguous URL routing (fred=%s tickers=%s) for qid=%s — skipping",
            fred,
            tickers,
            getattr(question, "id_of_question", None),
        )
        return None

    lowered = text.lower()
    matches = [e for e in _TEMPLATE_REGISTRY if any(kw in lowered for kw in e.keywords)]
    if not matches:
        return None
    if len(matches) > 1:
        logger.info(
            "ts_anchor: ambiguous keyword routing (%s) for qid=%s — skipping",
            [m.series_id for m in matches],
            getattr(question, "id_of_question", None),
        )
        return None

    entry = matches[0]
    spec = _single_spec(entry.series_id, entry.source, text)
    return _Route(
        kind="single",
        spec=spec,
        label=entry.label,
        model_target=entry.model_target,
        is_max=is_max,
        note=entry.note,
    )


# ---------------------------------------------------------------------------
# Deterministic estimators (pure numpy, ported from estimators.py)
# ---------------------------------------------------------------------------


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
    window-max / window-anchor ratio (or difference) applied to the last value."""
    windows = np.lib.stride_tricks.sliding_window_view(y, h)  # (n-h+1, h)
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


# ---------------------------------------------------------------------------
# Rendering (self-budgeted; constants cap section + table sizes)
# ---------------------------------------------------------------------------


def _fmt(v: float) -> str:
    """Sensible sig figs: thousands-separated for large magnitudes, 4 sig figs otherwise."""
    a = abs(float(v))
    if a >= 10000:
        return f"{v:,.0f}"
    if a >= 100:
        return f"{v:,.1f}"
    return f"{v:.4g}"


_FREQ_UNIT: dict[Freq, str] = {"daily": "trading-day", "weekly": "week", "monthly": "month"}


def _history_lines(series: pd.Series, n: int, header: str) -> str:
    tail = series.tail(n)
    dates = pd.DatetimeIndex(tail.index).strftime("%Y-%m-%d")
    values = tail.to_numpy(dtype="float64")
    rows = [f"  - {d}: {_fmt(v)}" for d, v in zip(dates, values, strict=True)]
    return f"- {header}:\n" + "\n".join(rows)


def _downsample_last(series: pd.Series, rule: str) -> pd.Series:
    """Keep the last real observation within each calendar period, KEEPING its true
    observation date. Unlike ``resample(...).last()`` this never labels a row by a
    bucket-end date that postdates the ceiling — load-bearing for a leakage-safe
    provider (a Sunday week-end or a month-end label after the fetch ceiling would
    look like future data even though the value is genuine)."""
    periods = pd.DatetimeIndex(series.index).to_period(rule)
    keep = ~periods.duplicated(keep="last")
    return series[keep]


def _multi_res_history(series: pd.Series, freq: Freq) -> list[str]:
    """Native + coarser down-samples per frequency, using the per-resolution row caps."""
    blocks: list[str] = []
    if freq == "daily":
        blocks.append(_history_lines(series, TS_ANCHOR_NATIVE_TABLE_ROWS, "Last daily observations"))
        weekly = _downsample_last(series, "W")
        blocks.append(_history_lines(weekly, TS_ANCHOR_WEEKLY_TABLE_ROWS, "Weekly (last obs of week)"))
        monthly = _downsample_last(series, "M")
        blocks.append(_history_lines(monthly, TS_ANCHOR_MONTHLY_TABLE_ROWS, "Monthly (last obs of month)"))
    elif freq == "weekly":
        blocks.append(_history_lines(series, TS_ANCHOR_WEEKLY_TABLE_ROWS, "Last weekly observations"))
        monthly = _downsample_last(series, "M")
        blocks.append(_history_lines(monthly, TS_ANCHOR_MONTHLY_TABLE_ROWS, "Monthly (last obs of month)"))
    else:  # monthly
        blocks.append(_history_lines(series, TS_ANCHOR_MONTHLY_TABLE_ROWS, "Last monthly observations"))
    return blocks


def _fifty_two_week_line(series: pd.Series, ceiling: date, last: float) -> str:
    cutoff = pd.Timestamp(ceiling) - pd.Timedelta(days=365)
    window = series[series.index >= cutoff]
    if window.empty:
        window = series
    low = float(window.min())
    high = float(window.max())
    span = high - low
    pct = f"{(last - low) / span * 100:.0f}% of the way up the range" if span > 0 else "range is flat"
    return f"- 52-week range: {_fmt(low)} – {_fmt(high)} (latest sits {pct})"


def _realized_vol_line(series: pd.Series) -> str | None:
    """30-day annualized realized volatility (daily financial series only)."""
    returns = series.pct_change().dropna()
    if len(returns) < REALIZED_VOL_WINDOW:
        return None
    recent = returns.tail(REALIZED_VOL_WINDOW)
    annualized = float(recent.std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100.0)
    return f"- 30-day annualized realized volatility: {annualized:.1f}%"


def _band_line(
    kind: str, freq: Freq, h: int, lookback_years: int, band: tuple[float, float, float], last: float
) -> str:
    unit = _FREQ_UNIT[freq]
    p10, p50, p90 = band
    return (
        f"- Horizon-matched empirical band (over all {h}-{unit} {kind} windows in the last "
        f"~{lookback_years} years, applied to the latest value {_fmt(last)}):\n"
        f"  - P10 / P50 / P90 → {_fmt(p10)} / {_fmt(p50)} / {_fmt(p90)}"
    )


def _render_single(
    series: pd.Series,
    *,
    route: _Route,
    ceiling: date,
    calendar_days: int,
) -> str:
    freq = _detect_freq(pd.DatetimeIndex(series.index))
    last = float(series.iloc[-1])
    last_date = pd.DatetimeIndex(series.index)[-1].strftime("%Y-%m-%d")
    h = horizon_steps(freq, calendar_days)
    use_log = bool(np.all(series.to_numpy(dtype="float64") > 0.0))
    # A forward-window-max question (from title framing OR a High-column yfinance
    # spec) resolves on the max over the window, not the period-end level.
    is_max = route.is_max or route.spec.column == "High"

    parts: list[str] = [f"**{route.label}** — latest {_fmt(last)} (as of {last_date}; series frequency: {freq})"]
    if route.note:
        parts.append(f"- Note: {route.note}")
    parts.extend(_multi_res_history(series, freq))
    parts.append(_fifty_two_week_line(series, ceiling, last))

    y = series.to_numpy(dtype="float64")
    if route.model_target and y.size > h:
        if is_max:
            band = _empirical_max_band(y, h, use_log=use_log, last=last)
            parts.append(_band_line("forward-max", freq, h, TS_ANCHOR_LOOKBACK_YEARS, band, last))
        else:
            band = _empirical_change_band(y, h, use_log=use_log, anchor=last)
            parts.append(_band_line("change", freq, h, TS_ANCHOR_LOOKBACK_YEARS, band, last))
    elif route.model_target:
        parts.append(f"- (Horizon {h} exceeds available history; empirical band withheld.)")

    if freq == "daily" and use_log:
        vol_line = _realized_vol_line(series)
        if vol_line:
            parts.append(vol_line)

    parts.append(f"\n_{PROVENANCE_FOOTER}_")
    return "\n".join(parts)


def _render_spread(
    series_a: pd.Series,
    series_b: pd.Series,
    *,
    route: _Route,
    calendar_days: int,
) -> str:
    spread_series = _build_spread_series(series_a, series_b)  # raises ValueError on bad legs
    freq = _detect_freq(pd.DatetimeIndex(spread_series.index))
    h = horizon_steps(freq, calendar_days)
    y = spread_series.to_numpy(dtype="float64")
    if y.size <= h:
        raise ValueError(f"spread history length {y.size} too short for horizon {h}")
    # Re-anchor to 0 at the forecast ceiling: the band is the forward-window
    # relative return (pp), which is what the question resolves on.
    band = _empirical_change_band(y, h, use_log=False, anchor=0.0)

    last_a = float(series_a.iloc[-1])
    last_b = float(series_b.iloc[-1])
    date_a = pd.DatetimeIndex(series_a.index)[-1].strftime("%Y-%m-%d")
    date_b = pd.DatetimeIndex(series_b.index)[-1].strftime("%Y-%m-%d")
    parts: list[str] = [
        f"**Relative-return spread: {route.label} vs {route.label_b}** "
        f"(ret[{route.label}] − ret[{route.label_b}] over the forecast window, in percentage points)",
        f"- {route.label} latest: {_fmt(last_a)} (as of {date_a})",
        f"- {route.label_b} latest: {_fmt(last_b)} (as of {date_b})",
    ]
    parts.append(_history_lines(series_a, TS_ANCHOR_NATIVE_TABLE_ROWS, f"{route.label} recent"))
    parts.append(_history_lines(series_b, TS_ANCHOR_NATIVE_TABLE_ROWS, f"{route.label_b} recent"))
    unit = _FREQ_UNIT[freq]
    parts.append(
        f"- Forward {h}-{unit} relative-return band (pp; ~mean-zero prior, empirical over the last "
        f"~{TS_ANCHOR_SPREAD_LOOKBACK_YEARS} years):\n"
        f"  - P10 / P50 / P90 → {_fmt(band[0])} / {_fmt(band[1])} / {_fmt(band[2])}"
    )
    parts.append(f"\n_{PROVENANCE_FOOTER}_")
    return "\n".join(parts)


def _truncate_section(text: str) -> str:
    """Hard char-budget backstop (row caps normally keep it well under)."""
    if len(text) <= TS_ANCHOR_SECTION_MAX_CHARS:
        return text
    marker = "\n[truncated — time-series anchor section budget]"
    return text[: TS_ANCHOR_SECTION_MAX_CHARS - len(marker)].rstrip() + marker


def _maybe_stash_single_chart(
    series: pd.Series, *, route: _Route, question: MetaculusQuestion, as_of: datetime, calendar_days: int
) -> None:
    """Render + stash the anchor chart for a single LEVEL question when the chart
    flag is on. No-op for max-window / spread / no-band cases (v1 charts only the
    level shape, where the ribbon reads cleanly). Chart-render failures are
    swallowed so a plotting hiccup never breaks the text section — the chart is a
    strict add-on and the caller's soft-fail only guards the text.
    """
    if not env_flag_enabled(TS_ANCHOR_CHART_ENABLED_ENV):
        return
    qid = getattr(question, "id_of_question", None)
    if qid is None:
        return
    is_max = route.is_max or route.spec.column == "High"
    if is_max or not route.model_target:
        return

    freq = _detect_freq(pd.DatetimeIndex(series.index))
    h = horizon_steps(freq, calendar_days)
    y = series.to_numpy(dtype="float64")
    if y.size <= h:
        return  # band withheld in the text too; nothing to chart
    last = float(series.iloc[-1])
    use_log = bool(np.all(y > 0.0))
    band = _empirical_change_band(y, h, use_log=use_log, anchor=last)

    from metaculus_bot.research.ts_chart import (
        render_anchor_chart,  # noqa: PLC0415, HARNESS-SCAN-EXEMPT-function-level-import  # matplotlib off the cold path
    )

    as_of_ts = pd.Timestamp(as_of).tz_localize(None) if pd.Timestamp(as_of).tzinfo else pd.Timestamp(as_of)
    try:
        _session_charts[qid] = render_anchor_chart(
            series,
            as_of=as_of_ts,
            horizon_end=_horizon_end_date(as_of_ts, freq, h),
            band=band,
            title=route.label,
        )
    except (ValueError, RuntimeError) as exc:
        logger.warning("ts_anchor: chart render failed for qid=%s (%s): %s", qid, type(exc).__name__, exc)


def build_anchor_section(question: MetaculusQuestion, as_of: datetime) -> str:
    """Synchronous core: route, fetch point-in-time, render. Returns "" if unroutable.

    Runs in a worker thread (all HTTP is blocking). Raises FetchError/ValueError on
    a genuine data problem — the async wrapper soft-fails those to "".
    """
    scheduled = getattr(question, "scheduled_resolution_time", None)
    if not isinstance(scheduled, datetime):
        return ""
    ceiling = as_of.date()
    calendar_days = max(1, (scheduled.date() - ceiling).days)

    route = route_question(question)
    if route is None:
        return ""

    if route.kind == "spread":
        assert route.spec_b is not None  # kind=="spread" always sets spec_b
        series_a = fetch_series(route.spec, ceiling, lookback_years=TS_ANCHOR_SPREAD_LOOKBACK_YEARS)
        series_b = fetch_series(route.spec_b, ceiling, lookback_years=TS_ANCHOR_SPREAD_LOOKBACK_YEARS)
        return _truncate_section(_render_spread(series_a, series_b, route=route, calendar_days=calendar_days))

    series = fetch_series(route.spec, ceiling, lookback_years=TS_ANCHOR_LOOKBACK_YEARS)
    _maybe_stash_single_chart(series, route=route, question=question, as_of=as_of, calendar_days=calendar_days)
    return _truncate_section(_render_single(series, route=route, ceiling=ceiling, calendar_days=calendar_days))


# ---------------------------------------------------------------------------
# Per-session cache
# ---------------------------------------------------------------------------

# Rendered-section cache keyed (qid, as_of_iso) — the as_of leg keeps a backtest
# at one as-of from reusing a section computed at another.
_SECTION_CACHE: dict[tuple[int, str], str] = {}

# Chart side-channel: qid -> base64 PNG. Populated by build_anchor_section only
# when TS_ANCHOR_CHART_ENABLED is on AND the question routed to a single LEVEL
# series (v1 skips max-window / spread charts). The provider returns only the
# text section; the forecaster pulls the image from here to attach to each base
# model's vision message. Never read by the stacker / summarizer / gap-fill.
_session_charts: dict[int, str] = {}


def _reset_session_caches() -> None:
    """Clear the section + chart caches and the underlying series cache (tests + session start)."""
    _SECTION_CACHE.clear()
    _session_charts.clear()
    _reset_series_cache()


def _as_of_iso(as_of: datetime) -> str:
    aware = as_of if as_of.tzinfo else as_of.replace(tzinfo=UTC)
    return aware.astimezone(UTC).isoformat()


# ---------------------------------------------------------------------------
# ResearchCallable factory
# ---------------------------------------------------------------------------


def timeseries_anchor_provider(is_benchmarking: bool = False) -> ResearchCallable:
    """Factory for the time-series-anchor research callable.

    Gated on ``TS_ANCHOR_ENABLED`` (double-gated: orchestrator selects on the flag,
    and ``_fetch`` re-checks it). Unlike other backtest-sensitive providers this one
    does NOT hard-disable under ``is_benchmarking`` — it is leakage-safe by pinning
    ``as_of`` to ``question.open_time`` (live: ``datetime.now(UTC)``) and fetching the
    series point-in-time up to that date, with ALFRED vintages for revising macro
    series. Soft-fails to "" on any per-question fetch/data error or timeout.
    """

    async def _fetch(question: MetaculusQuestion) -> str:
        if not env_flag_enabled(TS_ANCHOR_ENABLED_ENV):
            return ""  # noqa: ASYNC910
        if not isinstance(question, NumericQuestion):
            return ""  # noqa: ASYNC910

        if is_benchmarking:
            open_time = getattr(question, "open_time", None)
            if not isinstance(open_time, datetime):
                logger.warning(
                    "ts_anchor: is_benchmarking but qid=%s has no open_time; skipping (leakage-safe)",
                    getattr(question, "id_of_question", None),
                )
                return ""  # noqa: ASYNC910
            as_of = open_time
        else:
            as_of = datetime.now(UTC)

        qid = getattr(question, "id_of_question", None)
        cache_key = (qid, _as_of_iso(as_of)) if qid is not None else None
        if cache_key is not None and cache_key in _SECTION_CACHE:
            return _SECTION_CACHE[cache_key]  # noqa: ASYNC910

        try:
            section = await asyncio.wait_for(
                asyncio.to_thread(build_anchor_section, question, as_of),
                timeout=TS_ANCHOR_TIMEOUT,
            )
        except (FetchError, ValueError, asyncio.TimeoutError) as exc:
            logger.warning("ts_anchor: soft-fail for qid=%s (%s): %s", qid, type(exc).__name__, exc)
            return ""  # noqa: ASYNC910

        if cache_key is not None:
            _SECTION_CACHE[cache_key] = section
        return section

    return _fetch
