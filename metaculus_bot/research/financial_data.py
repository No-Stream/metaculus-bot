"""Financial data research provider using yfinance and FRED.

Fetches real price/indicator data for questions involving trackable financial metrics.
Follows the same factory-function-returning-ResearchCallable pattern as other providers.
"""

import asyncio
import logging
import os
import re
from datetime import UTC, datetime, timedelta
from typing import cast
from urllib.parse import unquote

import numpy as np
import pandas as pd
import yfinance
from forecasting_tools import GeneralLlm
from forecasting_tools.data_models.questions import MetaculusQuestion
from fredapi import Fred

from metaculus_bot.constants import (
    FINANCIAL_CLASSIFIER_MODEL,
    FINANCIAL_CLASSIFIER_TIMEOUT,
    FINANCIAL_YFINANCE_LOOKBACK_DAYS,
    FINANCIAL_YFINANCE_RECENT_DAYS,
    FRED_API_KEY_ENV,
)
from metaculus_bot.fallback_openrouter import build_llm_with_openrouter_fallback
from metaculus_bot.llm_retry import invoke_with_transient_retry
from metaculus_bot.research.provider_diagnostics import record_provider_detail
from metaculus_bot.research.providers import ResearchCallable
from metaculus_bot.research.ts_fetch import FRED_NON_REVISING_SERIES, FetchError, SeriesSpec, fetch_series

logger: logging.Logger = logging.getLogger(__name__)

# FRED series IDs are alphanumeric + underscore (e.g. DGS10, BAMLH0A0HYM2, T10Y2Y).
# Yahoo tickers add `^`, `=`, `.`, `-` (e.g. ^TNX, CL=F, BTC-USD, EURUSD=X).
_FRED_SERIES_URL_RE = re.compile(r"fred\.stlouisfed\.org/series/([A-Za-z0-9_]+)")
_YAHOO_TICKER_URL_RE = re.compile(r"finance\.yahoo\.com/quote/([A-Za-z0-9.^=\-]+)")

# Full-string char-class guards (same classes the extraction regexes enforce), used
# to sanitize classifier-emitted IDs — which come from comma-splitting with NO
# char validation — before they reach the fetch set and the HTML-comment marker.
# A `-->` in a classifier token would otherwise close the comment and leak its tail
# as visible markdown in the published Metaculus comment.
_TICKER_CHARS_RE = re.compile(r"^[A-Za-z0-9.^=\-]+$")
_FRED_CHARS_RE = re.compile(r"^[A-Za-z0-9_]+$")

# Single source of truth for the reference identifiers: id -> human label, grouped
# by category. The KNOWN_* frozensets AND the CLASSIFIER_PROMPT reference table are
# both DERIVED from these dicts, so the allowlist and the prompt cannot drift apart.
# Frozensets feed the soft-fail flagging (unrecognized classifier IDs surface as
# WARNINGs + in the routing marker, never silently dropped); labels feed the prompt.
_TICKER_GROUPS: dict[str, dict[str, str]] = {
    "Stock indices": {
        "^GSPC": "S&P 500",
        "^DJI": "Dow Jones",
        "^IXIC": "Nasdaq",
        "^RUT": "Russell 2000",
        "^FTSE": "FTSE 100",
        "^N225": "Nikkei",
    },
    "Stocks": {
        "AAPL": "",
        "MSFT": "",
        "GOOGL": "",
        "AMZN": "",
        "NVDA": "",
        "TSLA": "",
        "META": "",
        "BRK-B": "",
        "JPM": "",
        "V": "",
    },
    "Commodities": {
        "CL=F": "crude oil",
        "GC=F": "gold",
        "SI=F": "silver",
        "NG=F": "natural gas",
        "HG=F": "copper",
    },
    "Crypto": {
        "BTC-USD": "Bitcoin",
        "ETH-USD": "Ethereum",
    },
    "Bonds/Rates": {
        "^TNX": "10Y Treasury yield",
        "^FVX": "5Y Treasury yield",
        "^TYX": "30Y Treasury yield",
    },
    "Currencies": {
        "EURUSD=X": "",
        "GBPUSD=X": "",
        "USDJPY=X": "",
        "DX-Y.NYB": "US Dollar Index",
    },
}
_FRED_GROUPS: dict[str, dict[str, str]] = {
    "labor": {"UNRATE": "unemployment rate", "PAYEMS": "nonfarm payrolls"},
    "inflation": {"CPIAUCSL": "CPI all items", "CPILFESL": "core CPI", "PCEPI": "PCE price index"},
    "output": {"GDP": "gross domestic product", "GDPC1": "real GDP"},
    "rates": {"FEDFUNDS": "federal funds rate", "DFF": "daily fed funds"},
    "treasury": {"DGS10": "10Y Treasury rate", "DGS2": "2Y Treasury rate", "T10Y2Y": "10Y-2Y spread"},
    "housing": {"CSUSHPISA": "Case-Shiller home price index", "HOUST": "housing starts"},
    "consumer": {"UMCSENT": "consumer sentiment", "RSAFS": "retail sales"},
    "money": {"M2SL": "M2 money supply", "WALCL": "Fed balance sheet"},
}

TICKER_LABELS: dict[str, str] = {tid: label for group in _TICKER_GROUPS.values() for tid, label in group.items()}
FRED_LABELS: dict[str, str] = {sid: label for group in _FRED_GROUPS.values() for sid, label in group.items()}
KNOWN_TICKERS: frozenset[str] = frozenset(TICKER_LABELS)
KNOWN_FRED_SERIES: frozenset[str] = frozenset(FRED_LABELS)


def _render_ticker(tid: str, label: str) -> str:
    """Render `id (label)` when a label exists, else the bare id (stocks have none)."""
    return f"{tid} ({label})" if label else tid


def _build_ticker_reference_lines() -> str:
    """Build the prompt's grouped ticker reference lines from _TICKER_GROUPS."""
    return "\n".join(
        f"{group_name}: " + ", ".join(_render_ticker(tid, label) for tid, label in members.items())
        for group_name, members in _TICKER_GROUPS.items()
    )


def _build_fred_reference_lines() -> str:
    """Build the prompt's FRED reference bullet lines from _FRED_GROUPS."""
    return "\n".join(
        "- " + ", ".join(f"{sid} ({label})" for sid, label in members.items()) for members in _FRED_GROUPS.values()
    )


def _dedupe_preserving_order(items: list[str]) -> list[str]:
    """Drop duplicates while keeping first-seen order (dict preserves insertion)."""
    return list(dict.fromkeys(items))


def _sanitize_classifier_ids(items: list[str], char_re: re.Pattern[str], kind: str) -> list[str]:
    """Drop classifier IDs that don't fully match the extraction char class (log-and-skip).

    Classifier IDs come from comma-splitting with no char validation, so a garbled
    token (e.g. one containing `-->`) could close the routing HTML comment and leak
    its tail as visible markdown, or be sent pointlessly to yfinance/FRED. Filter to
    the same char classes the URL-extraction regexes enforce, warning on each drop so
    a malformed classifier emission is visible rather than silently dropped.
    """
    sanitized: list[str] = []
    for item in items:
        if char_re.fullmatch(item):
            sanitized.append(item)
        else:
            logger.warning("financial classifier emitted malformed %s %r — dropping", kind, item)
    return sanitized


def extract_financial_identifiers_from_criteria(text: str) -> dict[str, list[str]]:
    """Deterministically extract the resolving FRED series / Yahoo tickers from URLs.

    Resolution criteria usually name the exact source the question resolves on
    (e.g. https://fred.stlouisfed.org/series/DGS10). Extracting these directly
    guarantees the resolving series fires regardless of the LLM classifier's guess.

    URL-decodes first so `%5ETNX` -> `^TNX` matches the Yahoo ticker pattern.
    Returns {"tickers": [...], "fred_series": [...]}, deduped, order-preserving.
    """
    decoded = unquote(text)
    fred_series = _dedupe_preserving_order(_FRED_SERIES_URL_RE.findall(decoded))
    # The Yahoo char class includes `.` with no right boundary, so a sentence-final
    # URL captures the trailing period (e.g. `.../quote/%5ETNX.` -> `^TNX.`), which
    # isn't in KNOWN_TICKERS and fails the yfinance lookup. `.rstrip(".")` only trims
    # trailing dots — internal dots (e.g. `DX-Y.NYB`) are preserved.
    tickers = _dedupe_preserving_order([t.rstrip(".") for t in _YAHOO_TICKER_URL_RE.findall(decoded)])
    return {"tickers": tickers, "fred_series": fred_series}


# Built from _TICKER_GROUPS / _FRED_GROUPS so the prompt's reference table and the
# KNOWN_* allowlist share one source of truth (the f-string-injected blocks carry no
# `{...}` of their own; the trailing {question_text}/{resolution_criteria}/{fine_print}
# remain str.format placeholders filled by _classify_financial_question).
CLASSIFIER_PROMPT = f"""You are a classifier that determines whether a forecasting question involves financial markets or economic indicators that can be looked up via stock/index/commodity tickers or FRED economic data series.

Respond in EXACTLY this format (3 lines, no extra text):
FINANCIAL: YES or NO
TICKERS: comma-separated yfinance tickers, or NONE
FRED_SERIES: comma-separated FRED series IDs, or NONE

REFERENCE TABLE of common tickers and FRED series:

{_build_ticker_reference_lines()}

FRED series:
{_build_fred_reference_lines()}

Only output YES if there are specific tickers or FRED series that would provide useful data.
If the question is about general economic trends without specific measurable indicators, output NO.

Question: {{question_text}}

Resolution criteria (may name the exact resolving source/series):
{{resolution_criteria}}
{{fine_print}}"""


async def _classify_financial_question(
    question_text: str,
    classifier_llm: GeneralLlm,
    resolution_criteria: str = "",
    fine_print: str = "",
) -> tuple[dict[str, list[str]] | None, str | None]:
    """Use an LLM to determine if a question involves trackable financial/economic data.

    Returns ``(classification, error_reason)``: the classification is
    ``{"tickers": [...], "fred_series": [...]}`` or None when the question isn't
    financial, and ``error_reason`` is the exception type name when the call FAILED
    (None otherwise). Resolution criteria / fine print are passed through so the
    classifier also sees the resolving source (belt-and-suspenders to the deterministic
    extraction).

    The two None cases have to be told apart by the caller: "read as non-financial" is a
    normal outcome, while a dead classifier (a model retirement — the 2026-05-15 grok 404
    precedent — a schema change, a quota) is a lost source that must render on the
    diagnostics line. They used to be indistinguishable, so a persistently broken
    classifier looked exactly like a question with no financial angle.
    """
    try:
        prompt = CLASSIFIER_PROMPT.format(
            question_text=question_text,
            resolution_criteria=resolution_criteria,
            fine_print=fine_print,
        )
        # Wrapped with the elapsed-gated transient retry (litellm #14895): the
        # classifier LLM is built allowed_tries=1, so this wrapper is its sole
        # retry layer and also supplies the wall-clock cap the call previously
        # lacked. FINANCIAL_CLASSIFIER_TIMEOUT doubles as the wall cap (no
        # separate constant exists; the per-request timeout is the natural bound).
        response = await invoke_with_transient_retry(
            lambda: classifier_llm.invoke(prompt),
            wall_timeout=FINANCIAL_CLASSIFIER_TIMEOUT,
            label="financial_classifier",
        )
    except Exception as exc:
        logger.warning("Financial classifier LLM call failed", exc_info=True)
        return (None, type(exc).__name__)

    return (_parse_classifier_response(response), None)


def _parse_classifier_response(response: str) -> dict[str, list[str]] | None:
    """Parse the structured 3-line classifier response into a dict or None."""
    lines: dict[str, str] = {}
    for line in response.strip().splitlines():
        line = line.strip()
        if ":" in line:
            key, _, value = line.partition(":")
            lines[key.strip().upper()] = value.strip().upper()

    financial_flag = lines.get("FINANCIAL", "")
    if not financial_flag.startswith("YES"):
        return None

    tickers = _parse_csv_field(lines.get("TICKERS", "NONE"))
    fred_series = _parse_csv_field(lines.get("FRED_SERIES", "NONE"))

    if not tickers and not fred_series:
        return None

    logger.debug(f"Financial classifier: {tickers=}, {fred_series=}")
    return {"tickers": tickers, "fred_series": fred_series}


def _parse_csv_field(raw: str) -> list[str]:
    """Parse a comma-separated field, returning [] for 'NONE' or empty."""
    if not raw or raw == "NONE":
        return []
    items = [item.strip() for item in raw.split(",")]
    return [item for item in items if item and item != "NONE"]


def _fetch_yfinance_data(ticker: str, *, as_of: datetime | None = None, is_benchmarking: bool = False) -> str:
    """Fetch price data and key metrics for a single ticker via yfinance.

    Sync function -- caller wraps in asyncio.to_thread().
    Returns formatted markdown or "" on any failure.

    Live (``is_benchmarking=False``): unchanged — a trailing ``period="365d"``
    window and the live ``.info`` fundamentals. Backtest (``is_benchmarking=True``):
    ceiling the price history to ``as_of`` via ``history(start=..., end=...)`` (end
    exclusive) and SKIP the ``.info`` call entirely — ``.info`` has no historical
    mode, so its market cap / P/E / current price would leak TODAY's values into a
    question resolved months ago. Current price then comes from the last ceilinged
    close, and every derived stat (returns / vol / 52wk) computes off that frame.
    """
    try:
        ticker_obj = yfinance.Ticker(ticker)
        if is_benchmarking:
            assert as_of is not None, "benchmarking yfinance fetch requires as_of"
            start = (as_of - timedelta(days=FINANCIAL_YFINANCE_LOOKBACK_DAYS)).date()
            end = (as_of + timedelta(days=1)).date()  # yfinance end is EXCLUSIVE → +1d makes as_of inclusive
            history = ticker_obj.history(start=start.isoformat(), end=end.isoformat())
        else:
            history = ticker_obj.history(period=f"{FINANCIAL_YFINANCE_LOOKBACK_DAYS}d")

        if history.empty:
            logger.warning(f"yfinance returned empty history for {ticker=}")
            return ""

        # Skip the live `.info` call under benchmarking (no historical mode → leakage).
        info: dict = {} if is_benchmarking else (ticker_obj.info or {})
        close = history["Close"]
        current_price = close.iloc[-1]

        parts = [f"### {ticker}"]
        name = info.get("shortName", "")
        if name:
            parts.append(f"**{name}**")
        parts.append(f"- Current price: {current_price:.2f}")

        # Period returns
        returns_section = _compute_period_returns(close)
        if returns_section:
            parts.append(returns_section)

        # Volatility (30-day annualized)
        if len(close) >= FINANCIAL_YFINANCE_RECENT_DAYS:
            daily_returns = close.pct_change().dropna()
            recent_daily = daily_returns.iloc[-FINANCIAL_YFINANCE_RECENT_DAYS:]
            annualized_vol = recent_daily.std() * np.sqrt(252) * 100
            parts.append(f"- 30-day annualized volatility: {annualized_vol:.1f}%")

        # 52-week range
        year_slice = close.iloc[-min(252, len(close)) :]
        low_52w = year_slice.min()
        high_52w = year_slice.max()
        parts.append(f"- 52-week range: {low_52w:.2f} - {high_52w:.2f}")

        # Optional fundamentals from .info (live only; `info` is {} under benchmarking).
        if is_benchmarking:
            parts.append("- Fundamentals: [omitted under backtest — .info has no historical mode]")
        else:
            fundamentals = _format_fundamentals(info)
            if fundamentals:
                parts.append(fundamentals)

        # Last 5 closing prices
        last_5 = close.tail(5)
        closing_lines = [
            f"  - {cast(pd.Timestamp, date).strftime('%Y-%m-%d')}: {price:.2f}" for date, price in last_5.items()
        ]
        parts.append("- Last 5 closes:\n" + "\n".join(closing_lines))

        return "\n".join(parts)

    except Exception:
        logger.warning(f"yfinance fetch failed for {ticker=}", exc_info=True)
        return ""


def _compute_period_returns(close: pd.Series) -> str:
    """Compute returns over standard periods, returning a formatted string."""
    periods = [
        ("1d", 1),
        ("1w", 5),
        ("1m", 21),
        ("3m", 63),
        ("6m", 126),
        ("1y", 252),
    ]
    lines = []
    for label, days in periods:
        if len(close) > days:
            start_price = close.iloc[-(days + 1)]
            end_price = close.iloc[-1]
            pct_change = (end_price / start_price - 1) * 100
            lines.append(f"  - {label}: {pct_change:+.2f}%")
    if lines:
        return "- Period returns:\n" + "\n".join(lines)
    return ""


def _format_fundamentals(info: dict) -> str:
    """Extract optional fundamental metrics from yfinance .info dict."""
    lines = []
    pe = info.get("trailingPE")
    if pe is not None:
        lines.append(f"  - P/E ratio: {pe:.1f}")
    market_cap = info.get("marketCap")
    if market_cap is not None:
        if market_cap >= 1e12:
            lines.append(f"  - Market cap: ${market_cap / 1e12:.2f}T")
        elif market_cap >= 1e9:
            lines.append(f"  - Market cap: ${market_cap / 1e9:.2f}B")
        else:
            lines.append(f"  - Market cap: ${market_cap / 1e6:.0f}M")
    fwd_eps = info.get("forwardEps")
    if fwd_eps is not None:
        lines.append(f"  - Forward EPS: {fwd_eps:.2f}")
    if lines:
        return "- Fundamentals:\n" + "\n".join(lines)
    return ""


def _render_fred_series(series_id: str, data: pd.Series, title: str) -> str:
    """Render the derived-stat markdown block for a FRED series.

    Shared by the live (fredapi) and benchmarking (keyless ts_fetch) paths so the two
    render identically — latest/previous value, MoM + YoY change, last 6 observations.
    ``data`` must already be dropna'd and sorted ascending by date.
    """
    parts = [f"### {series_id} ({title})"]

    latest_value = data.iloc[-1]
    latest_date = data.index[-1]
    parts.append(f"- Latest value: {latest_value:.4g} ({latest_date.strftime('%Y-%m-%d')})")

    if len(data) >= 2:
        previous_value = data.iloc[-2]
        parts.append(f"- Previous value: {previous_value:.4g}")

    # Month-over-month change (if monthly-ish frequency)
    if len(data) >= 2:
        mom_change = latest_value - data.iloc[-2]
        mom_pct = (mom_change / abs(float(data.iloc[-2]))) * 100 if data.iloc[-2] != 0 else 0
        parts.append(f"- Change from previous: {mom_change:+.4g} ({mom_pct:+.2f}%)")

    # Year-over-year change via a DATE-based lookup, not a fixed observation
    # offset: `data.iloc[-13]` is one year back only on a monthly series; on a
    # daily FRED series (DGS10, DGS2, T10Y2Y, ...) 13 observations is ~2.5 weeks,
    # which would be mislabeled "year-over-year" in a live forecaster prompt. Take
    # the most recent observation at or before ~365 days ago (label slice is
    # inclusive; data is sorted ascending). Omit the line entirely when no such
    # observation exists (series shorter than a year).
    year_ago = latest_date - pd.Timedelta(days=365)
    prior = data.loc[:year_ago]
    if not prior.empty:
        yoy_value = prior.iloc[-1]
        yoy_change = latest_value - yoy_value
        yoy_pct = (yoy_change / abs(float(yoy_value))) * 100 if yoy_value != 0 else 0
        parts.append(f"- Year-over-year change: {yoy_change:+.4g} ({yoy_pct:+.2f}%)")

    # Last 6 observations
    last_6 = data.tail(6)
    obs_lines = [f"  - {cast(pd.Timestamp, date).strftime('%Y-%m-%d')}: {val:.4g}" for date, val in last_6.items()]
    parts.append("- Recent observations:\n" + "\n".join(obs_lines))

    return "\n".join(parts)


def _fetch_fred_data(series_id: str, api_key: str) -> str:
    """Fetch economic data for a single FRED series (live path, fredapi).

    Sync function -- caller wraps in asyncio.to_thread().
    Returns formatted markdown or "" on any failure.
    """
    try:
        fred = Fred(api_key=api_key)
        data = fred.get_series(series_id)

        if data.empty:
            logger.warning(f"FRED returned empty data for {series_id=}")
            return ""

        data = data.dropna()
        if data.empty:
            return ""

        # Title is best-effort enrichment; fall back to the raw series_id if FRED metadata lookup fails.
        title = series_id
        try:
            info_df = fred.get_series_info(series_id)
            if isinstance(info_df, pd.DataFrame) and "title" in info_df.columns:
                title = cast(pd.Series, info_df["title"]).iloc[0]
            elif isinstance(info_df, pd.Series) and "title" in info_df.index:
                title = info_df["title"]
        except Exception:
            logger.debug(f"FRED series title lookup failed for {series_id=}", exc_info=True)

        return _render_fred_series(series_id, data, title)

    except Exception:
        logger.warning(f"FRED fetch failed for {series_id=}", exc_info=True)
        return ""


def _fetch_fred_data_ceiling(series_id: str, as_of: datetime) -> str:
    """Point-in-time FRED fetch for backtests, via the keyless ts_fetch path.

    Sync function -- caller wraps in asyncio.to_thread(). Reuses ``ts_fetch.fetch_series``
    (fredgraph / ALFRED-vintage), so revised macro series (CPI, payrolls, GDP) return the
    vintage KNOWN at ``as_of`` instead of today's revisions — a plain observation_end on
    fredapi would still leak those. Keyless, so it works in CI without FRED_API_KEY.

    Non-title header: get_series_info needs an API key, so the backtest block reuses the
    series_id as the title — identical to the live path's metadata-failure fallback.
    Returns formatted markdown or "" on any fetch/data error.
    """
    try:
        # Default to ALFRED vintages for every revising macro series; only the curated
        # non-revising allowlist is safe on plain fredgraph (same decision the anchor makes).
        revises = series_id.upper() not in FRED_NON_REVISING_SERIES
        spec = SeriesSpec(source="fred", series_id=series_id, revises=revises)
        data = fetch_series(spec, as_of.date())
        return _render_fred_series(series_id, data, series_id)
    except (FetchError, ValueError):
        logger.warning(f"FRED ceilinged fetch failed for {series_id=}", exc_info=True)
        return ""


def financial_data_provider(is_benchmarking: bool = False) -> ResearchCallable:
    """Factory function returning an async research callable for financial/economic data.

    The callable:
    1. Classifies whether the question involves financial data (via cheap LLM).
    2. Fetches relevant data from yfinance and/or FRED in parallel.
    3. Combines results into structured markdown.

    FRED gracefully degrades if FRED_API_KEY is not set.

    Backtest-safe like ``timeseries_anchor``: under ``is_benchmarking`` every fetch is
    ceilinged to ``question.open_time`` (NOT the resolution time — post-open data can
    contain the resolution). Live (``is_benchmarking=False``) fetches are unchanged: a
    trailing yfinance ``period=`` window and the fredapi path. yfinance ceilings via
    ``history(start=..., end=...)`` and skips the leaky ``.info`` fundamentals; FRED
    routes through the keyless ``ts_fetch`` fredgraph / ALFRED-vintage path so revised
    macro series return the vintage known at forecast time rather than today's revisions.
    """
    classifier_llm = build_llm_with_openrouter_fallback(
        model=FINANCIAL_CLASSIFIER_MODEL,
        # temperature=None defers reasoning models to provider defaults; redundant
        # on ft 0.2.92 (GeneralLlm ctor default is already None). No top_p.
        temperature=None,
        max_tokens=500,
        reasoning={"effort": "low"},
        timeout=FINANCIAL_CLASSIFIER_TIMEOUT,
        # allowed_tries=1 so the elapsed-gated transient retry in
        # _classify_financial_question is the SOLE retry layer. Without this the
        # builder defaults to allowed_tries=2, whose unguarded tenacity would
        # retry a slow stall — the exact failure mode the elapsed gate prevents.
        allowed_tries=1,
    )

    async def _fetch(question: MetaculusQuestion) -> str:
        # Ceiling every fetch to open_time under benchmarking (leakage-safe, mirrors
        # timeseries_anchor). A missing open_time can't be ceilinged, so we bail rather
        # than risk fetching today's data into a resolved question.
        if is_benchmarking:
            open_time = getattr(question, "open_time", None)
            if not isinstance(open_time, datetime):
                logger.warning(
                    "financial_data: is_benchmarking but qid=%s has no open_time; skipping (leakage-safe)",
                    getattr(question, "id_of_question", None),
                )
                return ""
            as_of = open_time
        else:
            as_of = datetime.now(UTC)

        criteria_text = f"{question.resolution_criteria or ''}\n{question.fine_print or ''}"
        extracted = extract_financial_identifiers_from_criteria(criteria_text)

        classification, classifier_error = await _classify_financial_question(
            question.question_text,
            classifier_llm,
            resolution_criteria=question.resolution_criteria or "",
            fine_print=question.fine_print or "",
        )
        qid = getattr(question, "id_of_question", None)
        if classifier_error is not None:
            # Recorded HERE, above the empty-fetch-set early return below, because that
            # return happens before the per-identifier record_provider_detail call at the
            # end — so a dead classifier on a question with no extracted identifiers
            # produced NO diagnostics detail at all and read as "no financial angle".
            # A later successful record for this qid overwrites the entry, which is
            # correct: if identifiers were still fetched, their per-source map is the
            # fuller picture and the classifier loss shows up there instead.
            record_provider_detail(qid, "financial_data", {"sources": {"classifier": f"error({classifier_error})"}})

        # Sanitize classifier IDs (F2) BEFORE they reach the fetch set, marker, or
        # unknown-flagging: they come from comma-splitting with no char validation,
        # unlike the regex-constrained extracted IDs. A malformed token (e.g. one
        # containing `-->`) would otherwise leak into the HTML-comment marker.
        classifier_tickers = _sanitize_classifier_ids(
            classification["tickers"] if classification else [], _TICKER_CHARS_RE, "ticker"
        )
        classifier_fred = _sanitize_classifier_ids(
            classification["fred_series"] if classification else [], _FRED_CHARS_RE, "FRED series"
        )

        # Extraction is ADDITIVE, not a replacement: the classifier may legitimately
        # add a Yahoo proxy for richer context, but extracted IDs guarantee the
        # resolving series is in the fetch set.
        tickers = _dedupe_preserving_order(classifier_tickers + extracted["tickers"])
        fred_series = _dedupe_preserving_order(classifier_fred + extracted["fred_series"])

        # The deterministic extraction is the load-bearing guarantee: even when the
        # classifier returns None (question read as non-financial) or misroutes, the
        # source the question RESOLVES ON must still fire. So we only bail when the
        # merged fetch set is empty.
        if not tickers and not fred_series:
            return ""

        # Soft-fail loudly: classifier IDs not in the reference allowlist (and not
        # independently confirmed by extraction) are flagged but still fetched —
        # a valid-but-unlisted series shouldn't be dropped, just made visible.
        unknown = _flag_unknown_classifier_ids(classifier_tickers, classifier_fred, extracted)

        tasks: list[asyncio.Task] = []
        identifiers: list[str] = []  # the ticker/FRED id each task fetches, parallel to tasks

        for ticker in tickers:
            tasks.append(
                asyncio.ensure_future(
                    asyncio.to_thread(_fetch_yfinance_data, ticker, as_of=as_of, is_benchmarking=is_benchmarking)
                )
            )
            identifiers.append(ticker)

        if is_benchmarking:
            # Keyless ceilinged path — no FRED_API_KEY needed, works in CI, and returns
            # point-in-time vintages instead of today's revisions.
            for series_id in fred_series:
                tasks.append(asyncio.ensure_future(asyncio.to_thread(_fetch_fred_data_ceiling, series_id, as_of)))
                identifiers.append(series_id)
        else:
            fred_api_key = os.getenv(FRED_API_KEY_ENV)
            if fred_api_key:
                for series_id in fred_series:
                    tasks.append(asyncio.ensure_future(asyncio.to_thread(_fetch_fred_data, series_id, fred_api_key)))
                    identifiers.append(series_id)
            elif fred_series:
                logger.info(f"FRED_API_KEY not set, skipping {len(fred_series)} FRED series fetches")

        if not tasks:
            return ""

        results = await asyncio.gather(*tasks, return_exceptions=True)

        non_empty_results = []
        # Per-identifier outcome for the diagnostics block: a requested ticker/FRED
        # series that errored or returned no data is a lost source, so a partial
        # financial fetch stays visible even when other identifiers succeed.
        sources: dict[str, str] = {}
        for identifier, result in zip(identifiers, results, strict=True):
            if isinstance(result, Exception):
                logger.warning(f"Financial data fetch task failed: {result}")
                sources[identifier] = "error"
                continue
            if isinstance(result, str) and result.strip():
                non_empty_results.append(result)
                sources[identifier] = "ok"
            else:
                sources[identifier] = "empty"

        record_provider_detail(qid, "financial_data", {"sources": sources})

        if not non_empty_results:
            return ""

        marker = _build_routing_marker(fred_series, tickers, extracted, unknown)
        return "\n\n".join(non_empty_results) + marker

    return _fetch


def _flag_unknown_classifier_ids(
    classifier_tickers: list[str],
    classifier_fred: list[str],
    extracted: dict[str, list[str]],
) -> list[str]:
    """Return classifier IDs not in the reference allowlist nor confirmed by extraction.

    Logs a WARNING per unrecognized ID (soft-fail loudly). Caller still fetches them.
    """
    unknown: list[str] = []
    for ticker in classifier_tickers:
        if ticker not in KNOWN_TICKERS and ticker not in extracted["tickers"]:
            unknown.append(ticker)
            logger.warning("financial classifier emitted unrecognized ticker %r — fetching anyway but flagging", ticker)
    for series_id in classifier_fred:
        if series_id not in KNOWN_FRED_SERIES and series_id not in extracted["fred_series"]:
            unknown.append(series_id)
            logger.warning(
                "financial classifier emitted unrecognized FRED series %r — fetching anyway but flagging", series_id
            )
    return unknown


def _build_routing_marker(
    fred_series: list[str],
    tickers: list[str],
    extracted: dict[str, list[str]],
    unknown: list[str],
) -> str:
    """Build a forecaster-invisible, greppable routing marker (Part D observability).

    An HTML comment is invisible in rendered markdown but survives verbatim into
    research_text — the cached blob, the persisted artifact, and the Metaculus
    comment — so the routing decision is durable and auditable without changing
    the ResearchCallable signature.
    """
    return (
        f"\n\n<!-- financial_routing: fred=[{','.join(fred_series)}] tickers=[{','.join(tickers)}] "
        f"extracted_fred=[{','.join(extracted['fred_series'])}] "
        f"extracted_tickers=[{','.join(extracted['tickers'])}] unknown=[{','.join(unknown)}] -->"
    )
