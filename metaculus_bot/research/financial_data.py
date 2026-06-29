"""Financial data research provider using yfinance and FRED.

Fetches real price/indicator data for questions involving trackable financial metrics.
Follows the same factory-function-returning-ResearchCallable pattern as other providers.
"""

import asyncio
import logging
import os
import re
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
from metaculus_bot.research.providers import ResearchCallable

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
) -> dict[str, list[str]] | None:
    """Use an LLM to determine if a question involves trackable financial/economic data.

    Returns {"tickers": [...], "fred_series": [...]} or None if not financial.
    Resolution criteria / fine print are passed through so the classifier also
    sees the resolving source (belt-and-suspenders to the deterministic extraction).
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
    except Exception:
        logger.warning("Financial classifier LLM call failed", exc_info=True)
        return None

    return _parse_classifier_response(response)


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


def _fetch_yfinance_data(ticker: str) -> str:
    """Fetch price data and key metrics for a single ticker via yfinance.

    Sync function -- caller wraps in asyncio.to_thread().
    Returns formatted markdown or "" on any failure.
    """
    try:
        ticker_obj = yfinance.Ticker(ticker)
        period = f"{FINANCIAL_YFINANCE_LOOKBACK_DAYS}d"
        history = ticker_obj.history(period=period)

        if history.empty:
            logger.warning(f"yfinance returned empty history for {ticker=}")
            return ""

        info = ticker_obj.info or {}
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

        # Optional fundamentals from .info
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


def _fetch_fred_data(series_id: str, api_key: str) -> str:
    """Fetch economic data for a single FRED series.

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

        # Year-over-year change (try ~12 periods back)
        if len(data) >= 13:
            yoy_value = data.iloc[-13]
            yoy_change = latest_value - yoy_value
            yoy_pct = (yoy_change / abs(float(yoy_value))) * 100 if yoy_value != 0 else 0
            parts.append(f"- Year-over-year change: {yoy_change:+.4g} ({yoy_pct:+.2f}%)")

        # Last 6 observations
        last_6 = data.tail(6)
        obs_lines = [f"  - {cast(pd.Timestamp, date).strftime('%Y-%m-%d')}: {val:.4g}" for date, val in last_6.items()]
        parts.append("- Recent observations:\n" + "\n".join(obs_lines))

        return "\n".join(parts)

    except Exception:
        logger.warning(f"FRED fetch failed for {series_id=}", exc_info=True)
        return ""


def financial_data_provider() -> ResearchCallable:
    """Factory function returning an async research callable for financial/economic data.

    The callable:
    1. Classifies whether the question involves financial data (via cheap LLM).
    2. Fetches relevant data from yfinance and/or FRED in parallel.
    3. Combines results into structured markdown.

    FRED gracefully degrades if FRED_API_KEY is not set.
    """
    classifier_llm = build_llm_with_openrouter_fallback(
        model=FINANCIAL_CLASSIFIER_MODEL,
        temperature=0.0,
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
        criteria_text = f"{question.resolution_criteria or ''}\n{question.fine_print or ''}"
        extracted = extract_financial_identifiers_from_criteria(criteria_text)

        classification = await _classify_financial_question(
            question.question_text,
            classifier_llm,
            resolution_criteria=question.resolution_criteria or "",
            fine_print=question.fine_print or "",
        )

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

        fred_api_key = os.getenv(FRED_API_KEY_ENV)

        tasks: list[asyncio.Task] = []

        for ticker in tickers:
            tasks.append(asyncio.ensure_future(asyncio.to_thread(_fetch_yfinance_data, ticker)))

        if fred_api_key:
            for series_id in fred_series:
                tasks.append(asyncio.ensure_future(asyncio.to_thread(_fetch_fred_data, series_id, fred_api_key)))
        elif fred_series:
            logger.info(f"FRED_API_KEY not set, skipping {len(fred_series)} FRED series fetches")

        if not tasks:
            return ""

        results = await asyncio.gather(*tasks, return_exceptions=True)

        non_empty_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Financial data fetch task failed: {result}")
                continue
            if isinstance(result, str) and result.strip():
                non_empty_results.append(result)

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
