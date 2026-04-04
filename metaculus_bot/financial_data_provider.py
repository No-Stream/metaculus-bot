"""Financial data research provider using yfinance and FRED.

Fetches real price/indicator data for questions involving trackable financial metrics.
Follows the same factory-function-returning-ResearchCallable pattern as other providers.
"""

import asyncio
import logging
import os

import numpy as np
import pandas as pd
import yfinance
from forecasting_tools import GeneralLlm
from fredapi import Fred

from metaculus_bot.constants import (
    FINANCIAL_CLASSIFIER_MODEL,
    FINANCIAL_CLASSIFIER_TIMEOUT,
    FINANCIAL_YFINANCE_LOOKBACK_DAYS,
    FINANCIAL_YFINANCE_RECENT_DAYS,
    FRED_API_KEY_ENV,
)
from metaculus_bot.fallback_openrouter import build_llm_with_openrouter_fallback
from metaculus_bot.research_providers import ResearchCallable

logger: logging.Logger = logging.getLogger(__name__)

CLASSIFIER_PROMPT = """You are a classifier that determines whether a forecasting question involves financial markets or economic indicators that can be looked up via stock/index/commodity tickers or FRED economic data series.

Respond in EXACTLY this format (3 lines, no extra text):
FINANCIAL: YES or NO
TICKERS: comma-separated yfinance tickers, or NONE
FRED_SERIES: comma-separated FRED series IDs, or NONE

REFERENCE TABLE of common tickers and FRED series:

Stock indices: ^GSPC (S&P 500), ^DJI (Dow Jones), ^IXIC (Nasdaq), ^RUT (Russell 2000), ^FTSE (FTSE 100), ^N225 (Nikkei)
Stocks: AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA, META, BRK-B, JPM, V
Commodities: CL=F (crude oil), GC=F (gold), SI=F (silver), NG=F (natural gas), HG=F (copper)
Crypto: BTC-USD (Bitcoin), ETH-USD (Ethereum)
Bonds/Rates: ^TNX (10Y Treasury yield), ^FVX (5Y Treasury yield), ^TYX (30Y Treasury yield)
Currencies: EURUSD=X, GBPUSD=X, USDJPY=X, DX-Y.NYB (US Dollar Index)

FRED series:
- UNRATE (unemployment rate), PAYEMS (nonfarm payrolls)
- CPIAUCSL (CPI all items), CPILFESL (core CPI), PCEPI (PCE price index)
- GDP (gross domestic product), GDPC1 (real GDP)
- FEDFUNDS (federal funds rate), DFF (daily fed funds)
- DGS10 (10Y Treasury rate), DGS2 (2Y Treasury rate), T10Y2Y (10Y-2Y spread)
- CSUSHPISA (Case-Shiller home price index), HOUST (housing starts)
- UMCSENT (consumer sentiment), RSAFS (retail sales)
- M2SL (M2 money supply), WALCL (Fed balance sheet)

Only output YES if there are specific tickers or FRED series that would provide useful data.
If the question is about general economic trends without specific measurable indicators, output NO.

Question: {question_text}"""


async def _classify_financial_question(
    question_text: str,
    classifier_llm: GeneralLlm,
) -> dict[str, list[str]] | None:
    """Use an LLM to determine if a question involves trackable financial/economic data.

    Returns {"tickers": [...], "fred_series": [...]} or None if not financial.
    """
    try:
        prompt = CLASSIFIER_PROMPT.format(question_text=question_text)
        response = await classifier_llm.invoke(prompt)
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
        closing_lines = [f"  - {date.strftime('%Y-%m-%d')}: {price:.2f}" for date, price in last_5.items()]
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

        # Try to get series title
        title = series_id
        try:
            info_df = fred.get_series_info(series_id)
            if isinstance(info_df, pd.DataFrame) and "title" in info_df.columns:
                title = info_df["title"].iloc[0]
            elif isinstance(info_df, pd.Series) and "title" in info_df.index:
                title = info_df["title"]
        except Exception:
            pass

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
            mom_pct = (mom_change / abs(data.iloc[-2])) * 100 if data.iloc[-2] != 0 else 0
            parts.append(f"- Change from previous: {mom_change:+.4g} ({mom_pct:+.2f}%)")

        # Year-over-year change (try ~12 periods back)
        if len(data) >= 13:
            yoy_value = data.iloc[-13]
            yoy_change = latest_value - yoy_value
            yoy_pct = (yoy_change / abs(yoy_value)) * 100 if yoy_value != 0 else 0
            parts.append(f"- Year-over-year change: {yoy_change:+.4g} ({yoy_pct:+.2f}%)")

        # Last 6 observations
        last_6 = data.tail(6)
        obs_lines = [f"  - {date.strftime('%Y-%m-%d')}: {val:.4g}" for date, val in last_6.items()]
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
        timeout=FINANCIAL_CLASSIFIER_TIMEOUT,
    )

    async def _fetch(question_text: str) -> str:
        classification = await _classify_financial_question(question_text, classifier_llm)
        if classification is None:
            return ""

        tickers = classification["tickers"]
        fred_series = classification["fred_series"]
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

        return "\n\n".join(non_empty_results)

    return _fetch
