"""Financial data research provider using yfinance and FRED.

Fetches real price/indicator data for questions involving trackable financial metrics.
Follows the same factory-function-returning-ResearchCallable pattern as other providers.
"""

import asyncio
import logging
import os
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, cast
from urllib.parse import unquote
from xml.etree.ElementTree import ParseError

import pandas as pd
import yfinance
from forecasting_tools import GeneralLlm
from forecasting_tools.data_models.questions import MetaculusQuestion
from fredapi import Fred

from metaculus_bot.constants import (
    FINANCIAL_CLASSIFIER_MODEL,
    FINANCIAL_CLASSIFIER_TIMEOUT,
    FINANCIAL_FRED_VINTAGE_PRINTS,
    FINANCIAL_VARIANCE_RATIO_FLOOR,
    FINANCIAL_VARIANCE_RATIO_LAG,
    FINANCIAL_YFINANCE_LOOKBACK_DAYS,
    FINANCIAL_YFINANCE_RECENT_DAYS,
    FRED_API_KEY_ENV,
    MAX_FINANCIAL_IDENTIFIERS,
)
from metaculus_bot.fallback_openrouter import build_llm_with_openrouter_fallback
from metaculus_bot.llm_retry import invoke_with_transient_retry
from metaculus_bot.research.noise_flag import noise_flag_line, screen_for_quote_noise
from metaculus_bot.research.provider_diagnostics import record_provider_detail
from metaculus_bot.research.providers import ResearchCallable
from metaculus_bot.research.ts_estimators import (
    CALENDAR_DAYS_PER_YEAR,
    TRADING_DAYS_PER_YEAR,
    annualized_realized_vol_pct,
    daily_step_unit,
    observed_periods_per_year,
    stale_latest_age_days,
)
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


@dataclass(frozen=True)
class CurrencyPeg:
    """A hard peg on a USD FX cross: where the pair's real dynamics live, and the regime.

    ``anchor_ticker`` is the LIQUID cross the pegged leg is fixed to, which is the honest
    read of the pegged pair's volatility and percent moves; ``None`` means the leg is
    pegged to the USD leg itself (HKD, AED, SAR, QAR), where no substitute cross exists and
    the only honest statement is that the pair has no independent market dynamics.
    ``regime`` names the peg rate and its start date and is rendered verbatim.

    Every ``anchor_ticker`` is Yahoo's per-USD ``XXX=X`` form — units of the anchor currency
    per US dollar, the same way round as the ``USD<currency>=X`` pair the question resolves
    on. That orientation is the whole reason the block can hand the anchor's SIGNED percent
    moves and its volatility to a forecaster as the pegged pair's own; only the price LEVELS
    are a different quantity (and only when the peg is not at par). An inverted anchor such
    as ``EURUSD=X`` (US dollars per euro) would move the opposite way and is forbidden by
    test — see ``test_every_anchor_is_the_per_usd_form``.
    """

    currency: str
    regime: str
    anchor_ticker: str | None


def _peg_entries(peg: CurrencyPeg) -> dict[str, CurrencyPeg]:
    """Both Yahoo spellings of USD/<currency>: the explicit pair and the USD-implied form.

    Yahoo serves the same cross as ``USDSZL=X`` and as ``SZL=X``; a question's resolution
    URL may cite either, and a lookup that knew only one would miss the peg silently.
    """
    return {f"USD{peg.currency}=X": peg, f"{peg.currency}=X": peg}


# Currencies whose USD cross is a fixed or band-bounded quote rather than a traded market,
# so its measured daily volatility is largely vendor noise. q44797 is the realized failure:
# `USDSZL=X`'s 17.8% "30-day annualized volatility" went to all six forecasters, 79% of that
# series' return variance was quote noise, and the like-for-like figure off the liquid rand
# cross was 10.6% — the single cheapest fix point in that question's ~32 peer-point loss.
# Deliberately a STATIC table, not a correlation detector: the failure class is hard pegs,
# which are published policy and do not need inferring. Peg facts verified 2026-09-01
# against each regime's own authority (HKMA, ECB/Danmarks Nationalbank, BCEAO, the CMA
# treaty history, and Reuters' FX-regime factbox for the Gulf pegs).
HARD_PEG_ANCHORS: dict[str, CurrencyPeg] = {
    # Common Monetary Area: each member issues its own currency at par with the rand, so
    # USD/<member> IS USD/ZAR plus quote noise. Dates are each currency's own par entry.
    **_peg_entries(
        CurrencyPeg(
            "SZL", "fixed at par with the South African rand since 1974 under the Common Monetary Area", "ZAR=X"
        )
    ),
    **_peg_entries(
        CurrencyPeg(
            "LSL", "fixed at par with the South African rand since 1980 under the Common Monetary Area", "ZAR=X"
        )
    ),
    **_peg_entries(
        CurrencyPeg(
            "NAD", "fixed at par with the South African rand since 1993 under the Common Monetary Area", "ZAR=X"
        )
    ),
    # Pegged to the euro, so the euro's own USD cross carries the dynamics. The anchor is
    # `EUR=X` — Yahoo's per-USD spelling, euros per US dollar (~0.86) — NOT `EURUSD=X`, which
    # quotes US dollars per euro (~1.16) and therefore moves the OPPOSITE way to USD/DKK:
    # USD/DKK = 7.46038 * (EUR per USD), so a 2% rise in `EUR=X` is a ~2% rise in the pegged
    # pair, while a 2% rise in `EURUSD=X` is a ~2% FALL in it. With the per-USD form the
    # anchor's signed percent moves and its volatility are the pegged pair's own; only the
    # levels differ.
    **_peg_entries(
        CurrencyPeg(
            "DKK",
            "held near the ERM II central rate of 7.46038 per euro (band 7.29252-7.62824) under Denmark's "
            "fixed-exchange-rate policy, in force since the 1980s",
            "EUR=X",
        )
    ),
    **_peg_entries(
        CurrencyPeg(
            "XOF",
            "fixed at 655.957 per euro since 1 January 1999, guaranteed by the French Treasury",
            "EUR=X",
        )
    ),
    **_peg_entries(
        CurrencyPeg(
            "XAF",
            "fixed at 655.957 per euro since 1 January 1999, guaranteed by the French Treasury",
            "EUR=X",
        )
    ),
    # Interchangeable at par with the Singapore dollar, which is the traded cross.
    **_peg_entries(
        CurrencyPeg(
            "BND",
            "interchangeable at par with the Singapore dollar under the 1967 Currency Interchangeability Agreement",
            "SGD=X",
        )
    ),
    # Pegged to the USD leg itself: no third currency to read instead.
    **_peg_entries(
        CurrencyPeg(
            "HKD",
            "held inside the HKMA's 7.75-7.85 Convertibility Zone (linked to the US dollar at 7.80 since "
            "17 October 1983; the two-sided band since May 2005)",
            None,
        )
    ),
    **_peg_entries(CurrencyPeg("AED", "pegged to the US dollar at 3.6725 since November 1997", None)),
    **_peg_entries(CurrencyPeg("SAR", "pegged to the US dollar at 3.75 since 1986", None)),
    **_peg_entries(
        CurrencyPeg("QAR", "pegged to the US dollar at 3.64 since 1980 (official band 3.6385-3.6415)", None)
    ),
}


def _peg_for_ticker(ticker: str) -> CurrencyPeg | None:
    """The peg record for a Yahoo FX ticker, or None when the pair is a traded market."""
    return HARD_PEG_ANCHORS.get(ticker.upper())


def _peg_disclosure_lines(ticker: str, peg: CurrencyPeg) -> list[str]:
    """The forecaster-facing peg warning: what is fixed, and what to read instead.

    The regime's subject is the CURRENCY, not the USD cross: every ``regime`` string states a
    rate against the peg's own reference (7.46038 DKK per euro, 655.957 XOF per euro), which
    is not the USD cross's level — rendering it as "USD/XOF is fixed at 655.957 per euro" put
    a ~16% level error directly above that block's own "Latest price: 565.31".
    """
    pair = f"USD/{peg.currency}"
    lines = [
        f"- ⚠ Pegged pair: {pair} — {peg.currency} is {peg.regime}. The volatility, 52-week "
        f"range and daily closes in this block are a thin vendor quote on a fixed cross, so "
        f"most of their day-to-day movement is quote noise rather than exchange-rate risk."
    ]
    if peg.anchor_ticker is None:
        lines.append(
            f"- There is no liquid third-currency cross to read instead — {peg.currency} is pegged to the US "
            "dollar itself, so this pair has no independent market dynamics. Do not size a forecast interval "
            "from any volatility figure below."
        )
    else:
        anchor_currency = peg.anchor_ticker.split("=")[0]
        lines.append(
            f"- Peg anchor: `{peg.anchor_ticker}` is the liquid cross the peg is fixed to, and its block is "
            f"appended directly below. It quotes {anchor_currency} per US dollar — the same way round as "
            f"{pair}, so read ITS volatility and signed percent moves as {pair}'s own."
        )
    return lines


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


def _cap_identifiers(
    tickers: list[str],
    fred_series: list[str],
    extracted: dict[str, list[str]],
) -> tuple[list[str], list[str]]:
    """Bound the total fetch set to ``MAX_FINANCIAL_IDENTIFIERS``, dropping classifier IDs first.

    Each identifier becomes its own ``asyncio.to_thread``, and the fan-out was
    previously unbounded — one over-eager classification could queue arbitrarily many
    blocking calls into the process-wide default executor that ts_fetch,
    resolution_source, the agentic fetch ladder, and the /auth/key probe all share.
    Queued tasks burn their ``wait_for`` budget without executing, so the damage lands on
    unrelated providers and other questions.

    Deliberately asymmetric about WHAT it drops: URL-EXTRACTED identifiers are kept
    unconditionally, because they are the load-bearing guarantee that the source a
    question actually resolves on is fetched even when the classifier misroutes (the
    invariant the merged-set bail below rests on). Only classifier-added extras are
    trimmed, from the tail, and the drop is logged so an over-eager classification is
    visible rather than silent.
    """
    total = len(tickers) + len(fred_series)
    if total <= MAX_FINANCIAL_IDENTIFIERS:
        return (tickers, fred_series)

    extracted_tickers = set(
        extracted["tickers"]
    )  # HARNESS-SCAN-EXEMPT-object-explosion: short id list, not a frame column
    extracted_fred = set(
        extracted["fred_series"]
    )  # HARNESS-SCAN-EXEMPT-object-explosion: short id list, not a frame column
    budget = MAX_FINANCIAL_IDENTIFIERS - len(extracted_tickers) - len(extracted_fred)
    dropped: list[str] = []

    def keep(identifiers: list[str], protected: set[str]) -> list[str]:
        nonlocal budget
        kept: list[str] = []
        for identifier in identifiers:
            if identifier in protected:
                kept.append(identifier)  # extracted: never dropped
            elif budget > 0:
                kept.append(identifier)
                budget -= 1
            else:
                dropped.append(identifier)
        return kept

    kept_tickers = keep(tickers, extracted_tickers)
    kept_fred = keep(fred_series, extracted_fred)
    logger.warning(
        "financial_data: capped fetch set at %d identifiers (%d requested); dropped classifier-only "
        "IDs %s. URL-extracted IDs are never dropped.",
        MAX_FINANCIAL_IDENTIFIERS,
        total,
        dropped,
    )
    return (kept_tickers, kept_fred)


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
    except Exception as exc:  # HARNESS-SCAN-EXEMPT-broad-except  # provider soft-fail boundary, logged
        logger.warning("Financial classifier LLM call failed", exc_info=True)
        return (None, type(exc).__name__)

    return (_parse_classifier_response(response), None)


def _parse_classifier_response(response: str) -> dict[str, list[str]] | None:
    """Parse the structured 3-line classifier response into a dict or None."""
    lines: dict[str, str] = {}
    for raw_line in response.strip().splitlines():
        line = raw_line.strip()
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


def _yfinance_history(ticker_obj: Any, window_end: datetime, *, is_benchmarking: bool) -> pd.DataFrame:
    """Daily bars from ``FINANCIAL_YFINANCE_LOOKBACK_DAYS`` before ``window_end``.

    Benchmarking additionally ceilings the window at ``window_end``; live leaves
    ``end`` unset so yfinance includes today's partial bar.
    """
    start = (window_end - timedelta(days=FINANCIAL_YFINANCE_LOOKBACK_DAYS)).date()
    if is_benchmarking:
        end = (window_end + timedelta(days=1)).date()  # yfinance end is EXCLUSIVE → +1d makes as_of inclusive
        return ticker_obj.history(start=start.isoformat(), end=end.isoformat())
    return ticker_obj.history(start=start.isoformat())


def _yfinance_latest_lines(
    ticker: str,
    close: pd.Series,
    reference_date: date,
    *,
    name: str,
    periods_per_year: int,
    is_benchmarking: bool,
) -> list[str]:
    """Header, optional name, the dated latest price, and the stale-observation warning."""
    parts = [f"### {ticker}"]
    if name:
        parts.append(f"**{name}**")
    current_price = close.iloc[-1]
    latest_obs_date = cast(pd.Timestamp, close.index[-1]).date()
    # Every "latest" carries its observation date: an undated headline reads as live
    # even when Yahoo's newest bar is days old (a weekend Friday close, or a
    # null-close hole silently dropped from the frame).
    latest_line = f"- Latest price: {current_price:.2f} (as of {latest_obs_date.isoformat()})"
    if not is_benchmarking and latest_obs_date == reference_date:
        latest_line += " — today's bar, in progress"
    parts.append(latest_line)

    stale_age = stale_latest_age_days(latest_obs_date, reference_date, periods_per_year)
    if stale_age is not None:
        unit = daily_step_unit(periods_per_year)
        parts.append(
            f"- ⚠ Latest observation is {stale_age} days old — beyond what a {unit} cadence "
            "explains; treat the latest price as stale."
        )
        logger.warning(
            f"FINANCIAL_STALE_LATEST: surface=financial_data symbol={ticker} age_d={stale_age} cadence={unit}"
        )
    return parts


def _volatility_lines(close: pd.Series, periods_per_year: int, *, symbol: str) -> list[str]:
    """Annualized volatility at two horizons, plus the vendor-noise flag when it fires.

    Two horizons because a single 30-row window is both a noisy estimate and, on a thin
    series, a systematically inflated one: q44797 published a 43-day forecast sized off a
    17.8% figure computed on 30 rows of a pegged cross, where the same series over a year
    read 15.2% and the liquid anchor read 10.6-12.9%.

    The noise flag is the variance-ratio screen (``FINANCIAL_VARIANCE_RATIO_*``). When it
    fires the ordering inverts: the noise-robust volatility (measured on multi-day returns,
    over which the reversing component cancels) leads, the long window follows, and the
    short window comes last carrying the noise-suspect label — so the number nearest the top
    is the one a forecaster should size an interval from. Unflagged, the short window stays
    first, exactly as before.
    """
    unit = daily_step_unit(periods_per_year)
    # Volatility over the trailing FINANCIAL_YFINANCE_RECENT_DAYS observations — the
    # shared estimator (ts_estimators), so this line and the anchor stack's vol note
    # cannot drift apart again; None when the return sample is shorter than the window
    # (a vol wearing the window's label without its sample size). Name the step unit:
    # FINANCIAL_YFINANCE_RECENT_DAYS is a ROW count, which is six calendar weeks on an
    # exchange-traded series and 30 calendar days on a 24/7 one, so a bare "30-day" label
    # was itself a row count posing as a calendar window.
    short_vol = annualized_realized_vol_pct(
        close, window=FINANCIAL_YFINANCE_RECENT_DAYS, periods_per_year=periods_per_year
    )
    if short_vol is None:
        return []
    short_line = f"- {FINANCIAL_YFINANCE_RECENT_DAYS}-{unit} annualized volatility: {short_vol:.1f}%"

    # The long horizon: one year of returns, or everything the fetch actually holds when
    # that is less. Capped at a year so the label stays a period a forecaster can reason
    # about, and skipped when it would not clear the short window (two windows of the same
    # length are one number printed twice).
    long_window = min(len(close) - 1, periods_per_year)
    long_vol = (
        annualized_realized_vol_pct(close, window=long_window, periods_per_year=periods_per_year)
        if long_window > FINANCIAL_YFINANCE_RECENT_DAYS
        else None
    )
    long_line = None if long_vol is None else f"- {long_window}-{unit} annualized volatility: {long_vol:.1f}%"

    screen = screen_for_quote_noise(close, periods_per_year=periods_per_year)
    if screen is None:
        return [short_line] if long_line is None else [short_line, long_line]

    robust_vol = screen.robust_vol_pct
    flagged = [
        f"- ⚠ Vendor-noise flag: variance ratio VR({FINANCIAL_VARIANCE_RATIO_LAG}) = {screen.ratio:.2f} over "
        f"{len(close) - 1} {unit} returns. A random walk reads ~1.0; below "
        f"{FINANCIAL_VARIANCE_RATIO_FLOOR:.2f} most of each day's move is reversed the next, which is quote "
        "noise on an illiquid or fixed cross rather than genuine price movement, and it inflates every "
        "volatility computed from one-day returns."
    ]
    if robust_vol is not None:
        flagged.append(
            f"- Noise-robust annualized volatility, from overlapping {FINANCIAL_VARIANCE_RATIO_LAG}-"
            f"{unit} returns (the horizon over which that reversing component cancels): "
            f"{robust_vol:.1f}% — size intervals from THIS figure, not the one-day-return ones below."
        )
    if long_line is not None:
        flagged.append(f"{long_line} (from one-day returns, noise included)")
    flagged.append(f"{short_line} (from one-day returns, noise included; noise-suspect)")
    logger.info(
        noise_flag_line(screen, surface="financial_data", symbol=symbol, short_vol=short_vol, long_vol=long_vol)
    )
    return flagged


def _yfinance_stats_lines(close: pd.Series, periods_per_year: int, *, symbol: str) -> list[str]:
    """Period returns, annualized volatility, and the 52-week range."""
    parts: list[str] = []
    # Period returns
    returns_section = _compute_period_returns(close, periods_per_year)
    if returns_section:
        parts.append(returns_section)

    parts.extend(_volatility_lines(close, periods_per_year, symbol=symbol))

    # 52-week range, windowed by DATE like the period returns: a row-count slice
    # under a fixed "52-week" label spans ~13 months on a gapped 24/7 series and
    # over a year on a holiday-bearing exchange index. Never empty — the last
    # observation is always inside its own trailing year.
    year_slice = close[close.index >= close.index[-1] - pd.Timedelta(days=CALENDAR_DAYS_PER_YEAR)]
    low_52w = year_slice.min()
    high_52w = year_slice.max()
    parts.append(f"- 52-week range: {low_52w:.2f} - {high_52w:.2f}")
    return parts


def _yfinance_fundamentals_lines(close: pd.Series, info: dict, *, is_benchmarking: bool) -> list[str]:
    """The .info fundamentals block (live only) and the last five closes."""
    parts: list[str] = []
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
    return parts


def _fetch_yfinance_data(ticker: str, *, as_of: datetime | None = None, is_benchmarking: bool = False) -> str:
    """One ticker's markdown block, plus the peg anchor's block when the pair is pegged.

    Sync function -- caller wraps in asyncio.to_thread(). Returns "" when the ticker's own
    fetch fails, exactly as before; a peg anchor that fails to fetch degrades to a visible
    one-line notice rather than taking the pegged pair's block down with it.

    The anchor is rendered BESIDE the pegged pair, never substituted for it: the question
    resolves on the pegged cross, so its own quote has to stay on the page. Only the
    interpretation changes, and the peg disclosure inside the block says so.
    """
    block = _render_yfinance_block(ticker, as_of=as_of, is_benchmarking=is_benchmarking)
    if not block:
        return ""
    peg = _peg_for_ticker(ticker)
    if peg is None or peg.anchor_ticker is None:
        return block
    # Recursion is bounded at one level by construction: no anchor ticker is itself a key of
    # HARD_PEG_ANCHORS (asserted in tests), so the anchor's own render finds no peg.
    anchor_block = _render_yfinance_block(peg.anchor_ticker, as_of=as_of, is_benchmarking=is_benchmarking)
    if not anchor_block:
        logger.warning(f"peg anchor fetch returned nothing for {peg.anchor_ticker=} (pegged {ticker=})")
        return (
            f"{block}\n- ⚠ The peg anchor `{peg.anchor_ticker}` could not be fetched, so no clean read of "
            "this pair's dynamics is available in this block."
        )
    return (
        f"{block}\n\n_Peg anchor for {ticker} — the liquid cross the peg is fixed to. Its volatility and "
        f"percent moves are the honest read of USD/{peg.currency}'s; its price LEVELS are a different "
        f"quantity unless the peg is at par._\n{anchor_block}"
    )


def _render_yfinance_block(ticker: str, *, as_of: datetime | None = None, is_benchmarking: bool = False) -> str:
    """Fetch price data and key metrics for a single ticker via yfinance.

    Returns formatted markdown or "" on any failure.

    Both paths fetch by explicit calendar start date, ``as_of`` -
    ``FINANCIAL_YFINANCE_LOOKBACK_DAYS`` (``as_of`` defaults to now; the live provider
    passes now explicitly). A bare ``period="Nd"`` is deliberately avoided: Yahoo's
    chart API reads that custom range as N trading BARS for listed assets but ~N
    calendar DATES for 24/7 ones — one integer under two unit systems. Live
    (``is_benchmarking=False``) leaves ``end`` unset (yfinance defaults it to now, so
    today's partial bar stays included) and reads the live ``.info`` fundamentals.
    Backtest (``is_benchmarking=True``) ceilings the history at ``as_of`` via ``end``
    and SKIPs the ``.info`` call entirely — ``.info`` has no historical mode, so its
    market cap / P/E / current price would leak TODAY's values into a question
    resolved months ago. The latest price then comes from the last ceilinged close,
    and every derived stat (returns / vol / 52wk) computes off that frame.
    """
    try:
        ticker_obj = yfinance.Ticker(ticker)
        if is_benchmarking:
            assert as_of is not None, "benchmarking yfinance fetch requires as_of"
        window_end = as_of if as_of is not None else datetime.now(UTC)
        history = _yfinance_history(ticker_obj, window_end, is_benchmarking=is_benchmarking)

        if history.empty:
            logger.warning(f"yfinance returned empty history for {ticker=}")
            return ""

        # Skip the live `.info` call under benchmarking (no historical mode → leakage).
        info: dict = {} if is_benchmarking else (ticker_obj.info or {})
        # Yahoo can serve a bar whose close is null (a consolidation hole); yfinance's
        # keepna=False default usually deletes the row, but a NaN that does arrive would
        # poison every tail-anchored stat below, so drop them here too.
        close = history["Close"].dropna()
        if close.empty:
            logger.warning(f"yfinance returned no non-null closes for {ticker=}")
            return ""
        # yfinance serves listed-asset bars on a tz-aware exchange-local index while
        # window_end is UTC; the two dates disagree for part of every day, so age and
        # the partial-bar check compare in the index's own timezone (a 00:03 UTC cron
        # would otherwise inflate every US-equity age by one).
        index_tz = cast(pd.DatetimeIndex, close.index).tz
        reference_date = window_end.astimezone(index_tz).date() if index_tz is not None else window_end.date()

        periods_per_year = observed_periods_per_year(close.index)
        parts = _yfinance_latest_lines(
            ticker,
            close,
            reference_date,
            name=info.get("shortName", ""),
            periods_per_year=periods_per_year,
            is_benchmarking=is_benchmarking,
        )
        # Before the stats, because it changes how every one of them reads.
        peg = _peg_for_ticker(ticker)
        if peg is not None:
            parts.extend(_peg_disclosure_lines(ticker, peg))
        parts.extend(_yfinance_stats_lines(close, periods_per_year, symbol=ticker))
        parts.extend(_yfinance_fundamentals_lines(close, info, is_benchmarking=is_benchmarking))
        return "\n".join(parts)

    except Exception:  # HARNESS-SCAN-EXEMPT-broad-except  # provider soft-fail boundary, logged
        logger.warning(f"yfinance fetch failed for {ticker=}", exc_info=True)
        return ""


# Annualization bases for a daily-bar series: exchange-traded assets print ~252
# rows/year (5 trading days a week), 24/7 markets (crypto) print ~365. Annualizing
# a 24/7 series with sqrt(252) understates vol by ~17% (sqrt(252/365) ~= 0.83), and
# row-count windows labeled "1y"/"52-week" then span only ~8.2 calendar months —
# both bit q44882 (ETH-USD). The two constants AND the observed-density read that picks
# between them are the package's shared definitions, imported from ts_estimators above, so a
# correction cannot miss a copy — the ts-anchor stack had exactly that miss until 2026-08-25.

# Calendar days back behind each period-return label, on BOTH bases. Each count is
# resolved to a target DATE and matched at-or-before — never used as a row offset,
# because a gapped index (Yahoo's dropped null-close bar, an unscheduled closure)
# shifts every row offset by the hole count, which is how a "1d" return spanning 53
# hours once rendered on a BTC-USD snapshot. Calendar days rather than business days
# deliberately: `pd.offsets.BDay` skips only weekends and knows nothing about market
# HOLIDAYS, so a business-day 1y target on a NYSE-shaped index landed ~352 calendar
# days back under a label claiming a year, at slip 0 — silently. With calendar
# targets, every label is a true calendar period on both bases, and the weekend or
# holiday under a 252-basis target is absorbed by the at-or-before match within the
# slip grace below.
_PERIOD_TARGET_DAYS: list[tuple[str, int]] = [
    ("1d", 1),
    ("1w", 7),
    ("1m", 30),
    ("3m", 91),
    ("6m", 182),
    ("1y", CALENDAR_DAYS_PER_YEAR),
]

# Calendar days a period-return match may land before its target date before the label
# discloses the actual span. On the 252 basis a target can fall on a weekend day with a
# market holiday under the adjacent Friday — a 3-day slip that is routine, not a
# mislabel. On the 365 basis every date should print a bar, so ANY slip is a data gap
# the label must own up to.
_PERIOD_SLIP_GRACE_DAYS: dict[int, int] = {TRADING_DAYS_PER_YEAR: 3, CALENDAR_DAYS_PER_YEAR: 0}


def _compute_period_returns(close: pd.Series, periods_per_year: int) -> str:
    """Compute returns over standard periods via date-based at-or-before lookups.

    ``periods_per_year`` is REQUIRED (no trading-day default): it selects the slip
    grace that separates routine weekend/holiday slippage from a data gap, so a caller
    that forgets it would silently reproduce the q44882 mislabelling on a 24/7 series.
    Each label's start value is the newest observation at or before ``last - days``
    (the same pattern as the FRED YoY lookup below); a label whose match lands beyond
    the basis's slip grace discloses the actual span ("1d (actual 2d)") instead of
    wearing a period it doesn't cover. A label with no observation at or before its
    target is omitted, exactly as the old too-few-rows guard omitted it.
    """
    last_ts = cast(pd.Timestamp, close.index[-1])
    end_price = close.iloc[-1]
    lines = []
    for label, days in _PERIOD_TARGET_DAYS:
        target = last_ts - pd.Timedelta(days=days)
        prior = close.loc[:target]
        if prior.empty:
            continue
        start_ts = cast(pd.Timestamp, prior.index[-1])
        pct_change = (end_price / prior.iloc[-1] - 1) * 100
        slipped = (target - start_ts).days > _PERIOD_SLIP_GRACE_DAYS[periods_per_year]
        span = f" (actual {(last_ts - start_ts).days}d)" if slipped else ""
        lines.append(f"  - {label}{span}: {pct_change:+.2f}%")
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


def _pct_clause(change: float, base: float) -> str:
    """`` (+1.23%)`` for a percent change off ``base``, or "" when there is no percent.

    A base of exactly 0 has NO percent change — it is undefined, not 0.00%. FRED spread
    and rate-difference series (T10Y2Y, T10Y3M) cross zero routinely, and the old
    ``else 0`` rendered ``+0.31 (+0.00%)`` there: a fabricated "unchanged" reading sitting
    beside a genuine absolute move, in a forecaster prompt. Omitting the clause leaves the
    absolute change, which is the part that was actually measured.
    """
    base_value = float(base)
    if base_value == 0:
        return ""
    return f" ({(change / abs(base_value)) * 100:+.2f}%)"


# Decimals kept on a rendered FRED level or change. Six covers everything FRED publishes
# (index levels at three, most rates at two, a few series at four) and trailing zeros are
# stripped, so a rate still renders "4.2". Replaces `:.4g`, which rounded a Case-Shiller
# level of 331.893 to "331.9" on q44944 — a question whose displayed range was four index
# points wide with 0.02-point buckets, so the digits `:.4g` threw away were the whole
# forecast. `:.4g` also flipped to scientific notation on large series (WALCL's ~6.7e6
# rendered as "6.7e+06"), which this format never does.
_FRED_VALUE_DECIMALS = 6


def _format_fred_value(value: float) -> str:
    """A FRED level at its published precision: fixed-point, no scientific notation.

    Also cleans up float subtraction artifacts for free — a change computed as
    331.893 - 331.02 = 0.8729999999999905 renders "0.873".
    """
    text = f"{float(value):.{_FRED_VALUE_DECIMALS}f}".rstrip("0").rstrip(".")
    return text if text not in {"", "-", "-0"} else "0"


def _format_fred_change(change: float) -> str:
    """A signed change at the same precision (``:+`` cannot drive a custom formatter).

    A change that rounds to zero at this precision renders "+0", never "-0": the sign of a
    quantity too small to display is not information.
    """
    magnitude = _format_fred_value(abs(float(change)))
    sign = "-" if float(change) < 0 and magnitude != "0" else "+"
    return f"{sign}{magnitude}"


def _first_release_lines(data: pd.Series, first_releases: pd.Series) -> list[str]:
    """The first-release-versus-current-vintage table for the most recent prints.

    A question on a revising FRED series resolves on the value the agency PUBLISHES, i.e.
    the first release, while every level rendered above it is today's revised vintage —
    q44944 resolved on a first-release Case-Shiller print, and a revision-adjusted anchor
    was worth +66.6 spot peer there. The gap between the two is a signed, measurable
    quantity, so it is rendered rather than left as a symmetric-noise assumption.

    Empty list when the two series share no dated observation, or when the LATEST print we
    render a level for has no first release in hand: a table of older prints under a
    "recent prints" label would be a different claim than the one being made.
    """
    paired = pd.concat([data.rename("current"), first_releases.rename("first")], axis=1, join="inner").dropna()
    if paired.empty or data.index[-1] not in paired.index:
        return []
    recent = paired.tail(FINANCIAL_FRED_VINTAGE_PRINTS)
    dates = pd.DatetimeIndex(recent.index).strftime("%Y-%m-%d")
    current_values = recent["current"].to_numpy(dtype="float64")
    first_values = recent["first"].to_numpy(dtype="float64")
    revisions = current_values - first_values
    rows = [
        f"  - {obs_date}: first release {_format_fred_value(first)} → current vintage "
        f"{_format_fred_value(current)} "
        + ("(unrevised)" if _format_fred_change(revision) == "+0" else f"(revised {_format_fred_change(revision)})")
        for obs_date, first, current, revision in zip(dates, first_values, current_values, revisions, strict=True)
    ]
    revised_up = int((revisions > 0).sum())
    revised_down = int((revisions < 0).sum())
    unchanged = len(revisions) - revised_up - revised_down
    direction = (
        f"  - Of these {len(revisions)} prints, {revised_up} were revised up, {revised_down} down and "
        f"{unchanged} not at all; mean revision {_format_fred_change(float(revisions.mean()))}."
    )
    return [
        "- First release vs current vintage (this series revises: a question resolving on the published print "
        "resolves on the FIRST release, while every level above is today's revised vintage):",
        "\n".join([*rows, direction]),
        # Mandatory guard from the q44944 dossier's "two levers still on the table" table:
        # each lever alone roughly doubled the score, stacking them overshot by 0.7 index
        # points and lost 15 spot-peer.
        "- ⚠ Do not double-count: adjusting for the revision direction and leaning on a same-source leading "
        "indicator (e.g. ICE HPI for Case-Shiller) partly measure the SAME underlying data. Apply one of them, "
        "not both.",
    ]


def _render_fred_series(series_id: str, data: pd.Series, title: str, *, first_releases: pd.Series | None = None) -> str:
    """Render the derived-stat markdown block for a FRED series.

    Shared by the live (fredapi) and benchmarking (keyless ts_fetch) paths so the two
    render identically — latest/previous value, MoM + YoY change, last 6 observations, and
    the first-release table when the caller could fetch one. ``data`` must already be
    dropna'd and sorted ascending by date.
    """
    parts = [f"### {series_id} ({title})"]

    latest_value = data.iloc[-1]
    latest_date = data.index[-1]
    parts.append(f"- Latest value: {_format_fred_value(latest_value)} ({latest_date.strftime('%Y-%m-%d')})")

    if len(data) >= 2:
        previous_value = data.iloc[-2]
        parts.append(f"- Previous value: {_format_fred_value(previous_value)}")

    # Change from the previous OBSERVATION — a row step, whatever the series'
    # cadence (monthly CPI, quarterly GDP, weekly claims) — which is exactly what
    # the rendered "Change from previous" label claims and no more.
    if len(data) >= 2:
        mom_change = latest_value - data.iloc[-2]
        parts.append(
            f"- Change from previous: {_format_fred_change(mom_change)}{_pct_clause(mom_change, data.iloc[-2])}"
        )

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
        parts.append(f"- Year-over-year change: {_format_fred_change(yoy_change)}{_pct_clause(yoy_change, yoy_value)}")

    # Last 6 observations
    last_6 = data.tail(6)
    obs_lines = [
        f"  - {cast(pd.Timestamp, date).strftime('%Y-%m-%d')}: {_format_fred_value(val)}"
        for date, val in last_6.items()
    ]
    parts.append("- Recent observations:\n" + "\n".join(obs_lines))

    if first_releases is not None:
        parts.extend(_first_release_lines(data, first_releases))

    return "\n".join(parts)


def _fetch_fred_first_releases(fred: Fred, series_id: str, observation_start: pd.Timestamp) -> pd.Series | None:
    """First-published value per observation since ``observation_start``, or None.

    ``output_type=4`` is ALFRED's "Observations, Initial Release Only" format: one row per
    observation carrying the value first released for it, revisions omitted (ALFRED's
    download-data help, read 2026-09-01: "This output format only contains the first
    released values for each observation ... the realtime_start_date column contains the
    dates when initial values were released to the public"). One row per observation is what
    makes this the safe request shape — ``fredapi.get_series_first_release`` instead pulls
    EVERY revision of every observation and takes the first per date, which on a series that
    restates its whole history each month (Case-Shiller's seasonal factors) is six figures
    of rows against FRED's 100k response cap, silently truncated at the oldest end.

    The real-time window is opened to FRED's full range because both bounds default to
    TODAY, and a real-time period of today would restrict the answer to values whose
    real-time period still contains today — i.e. only the prints that were never revised,
    which are exactly the ones with nothing to report. ``_first_release_lines`` re-checks
    that the latest rendered observation is present, so if that reading of the parameter
    interaction is ever wrong the table is dropped rather than rendered stale.

    ``None`` on any FRED error, so the series' primary block still renders — this table is
    enrichment and must never be able to take the source itself down. The three ways this
    call can fail are ``ValueError`` (``fredapi`` re-raises the API's own error message that
    way), ``OSError`` (``URLError``/``HTTPError`` transport failures), and ``ParseError``
    (``fredapi`` runs ``ET.fromstring`` over the response body, including an error body,
    which is not XML if a proxy or status page answers instead).
    """
    try:
        first_releases = fred.get_series(
            series_id,
            observation_start=observation_start,
            output_type=4,
            realtime_start=Fred.earliest_realtime_start,
            realtime_end=Fred.latest_realtime_end,
        )
    except (ValueError, OSError, ParseError):
        logger.warning(f"FRED first-release (ALFRED vintage) fetch failed for {series_id=}", exc_info=True)
        return None
    cleaned = first_releases.dropna()
    if cleaned.empty:
        logger.info(f"FRED first-release fetch returned no observations for {series_id=}")
        return None
    return cleaned.sort_index()


def _fetch_fred_data(series_id: str, api_key: str, *, is_resolving_source: bool = False) -> str:
    """Fetch economic data for a single FRED series (live path, fredapi).

    Sync function -- caller wraps in asyncio.to_thread().
    Returns formatted markdown or "" on any failure.

    ``is_resolving_source`` marks a series the QUESTION resolves on (URL-extracted from the
    resolution criteria, not merely named by the classifier). Those get the extra
    first-release/vintage fetch, since the revision channel only matters for the series the
    question grades against, and every extra identifier would otherwise cost another HTTP
    round trip inside this thread.
    """
    try:
        fred = Fred(api_key=api_key)
        data = fred.get_series(series_id)

        if data.empty:
            logger.warning(f"FRED returned empty data for {series_id=}")
            return ""

        # Sorted here, not just dropna'd, because everything downstream reads position as
        # date order: `iloc[-1]`/`iloc[-2]` as latest/previous, the YoY label slice (which
        # RAISES on a non-monotonic DatetimeIndex and would soft-fail the whole block), and
        # the first-release table's `tail()` — `pd.concat(join="inner")` takes the LEFT
        # operand's order, so sorting the vintage series alone establishes nothing.
        data = data.dropna().sort_index()
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
        except Exception:  # HARNESS-SCAN-EXEMPT-broad-except  # cosmetic title lookup, logged
            logger.debug(f"FRED series title lookup failed for {series_id=}", exc_info=True)

        first_releases = None
        # Series on the non-revising allowlist cannot revise, so a first-release table there
        # would be a column of zeros dressed as a finding.
        if is_resolving_source and series_id.upper() not in FRED_NON_REVISING_SERIES and len(data) >= 2:
            # Bound the request to the prints the table renders, taken off the dates we
            # already hold — cadence-agnostic, so it is the last four months on CPI and the
            # last four quarters on GDP without either being hardcoded.
            observation_start = cast(pd.Timestamp, data.index[-min(FINANCIAL_FRED_VINTAGE_PRINTS, len(data))])
            first_releases = _fetch_fred_first_releases(fred, series_id, observation_start)

        return _render_fred_series(series_id, data, title, first_releases=first_releases)

    except Exception:  # HARNESS-SCAN-EXEMPT-broad-except  # provider soft-fail boundary, logged
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

    No first-release/vintage table here, so a backtest cannot measure that feature (the same
    limitation prediction_market and resolution_source carry). The keyless alfredgraph CSV
    serves exactly one thing — the series AS OF a given vintage date — so pinning each
    print's FIRST release would need one fetch per print at release dates this path does not
    know; the real-time query that answers it in one request needs FRED_API_KEY, which this
    path exists to do without.
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


def _resolve_fetch_as_of(question: MetaculusQuestion, *, is_benchmarking: bool) -> datetime | None:
    """The window ceiling for every fetch, or None when benchmarking cannot be made leakage-safe.

    Ceiling every fetch to open_time under benchmarking (leakage-safe, mirrors
    timeseries_anchor). A missing open_time can't be ceilinged, so we bail rather
    than risk fetching today's data into a resolved question.
    """
    if not is_benchmarking:
        return datetime.now(UTC)
    open_time = getattr(question, "open_time", None)
    if not isinstance(open_time, datetime):
        logger.warning(
            "financial_data: is_benchmarking but qid=%s has no open_time; skipping (leakage-safe)",
            getattr(question, "id_of_question", None),
        )
        return None
    return open_time


def _build_financial_fetch_jobs(
    tickers: list[str],
    fred_series: list[str],
    *,
    as_of: datetime,
    is_benchmarking: bool,
    resolving_fred_series: frozenset[str] = frozenset(),
) -> list[tuple[str, asyncio.Task]]:
    """Spawn one fetch task per identifier, each paired with the ticker/FRED id it fetches.

    ``resolving_fred_series`` holds the UPPER-CASED ids the resolution criteria cited by URL
    — the series a question actually grades against — which is what earns the extra
    first-release/vintage fetch on the live path.
    """
    jobs: list[tuple[str, asyncio.Task]] = [
        (
            ticker,
            asyncio.ensure_future(
                asyncio.to_thread(_fetch_yfinance_data, ticker, as_of=as_of, is_benchmarking=is_benchmarking)
            ),
        )
        for ticker in tickers
    ]
    if is_benchmarking:
        # Keyless ceilinged path — no FRED_API_KEY needed, works in CI, and returns
        # point-in-time vintages instead of today's revisions.
        jobs.extend(
            (series_id, asyncio.ensure_future(asyncio.to_thread(_fetch_fred_data_ceiling, series_id, as_of)))
            for series_id in fred_series
        )
        return jobs
    fred_api_key = os.getenv(FRED_API_KEY_ENV)
    if fred_api_key:
        jobs.extend(
            (
                series_id,
                asyncio.ensure_future(
                    asyncio.to_thread(
                        _fetch_fred_data,
                        series_id,
                        fred_api_key,
                        is_resolving_source=series_id.upper() in resolving_fred_series,
                    )
                ),
            )
            for series_id in fred_series
        )
    elif fred_series:
        logger.info(f"FRED_API_KEY not set, skipping {len(fred_series)} FRED series fetches")
    return jobs


async def _gather_financial_results(jobs: list[tuple[str, asyncio.Task]]) -> tuple[list[str], dict[str, str]]:
    """Await every fetch, returning the non-empty sections plus a per-identifier outcome map.

    Per-identifier outcome for the diagnostics block: a requested ticker/FRED
    series that errored or returned no data is a lost source, so a partial
    financial fetch stays visible even when other identifiers succeed.
    """
    results = await asyncio.gather(*(task for _, task in jobs), return_exceptions=True)
    non_empty_results: list[str] = []
    sources: dict[str, str] = {}
    for (identifier, _), result in zip(jobs, results, strict=True):
        if isinstance(result, Exception):
            logger.warning(f"Financial data fetch task failed: {result}")
            sources[identifier] = "error"
            continue
        if isinstance(result, str) and result.strip():
            non_empty_results.append(result)
            sources[identifier] = "ok"
        else:
            sources[identifier] = "empty"
    return non_empty_results, sources


def financial_data_provider(is_benchmarking: bool = False) -> ResearchCallable:
    """Factory function returning an async research callable for financial/economic data.

    The callable:
    1. Classifies whether the question involves financial data (via cheap LLM).
    2. Fetches relevant data from yfinance and/or FRED in parallel.
    3. Combines results into structured markdown.

    FRED gracefully degrades if FRED_API_KEY is not set.

    Backtest-safe like ``timeseries_anchor``: under ``is_benchmarking`` every fetch is
    ceilinged to ``question.open_time`` (NOT the resolution time — post-open data can
    contain the resolution). Both modes fetch yfinance by explicit start date
    (``as_of`` - ``FINANCIAL_YFINANCE_LOOKBACK_DAYS``); benchmarking additionally
    ceilings the window via ``end`` and skips the leaky ``.info`` fundamentals. FRED:
    live uses the fredapi path, while benchmarking routes through the keyless
    ``ts_fetch`` fredgraph / ALFRED-vintage path so revised macro series return the
    vintage known at forecast time rather than today's revisions.
    """
    classifier_llm = build_llm_with_openrouter_fallback(
        model=FINANCIAL_CLASSIFIER_MODEL,
        role="financial_classifier",
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
        as_of = _resolve_fetch_as_of(question, is_benchmarking=is_benchmarking)
        if as_of is None:
            return ""

        extracted = extract_financial_identifiers_from_criteria(
            f"{question.resolution_criteria or ''}\n{question.fine_print or ''}"
        )

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
        tickers, fred_series = _cap_identifiers(tickers, fred_series, extracted)

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

        jobs = _build_financial_fetch_jobs(
            tickers,
            fred_series,
            as_of=as_of,
            is_benchmarking=is_benchmarking,
            resolving_fred_series=frozenset(sid.upper() for sid in extracted["fred_series"]),
        )
        if not jobs:
            return ""

        non_empty_results, sources = await _gather_financial_results(jobs)
        record_provider_detail(qid, "financial_data", {"sources": sources})

        if not non_empty_results:
            return ""

        return "\n\n".join(non_empty_results) + _build_routing_marker(fred_series, tickers, extracted, unknown)

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
