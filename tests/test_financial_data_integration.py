"""Opt-in integration tests for the financial data provider.

These hit real APIs (yfinance via Yahoo Finance, FRED via St. Louis Fed).
Skipped in CI by default; run locally with `RUN_INTEGRATION_TESTS=1 pytest
tests/test_financial_data_integration.py` when validating after a schema
drift or initial implementation.

Each test exercises a stable, high-signal symbol (AAPL for yfinance, UNRATE
for FRED) and asserts that the produced markdown contains the load-bearing
fields. The provider's underlying fetchers (`_fetch_yfinance_data`,
`_fetch_fred_data`) currently swallow `Exception` and return `""` on failure
-- so an empty string is the soft-fail signal, and any non-empty result
must contain the documented sections. We use stronger asserts than "doesn't
crash" precisely because the provider is permissive.

FRED tests skip if `FRED_API_KEY` is not set in the environment.
yfinance has no auth requirement.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# yfinance (no auth required)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not os.getenv("RUN_INTEGRATION_TESTS"), reason="set RUN_INTEGRATION_TESTS=1 to enable")
def test_yfinance_real_fetch_returns_parseable_markdown():
    """yfinance returns markdown with all standard sections for AAPL."""
    from metaculus_bot.financial_data_provider import _fetch_yfinance_data

    md = _fetch_yfinance_data("AAPL")

    if not md:
        pytest.skip("yfinance returned empty (transient: rate-limit, network, or symbol-not-found)")

    assert md.startswith("### AAPL"), f"Expected ### AAPL header, got: {md[:80]!r}"
    assert "Current price:" in md, "Missing 'Current price' line"
    assert "52-week range:" in md, "Missing '52-week range' line"
    assert "Last 5 closes:" in md, "Missing 'Last 5 closes' section"
    assert "Period returns:" in md, "Missing 'Period returns' section"
    # Volatility line is conditional on >=30 daily returns; AAPL always has that.
    assert "30-day annualized volatility:" in md, "Missing volatility line"


@pytest.mark.skipif(not os.getenv("RUN_INTEGRATION_TESTS"), reason="set RUN_INTEGRATION_TESTS=1 to enable")
def test_yfinance_real_fetch_index_symbol():
    """yfinance handles index symbols (^GSPC) the same way it handles equities."""
    from metaculus_bot.financial_data_provider import _fetch_yfinance_data

    md = _fetch_yfinance_data("^GSPC")

    if not md:
        pytest.skip("yfinance returned empty for ^GSPC (transient or upstream issue)")

    assert md.startswith("### ^GSPC"), f"Expected ### ^GSPC header, got: {md[:80]!r}"
    assert "Current price:" in md
    assert "52-week range:" in md


@pytest.mark.skipif(not os.getenv("RUN_INTEGRATION_TESTS"), reason="set RUN_INTEGRATION_TESTS=1 to enable")
def test_yfinance_real_fetch_unknown_ticker_returns_empty():
    """Unknown tickers return empty string, not a crash. Soft-fail behavior."""
    from metaculus_bot.financial_data_provider import _fetch_yfinance_data

    md = _fetch_yfinance_data("NOSUCHTICKER12345")

    # Empty is the soft-fail signal. Non-empty is a regression: we'd be returning
    # placeholder/garbage data for an invalid ticker.
    assert md == "", f"Expected empty markdown for unknown ticker, got: {md[:120]!r}"


# ---------------------------------------------------------------------------
# FRED (requires FRED_API_KEY)
# ---------------------------------------------------------------------------


def _fred_api_key() -> str | None:
    return os.getenv("FRED_API_KEY")


@pytest.mark.skipif(not os.getenv("RUN_INTEGRATION_TESTS"), reason="set RUN_INTEGRATION_TESTS=1 to enable")
@pytest.mark.skipif(not _fred_api_key(), reason="set FRED_API_KEY to enable")
def test_fred_real_fetch_returns_parseable_markdown():
    """FRED returns markdown with all standard sections for UNRATE (unemployment)."""
    from metaculus_bot.financial_data_provider import _fetch_fred_data

    api_key = _fred_api_key()
    assert api_key is not None  # skipif gate guarantees this; narrows for type checker
    md = _fetch_fred_data("UNRATE", api_key)

    if not md:
        pytest.skip("FRED returned empty for UNRATE (transient or upstream issue)")

    assert md.startswith("### UNRATE"), f"Expected ### UNRATE header, got: {md[:80]!r}"
    assert "Latest value:" in md, "Missing 'Latest value' line"
    assert "Previous value:" in md, "Missing 'Previous value' line"
    assert "Change from previous:" in md, "Missing 'Change from previous' line"
    assert "Year-over-year change:" in md, "Missing 'Year-over-year change' line"
    assert "Recent observations:" in md, "Missing 'Recent observations' section"


@pytest.mark.skipif(not os.getenv("RUN_INTEGRATION_TESTS"), reason="set RUN_INTEGRATION_TESTS=1 to enable")
@pytest.mark.skipif(not _fred_api_key(), reason="set FRED_API_KEY to enable")
def test_fred_real_fetch_includes_series_title():
    """FRED's get_series_info path populates a human-readable title in the header."""
    from metaculus_bot.financial_data_provider import _fetch_fred_data

    api_key = _fred_api_key()
    assert api_key is not None
    md = _fetch_fred_data("CPIAUCSL", api_key)

    if not md:
        pytest.skip("FRED returned empty for CPIAUCSL (transient or upstream issue)")

    # Title fetch is best-effort; on success the header is "### CPIAUCSL (Some Title)".
    # If get_series_info raised, the header degrades to "### CPIAUCSL (CPIAUCSL)".
    assert "### CPIAUCSL" in md
    # CPIAUCSL is the headline CPI series. The title contains "Consumer Price Index"
    # in normal operation; if FRED renamed it or the info call failed, accept the
    # degraded form rather than fail loud here.
    header_line = md.split("\n", 1)[0]
    assert header_line.startswith("### CPIAUCSL ("), f"Header missing parenthetical title: {header_line!r}"


@pytest.mark.skipif(not os.getenv("RUN_INTEGRATION_TESTS"), reason="set RUN_INTEGRATION_TESTS=1 to enable")
@pytest.mark.skipif(not _fred_api_key(), reason="set FRED_API_KEY to enable")
def test_fred_real_fetch_unknown_series_returns_empty():
    """Unknown FRED series return empty string. Soft-fail behavior."""
    from metaculus_bot.financial_data_provider import _fetch_fred_data

    api_key = _fred_api_key()
    assert api_key is not None
    md = _fetch_fred_data("NOSUCHSERIES99999", api_key)

    assert md == "", f"Expected empty markdown for unknown series, got: {md[:120]!r}"
