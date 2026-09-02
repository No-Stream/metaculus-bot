"""Opt-in integration tests for the financial data provider.

These hit real APIs (yfinance via Yahoo Finance, FRED via St. Louis Fed).
Skipped in CI by default; run locally with `RUN_INTEGRATION_TESTS=1 pytest
tests/test_financial_data_integration.py` when validating after a schema
drift or initial implementation.

Each test exercises a stable, high-signal symbol (AAPL for yfinance, UNRATE
for FRED) and asserts that the produced markdown contains the load-bearing
fields. The provider's underlying fetchers (`financial_data._fetch_yfinance_data`,
`fred_rendering._fetch_fred_data`) currently swallow `Exception` and return `""` on failure
-- so an empty string is the soft-fail signal, and any non-empty result
must contain the documented sections. We use stronger asserts than "doesn't
crash" precisely because the provider is permissive.

FRED tests skip if `FRED_API_KEY` is not set in the environment.
yfinance has no auth requirement. Both APIs are free — FRED's key is free to obtain
and its calls are unmetered — so these are outside the repo's cost gate.

`allow_network` is REQUIRED alongside the env gate. The autouse `_block_network_egress`
guard in `tests/conftest.py` blocks every non-localhost connect for a test carrying
neither `allow_network` nor `live`, so without the marker these died on a blocked socket
even with `RUN_INTEGRATION_TESTS=1` set and the whole file was unrunnable. Same defect
found and fixed in `test_prediction_market_integration.py` on 2026-08-03.
"""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.allow_network]

# Spelled out so a run log states WHY this is off rather than just that it is: both
# upstreams are free (yfinance needs no auth; a FRED key is free and its calls are
# unmetered), so the gate keeps a network round-trip out of the dev loop and out of
# CI's failure surface, NOT cost or credentials. Markdown-shape coverage over stubbed
# responses runs unconditionally in tests/test_financial_data_provider.py, so what
# this file uniquely adds is "the live API still answers this shape TODAY".
_SKIP_REASON = (
    "opt-in live-API check: set RUN_INTEGRATION_TESTS=1 to enable. Free (yfinance needs "
    "no auth, FRED keys are free and unmetered) but network-dependent, so it is off by "
    "default; provider parsing coverage runs unconditionally in "
    "tests/test_financial_data_provider.py."
)


# ---------------------------------------------------------------------------
# yfinance (no auth required)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not os.getenv("RUN_INTEGRATION_TESTS"), reason=_SKIP_REASON)
def test_yfinance_real_fetch_returns_parseable_markdown():
    """yfinance returns markdown with all standard sections for AAPL."""
    from metaculus_bot.research.financial_data import _fetch_yfinance_data

    md = _fetch_yfinance_data("AAPL")

    if not md:
        pytest.skip("yfinance returned empty (transient: rate-limit, network, or symbol-not-found)")

    assert md.startswith("### AAPL"), f"Expected ### AAPL header, got: {md[:80]!r}"
    assert "Latest price:" in md, "Missing dated 'Latest price' line"
    assert "(as of " in md, "Latest price must carry its observation date"
    assert "52-week range:" in md, "Missing '52-week range' line"
    assert "Last 5 closes:" in md, "Missing 'Last 5 closes' section"
    assert "Period returns:" in md, "Missing 'Period returns' section"
    # Volatility line is conditional on >=30 daily returns; AAPL always has that. AAPL is
    # exchange-traded, so the label must name TRADING days — a bare "30-day" would mean the
    # observed-density read misfired and annualized a business-day series on the 365 basis.
    assert "30-trading-day annualized volatility:" in md, "Missing volatility line"


@pytest.mark.skipif(not os.getenv("RUN_INTEGRATION_TESTS"), reason=_SKIP_REASON)
def test_yfinance_real_fetch_index_symbol():
    """yfinance handles index symbols (^GSPC) the same way it handles equities."""
    from metaculus_bot.research.financial_data import _fetch_yfinance_data

    md = _fetch_yfinance_data("^GSPC")

    if not md:
        pytest.skip("yfinance returned empty for ^GSPC (transient or upstream issue)")

    assert md.startswith("### ^GSPC"), f"Expected ### ^GSPC header, got: {md[:80]!r}"
    assert "Latest price:" in md
    assert "52-week range:" in md


@pytest.mark.skipif(not os.getenv("RUN_INTEGRATION_TESTS"), reason=_SKIP_REASON)
def test_yfinance_real_fetch_unknown_ticker_returns_empty():
    """Unknown tickers return empty string, not a crash. Soft-fail behavior."""
    from metaculus_bot.research.financial_data import _fetch_yfinance_data

    md = _fetch_yfinance_data("NOSUCHTICKER12345")

    # Empty is the soft-fail signal. Non-empty is a regression: we'd be returning
    # placeholder/garbage data for an invalid ticker.
    assert md == "", f"Expected empty markdown for unknown ticker, got: {md[:120]!r}"


# ---------------------------------------------------------------------------
# FRED (requires FRED_API_KEY)
# ---------------------------------------------------------------------------


def _fred_api_key() -> str | None:
    return os.getenv("FRED_API_KEY")


@pytest.mark.skipif(not os.getenv("RUN_INTEGRATION_TESTS"), reason=_SKIP_REASON)
@pytest.mark.skipif(not _fred_api_key(), reason="set FRED_API_KEY to enable")
def test_fred_real_fetch_returns_parseable_markdown():
    """FRED returns markdown with all standard sections for UNRATE (unemployment)."""
    from metaculus_bot.research.fred_rendering import _fetch_fred_data

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


@pytest.mark.skipif(not os.getenv("RUN_INTEGRATION_TESTS"), reason=_SKIP_REASON)
@pytest.mark.skipif(not _fred_api_key(), reason="set FRED_API_KEY to enable")
def test_fred_real_fetch_includes_series_title():
    """FRED's get_series_info path populates a human-readable title in the header."""
    from metaculus_bot.research.fred_rendering import _fetch_fred_data

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


@pytest.mark.skipif(not os.getenv("RUN_INTEGRATION_TESTS"), reason=_SKIP_REASON)
@pytest.mark.skipif(not _fred_api_key(), reason="set FRED_API_KEY to enable")
def test_fred_real_fetch_renders_the_first_release_table_for_a_resolving_series():
    """The one live check the first-release table needs, and the reason it exists.

    Everything about that table is covered offline except a single inference: that
    ``output_type=4`` with the real-time window opened to FRED's full range really does
    return each observation's INITIAL release, including for prints that have since been
    revised. ALFRED's download-data help says so in prose; only a live call proves it.
    CSUSHPISA is the right probe because it revises every month and it is the series the
    q44944 finding came from.

    If FRED's parameter semantics differ from that reading, the guard in
    ``_first_release_lines`` drops the table (the latest print would be missing from the
    response) and this test fails on the missing header rather than the bot rendering
    something stale.
    """
    from metaculus_bot.research.fred_rendering import _fetch_fred_data

    api_key = _fred_api_key()
    assert api_key is not None
    md = _fetch_fred_data("CSUSHPISA", api_key, is_resolving_source=True)

    if not md:
        pytest.skip("FRED returned empty for CSUSHPISA (transient or upstream issue)")

    assert "- First release vs current vintage" in md, (
        "no first-release table: either output_type=4 did not return revised prints' initial "
        "releases, or the latest print was absent from the response"
    )
    assert "first release" in md
    assert "current vintage" in md
    assert "⚠ Do not double-count" in md, "the double-count guard is mandatory in this block"


@pytest.mark.skipif(not os.getenv("RUN_INTEGRATION_TESTS"), reason=_SKIP_REASON)
@pytest.mark.skipif(not _fred_api_key(), reason="set FRED_API_KEY to enable")
def test_fred_real_fetch_unknown_series_returns_empty():
    """Unknown FRED series return empty string. Soft-fail behavior."""
    from metaculus_bot.research.fred_rendering import _fetch_fred_data

    api_key = _fred_api_key()
    assert api_key is not None
    md = _fetch_fred_data("NOSUCHSERIES99999", api_key)

    assert md == "", f"Expected empty markdown for unknown series, got: {md[:120]!r}"
