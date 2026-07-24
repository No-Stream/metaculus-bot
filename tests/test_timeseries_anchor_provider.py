"""Tests for the time-series-anchor research provider (FRED / yfinance empirical band).

All HTTP is mocked. Two seams, matching the module's own layering:

- Fetch-path tests patch ``ts_fetch._http_get`` (the single synchronous HTTP seam that
  returns raw CSV bytes) so the real ``fetch_series`` parse + leakage-guard runs.
- Routing / render / provider tests monkeypatch ``timeseries_anchor.fetch_series`` with a
  canned synthetic series, so no network and a deterministic band.

Coverage (one behavior per test):
- Routing: FRED URL, Yahoo URL (single), two Yahoo tickers -> spread, template keyword,
  miss -> None, ambiguous URL -> None, ambiguous keyword -> None, "highest" -> High column.
- Fetch layer: fredgraph for non-revising vs alfredgraph (vintage) for revising; "." ->
  NaN dropped; malformed HTML body -> FetchError; post-ceiling row -> LeakageError; cache
  reuse (one HTTP call for a repeat key).
- Render: latest-value first line + P10/P50/P90 band line (single); both legs + band
  (spread); model_target=False withholds the band; section char budget truncates.
- Provider: disabled flag -> "" (even when routable); non-numeric question -> "";
  is_benchmarking=True uses ``question.open_time`` as the fetch ceiling (does NOT
  short-circuit like prediction_market — this provider is backtest-safe); env-flag gate is
  checked BEFORE the is_benchmarking as_of logic; malformed fetch -> "" + WARNING;
  two calls -> byte-identical output (determinism).
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from forecasting_tools import BinaryQuestion, NumericQuestion

from metaculus_bot.research import timeseries_anchor as ts
from metaculus_bot.research import ts_fetch as tf
from metaculus_bot.research.timeseries_anchor import (
    _apply_derivation,
    _band_misses_bounds,
    _build_spread_series,
    _empirical_change_band,
    _empirical_max_band,
    _n_eff,
    _realized_max_floor,
    _render_single,
    _render_spread,
    _reset_session_caches,
    _Route,
    _truncate_section,
    build_anchor_section,
    horizon_steps,
    route_question,
    timeseries_anchor_provider,
)
from metaculus_bot.research.ts_fetch import (
    ALFRED_CSV_URL,
    FRED_CSV_URL,
    FetchError,
    LeakageError,
    SeriesSpec,
    fetch_series,
)


# Test isolation: the provider keeps a rendered-section cache and the fetch layer
# keeps a parsed-series cache. Both bleed across tests otherwise.
@pytest.fixture(autouse=True)
def _reset_provider_caches():
    _reset_session_caches()
    yield
    _reset_session_caches()


# Fake synchronous HTTP seam (returns CSV bytes; mirrors FakeSession's dispatch).


class FakeHttp:
    """Drop-in for ``ts_fetch._http_get`` dispatching by URL prefix to the raw CSV
    bytes that prefix should return."""

    def __init__(self, handlers: dict[str, bytes]):
        self._handlers = handlers
        self.calls: list[tuple[str, dict[str, str]]] = []

    def __call__(self, url: str, params: dict[str, str]) -> bytes:
        self.calls.append((url, dict(params)))
        for prefix, body in self._handlers.items():
            if url.startswith(prefix):
                return body
        raise AssertionError(f"no handler for URL {url}")


def _csv(header_value_col: str, rows: list[tuple[str, str]]) -> bytes:
    body = f"observation_date,{header_value_col}\n" + "".join(f"{d},{v}\n" for d, v in rows)
    return body.encode("utf-8")


# Synthetic series + question factories.


def _daily_positive_series(name: str, *, seed: int = 0, end: str = "2026-06-30", years: int = 6) -> pd.Series:
    """A strictly-positive daily business-day series, deterministic per seed."""
    end_ts = pd.Timestamp(end)
    idx = pd.bdate_range(end_ts - pd.Timedelta(days=round(years * 365.25)), end_ts)
    rng = np.random.default_rng(seed)
    walk = 20.0 + np.cumsum(rng.normal(0.0, 0.3, len(idx)))
    return pd.Series(np.abs(walk) + 8.0, index=idx, name=name)


def _monthly_series(name: str, *, seed: int = 0, end: str = "2026-06-01", n: int = 96) -> pd.Series:
    """A strictly-positive monthly (month-start) series, deterministic per seed. n months
    ending at ``end`` — long enough that a small monthly horizon leaves ample overlap."""
    idx = pd.date_range(end=pd.Timestamp(end), periods=n, freq="MS")
    rng = np.random.default_rng(seed)
    walk = 200.0 + np.cumsum(rng.normal(0.0, 1.0, n))
    return pd.Series(np.abs(walk) + 50.0, index=idx, name=name)


def _make_numeric_q(
    *,
    qid: int = 7001,
    question_text: str = "What will X be?",
    resolution_criteria: str = "rc",
    fine_print: str = "",
    open_time: datetime | None = None,
    scheduled_resolution_time: datetime | None = datetime(2027, 1, 1, tzinfo=UTC),
    lower_bound: float = 0.0,
    upper_bound: float = 1000.0,
    open_lower_bound: bool = False,
    open_upper_bound: bool = False,
) -> MagicMock:
    """A ``MagicMock(spec=NumericQuestion)`` with the fields the provider reads set to
    real values (unset MagicMock attrs are truthy and would corrupt routing / isinstance,
    and the bounds backstop needs real numeric bounds). The wide default range [0, 1000]
    comfortably contains the synthetic-series bands, so the backstop is a no-op unless a
    test opts into a mismatched range."""
    q = MagicMock(spec=NumericQuestion)
    q.id_of_question = qid
    q.question_text = question_text
    q.resolution_criteria = resolution_criteria
    q.fine_print = fine_print
    q.title = question_text
    q.open_time = open_time if open_time is not None else datetime(2026, 3, 15, tzinfo=UTC)
    q.scheduled_resolution_time = scheduled_resolution_time
    q.lower_bound = lower_bound
    q.upper_bound = upper_bound
    q.open_lower_bound = open_lower_bound
    q.open_upper_bound = open_upper_bound
    q.page_url = f"https://www.metaculus.com/questions/{qid}/"
    return q


# A resolution-criteria string that routes deterministically to a non-revising FRED series.
_DGS10_RC = "Resolves per https://fred.stlouisfed.org/series/DGS10 on the resolution date."


# Routing.
class TestRouting:
    def test_routes_via_fred_url(self):
        route = route_question(_make_numeric_q(resolution_criteria=_DGS10_RC))
        assert route is not None
        assert route.kind == "single"
        assert route.spec.source == "fred"
        assert route.spec.series_id == "DGS10"
        assert route.spec.revises is False  # DGS10 is in the non-revising allowlist

    def test_routes_via_fred_url_revising_series_uses_alfred_spec(self):
        # The URL pins CPIAUCSL, but its mom_pct derivation is now gated: the question must
        # ask for the MoM quantity, else the URL route skips (see the YoY-URL test below).
        # A MoM CPI question citing the URL routes and, being a revising series, uses ALFRED.
        qt = "What will month-over-month CPI inflation be in the United States for December 2026?"
        rc = "Resolves per https://fred.stlouisfed.org/series/CPIAUCSL."
        route = route_question(_make_numeric_q(question_text=qt, resolution_criteria=rc))
        assert route is not None
        assert route.spec.series_id == "CPIAUCSL"
        assert route.derivation == "mom_pct"  # registry derivation carried through the URL route
        assert route.spec.revises is True  # CPIAUCSL revises -> ALFRED vintage fetch

    def test_url_cited_unlisted_series_defaults_to_alfred_vintage(self):
        # A revising series NOT in the (small) non-revising allowlist — INDPRO revises
        # heavily but was never enumerated. The allowlist default (revises=True) sends
        # it to ALFRED point-in-time, so a plain-fredgraph revision leak is impossible.
        rc = "Resolves per https://fred.stlouisfed.org/series/INDPRO on the resolution date."
        route = route_question(_make_numeric_q(resolution_criteria=rc))
        assert route is not None
        assert route.spec.series_id == "INDPRO"
        assert route.spec.revises is True  # unlisted -> fail-safe ALFRED vintage

    def test_url_cited_registry_series_carries_registry_derivation(self):
        # F10: a CHANGE-phrased question citing the PAYEMS FRED URL (resolves on the MoM
        # jobs-added change, scaled x1000) inherits the registry's derivation/scale/label,
        # NOT a raw unscaled level band. The mom_diff derivation is now gated on the URL
        # route too (see the level-phrased-URL test below), so the question must ask for the
        # change quantity for this to route.
        qt = "What will the change in nonfarm payroll employment be for December 2026?"
        rc = "Resolves per https://fred.stlouisfed.org/series/PAYEMS on the release date."
        route = route_question(_make_numeric_q(question_text=qt, resolution_criteria=rc))
        assert route is not None
        assert route.spec.series_id == "PAYEMS"
        assert route.derivation == "mom_diff"  # from the registry, not the 'level' default
        assert route.scale == 1000.0
        assert "payrolls" in route.label.lower()  # registry label, not the bare series_id

    def test_url_cited_registry_series_skips_when_quantity_gate_fails(self):
        # A PAYEMS FRED URL on a payroll-LEVEL question (no change language): the mom_diff
        # derivation doesn't fit the level quantity, so the URL route skips entirely rather
        # than fall back to a misleading raw-level band.
        qt = "What will total nonfarm payroll employment (in thousands of persons) be?"
        rc = "Resolves per https://fred.stlouisfed.org/series/PAYEMS on the release date."
        assert route_question(_make_numeric_q(question_text=qt, resolution_criteria=rc)) is None

    def test_url_cited_registry_series_carries_unit_scale(self):
        # BOPGTB is a plain level but unit-scaled (millions -> billions). A cited URL must
        # still carry scale=0.001 from the registry.
        rc = "Resolves per https://fred.stlouisfed.org/series/BOPGTB on the release date."
        route = route_question(_make_numeric_q(resolution_criteria=rc))
        assert route is not None
        assert route.spec.series_id == "BOPGTB"
        assert route.derivation == "level"
        assert route.scale == pytest.approx(0.001)

    def test_url_cited_unregistered_series_keeps_level_defaults(self):
        # INDPRO is not in the registry -> URL routing keeps the dataclass defaults
        # (level, scale=1.0, label==series_id): the fix only carries metadata for MATCHES.
        rc = "Resolves per https://fred.stlouisfed.org/series/INDPRO on the release date."
        route = route_question(_make_numeric_q(resolution_criteria=rc))
        assert route is not None
        assert route.spec.series_id == "INDPRO"
        assert route.derivation == "level"
        assert route.scale == 1.0
        assert route.label == "INDPRO"

    def test_routes_via_yahoo_url_single(self):
        # %5E is the URL-encoded caret for ^VIX; the extractor url-decodes before matching.
        rc = "Tracks https://finance.yahoo.com/quote/%5EVIX at close."
        route = route_question(_make_numeric_q(resolution_criteria=rc))
        assert route is not None
        assert route.kind == "single"
        assert route.spec.source == "yfinance"
        assert route.spec.series_id == "^VIX"

    def test_two_yahoo_tickers_route_to_spread(self):
        # The relative-return family (all 47 observed two-ticker questions phrase it "X's
        # returns exceed Y's"): two Yahoo tickers + relative-return wording -> spread.
        qt = "How much will CL=F's returns exceed ^GSPC's over the window?"
        rc = "return(https://finance.yahoo.com/quote/CL=F) minus return(https://finance.yahoo.com/quote/%5EGSPC)."
        route = route_question(_make_numeric_q(question_text=qt, resolution_criteria=rc))
        assert route is not None
        assert route.kind == "spread"
        assert route.spec.series_id == "CL=F"
        assert route.spec_b is not None
        assert route.spec_b.series_id == "^GSPC"

    def test_two_yahoo_tickers_relative_outperform_wording_routes_to_spread(self):
        # "outperform" is part of the relative-return keyword set -> spread.
        qt = "Will NVDA outperform AAPL over the window?"
        rc = "Compares https://finance.yahoo.com/quote/NVDA and https://finance.yahoo.com/quote/AAPL."
        route = route_question(_make_numeric_q(question_text=qt, resolution_criteria=rc))
        assert route is not None
        assert route.kind == "spread"
        assert route.spec.series_id == "NVDA"
        assert route.spec_b is not None
        assert route.spec_b.series_id == "AAPL"

    def test_two_yahoo_tickers_ratio_wording_skips(self, caplog):
        # A gold/silver RATIO question (~85x) cites two Yahoo tickers but asks for a ratio,
        # not a relative return -> the mean-zero pp band would be wrong-unit, so skip + log.
        qt = "What will the gold-to-silver price ratio be on the resolution date?"
        rc = "Ratio of https://finance.yahoo.com/quote/GC=F to https://finance.yahoo.com/quote/SI=F."
        with caplog.at_level(logging.INFO):
            assert route_question(_make_numeric_q(question_text=qt, resolution_criteria=rc)) is None
        assert any("no relative-return wording" in r.message for r in caplog.records)

    def test_two_yahoo_tickers_price_difference_wording_skips(self, caplog):
        # A price-DIFFERENCE question (dollars) cites two Yahoo tickers -> skip (wrong unit).
        qt = "What will the price of gold minus the price of silver be, in dollars?"
        rc = "https://finance.yahoo.com/quote/GC=F minus https://finance.yahoo.com/quote/SI=F."
        with caplog.at_level(logging.INFO):
            assert route_question(_make_numeric_q(question_text=qt, resolution_criteria=rc)) is None
        assert any("no relative-return wording" in r.message for r in caplog.records)

    def test_two_yahoo_tickers_single_level_wording_skips(self):
        # A single ticker's LEVEL, with a second ticker cited only as context -> the level
        # (dollars) is wrong-unit for the spread band, and there's no relret wording -> skip.
        qt = "What will the closing price of NVDA be, with AAPL cited for comparison?"
        rc = "See https://finance.yahoo.com/quote/NVDA and https://finance.yahoo.com/quote/AAPL."
        assert route_question(_make_numeric_q(question_text=qt, resolution_criteria=rc)) is None

    def test_two_yahoo_tickers_return_to_level_wording_skips(self, caplog):
        # "return TO <level>" is a LEVEL question (dollars), not a relative return: the bare
        # "return" substring used to route this to the mean-zero pp spread band (wrong-unit).
        # The word-boundary gate's "(?!\s+to\b)" lookahead excludes it -> skip + log.
        qt = "Will the price of NVDA return to $400 before AAPL does?"
        rc = "Compares https://finance.yahoo.com/quote/NVDA and https://finance.yahoo.com/quote/AAPL."
        with caplog.at_level(logging.INFO):
            assert route_question(_make_numeric_q(question_text=qt, resolution_criteria=rc)) is None
        assert any("no relative-return wording" in r.message for r in caplog.records)

    def test_two_yahoo_tickers_singular_return_phrasing_routes_to_spread(self):
        # Guards against a plural-only fix: the SINGULAR "return(URL) minus return(URL)"
        # criteria wording (no plural "returns" anywhere in the text) must still route to
        # spread -- "return(" is followed by "(", not " to", so the lookahead lets it match.
        qt = "Where will the CL=F versus ^GSPC figure land over the window?"
        rc = "return(https://finance.yahoo.com/quote/CL=F) minus return(https://finance.yahoo.com/quote/%5EGSPC)."
        route = route_question(_make_numeric_q(question_text=qt, resolution_criteria=rc))
        assert route is not None
        assert route.kind == "spread"
        assert route.spec.series_id == "CL=F"
        assert route.spec_b is not None
        assert route.spec_b.series_id == "^GSPC"

    def test_routes_via_template_keyword(self):
        route = route_question(_make_numeric_q(question_text="Where will the 10-year treasury yield close?"))
        assert route is not None
        assert route.spec.series_id == "DGS10"
        assert "10-Year Treasury" in route.label

    def test_miss_returns_none(self):
        assert route_question(_make_numeric_q(question_text="Who wins the 2028 election?")) is None

    def test_ambiguous_url_returns_none(self, caplog):
        # Two DIFFERENT fred series cited -> not a 2-ticker spread, not a single -> ambiguous.
        rc = "https://fred.stlouisfed.org/series/DGS10 and https://fred.stlouisfed.org/series/UNRATE"
        with caplog.at_level(logging.INFO):
            assert route_question(_make_numeric_q(resolution_criteria=rc)) is None
        assert any("ambiguous URL routing" in r.message for r in caplog.records)

    def test_ambiguous_keyword_returns_none(self, caplog):
        # Two ungated level entries both match (10-year treasury + high yield spread) ->
        # ambiguous -> None. (The CPI entry, which used to serve this test, is now gated on
        # MoM language, so it no longer matches a bare "cpi versus ..." string.)
        q = _make_numeric_q(question_text="the 10-year treasury yield versus the high yield spread")
        with caplog.at_level(logging.INFO):
            assert route_question(q) is None
        assert any("ambiguous keyword routing" in r.message for r in caplog.records)

    def test_highest_framing_selects_high_column(self):
        q = _make_numeric_q(question_text="What is the highest VIX value this year?")
        route = route_question(q)
        assert route is not None
        assert route.spec.column == "High"
        assert route.is_max is True

    def test_sp500_keyword_routes_to_gspc_level(self):
        route = route_question(_make_numeric_q(question_text="Where will the S&P 500 close on Dec 31?"))
        assert route is not None
        assert route.spec.source == "yfinance"
        assert route.spec.series_id == "^GSPC"
        assert route.derivation == "level"

    def test_bitcoin_highest_routes_to_btc_high_column_max(self):
        route = route_question(_make_numeric_q(question_text="What is the highest price of Bitcoin in 2025?"))
        assert route is not None
        assert route.spec.series_id == "BTC-USD"
        assert route.spec.column == "High"  # max/highest framing -> daily High
        assert route.is_max is True

    def test_silver_highest_routes_to_high_column(self):
        route = route_question(_make_numeric_q(question_text="Highest silver price per troy oz in April?"))
        assert route is not None
        assert route.spec.series_id == "SI=F"
        assert route.spec.column == "High"

    def test_gold_level_routes_to_close_column(self):
        route = route_question(_make_numeric_q(question_text="What will the price of gold be on the date?"))
        assert route is not None
        assert route.spec.series_id == "GC=F"
        assert route.spec.column == "Close"  # no max framing -> Close

    def test_case_shiller_routes_and_revises(self):
        route = route_question(
            _make_numeric_q(question_text="What will the Case-Shiller national home price index be?")
        )
        assert route is not None
        assert route.spec.series_id == "CSUSHPISA"
        assert route.spec.revises is True  # not in the non-revising allowlist -> ALFRED vintage

    def test_fed_funds_upper_routes_and_non_revising(self):
        route = route_question(
            _make_numeric_q(question_text="What will the federal funds target range upper limit be?")
        )
        assert route is not None
        assert route.spec.series_id == "DFEDTARU"
        assert route.spec.revises is True  # DFEDTARU is NOT in the non-revising allowlist

    def test_average_weekly_hours_routes_and_revises(self):
        route = route_question(_make_numeric_q(question_text="What will average weekly hours (all employees) be?"))
        assert route is not None
        assert route.spec.series_id == "AWHAETP"
        assert route.spec.revises is True

    def test_australia_unemployment_routes_to_aus_series(self):
        route = route_question(_make_numeric_q(question_text="What will the Australian unemployment rate be?"))
        assert route is not None
        assert route.spec.series_id == "LRHUTTTTAUM156S"

    def test_us_unemployment_excludes_australia_entry(self):
        # "unemployment rate" alone must route to US UNRATE, not double-match the AUS entry.
        route = route_question(_make_numeric_q(question_text="What will the US unemployment rate be?"))
        assert route is not None
        assert route.spec.series_id == "UNRATE"

    def test_payrolls_routes_to_mom_diff_with_scale(self):
        route = route_question(_make_numeric_q(question_text="How many nonfarm payrolls jobs were added?"))
        assert route is not None
        assert route.spec.series_id == "PAYEMS"
        assert route.derivation == "mom_diff"
        assert route.scale == 1000.0
        assert route.model_target is True  # no longer a level-with-caveat skip

    def test_cpi_routes_to_mom_pct(self):
        route = route_question(_make_numeric_q(question_text="What will month-over-month CPI inflation be?"))
        assert route is not None
        assert route.spec.series_id == "CPIAUCSL"
        assert route.derivation == "mom_pct"

    def test_gasoline_routes_to_monthly_avg(self):
        route = route_question(
            _make_numeric_q(question_text="What will the monthly average regular gasoline price be?")
        )
        assert route is not None
        assert route.spec.series_id == "GASREGW"
        assert route.derivation == "monthly_avg"

    def test_goods_trade_balance_routes_with_unit_scale(self):
        route = route_question(_make_numeric_q(question_text="What will the US advance goods trade balance be?"))
        assert route is not None
        assert route.spec.series_id == "BOPGTB"
        assert route.derivation == "level"
        assert route.scale == pytest.approx(0.001)  # millions of USD -> billions


# Derivation gate: a NON-"level" registry entry (CPI mom_pct, PAYEMS mom_diff, gasoline
# monthly_avg) must only fire when the question actually asks for that derived quantity.
# The overreach it fixes: an index-LEVEL, year-over-year, or foreign-country CPI question
# (real tournament questions — UK q41640, Egypt q41634) used to route to US CPIAUCSL
# mom_pct and get an empirical band in the wrong units / wrong country. Phrasings below
# mirror the real tournament questions (scratch/ts_anchor_gate_2026-07-16/ts_labeled.json).
class TestDerivationGating:
    def test_cpi_intended_mom_us_question_routes(self):
        # qid 39567/41681 — the intended recurring US MoM headline-CPI family.
        qt = (
            "What will the seasonally adjusted month over month headline CPI inflation be in the "
            "United States in the following months? (Sep-25)"
        )
        route = route_question(_make_numeric_q(question_text=qt))
        assert route is not None
        assert route.spec.series_id == "CPIAUCSL"
        assert route.derivation == "mom_pct"

    def test_cpi_index_level_question_skips(self):
        qt = "What will the CPI-U index level (1982-84=100) be for December 2026?"
        assert route_question(_make_numeric_q(question_text=qt)) is None

    def test_cpi_yoy_question_skips(self):
        qt = "What will the year-over-year CPI inflation rate be in the United States in December 2026?"
        assert route_question(_make_numeric_q(question_text=qt)) is None

    def test_uk_cpi_12_month_yoy_question_skips(self):
        # qid 41640 — routes to US CPIAUCSL mom_pct today; wrong units AND wrong country.
        qt = (
            "What will be the UK Consumer Prices Index (CPI) 12-month rate (year-over-year percent change) "
            "for March 2026, as reported by the UK Office for National Statistics (ONS)?"
        )
        assert route_question(_make_numeric_q(question_text=qt)) is None

    def test_egypt_urban_cpi_question_skips(self):
        # qid 41634 — foreign CPI; must not inherit the US MoM-% band.
        qt = (
            "What is Egypt's annual inflation rate (year-over-year percent change) for the Urban "
            "Consumer Price Index (Urban CPI) for March 2026?"
        )
        assert route_question(_make_numeric_q(question_text=qt)) is None

    def test_cpi_url_yoy_question_skips(self):
        # A YoY question citing the CPIAUCSL FRED URL: the URL route is gated too, so it
        # skips rather than hand back a US-MoM-% band for a year-over-year quantity.
        qt = "What will the year-over-year CPI inflation rate be for December 2026?"
        rc = "Resolves per https://fred.stlouisfed.org/series/CPIAUCSL on the release date."
        assert route_question(_make_numeric_q(question_text=qt, resolution_criteria=rc)) is None

    def test_payrolls_change_intended_phrasing_routes(self):
        # qid 40100/38829 — the intended recurring "change in nonfarm payroll employment" family.
        qt = "What will be the change in seasonally adjusted nonfarm payroll employment in the following months? (Dec-25)"
        route = route_question(_make_numeric_q(question_text=qt))
        assert route is not None
        assert route.spec.series_id == "PAYEMS"
        assert route.derivation == "mom_diff"
        assert route.scale == 1000.0

    def test_payrolls_level_question_skips(self):
        qt = "What will total nonfarm payroll employment (in thousands of persons) be in December 2026?"
        assert route_question(_make_numeric_q(question_text=qt)) is None

    def test_gasoline_point_in_time_question_skips(self):
        # A spot gasoline-price question resolves on a point value, not a monthly mean, so
        # the monthly_avg entry must not fire (no monthly-period language).
        qt = "What will the price of regular gasoline be on December 31, 2026?"
        assert route_question(_make_numeric_q(question_text=qt)) is None

    def test_gasoline_for_the_month_intended_routes(self):
        # qid 41795/41791/41785 — the intended "national average price ... for the month of X" family.
        qt = (
            "What will be the US national average price for regular gasoline (dollars per gallon) "
            "for the month of April 2026?"
        )
        route = route_question(_make_numeric_q(question_text=qt))
        assert route is not None
        assert route.spec.series_id == "GASREGW"
        assert route.derivation == "monthly_avg"

    def test_gasoline_national_average_spot_question_skips(self):
        # The GEOGRAPHIC "national average" descriptor is in every gasoline question; on a
        # point-in-time question it must NOT trigger monthly_avg (the bug F4 fixed — the old
        # bare-"average" gate matched here, and the ~$3/gal band can't be caught by bounds).
        qt = "What will the national average price of regular gasoline be on July 31, 2026?"
        assert route_question(_make_numeric_q(question_text=qt)) is None


# Fetch layer (real fetch_series over a faked _http_get).
class TestFetchLayer:
    def test_non_revising_hits_fredgraph_not_alfred(self, monkeypatch):
        fake = FakeHttp({FRED_CSV_URL: _csv("DGS10", [("2026-06-01", "4.20"), ("2026-06-02", "4.25")])})
        monkeypatch.setattr(tf, "_http_get", fake)

        series = fetch_series(SeriesSpec(source="fred", series_id="DGS10"), date(2026, 6, 30))

        assert len(series) == 2
        assert float(series.iloc[-1]) == pytest.approx(4.25)
        ((url, params),) = fake.calls
        assert url == FRED_CSV_URL  # fredgraph, not alfredgraph
        assert "vintage_date" not in params  # no vintage on a non-revising fetch

    def test_revising_hits_alfredgraph_with_vintage(self, monkeypatch):
        # ALFRED value column carries a vintage suffix; the parser matches by prefix.
        fake = FakeHttp({ALFRED_CSV_URL: _csv("CPIAUCSL_20260630", [("2026-05-01", "283.1"), ("2026-06-01", "283.9")])})
        monkeypatch.setattr(tf, "_http_get", fake)

        series = fetch_series(SeriesSpec(source="fred", series_id="CPIAUCSL", revises=True), date(2026, 6, 30))

        assert float(series.iloc[-1]) == pytest.approx(283.9)
        ((url, params),) = fake.calls
        assert url == ALFRED_CSV_URL
        # vintage defaults to the ceiling for a revising series with no explicit vintage.
        assert params["vintage_date"] == "2026-06-30"

    def test_missing_values_dropped(self, monkeypatch):
        rows = [("2026-06-01", "4.20"), ("2026-06-02", "."), ("2026-06-03", "4.30")]
        monkeypatch.setattr(tf, "_http_get", FakeHttp({FRED_CSV_URL: _csv("DGS10", rows)}))

        series = fetch_series(SeriesSpec(source="fred", series_id="DGS10"), date(2026, 6, 30))

        assert len(series) == 2  # the "." row is dropped, no interior NaN
        assert not series.isna().any()

    def test_malformed_html_body_raises_fetch_error(self, monkeypatch):
        monkeypatch.setattr(tf, "_http_get", FakeHttp({FRED_CSV_URL: b"<!DOCTYPE html><html>bad series id</html>"}))
        with pytest.raises(FetchError):
            fetch_series(SeriesSpec(source="fred", series_id="DGS10"), date(2026, 6, 30))

    def test_post_ceiling_row_raises_leakage_error(self, monkeypatch):
        # A row dated after the ceiling means the endpoint ignored the coed bound.
        rows = [("2026-06-01", "4.20"), ("2026-07-15", "4.30")]
        monkeypatch.setattr(tf, "_http_get", FakeHttp({FRED_CSV_URL: _csv("DGS10", rows)}))
        with pytest.raises(LeakageError):
            fetch_series(SeriesSpec(source="fred", series_id="DGS10"), date(2026, 6, 30))

    def test_cache_reuse_avoids_second_http_call(self, monkeypatch):
        fake = FakeHttp({FRED_CSV_URL: _csv("DGS10", [("2026-06-01", "4.20"), ("2026-06-02", "4.25")])})
        monkeypatch.setattr(tf, "_http_get", fake)

        first = fetch_series(SeriesSpec(source="fred", series_id="DGS10"), date(2026, 6, 30))
        second = fetch_series(SeriesSpec(source="fred", series_id="DGS10"), date(2026, 6, 30))

        assert len(fake.calls) == 1  # second call served from the in-memory cache
        pd.testing.assert_series_equal(first, second)


# yfinance fetch path (real fetch_series over a faked yfinance.Ticker).


def _yf_ohlc(dates: list[str], *, close: list[float], high: list[float]) -> pd.DataFrame:
    """Canned yfinance history frame: tz-aware DatetimeIndex + full OHLCV columns,
    mirroring what ``yfinance.Ticker(...).history()`` returns."""
    idx = pd.DatetimeIndex(pd.to_datetime(dates)).tz_localize("America/New_York")
    return pd.DataFrame(
        {"Open": close, "High": high, "Low": close, "Close": close, "Volume": [0] * len(dates)},
        index=idx,
    )


def _fake_yf_ticker(frame: pd.DataFrame) -> tuple[type, list[dict[str, str]]]:
    """Return a (Ticker-class, calls-list) pair; the class records every history() kwargs."""
    calls: list[dict[str, str]] = []

    class _Ticker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def history(self, **kwargs: str) -> pd.DataFrame:
            calls.append(kwargs)
            return frame

    return _Ticker, calls


class TestYfinanceFetch:
    def test_high_column_spec_reads_high(self, monkeypatch):
        frame = _yf_ohlc(["2026-06-29", "2026-06-30"], close=[18.0, 19.0], high=[20.0, 22.0])
        ticker, _ = _fake_yf_ticker(frame)
        monkeypatch.setattr("yfinance.Ticker", ticker)

        series = fetch_series(SeriesSpec(source="yfinance", series_id="^VIX", column="High"), date(2026, 6, 30))

        assert float(series.iloc[-1]) == pytest.approx(22.0)  # High, not Close
        assert float(series.iloc[0]) == pytest.approx(20.0)

    def test_default_spec_reads_close(self, monkeypatch):
        frame = _yf_ohlc(["2026-06-29", "2026-06-30"], close=[18.0, 19.0], high=[20.0, 22.0])
        ticker, _ = _fake_yf_ticker(frame)
        monkeypatch.setattr("yfinance.Ticker", ticker)

        series = fetch_series(SeriesSpec(source="yfinance", series_id="^VIX"), date(2026, 6, 30))

        assert float(series.iloc[-1]) == pytest.approx(19.0)  # Close (default column)

    def test_empty_frame_raises_fetch_error(self, monkeypatch):
        ticker, _ = _fake_yf_ticker(pd.DataFrame())
        monkeypatch.setattr("yfinance.Ticker", ticker)

        with pytest.raises(FetchError, match="empty history"):
            fetch_series(SeriesSpec(source="yfinance", series_id="^VIX"), date(2026, 6, 30))

    def test_missing_requested_column_raises_fetch_error(self, monkeypatch):
        # A frame with no High column, but the spec asks for High -> FetchError.
        frame = _yf_ohlc(["2026-06-30"], close=[19.0], high=[22.0]).drop(columns=["High"])
        ticker, _ = _fake_yf_ticker(frame)
        monkeypatch.setattr("yfinance.Ticker", ticker)

        with pytest.raises(FetchError, match="no 'High' column"):
            fetch_series(SeriesSpec(source="yfinance", series_id="^VIX", column="High"), date(2026, 6, 30))

    def test_ceiling_respected_end_to_end(self, monkeypatch):
        # yfinance end is EXCLUSIVE, so the fetch must request end = ceiling + 1 day,
        # and the returned series must carry no observation after the ceiling.
        frame = _yf_ohlc(["2026-06-28", "2026-06-29", "2026-06-30"], close=[17.0, 18.0, 19.0], high=[19.0, 20.0, 22.0])
        ticker, calls = _fake_yf_ticker(frame)
        monkeypatch.setattr("yfinance.Ticker", ticker)

        ceiling = date(2026, 6, 30)
        series = fetch_series(SeriesSpec(source="yfinance", series_id="^VIX"), ceiling)

        assert calls[0]["end"] == date(2026, 7, 1).isoformat()  # ceiling + 1 (exclusive end)
        assert series.index.max().date() <= ceiling  # leakage guard held on the yfinance path


# Render.
class TestRenderSingle:
    def test_latest_value_first_line_and_band_line(self):
        series = _daily_positive_series("^VIX")
        route = _Route(kind="single", spec=SeriesSpec(source="yfinance", series_id="^VIX"), label="CBOE VIX")

        out, band = _render_single(series, route=route, ceiling=date(2026, 6, 30), calendar_days=14)

        assert band is not None  # a model-target series longer than the horizon renders a band
        first_line = out.splitlines()[0]
        assert first_line.startswith("**CBOE VIX** — latest ")
        assert "as of 2026-06-30" in first_line
        assert "P10 / P50 / P90 →" in out
        assert PROVENANCE_MARKER in out
        # The band line reports both the raw overlapping-window count and the ~independent
        # count (n_obs // h). A daily 14-day horizon -> h=10 trading days.
        assert "overlapping windows" in out
        assert "independent" in out
        h = horizon_steps("daily", 14)
        assert f"~{series.size // h:,} independent" in out

    def test_note_rendered_and_band_withheld_when_not_model_target(self):
        series = _daily_positive_series("PAYEMS")
        route = _Route(
            kind="single",
            spec=SeriesSpec(source="fred", series_id="PAYEMS", revises=True),
            label="Total nonfarm payrolls",
            model_target=False,
            note="This is the payrolls LEVEL series.",
        )

        out, band = _render_single(series, route=route, ceiling=date(2026, 6, 30), calendar_days=30)

        assert band is None  # model_target=False -> no band computed, so nothing to return
        assert "- Note: This is the payrolls LEVEL series." in out
        # model_target=False -> no empirical band emitted at all.
        assert "P10 / P50 / P90" not in out
        assert "empirical band" not in out.lower()


class TestRenderSpread:
    def test_renders_both_legs_and_band(self):
        series_a = _daily_positive_series("CL=F", seed=1)
        series_b = _daily_positive_series("^GSPC", seed=2) * 40.0  # distinct level
        route = _Route(
            kind="spread",
            spec=SeriesSpec(source="yfinance", series_id="CL=F"),
            label="CL=F",
            spec_b=SeriesSpec(source="yfinance", series_id="^GSPC"),
            label_b="^GSPC",
        )

        out, band = _render_spread(series_a, series_b, route=route, calendar_days=14)

        assert band is not None  # the spread always emits a band; returned for the bounds backstop
        assert len(band) == 3  # P10/P50/P90
        assert "Relative-return spread: CL=F vs ^GSPC" in out
        assert "- CL=F latest:" in out
        assert "- ^GSPC latest:" in out
        assert "- CL=F recent:" in out
        assert "- ^GSPC recent:" in out
        assert "relative-return band" in out
        assert "P10 / P50 / P90 →" in out
        # The spread band line also reports the overlapping + ~independent window counts.
        assert "overlapping windows" in out
        assert "independent" in out
        # §g: spread sections carry an explicit mean-zero-prior disclaimer.
        assert "mean-zero by construction" in out
        assert "not a directional signal" in out


# Derived-target math: hand-confirmed reference values from the replay (Phase A).
class TestDerivedTargets:
    def test_mom_diff_scaled_first_difference(self):
        # PAYEMS-style: [100,110,105] x1000 -> diffs [10000, -5000].
        idx = pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01"])
        s = pd.Series([100.0, 110.0, 105.0], index=idx)
        out = _apply_derivation(s, "mom_diff", 1000.0)
        assert out.tolist() == pytest.approx([10000.0, -5000.0])

    def test_mom_pct_percent_change(self):
        # CPI-style: [100,110,105] -> MoM % [10.0, -4.5455].
        idx = pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01"])
        s = pd.Series([100.0, 110.0, 105.0], index=idx)
        out = _apply_derivation(s, "mom_pct", 1.0)
        assert out.tolist() == pytest.approx([10.0, -4.545454545454546])

    def test_monthly_avg_of_weekly(self):
        # Gasoline-style: weekly [3,4,5] in Jan + [15] in Feb -> {Jan 4.0, Feb 15.0}.
        idx = pd.to_datetime(["2026-01-05", "2026-01-12", "2026-01-19", "2026-02-02"])
        w = pd.Series([3.0, 4.0, 5.0, 15.0], index=idx)
        out = _apply_derivation(w, "monthly_avg", 1.0)
        assert [str(d.date()) for d in out.index] == ["2026-01-01", "2026-02-01"]
        assert out.tolist() == pytest.approx([4.0, 15.0])

    def test_level_scale_millions_to_billions(self):
        # BOPGTB-style unit conversion: millions of USD -> billions via scale=0.001.
        idx = pd.to_datetime(["2026-01-01"])
        out = _apply_derivation(pd.Series([-81800.0], index=idx), "level", 0.001)
        assert out.tolist() == pytest.approx([-81.8])

    def test_level_scale_one_is_identity(self):
        idx = pd.to_datetime(["2026-01-01", "2026-02-01"])
        s = pd.Series([4.1, 4.3], index=idx)
        out = _apply_derivation(s, "level", 1.0)
        pd.testing.assert_series_equal(out, s)


class TestRealizedMaxFloor:
    def test_floor_is_elapsed_window_max(self):
        idx = pd.date_range("2026-01-01", periods=10, freq="D")
        s = pd.Series([10.0, 12.0, 11.0, 15.0, 13.0, 9.0, 8.0, 7.0, 6.0, 5.0], index=idx)
        # Max over the elapsed portion [window_start, ceiling] = 15.0 (the fourth obs).
        floor = _realized_max_floor(s, window_start=date(2026, 1, 1), ceiling=date(2026, 1, 5))
        assert floor == pytest.approx(15.0)

    def test_no_floor_when_window_not_yet_open(self):
        # Benchmark path: window_start == ceiling -> no elapsed portion, no floor.
        idx = pd.date_range("2026-01-01", periods=5, freq="D")
        s = pd.Series([10.0, 12.0, 11.0, 15.0, 13.0], index=idx)
        assert _realized_max_floor(s, window_start=date(2026, 1, 3), ceiling=date(2026, 1, 3)) is None
        assert _realized_max_floor(s, window_start=None, ceiling=date(2026, 1, 3)) is None


class TestEffectiveWindowCount:
    """n_eff ~= n_obs // h captures that overlapping windows share observations, so
    the independent-sample count is far below the raw overlapping-window count at
    long horizons. Floored at 1 for degenerate inputs."""

    def test_long_horizon_collapses_to_few_independent_windows(self):
        # 15 years of daily data, 1-year (h=252 trading-day) horizon -> ~15 independent.
        assert _n_eff(15 * 252, 252) == 15

    def test_short_horizon_keeps_many_independent_windows(self):
        assert _n_eff(3780, 10) == 378

    def test_floored_at_one(self):
        # Fewer observations than the horizon (never happens where the band renders,
        # but the count must stay honest rather than go to zero).
        assert _n_eff(5, 10) == 1
        assert _n_eff(0, 10) == 1


class TestRenderDerived:
    def test_mom_diff_labels_derived_quantity_and_history(self):
        # A monthly level series routed through mom_diff x1000 must render the DERIVED
        # change values, not the raw level, and label them clearly.
        series = _monthly_series("PAYEMS")
        route = _Route(
            kind="single",
            spec=SeriesSpec(source="fred", series_id="PAYEMS", revises=True),
            label="Nonfarm payrolls — MoM change",
            derivation="mom_diff",
            scale=1000.0,
        )

        out, band = _render_single(series, route=route, ceiling=date(2026, 6, 1), calendar_days=30)

        assert band is not None
        assert "latest derived value" in out
        assert "month-over-month change" in out
        assert "from raw level" in out  # the raw level is still surfaced for context
        assert "(derived)" in out  # the history block is labeled as derived values
        assert "P10 / P50 / P90 →" in out
        assert "52-week range" not in out  # the level-only 52w line is skipped for derived Qs

    def test_mom_pct_renders_percent_change_band(self):
        series = _monthly_series("CPIAUCSL", seed=3)
        route = _Route(
            kind="single",
            spec=SeriesSpec(source="fred", series_id="CPIAUCSL", revises=True),
            label="CPI MoM % change",
            derivation="mom_pct",
        )

        out, band = _render_single(series, route=route, ceiling=date(2026, 6, 1), calendar_days=30)

        assert band is not None
        assert "month-over-month % change" in out
        assert "P10 / P50 / P90 →" in out

    def test_max_window_realized_floor_line_when_window_started(self):
        # A daily High series, max framing, window already open before the ceiling ->
        # the realized-max floor line appears and lifts the band.
        series = _daily_positive_series("BTC-USD", end="2026-06-30")
        route = _Route(
            kind="single",
            spec=SeriesSpec(source="yfinance", series_id="BTC-USD", column="High"),
            label="Bitcoin highest",
            is_max=True,
        )

        out, band = _render_single(
            series,
            route=route,
            ceiling=date(2026, 6, 30),
            calendar_days=30,
            window_start=date(2026, 1, 1),  # window opened months before the ceiling
        )

        assert band is not None
        assert "Realized max so far this window" in out
        assert "HARD LOWER BOUND" in out
        assert "forward-max" in out

    def test_max_window_no_floor_line_when_window_not_open(self):
        series = _daily_positive_series("^VIX", end="2026-06-30")
        route = _Route(
            kind="single",
            spec=SeriesSpec(source="yfinance", series_id="^VIX", column="High"),
            label="VIX max",
            is_max=True,
        )
        # Benchmark path: window_start == ceiling -> no elapsed portion, no floor line.
        out, band = _render_single(
            series,
            route=route,
            ceiling=date(2026, 6, 30),
            calendar_days=14,
            window_start=date(2026, 6, 30),
        )
        assert band is not None
        assert "Realized max so far this window" not in out
        assert "forward-max" in out


class TestTruncateSection:
    def test_section_char_budget_enforced(self, monkeypatch):
        monkeypatch.setattr(ts, "TS_ANCHOR_SECTION_MAX_CHARS", 120)
        text = "line\n" * 200  # ~1000 chars, well over the shrunken budget

        out = _truncate_section(text)

        assert len(out) <= 120
        assert out.endswith("[truncated — time-series anchor section budget]")

    def test_under_budget_passthrough(self):
        text = "short section"
        assert _truncate_section(text) == text


PROVENANCE_MARKER = "Statistical extrapolation of the resolution series' own history"


# Estimator math: known input -> hand-computed expected output. These pin the
# band/horizon arithmetic against silent mutations (e.g. a base/fwd swap or a
# swapped horizon constant) that a string-presence render test cannot see.
class TestEstimatorMath:
    def test_horizon_steps_matches_documented_formula(self):
        # daily: round(days * 252/365); weekly: round(days/7); monthly: round(days/30.4375).
        assert horizon_steps("daily", 30) == 21  # round(30 * 252 / 365) = round(20.7123)
        assert horizon_steps("weekly", 21) == 3  # round(21 / 7)
        assert horizon_steps("monthly", 90) == 3  # round(90 / 30.4375) = round(2.9569)

    def test_horizon_steps_floored_at_one(self):
        # round(5 / 30.4375) = round(0.164) = 0 -> floored to 1 (never a 0-step horizon).
        assert horizon_steps("monthly", 5) == 1

    def test_change_band_additive_ramp_is_exactly_h_step(self):
        # Constant-step additive ramp: every overlapping h-step change equals h*step
        # exactly, so P10=P50=P90 collapse. anchor=0 -> band returns the raw change,
        # which must be +h*step (a base/fwd swap would flip the sign to -h*step).
        step = 10.0
        y = np.arange(0, 20, dtype="float64") * step + 10.0  # 10, 20, 30, ...
        h = 3
        p10, p50, p90 = _empirical_change_band(y, h, use_log=False, anchor=0.0)
        assert p10 == pytest.approx(h * step)
        assert p50 == pytest.approx(h * step)
        assert p90 == pytest.approx(h * step)

    def test_change_band_log_branch_constant_ratio(self):
        # Constant-ratio positive series y = 100 * 1.01^t: every h-step log change is
        # exactly h*log(1.01), so the log-multiplicative band collapses to last*1.01^h.
        t = np.arange(0, 51, dtype="float64")
        ratio = 1.01
        y = 100.0 * ratio**t
        h = 3
        last = float(y[-1])
        expected = last * ratio**h
        p10, p50, p90 = _empirical_change_band(y, h, use_log=True, anchor=last)
        assert p10 == pytest.approx(expected)
        assert p50 == pytest.approx(expected)
        assert p90 == pytest.approx(expected)

    def test_max_band_hand_computed_window_max(self):
        # y=[1,3,2,5,4], h=2. Each window spans h+1=3 points (an h-step horizon, matching
        # the change band's y[i+h]-vs-y[i] span): [1,3,2],[3,2,5],[2,5,4] ->
        # window_max=[3,5,5], win_anchor=y[:3]=[1,3,2], diffs=[2,2,3].
        # sorted diffs=[2,2,3]; numpy linear quantiles at (.10,.50,.90) over n=3:
        #   .10 -> pos 0.2 -> 2.0;  .50 -> pos 1.0 -> 2.0;  .90 -> pos 1.8 -> 2.8.
        # anchor last=10 -> (12.0, 12.0, 12.8). A window_max/anchor swap flips the
        # diffs negative and this fails; a length-h (not h+1) window regresses to the
        # old [2,0,3,0] -> (10.0, 11.0, 12.7).
        y = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        p10, p50, p90 = _empirical_max_band(y, 2, use_log=False, last=10.0)
        assert p10 == pytest.approx(12.0)
        assert p50 == pytest.approx(12.0)
        assert p90 == pytest.approx(12.8)

    def test_max_band_log_branch_hand_computed(self):
        # F12: the use_log=True branch is the production path for every strictly-positive
        # financial series (VIX/BTC/gold "highest value" questions) — hand-pin it.
        # y=[1,2,1,4,2], h=2. Windows span h+1=3 points: [1,2,1],[2,1,4],[1,4,2] ->
        # window_max=[2,4,4], win_anchor=y[:3]=[1,2,1].
        # log-ratios = [ln(2/1), ln(4/2), ln(4/1)] = [ln2, ln2, 2*ln2].
        # sorted=[ln2, ln2, 2*ln2]; numpy linear quantiles over n=3:
        #   .10 -> pos 0.2 -> ln2;  .50 -> pos 1.0 -> ln2;  .90 -> pos 1.8 -> 1.8*ln2.
        # last=10, band = last*exp(r) -> (10*2, 10*2, 10*2^1.8) = (20.0, 20.0, ~34.822).
        y = np.array([1.0, 2.0, 1.0, 4.0, 2.0])
        p10, p50, p90 = _empirical_max_band(y, 2, use_log=True, last=10.0)
        assert p10 == pytest.approx(20.0)
        assert p50 == pytest.approx(20.0)
        assert p90 == pytest.approx(10.0 * 2.0**1.8)  # ~34.822

    def test_build_spread_series_relative_returns(self):
        # a=[10,20,25], b=[5,5,10] on aligned dates. rel = 100*[(logA-logA0)-(logB-logB0)]:
        #   t0: 0
        #   t1: 100*(log2 - 0)        = 100*ln(2)    ~= 69.3147
        #   t2: 100*(log2.5 - log2)   = 100*ln(1.25) ~= 22.3144
        idx = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"])
        a = pd.Series([10.0, 20.0, 25.0], index=idx)
        b = pd.Series([5.0, 5.0, 10.0], index=idx)
        spread = _build_spread_series(a, b)
        assert spread.iloc[0] == pytest.approx(0.0)
        assert spread.iloc[1] == pytest.approx(100.0 * float(np.log(2.0)))
        assert spread.iloc[2] == pytest.approx(100.0 * float(np.log(1.25)))

    def test_build_spread_series_disjoint_calendar_raises(self):
        # No overlapping dates -> inner join empty -> ValueError (the documented contract).
        a = pd.Series([1.0, 2.0], index=pd.to_datetime(["2026-01-01", "2026-01-02"]))
        b = pd.Series([1.0, 2.0], index=pd.to_datetime(["2026-02-01", "2026-02-02"]))
        with pytest.raises(ValueError, match="no overlapping dates"):
            _build_spread_series(a, b)


# Provider factory (flag gating, benchmark ceiling, soft-fail, determinism).
class TestProviderFactory:
    @pytest.mark.asyncio
    async def test_disabled_flag_returns_empty_even_when_routable(self, monkeypatch):
        """Env-flag gate: with TS_ANCHOR_ENABLED unset the provider short-circuits to ""
        WITHOUT touching fetch_series, even for a cleanly-routable question."""
        monkeypatch.delenv("TS_ANCHOR_ENABLED", raising=False)
        fetch_spy = MagicMock(side_effect=AssertionError("fetch_series must not run when disabled"))
        monkeypatch.setattr(ts, "fetch_series", fetch_spy)

        provider = timeseries_anchor_provider()
        result = await provider(_make_numeric_q(resolution_criteria=_DGS10_RC))

        assert result == ""
        fetch_spy.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_numeric_question_returns_empty(self, monkeypatch):
        monkeypatch.setenv("TS_ANCHOR_ENABLED", "true")
        binary_q = MagicMock(spec=BinaryQuestion)
        binary_q.id_of_question = 9
        binary_q.resolution_criteria = _DGS10_RC

        provider = timeseries_anchor_provider()
        assert await provider(binary_q) == ""

    @pytest.mark.asyncio
    async def test_enabled_flag_routes_fetches_and_renders(self, monkeypatch):
        monkeypatch.setenv("TS_ANCHOR_ENABLED", "true")
        monkeypatch.setattr(ts, "fetch_series", lambda *_a, **_k: _daily_positive_series("DGS10"))

        provider = timeseries_anchor_provider()
        out = await provider(_make_numeric_q(resolution_criteria=_DGS10_RC))

        assert isinstance(out, str)
        assert out  # non-empty section
        # DGS10 is a registry entry, so URL routing carries the registry's descriptive
        # label (F10 fix) rather than the bare series_id.
        assert out.splitlines()[0].startswith("**10-Year Treasury constant-maturity yield (%)** — latest ")
        assert "P10 / P50 / P90 →" in out

    @pytest.mark.asyncio
    async def test_is_benchmarking_uses_open_time_as_ceiling(self, monkeypatch):
        """Backtest-safe path: is_benchmarking=True does NOT short-circuit (unlike
        prediction_market) — it pins the fetch ceiling to question.open_time so series
        data known at forecast time IS the answer without leaking the resolution."""
        monkeypatch.setenv("TS_ANCHOR_ENABLED", "true")
        captured_ceilings: list[date] = []

        def _capturing_fetch(spec, ceiling, **_kwargs):
            captured_ceilings.append(ceiling)
            return _daily_positive_series("DGS10", end="2026-03-10")

        monkeypatch.setattr(ts, "fetch_series", _capturing_fetch)

        open_time = datetime(2026, 3, 15, tzinfo=UTC)
        q = _make_numeric_q(resolution_criteria=_DGS10_RC, open_time=open_time)

        provider = timeseries_anchor_provider(is_benchmarking=True)
        out = await provider(q)

        assert out  # still ran (not short-circuited)
        assert captured_ceilings == [open_time.date()]  # ceiling pinned to open_time

    @pytest.mark.asyncio
    async def test_env_flag_gate_precedes_is_benchmarking_logic(self, monkeypatch):
        """Ordering mirror: the env-flag gate is evaluated BEFORE the is_benchmarking
        as_of branch, so a disabled flag returns "" without ever reading open_time."""
        monkeypatch.delenv("TS_ANCHOR_ENABLED", raising=False)
        fetch_spy = MagicMock(side_effect=AssertionError("must not fetch when flag disabled"))
        monkeypatch.setattr(ts, "fetch_series", fetch_spy)

        # open_time deliberately absent — if the is_benchmarking branch ran first it would
        # log a warning; the flag gate must return "" before that.
        q = _make_numeric_q(resolution_criteria=_DGS10_RC)
        q.open_time = None

        provider = timeseries_anchor_provider(is_benchmarking=True)
        assert await provider(q) == ""
        fetch_spy.assert_not_called()

    @pytest.mark.asyncio
    async def test_malformed_fetch_soft_fails_to_empty_with_warning(self, monkeypatch, caplog):
        """A genuine fetch/data error (here: HTML instead of CSV) soft-fails to "" + WARNING;
        it never raises out of the provider."""
        monkeypatch.setenv("TS_ANCHOR_ENABLED", "true")
        monkeypatch.setattr(tf, "_http_get", FakeHttp({FRED_CSV_URL: b"<html>bad series id</html>"}))

        provider = timeseries_anchor_provider()
        with caplog.at_level(logging.WARNING):
            result = await provider(_make_numeric_q(resolution_criteria=_DGS10_RC))

        assert result == ""
        assert any("soft-fail" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_deterministic_output_across_calls(self, monkeypatch):
        """Same question + same series -> byte-identical section. Reset caches between
        the two calls so the second recomputes rather than reading the section cache."""
        monkeypatch.setenv("TS_ANCHOR_ENABLED", "true")
        monkeypatch.setattr(ts, "fetch_series", lambda *_a, **_k: _daily_positive_series("DGS10"))
        q = _make_numeric_q(resolution_criteria=_DGS10_RC)

        provider = timeseries_anchor_provider()
        first = await provider(q)
        _reset_session_caches()
        second = await provider(q)

        assert first == second
        assert first  # not the empty soft-fail

    @pytest.mark.asyncio
    async def test_leaky_fetch_soft_fails_to_empty(self, monkeypatch, caplog):
        """A post-ceiling row triggers the fetch layer's LeakageError, which the provider
        catches and soft-fails to "" — the render never reflects the leaked observation."""
        monkeypatch.setenv("TS_ANCHOR_ENABLED", "true")
        # The provider (live) uses as_of=now; a 2099 row is unambiguously post-ceiling.
        rows = [("2026-06-01", "4.20"), ("2099-01-01", "9.99")]
        monkeypatch.setattr(tf, "_http_get", FakeHttp({FRED_CSV_URL: _csv("DGS10", rows)}))

        provider = timeseries_anchor_provider()
        with caplog.at_level(logging.WARNING):
            result = await provider(_make_numeric_q(resolution_criteria=_DGS10_RC))

        assert result == ""
        assert "9.99" not in result

    @pytest.mark.asyncio
    async def test_missing_scheduled_resolution_time_returns_empty(self, monkeypatch):
        """build_anchor_section needs a real scheduled_resolution_time to size the horizon;
        without one it returns "" before fetching."""
        monkeypatch.setenv("TS_ANCHOR_ENABLED", "true")
        fetch_spy = MagicMock(side_effect=AssertionError("must not fetch without a horizon"))
        monkeypatch.setattr(ts, "fetch_series", fetch_spy)

        q = _make_numeric_q(resolution_criteria=_DGS10_RC, scheduled_resolution_time=None)
        provider = timeseries_anchor_provider()

        assert await provider(q) == ""
        fetch_spy.assert_not_called()


# Bounds-overlap backstop: a rendered P10-P90 band lying ENTIRELY outside the question's
# displayed range is a gross units/magnitude mismatch (level-vs-derived, wrong country), so
# build_anchor_section drops the section rather than feed a wrong-units anchor to the
# forecasters. Open / non-finite bounds impose no constraint on that side.
class TestBoundsBackstop:
    def test_band_none_never_misses(self):
        # No band rendered (not a model target, or horizon exceeds history) -> nothing to check.
        assert _band_misses_bounds(_make_numeric_q(lower_bound=0.0, upper_bound=1.0), None) is False

    def test_band_below_closed_bounds_misses(self):
        # The canonical bug shape: a ~0.3 band (MoM-%) vs an index-level range [250, 350].
        q = _make_numeric_q(lower_bound=250.0, upper_bound=350.0)
        assert _band_misses_bounds(q, (0.1, 0.3, 0.5)) is True

    def test_band_above_closed_bounds_misses(self):
        q = _make_numeric_q(lower_bound=250.0, upper_bound=350.0)
        assert _band_misses_bounds(q, (400.0, 450.0, 500.0)) is True

    def test_band_overlapping_bounds_does_not_miss(self):
        q = _make_numeric_q(lower_bound=0.0, upper_bound=1.0)
        assert _band_misses_bounds(q, (0.1, 0.3, 0.5)) is False

    def test_open_lower_bound_lifts_lower_constraint(self):
        # Value can settle below an OPEN lower edge, so a low band is not a mismatch.
        q = _make_numeric_q(lower_bound=250.0, upper_bound=350.0, open_lower_bound=True)
        assert _band_misses_bounds(q, (0.1, 0.3, 0.5)) is False

    def test_open_upper_bound_lifts_upper_constraint(self):
        q = _make_numeric_q(lower_bound=250.0, upper_bound=350.0, open_upper_bound=True)
        assert _band_misses_bounds(q, (400.0, 450.0, 500.0)) is False

    def test_non_finite_bounds_impose_no_constraint(self):
        q = _make_numeric_q(lower_bound=float("-inf"), upper_bound=float("inf"))
        assert _band_misses_bounds(q, (0.1, 0.3, 0.5)) is False

    def test_build_anchor_section_skips_on_bounds_mismatch(self, monkeypatch, caplog):
        # End-to-end: a ~0.3-magnitude series routed onto a level question with an
        # index-level range [250, 350] -> the section is dropped with a WARN.
        flat = pd.Series(
            np.full(400, 0.3, dtype="float64"),
            index=pd.bdate_range("2024-01-01", periods=400),
            name="DGS10",
        )
        monkeypatch.setattr(ts, "fetch_series", lambda *_a, **_k: flat)
        q = _make_numeric_q(resolution_criteria=_DGS10_RC, lower_bound=250.0, upper_bound=350.0)

        with caplog.at_level(logging.WARNING):
            out = build_anchor_section(q, datetime(2026, 3, 15, tzinfo=UTC))

        assert out == ""
        assert any("zero overlap with question bounds" in r.message for r in caplog.records)

    def test_build_anchor_section_renders_when_band_within_bounds(self, monkeypatch):
        # Same series, but a range that contains the ~0.3 band -> the section renders.
        flat = pd.Series(
            np.full(400, 0.3, dtype="float64"),
            index=pd.bdate_range("2024-01-01", periods=400),
            name="DGS10",
        )
        monkeypatch.setattr(ts, "fetch_series", lambda *_a, **_k: flat)
        q = _make_numeric_q(resolution_criteria=_DGS10_RC, lower_bound=0.0, upper_bound=1.0)

        out = build_anchor_section(q, datetime(2026, 3, 15, tzinfo=UTC))

        assert out  # non-empty; the band ~0.3 lies within [0, 1]
        assert "P10 / P50 / P90 →" in out

    # The spread path is now wired through the same backstop as the single path (part 2 of
    # the two-ticker fix). A relative-return-worded two-ticker question PASSES the wording
    # gate, but if its displayed bounds are a wrong-unit range (a ratio's [60, 110], not a
    # ±pp band) the backstop still drops the section — belt-and-suspenders behind the gate.
    @staticmethod
    def _spread_fetch(spec, _ceiling, **_kwargs):
        if spec.series_id == "CL=F":
            return _daily_positive_series("CL=F", seed=1)
        return _daily_positive_series("^GSPC", seed=2) * 40.0

    @staticmethod
    def _relret_two_ticker_q(*, lower_bound: float, upper_bound: float, qid: int = 8201) -> MagicMock:
        # Passes the relative-return wording gate (routes to spread), 14-day horizon -> a
        # ±few-pp mean-zero band.
        qt = "How much will CL=F's returns exceed ^GSPC's over the window?"
        rc = "return(https://finance.yahoo.com/quote/CL=F) minus return(https://finance.yahoo.com/quote/%5EGSPC)."
        return _make_numeric_q(
            qid=qid,
            question_text=qt,
            resolution_criteria=rc,
            scheduled_resolution_time=datetime(2026, 3, 29, tzinfo=UTC),  # 14 days past the 2026-03-15 as_of
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

    def test_build_anchor_section_spread_skips_on_bounds_mismatch(self, monkeypatch, caplog):
        # gold/silver-ratio-shaped bounds [60, 110] vs a ±few-pp mean-zero spread band ->
        # dropped by the newly-wired backstop with a spread-specific WARN.
        monkeypatch.setattr(ts, "fetch_series", self._spread_fetch)
        q = self._relret_two_ticker_q(lower_bound=60.0, upper_bound=110.0)

        with caplog.at_level(logging.WARNING):
            out = build_anchor_section(q, datetime(2026, 3, 15, tzinfo=UTC))

        assert out == ""
        assert any("zero overlap with question bounds" in r.message for r in caplog.records)

    def test_build_anchor_section_spread_renders_when_band_within_bounds(self, monkeypatch):
        # Same spread, but a pp-scale range [-50, 50] that contains the ±few-pp band -> renders.
        monkeypatch.setattr(ts, "fetch_series", self._spread_fetch)
        q = self._relret_two_ticker_q(lower_bound=-50.0, upper_bound=50.0, qid=8202)

        out = build_anchor_section(q, datetime(2026, 3, 15, tzinfo=UTC))

        assert out  # non-empty; the ±few-pp band lies within [-50, 50]
        assert "Relative-return spread: CL=F vs ^GSPC" in out
        assert "P10 / P50 / P90 →" in out


# Chart side-channel must respect the bounds backstop: build_anchor_section stashes the
# chart only on the success path, so a bounds-rejected (wrong-units) section leaves nothing
# for forecaster.py `_pull_research_chart` to attach to the base forecasters' vision message.
class TestChartBackstop:
    def _flat_series(self) -> pd.Series:
        return pd.Series(
            np.full(400, 0.3, dtype="float64"),
            index=pd.bdate_range("2024-01-01", periods=400),
            name="DGS10",
        )

    def test_chart_not_stashed_on_bounds_reject(self, monkeypatch):
        # Chart flag ON + a band that misses the bounds -> section suppressed AND no chart
        # stashed (the render is never even attempted on the reject path).
        monkeypatch.setenv("TS_ANCHOR_CHART_ENABLED", "true")
        monkeypatch.setattr(ts, "fetch_series", lambda *_a, **_k: self._flat_series())
        monkeypatch.setattr(
            "metaculus_bot.research.ts_chart.render_anchor_chart",
            MagicMock(side_effect=AssertionError("chart must not render on a bounds-rejected section")),
        )
        q = _make_numeric_q(resolution_criteria=_DGS10_RC, lower_bound=250.0, upper_bound=350.0)

        out = build_anchor_section(q, datetime(2026, 3, 15, tzinfo=UTC))

        assert out == ""
        assert ts._session_charts == {}  # nothing stashed for _pull_research_chart to attach

    def test_chart_stashed_on_success_path(self, monkeypatch):
        # The move to the success path must still stash the chart when the band is in-bounds.
        monkeypatch.setenv("TS_ANCHOR_CHART_ENABLED", "true")
        monkeypatch.setattr(ts, "fetch_series", lambda *_a, **_k: self._flat_series())
        monkeypatch.setattr(
            "metaculus_bot.research.ts_chart.render_anchor_chart",
            lambda *_a, **_k: "FAKE_PNG_BASE64",
        )
        q = _make_numeric_q(qid=8123, resolution_criteria=_DGS10_RC, lower_bound=0.0, upper_bound=1.0)

        out = build_anchor_section(q, datetime(2026, 3, 15, tzinfo=UTC))

        assert out  # non-empty success path
        assert ts._session_charts.get(8123) == "FAKE_PNG_BASE64"
