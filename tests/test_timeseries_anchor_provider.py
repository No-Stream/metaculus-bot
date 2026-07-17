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
    _render_single,
    _render_spread,
    _reset_session_caches,
    _Route,
    _truncate_section,
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


def _make_numeric_q(
    *,
    qid: int = 7001,
    question_text: str = "What will X be?",
    resolution_criteria: str = "rc",
    fine_print: str = "",
    open_time: datetime | None = None,
    scheduled_resolution_time: datetime | None = datetime(2027, 1, 1, tzinfo=UTC),
) -> MagicMock:
    """A ``MagicMock(spec=NumericQuestion)`` with the fields the provider reads set to
    real values (unset MagicMock attrs are truthy and would corrupt routing / isinstance)."""
    q = MagicMock(spec=NumericQuestion)
    q.id_of_question = qid
    q.question_text = question_text
    q.resolution_criteria = resolution_criteria
    q.fine_print = fine_print
    q.title = question_text
    q.open_time = open_time if open_time is not None else datetime(2026, 3, 15, tzinfo=UTC)
    q.scheduled_resolution_time = scheduled_resolution_time
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
        assert route.spec.revises is False  # DGS10 is not in the revising set

    def test_routes_via_fred_url_revising_series_uses_alfred_spec(self):
        rc = "Resolves per https://fred.stlouisfed.org/series/CPIAUCSL."
        route = route_question(_make_numeric_q(resolution_criteria=rc))
        assert route is not None
        assert route.spec.series_id == "CPIAUCSL"
        assert route.spec.revises is True  # CPIAUCSL revises -> ALFRED vintage fetch

    def test_routes_via_yahoo_url_single(self):
        # %5E is the URL-encoded caret for ^VIX; the extractor url-decodes before matching.
        rc = "Tracks https://finance.yahoo.com/quote/%5EVIX at close."
        route = route_question(_make_numeric_q(resolution_criteria=rc))
        assert route is not None
        assert route.kind == "single"
        assert route.spec.source == "yfinance"
        assert route.spec.series_id == "^VIX"

    def test_two_yahoo_tickers_route_to_spread(self):
        rc = "ret(https://finance.yahoo.com/quote/CL=F) minus ret(https://finance.yahoo.com/quote/%5EGSPC)."
        route = route_question(_make_numeric_q(resolution_criteria=rc))
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
        # Both the CPI and the treasury keyword registries match -> ambiguous.
        q = _make_numeric_q(question_text="cpi versus the 10-year treasury spread index")
        with caplog.at_level(logging.INFO):
            assert route_question(q) is None
        assert any("ambiguous keyword routing" in r.message for r in caplog.records)

    def test_highest_framing_selects_high_column(self):
        q = _make_numeric_q(question_text="What is the highest VIX value this year?")
        route = route_question(q)
        assert route is not None
        assert route.spec.column == "High"
        assert route.is_max is True


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


# Render.
class TestRenderSingle:
    def test_latest_value_first_line_and_band_line(self):
        series = _daily_positive_series("^VIX")
        route = _Route(kind="single", spec=SeriesSpec(source="yfinance", series_id="^VIX"), label="CBOE VIX")

        out = _render_single(series, route=route, ceiling=date(2026, 6, 30), calendar_days=14)

        first_line = out.splitlines()[0]
        assert first_line.startswith("**CBOE VIX** — latest ")
        assert "as of 2026-06-30" in first_line
        assert "P10 / P50 / P90 →" in out
        assert PROVENANCE_MARKER in out

    def test_note_rendered_and_band_withheld_when_not_model_target(self):
        series = _daily_positive_series("PAYEMS")
        route = _Route(
            kind="single",
            spec=SeriesSpec(source="fred", series_id="PAYEMS", revises=True),
            label="Total nonfarm payrolls",
            model_target=False,
            note="This is the payrolls LEVEL series.",
        )

        out = _render_single(series, route=route, ceiling=date(2026, 6, 30), calendar_days=30)

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

        out = _render_spread(series_a, series_b, route=route, calendar_days=14)

        assert "Relative-return spread: CL=F vs ^GSPC" in out
        assert "- CL=F latest:" in out
        assert "- ^GSPC latest:" in out
        assert "- CL=F recent:" in out
        assert "- ^GSPC recent:" in out
        assert "relative-return band" in out
        assert "P10 / P50 / P90 →" in out


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
        assert out.splitlines()[0].startswith("**DGS10** — latest ")
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
