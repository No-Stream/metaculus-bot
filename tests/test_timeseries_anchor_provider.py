"""Tests for the time-series-anchor provider: render, estimator math, provider, guards.

All HTTP is mocked. Two seams, matching the module's own layering:

- Provider soft-fail tests patch ``ts_fetch._http_get`` (the single synchronous HTTP seam
  that returns raw CSV bytes) so the real ``fetch_series`` parse + leakage guard runs and a
  genuine data error propagates into the provider's soft-fail path.
- Render / provider / guard tests monkeypatch ``timeseries_anchor.fetch_series`` with a
  canned synthetic series, so no network and a deterministic band. That target is where
  ``build_anchor_section`` LOOKS the symbol up, which is what makes the patch bite.

Coverage (one behavior per test):
- Render (``ts_render``): latest-value first line + P10/P50/P90 band line (single); both
  legs + band (spread); model_target=False withholds the band; derived-target labelling;
  the realized-max floor line; section char budget truncates.
- Estimator math (``ts_estimators``): horizon arithmetic, change/max bands (both the
  additive and log branches), the relative-return spread series, n_eff.
- Provider: disabled flag -> "" (even when routable); non-numeric question -> "";
  is_benchmarking=True uses ``question.open_time`` as the fetch ceiling (does NOT
  short-circuit like prediction_market — this provider is backtest-safe); env-flag gate is
  checked BEFORE the is_benchmarking as_of logic; malformed fetch -> "" + WARNING;
  two calls -> byte-identical output (determinism).
- Guards: the bounds-overlap magnitude backstop, the no-band-no-section rule, and the
  chart side-channel's success-path-only stashing.

The fetch layer itself is covered in ``test_ts_fetch.py`` and routing in
``test_ts_routing.py``; question mocks and the ``_http_get`` fake live in
``tests/ts_anchor_fakes.py``.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from forecasting_tools import BinaryQuestion

from metaculus_bot.research import timeseries_anchor as ts
from metaculus_bot.research import ts_fetch as tf
from metaculus_bot.research import ts_render as tsrender
from metaculus_bot.research.timeseries_anchor import (
    _band_misses_bounds,
    _reset_session_caches,
    build_anchor_section,
    timeseries_anchor_provider,
)
from metaculus_bot.research.ts_estimators import (
    _build_spread_series,
    _empirical_change_band,
    _empirical_max_band,
    _n_eff,
    horizon_steps,
)
from metaculus_bot.research.ts_fetch import FRED_CSV_URL, SeriesSpec
from metaculus_bot.research.ts_render import (
    _apply_derivation,
    _realized_max_floor,
    _render_single,
    _render_spread,
    _truncate_section,
)
from metaculus_bot.research.ts_routing import _Route
from tests.ts_anchor_fakes import _DGS10_RC, FakeHttp, _csv, _make_numeric_q


# Test isolation: the provider keeps a rendered-section cache and the fetch layer
# keeps a parsed-series cache. Both bleed across tests otherwise.
@pytest.fixture(autouse=True)
def _reset_provider_caches():
    _reset_session_caches()
    yield
    _reset_session_caches()


# Synthetic series factories.


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
        # Patched on ts_render, which is where _truncate_section reads the budget.
        monkeypatch.setattr(tsrender, "TS_ANCHOR_SECTION_MAX_CHARS", 120)
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

    def test_open_lower_bound_loosens_but_does_not_remove_the_constraint(self):
        # An OPEN lower edge means the outcome CAN settle below it, so the constraint has to
        # loosen — but mapping the edge to -inf removed it entirely, which disarmed the whole
        # backstop on the ~95% of real numeric questions that carry two open bounds. A
        # ~0.3-magnitude band against an index-level range is still the canonical wrong-units
        # shape whether or not the edge is open.
        q = _make_numeric_q(lower_bound=250.0, upper_bound=350.0, open_lower_bound=True)
        assert _band_misses_bounds(q, (0.1, 0.3, 0.5)) is True

    def test_open_upper_bound_loosens_but_does_not_remove_the_constraint(self):
        q = _make_numeric_q(lower_bound=250.0, upper_bound=350.0, open_upper_bound=True)
        assert _band_misses_bounds(q, (400.0, 450.0, 500.0)) is True

    def test_band_just_beyond_an_open_edge_is_tolerated(self):
        # The other half of "loosens": a band sitting a little past an OPEN edge is a normal
        # forecast (the outcome really can settle out there), not a units error. Only a band
        # clear of the range by more than the tolerance margin is suppressed.
        q = _make_numeric_q(lower_bound=250.0, upper_bound=350.0, open_lower_bound=True)
        assert _band_misses_bounds(q, (240.0, 245.0, 249.0)) is False

    def test_band_just_beyond_a_closed_edge_still_misses(self):
        # A CLOSED edge means the outcome cannot settle beyond it, so no tolerance is granted
        # there and the original zero-overlap rule stands unchanged.
        q = _make_numeric_q(lower_bound=250.0, upper_bound=350.0)
        assert _band_misses_bounds(q, (240.0, 245.0, 249.0)) is True

    def test_non_finite_bounds_impose_no_constraint(self):
        # A non-finite edge carries no span to scale a tolerance against, so it stays a
        # genuine no-constraint case (unlike an open-but-finite edge).
        q = _make_numeric_q(lower_bound=float("-inf"), upper_bound=float("inf"))
        assert _band_misses_bounds(q, (0.1, 0.3, 0.5)) is False

    def test_wrong_quantity_band_on_a_two_open_bound_question_is_caught(self):
        # qid 44868 ("the NOB spread", resolves in basis points around 53, displayed
        # [29.5, 70.5] with BOTH bounds open). These are the two wrong-quantity bands a
        # single-leg or unscaled route would produce. Pre-fix both returned False — the exact
        # hole that left the 548ba88 magnitude guard unable to police open-bound questions.
        q = _make_numeric_q(lower_bound=29.5, upper_bound=70.5, open_lower_bound=True, open_upper_bound=True)
        assert _band_misses_bounds(q, (4.40, 4.68, 4.95)) is True  # single-leg 10-year LEVEL, in percent
        assert _band_misses_bounds(q, (0.35, 0.53, 0.72)) is True  # spread in percent, not basis points

    def test_correct_quantity_band_on_the_same_question_is_kept(self):
        # Same question, the CORRECT basis-point spread band -> must not be suppressed.
        q = _make_numeric_q(lower_bound=29.5, upper_bound=70.5, open_lower_bound=True, open_upper_bound=True)
        assert _band_misses_bounds(q, (35.0, 53.0, 72.0)) is False

    @pytest.mark.parametrize(
        ("qid", "band", "lower_bound", "upper_bound"),
        [
            # Every band the anchor actually published in prod, read verbatim from the
            # archived `## Time Series Anchor` sections, with that question's real bounds.
            # All four carry two OPEN bounds, so they are precisely the population the
            # tolerance margin newly constrains — a too-tight threshold silently suppresses
            # healthy anchors, and silent suppression is the failure direction that would go
            # unnoticed. q44944's band sits partly above its own stale displayed range, which
            # is what makes it the binding case rather than a decorative one.
            (44943, (3.992, 4.200, 4.312), 3.9, 4.5),
            (44944, (330.9, 332.3, 335.4), 328.0, 332.0),
            (45114, (4.194, 4.336, 4.535), 4.2, 4.8),
            (45172, (18.70, 22.49, 30.83), 15.0, 25.0),
        ],
    )
    def test_real_published_bands_are_not_suppressed(
        self, qid: int, band: tuple[float, float, float], lower_bound: float, upper_bound: float
    ):
        q = _make_numeric_q(
            qid=qid,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            open_lower_bound=True,
            open_upper_bound=True,
        )
        assert _band_misses_bounds(q, band) is False

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
            # 14 calendar days past the as_of every caller below passes.
            scheduled_resolution_time=datetime(2026, 3, 29, tzinfo=UTC),
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


# A band is the anchor's whole quantitative payload, and the numeric prompt's anchor clause
# fires on the section HEADER alone — so a section without a band tells the forecaster to
# expect a P10/P50/P90 range that isn't there. Prod shipped exactly that once (qid 44803, a
# newly-listed ticker whose history was shorter than the horizon): a bare label, a 52-week
# range whose low WAS the latest value, and "empirical band withheld".
class TestNoBandNoSection:
    # A 28-calendar-day horizon is 19 TRADING steps on a daily series, so the history has to
    # be shorter than that (not merely shorter than 28) to reach the withheld branch.
    _HORIZON_CALENDAR_DAYS = 28
    _HISTORY_OBSERVATIONS = 15
    _AS_OF = datetime(2026, 3, 15, tzinfo=UTC)
    _RESOLVES = datetime(2026, 4, 12, tzinfo=UTC)  # _HORIZON_CALENDAR_DAYS past _AS_OF

    @classmethod
    def _short_history(cls) -> pd.Series:
        return pd.Series(
            np.linspace(150.0, 119.0, cls._HISTORY_OBSERVATIONS),
            index=pd.bdate_range("2026-02-23", periods=cls._HISTORY_OBSERVATIONS),
            name="SPCX",
        )

    def test_fixture_history_is_genuinely_shorter_than_the_horizon(self):
        # The fixture only exercises the withheld branch if y.size <= h; assert that rather
        # than trusting the arithmetic, so a constant change can't quietly make the tests
        # below pass for the wrong reason.
        assert self._HISTORY_OBSERVATIONS <= horizon_steps("daily", self._HORIZON_CALENDAR_DAYS)
        assert (self._RESOLVES.date() - self._AS_OF.date()).days == self._HORIZON_CALENDAR_DAYS

    def test_section_suppressed_when_horizon_exceeds_history(self, monkeypatch, caplog):
        monkeypatch.setattr(ts, "fetch_series", lambda *_a, **_k: self._short_history())
        q = _make_numeric_q(
            qid=44803,
            resolution_criteria=_DGS10_RC,
            scheduled_resolution_time=self._RESOLVES,
            lower_bound=50.0,
            upper_bound=250.0,
        )

        with caplog.at_level(logging.INFO):
            out = build_anchor_section(q, self._AS_OF)

        assert out == ""
        assert any("no empirical band" in r.message for r in caplog.records)

    def test_render_still_reports_the_withheld_sentinel(self):
        # The suppression belongs in build_anchor_section, not in _render_single: the renderer
        # keeps returning band=None plus its explanatory line so the caller (and any future
        # coverage instrumentation) can tell "no band" from "unroutable".
        route = _Route(
            kind="single",
            spec=SeriesSpec(source="yfinance", series_id="SPCX"),
            label="SPCX",
        )
        out, band = _render_single(
            self._short_history(),
            route=route,
            ceiling=self._AS_OF.date(),
            calendar_days=self._HORIZON_CALENDAR_DAYS,
        )
        assert band is None
        assert "empirical band withheld" in out

    def test_section_renders_when_history_covers_the_horizon(self, monkeypatch):
        # Control: same question shape, enough history -> band and section both survive, so
        # the suppression keys on the missing band rather than on anything else here.
        monkeypatch.setattr(ts, "fetch_series", lambda *_a, **_k: _daily_positive_series("SPCX"))
        q = _make_numeric_q(
            resolution_criteria=_DGS10_RC,
            scheduled_resolution_time=self._RESOLVES,
            lower_bound=0.0,
            upper_bound=1000.0,
        )

        out = build_anchor_section(q, self._AS_OF)

        assert out
        assert "P10 / P50 / P90 →" in out

    def test_non_model_target_route_still_renders_without_a_band(self, monkeypatch):
        # The suppression is gated on model_target, which is what separates the two reasons a
        # band can be absent. A NON-model-target route is deliberately band-free (context-only
        # history plus its explanatory note), so dropping that gate would silently delete a
        # whole intended section shape rather than only the broken one.
        route = _Route(
            kind="single",
            spec=SeriesSpec(source="fred", series_id="PAYEMS", revises=True),
            label="Total nonfarm payrolls",
            model_target=False,
            note="This is the payrolls LEVEL series.",
        )
        monkeypatch.setattr(ts, "route_question", lambda _q: route)
        monkeypatch.setattr(ts, "fetch_series", lambda *_a, **_k: _daily_positive_series("PAYEMS"))
        q = _make_numeric_q(scheduled_resolution_time=self._RESOLVES, lower_bound=0.0, upper_bound=1000.0)

        out = build_anchor_section(q, self._AS_OF)

        assert out
        assert "P10 / P50 / P90" not in out
        assert "- Note: This is the payrolls LEVEL series." in out


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
