"""Tests for the time-series-anchor chart side-channel.

Two surfaces:
- ``ts_chart.render_anchor_chart``: renders a valid, deterministic base64 PNG.
- ``timeseries_anchor``'s provider hook: populates ``_session_charts`` only when
  ``TS_ANCHOR_CHART_ENABLED`` is on AND the question routed to a single LEVEL
  series (never for max-window / spread / disabled).

No network, no LLM. Fetch is monkeypatched to a synthetic series.
"""

from __future__ import annotations

import base64
from datetime import UTC, datetime
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from forecasting_tools import NumericQuestion

from metaculus_bot.research import timeseries_anchor as ts
from metaculus_bot.research.ts_chart import render_anchor_chart

# PNG files start with this 8-byte signature.
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _daily_series(name: str = "DGS10", *, seed: int = 0, end: str = "2026-06-30", years: int = 6) -> pd.Series:
    end_ts = pd.Timestamp(end)
    idx = pd.bdate_range(end_ts - pd.Timedelta(days=round(years * 365.25)), end_ts)
    rng = np.random.default_rng(seed)
    walk = 20.0 + np.cumsum(rng.normal(0.0, 0.3, len(idx)))
    return pd.Series(np.abs(walk) + 8.0, index=idx, name=name)


def _make_numeric_q(
    *,
    qid: int = 8001,
    question_text: str = "What will X be?",
    resolution_criteria: str = "rc",
    fine_print: str = "",
    scheduled_resolution_time: datetime | None = datetime(2027, 1, 1, tzinfo=UTC),
) -> MagicMock:
    q = MagicMock(spec=NumericQuestion)
    q.id_of_question = qid
    q.question_text = question_text
    q.resolution_criteria = resolution_criteria
    q.fine_print = fine_print
    q.title = question_text
    q.open_time = datetime(2026, 3, 15, tzinfo=UTC)
    q.scheduled_resolution_time = scheduled_resolution_time
    return q


_DGS10_RC = "Resolves per https://fred.stlouisfed.org/series/DGS10 on the resolution date."


@pytest.fixture(autouse=True)
def _reset_caches():
    ts._reset_session_caches()
    yield
    ts._reset_session_caches()


class TestRenderAnchorChart:
    def test_renders_valid_png_base64(self):
        series = _daily_series()
        out = render_anchor_chart(
            series,
            as_of=pd.Timestamp("2026-06-30"),
            horizon_end=pd.Timestamp("2027-01-01"),
            band=(15.0, 20.0, 25.0),
            title="10-Year Treasury yield",
        )
        assert isinstance(out, str)
        raw = base64.b64decode(out, validate=True)  # raises binascii.Error on non-b64
        assert raw.startswith(_PNG_MAGIC)

    def test_deterministic_across_calls(self):
        series = _daily_series()

        def _render() -> str:
            return render_anchor_chart(
                series,
                as_of=pd.Timestamp("2026-06-30"),
                horizon_end=pd.Timestamp("2027-01-01"),
                band=(15.0, 20.0, 25.0),
                title="10-Year Treasury yield",
            )

        assert _render() == _render()


class TestProviderChartHook:
    @pytest.mark.asyncio
    async def test_chart_populated_when_flag_on_level_question(self, monkeypatch):
        monkeypatch.setenv("TS_ANCHOR_ENABLED", "true")
        monkeypatch.setenv("TS_ANCHOR_CHART_ENABLED", "true")
        monkeypatch.setattr(ts, "fetch_series", lambda *_a, **_k: _daily_series("DGS10"))

        q = _make_numeric_q(qid=8100, resolution_criteria=_DGS10_RC)
        provider = ts.timeseries_anchor_provider()
        out = await provider(q)

        assert out  # text section still produced
        assert 8100 in ts._session_charts
        raw = base64.b64decode(ts._session_charts[8100], validate=True)
        assert raw.startswith(_PNG_MAGIC)

    @pytest.mark.asyncio
    async def test_chart_not_populated_when_chart_flag_off(self, monkeypatch):
        """Anchor text still renders (TS_ANCHOR_ENABLED on) but no chart is stashed."""
        monkeypatch.setenv("TS_ANCHOR_ENABLED", "true")
        monkeypatch.delenv("TS_ANCHOR_CHART_ENABLED", raising=False)
        monkeypatch.setattr(ts, "fetch_series", lambda *_a, **_k: _daily_series("DGS10"))

        q = _make_numeric_q(qid=8101, resolution_criteria=_DGS10_RC)
        provider = ts.timeseries_anchor_provider()
        out = await provider(q)

        assert out
        assert ts._session_charts == {}

    @pytest.mark.asyncio
    async def test_chart_not_populated_for_max_window_question(self, monkeypatch):
        """v1 charts only the level shape — a 'highest' (forward-max) question routes
        to the High column and must NOT get a chart even with the flag on."""
        monkeypatch.setenv("TS_ANCHOR_ENABLED", "true")
        monkeypatch.setenv("TS_ANCHOR_CHART_ENABLED", "true")
        monkeypatch.setattr(ts, "fetch_series", lambda *_a, **_k: _daily_series("^VIX"))

        # "highest ... VIX" routes via the template registry to a High-column max question.
        q = _make_numeric_q(qid=8102, question_text="What is the highest VIX value this year?")
        provider = ts.timeseries_anchor_provider()
        out = await provider(q)

        assert out
        assert ts._session_charts == {}

    @pytest.mark.asyncio
    async def test_reset_session_caches_clears_charts(self, monkeypatch):
        monkeypatch.setenv("TS_ANCHOR_ENABLED", "true")
        monkeypatch.setenv("TS_ANCHOR_CHART_ENABLED", "true")
        monkeypatch.setattr(ts, "fetch_series", lambda *_a, **_k: _daily_series("DGS10"))

        provider = ts.timeseries_anchor_provider()
        await provider(_make_numeric_q(qid=8103, resolution_criteria=_DGS10_RC))
        assert 8103 in ts._session_charts

        ts._reset_session_caches()
        assert ts._session_charts == {}
