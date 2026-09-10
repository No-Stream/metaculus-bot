"""The three explicit ToolSpecs the gap-fill v2 driver gets, ready to append to its tool list.

The specs pair a driver-facing description and JSON schema with a handler that parses the ISO date
window off the model's arguments, calls the backend, and adapts the result to a ToolOutcome. The
market handler binds the per-question session, catalogue, dump and Kalshi semaphore.
"""

from __future__ import annotations

from datetime import date

import pytest

from metaculus_bot.research.known_api import backends, tools
from metaculus_bot.research.known_api.result import KnownApiResult

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _specs():
    return tools.build_known_api_tools(session=object())


class TestSpecShape:
    def test_three_named_specs_with_schemas(self):
        names = [spec.name for spec in _specs()]
        assert names == ["fred_series", "yahoo_history", "market_snapshot"]

    def test_yahoo_requires_a_ticker(self):
        yahoo = next(spec for spec in _specs() if spec.name == "yahoo_history")
        assert yahoo.parameters["required"] == ["ticker"]
        assert yahoo.parameters["additionalProperties"] is False

    def test_market_venue_is_an_enum(self):
        market = next(spec for spec in _specs() if spec.name == "market_snapshot")
        assert set(market.parameters["properties"]["venue"]["enum"]) == {
            "kalshi",
            "polymarket",
            "manifold",
            "predictit",
        }


class TestHandlers:
    async def test_fred_handler_parses_the_window_and_adapts(self, monkeypatch: pytest.MonkeyPatch):
        seen: dict[str, object] = {}

        async def _fake(*, series_id=None, search=None, start=None, end=None, first_release=False):
            seen.update(series_id=series_id, start=start, end=end, first_release=first_release)
            return KnownApiResult(status="ok", content_markdown="block", source_url="u", links=["u"])

        monkeypatch.setattr(backends, "fred_series", _fake)
        fred = next(spec for spec in _specs() if spec.name == "fred_series")

        outcome = await fred.handler(series_id="DGS30", start="2026-06-01", end="2026-07-31", first_release=True)

        assert outcome.method == "known_api"
        assert outcome.status == "ok"
        assert seen == {
            "series_id": "DGS30",
            "start": date(2026, 6, 1),
            "end": date(2026, 7, 31),
            "first_release": True,
        }

    async def test_fred_handler_tolerates_a_bad_date(self, monkeypatch: pytest.MonkeyPatch):
        seen: dict[str, object] = {}

        async def _fake(*, series_id=None, search=None, start=None, end=None, first_release=False):
            seen["start"] = start
            return KnownApiResult(status="ok", content_markdown="block", source_url="u")

        monkeypatch.setattr(backends, "fred_series", _fake)
        fred = next(spec for spec in _specs() if spec.name == "fred_series")

        await fred.handler(series_id="DGS30", start="not-a-date")

        assert seen["start"] is None

    async def test_market_handler_binds_the_session_and_catalogue(self, monkeypatch: pytest.MonkeyPatch):
        seen: dict[str, object] = {}
        session = object()
        catalogue = [{"title": "x", "event_ticker": "KX", "markets": []}]

        async def _fake(
            *, venue, market, session, kalshi_catalogue=None, predictit_markets=None, kalshi_detail_budget=None
        ):
            seen.update(venue=venue, market=market, session=session, kalshi_catalogue=kalshi_catalogue)
            return KnownApiResult(status="ok", content_markdown="snap", source_url="u")

        monkeypatch.setattr(backends, "market_snapshot", _fake)
        specs = tools.build_known_api_tools(session=session, kalshi_catalogue=catalogue)
        market = next(spec for spec in specs if spec.name == "market_snapshot")

        outcome = await market.handler(venue="kalshi", market="unemployment")

        assert outcome.status == "ok"
        assert seen["session"] is session
        assert seen["kalshi_catalogue"] == catalogue
