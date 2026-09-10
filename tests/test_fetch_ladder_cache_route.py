"""Acquisition-route invariants for shared fetch-ladder cache entries."""

from __future__ import annotations

from dataclasses import replace

import pytest

from metaculus_bot.research.fetch_ladder import direct_fetch, run_cache, rungs
from metaculus_bot.research.fetch_ladder.context import LadderContext
from metaculus_bot.research.fetch_ladder.ladder import fetch_url
from metaculus_bot.research.fetch_ladder.policy import GAP_FILL_FETCH_POLICY
from metaculus_bot.research.resolution_fetch_result import FetchResult, FetchRoute

_URL = "https://example.com/thin-report"


@pytest.mark.asyncio
@pytest.mark.parametrize("acquisition_route", [None, "meta_refresh"])
async def test_failed_render_does_not_make_a_direct_artifact_terminal(
    monkeypatch: pytest.MonkeyPatch,
    acquisition_route: FetchRoute | None,
) -> None:
    direct_calls = 0
    render_calls = 0

    async def direct(
        _session: object,
        url: str,
        _host_sems: dict[str, object],
        ctx: LadderContext,
    ) -> FetchResult:
        nonlocal direct_calls
        direct_calls += 1
        result = FetchResult(
            url=url,
            status="success",
            text="thin report",
            http_status=200,
            content_type="text/plain",
            escalate_rendered=True,
        )
        if acquisition_route is not None:
            ctx.start_rung(acquisition_route, result.status, url)
        ctx.capture_read(
            result,
            run_cache.TextRead(url=url, text="thin report", http_status=200, content_type="text/plain"),
        )
        return result

    async def failed_render(
        url: str,
        direct_result: FetchResult,
        _host_sems: dict[str, object],
        ctx: LadderContext,
    ) -> None:
        nonlocal render_calls
        render_calls += 1
        ctx.start_rung("rendered", direct_result.status, url)

    monkeypatch.setattr(direct_fetch, "_fetch_direct", direct)
    monkeypatch.setattr(rungs, "_rendered_rung", failed_render)
    policy = replace(GAP_FILL_FETCH_POLICY, rungs_enabled=frozenset({"rendered"}))
    first = await fetch_url(_URL, policy=policy, ctx=LadderContext(session=object(), host_sems={}))
    second = await fetch_url(_URL, policy=policy, ctx=LadderContext(session=object(), host_sems={}))

    assert first.text == second.text == "thin report"
    assert direct_calls == 1
    assert render_calls == 2
