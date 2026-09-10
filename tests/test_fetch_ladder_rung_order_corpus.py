"""The rung-order corpus: one synthetic direct outcome per (status, reason) pair, both callers.

The CI gate on the fetch-ladder unification. For every outcome the direct fetch can produce, it
drives the whole ladder with every escalation transport declining and records which transports
were dialed, in what order, plus the final route and status. The resolution-source fetcher's
column and the gap-fill v2 loop's column are pinned side by side in ``_EXPECTED``, so a change
that moves either caller's rung order, final route or final status fails here rather than in
production. Why this exists rather than trusting the two suites, and how the table was generated:
``scratch_docs_and_planning/fetch_ladder_progress_2026-09-10.md``, "The rung-order corpus".

Every observation is taken at TRANSPORT level (``fetch_impersonated``, ``render_page``, the
archive URL's own GET, ``run_url_context_read``, the derived feed's GET), never by stubbing a
rung, so every trigger predicate and every gate stays live and is part of what is pinned.

One synthetic direct outcome drives both columns, and two of its fields are per caller.
``escalate_rendered`` is the gap-fill verdict's own thin-content signal, which the fetcher's
verdict never sets, so the fetcher's column is always driven with it off. And a ``(status,
reason)`` pair only one verdict can produce still pins the other's column, because what the
corpus is about is what the DISPATCHER and the adapter do with an outcome, whoever produced it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pytest

from metaculus_bot.constants import GOOGLE_API_KEY_ENV, RESOLUTION_SOURCE_URL_CONTEXT_ENABLED_ENV
from metaculus_bot.research import derived_api, document_cache, impersonated_fetch, robots_policy
from metaculus_bot.research.agentic import tools as agentic_tools
from metaculus_bot.research.fetch_ladder import context, direct_fetch, guard, ladder, policy, rungs
from metaculus_bot.research.fetch_ladder.context import LadderContext
from metaculus_bot.research.http_fetch import reset_host_semaphores
from metaculus_bot.research.impersonated_fetch import ImpersonateDeclined
from metaculus_bot.research.resolution_fetch_result import FetchResult, FetchStatus, FetchStatusReason

_URL = "https://tracker.example.com/data"
_ENDPOINT_URL = "https://tracker.example.com/api/series.json"
_NOW = datetime(2026, 9, 10, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class _DirectCase:
    """One synthetic direct outcome, named for the corpus table."""

    name: str
    status: FetchStatus
    reason: FetchStatusReason | None = None
    http_status: int | None = None
    text: str = ""
    escalate_rendered: bool = False


# One case per (status, reason) pair, plus each HTTP status that maps to `blocked` or `error`.
_CASES: tuple[_DirectCase, ...] = (
    _DirectCase("success", "success", text="The tracker reported 41 cases in May 2026."),
    _DirectCase("success_thin", "success", text="41 cases", escalate_rendered=True),
    _DirectCase("blocked_403", "blocked", http_status=403),
    _DirectCase("blocked_406", "blocked", http_status=406),
    _DirectCase("blocked_429", "blocked", http_status=429),
    _DirectCase("blocked_self_ref", "blocked", reason="metaculus_self_ref"),
    _DirectCase("not_found", "not_found", http_status=404),
    _DirectCase("error_transport", "error"),
    _DirectCase("error_5xx", "error", http_status=503),
    _DirectCase("js_wall", "js_wall", http_status=200, escalate_rendered=True),
    _DirectCase("embed_shell", "no_resolving_content", reason="embed_shell", http_status=200, escalate_rendered=True),
    _DirectCase("thin_page", "no_resolving_content", reason="thin_page", http_status=200, escalate_rendered=True),
    # A fetcher-only withhold: a document the gap-fill verdict reads in full is a success.
    _DirectCase("no_matching_passage", "no_resolving_content", reason="no_matching_passage", http_status=200),
    _DirectCase("unsupported_type", "unsupported_type", http_status=200),
    _DirectCase("unsupported_budget_skipped", "unsupported_type", reason="budget_skipped", http_status=200),
    _DirectCase("unsupported_parse_contention", "unsupported_type", reason="parse_contention", http_status=200),
    _DirectCase("unreadable_no_text_layer", "unreadable_document", reason="no_text_layer", http_status=200),
    _DirectCase("unreadable_encrypted", "unreadable_document", reason="encrypted", http_status=200),
    _DirectCase("unreadable_malformed", "unreadable_document", reason="malformed", http_status=200),
    _DirectCase("empty_body", "empty_body", http_status=200, escalate_rendered=True),
    _DirectCase("throttled", "throttled", http_status=200),
    _DirectCase("ssrf_blocked", "ssrf_blocked"),
)


def _direct_result(case: _DirectCase, *, escalate_rendered: bool) -> FetchResult:
    return FetchResult(
        url=_URL,
        status=case.status,
        text="" if case.status == "throttled" else case.text,
        http_status=case.http_status,
        content_type="text/html",
        status_reason=case.reason,
        escalate_rendered=escalate_rendered,
        throttle_phrase="rate limit" if case.status == "throttled" else None,
        throttle_chars=42 if case.status == "throttled" else None,
    )


@dataclass(slots=True)
class _Dialed:
    """Which escalation transports one run reached, in the order it reached them."""

    order: list[str]

    def note(self, transport: str) -> None:
        self.order.append(transport)


class _NullSession:
    """A session the ladder opens and closes but never dials: every hop is doubled in the test."""

    async def __aenter__(self) -> _NullSession:
        await asyncio.sleep(0)
        return self

    async def __aexit__(self, *exc: object) -> None:
        await asyncio.sleep(0)


def _install_declining_transports(
    monkeypatch: pytest.MonkeyPatch, case: _DirectCase, dialed: _Dialed, *, escalate_rendered: bool
) -> None:
    """Answer the cited URL with ``case`` and make every escalation transport decline.

    The derived feed's endpoint is remembered up front, so the REUSE rung fires wherever the
    dispatcher reaches it rather than declining on an empty memo; its GET, the archive's and the
    robots pre-check's all go through the direct fetch, so one URL-keyed double records all three.
    ``escalate_rendered`` is the caller's own thin-content signal, which the fetcher never sets.
    """
    derived_api.remember_endpoint(_URL, _ENDPOINT_URL)

    async def _fake_direct(session: Any, url: str, host_sems: Any, ctx: context.LadderContext) -> FetchResult:
        del session, host_sems, ctx
        await asyncio.sleep(0)
        if url == _URL:
            return _direct_result(case, escalate_rendered=escalate_rendered)
        if "web.archive.org" in url:
            dialed.note("wayback")
        elif url == _ENDPOINT_URL:
            dialed.note("derived_api")
        return FetchResult(url=url, status="not_found", text="", http_status=404, content_type=None)

    async def _declining_render(url: str, **kwargs: Any) -> None:
        del url, kwargs
        dialed.note("rendered")
        await asyncio.sleep(0)

    async def _declining_impersonate(url: str, **kwargs: Any) -> Any:
        del kwargs
        dialed.note("impersonate")
        await asyncio.sleep(0)
        raise ImpersonateDeclined(f"declined for {url}")

    def _declining_reader(*args: Any, **kwargs: Any) -> tuple[str, int, list[str]]:
        del args, kwargs
        dialed.note("url_context")
        raise RuntimeError("the paid reader declined")

    monkeypatch.setattr(direct_fetch, "_fetch_direct", _fake_direct)
    monkeypatch.setattr(rungs, "render_page", _declining_render)
    monkeypatch.setattr(rungs, "fetch_impersonated", _declining_impersonate)
    monkeypatch.setattr(rungs, "run_url_context_read", _declining_reader)
    monkeypatch.setattr(impersonated_fetch, "IMPERSONATE_TRIGGER_STATUSES", frozenset({403}))
    monkeypatch.setenv(RESOLUTION_SOURCE_URL_CONTEXT_ENABLED_ENV, "true")
    monkeypatch.setenv(GOOGLE_API_KEY_ENV, "key")


_RESETS: tuple[Callable[[], None], ...] = (
    derived_api.reset_derived_endpoints,
    impersonated_fetch.reset_impersonation_memo,
    robots_policy.reset_robots_cache,
    reset_host_semaphores,
    document_cache.clear_document_cache,
    agentic_tools._FETCH_HOST_SEMAPHORES.clear,
)


@pytest.fixture(autouse=True)
def _reset_shared_state() -> Iterator[None]:
    """Drop every process-wide memo the ladder keeps, so one case cannot answer for the next."""
    for reset in _RESETS:
        reset()
    yield
    for reset in _RESETS:
        reset()


async def _observe_resolution_source(case: _DirectCase, monkeypatch: pytest.MonkeyPatch) -> tuple[str, ...]:
    """What the fetcher's ladder does with ``case``: the transports dialed, the route, the status."""
    dialed = _Dialed(order=[])
    _install_declining_transports(monkeypatch, case, dialed, escalate_rendered=False)
    monkeypatch.setattr(guard, "_get_session", _NullSession)
    result = await ladder.fetch_url(
        _URL,
        policy=policy.RESOLUTION_SOURCE_POLICY,
        ctx=context.LadderContext(query="cases", now=_NOW, host_sems={}),
    )
    return (*dialed.order, f"route={result.route}", f"status={result.status}")


async def _observe_gap_fill(case: _DirectCase, monkeypatch: pytest.MonkeyPatch) -> tuple[str, ...]:
    """What the loop's ``fetch`` tool does with ``case``: transports dialed, then status/method.

    The same ladder and the same declining transports as the fetcher's column, under
    ``GAP_FILL_FETCH_POLICY`` and through the loop's own handler, so what the two columns differ in
    is the preset rather than the harness. ``read_document`` is recorded rather than run, because a
    document escalation is a second tool call and not a rung.
    """
    dialed = _Dialed(order=[])
    _install_declining_transports(monkeypatch, case, dialed, escalate_rendered=case.escalate_rendered)
    monkeypatch.setattr(guard, "_get_session", _NullSession)

    async def _recording_read_document(url: str, ask: str, **kwargs: Any) -> agentic_tools.ToolOutcome:
        del url, ask, kwargs
        dialed.note("read_document")
        await asyncio.sleep(0)
        return agentic_tools.ToolOutcome(content_markdown="the reader ran", method="document")

    monkeypatch.setattr(agentic_tools, "read_document", _recording_read_document)
    outcome = await agentic_tools.fetch(_URL, question_topic="cases", ctx=LadderContext(now=_NOW, host_sems={}))
    return (*dialed.order, f"status={outcome.status}", f"method={outcome.method}")


_EXPECTED: dict[str, dict[str, tuple[str, ...]]] = {
    "success": {"resolution_source": ("route=direct", "status=success"), "gap_fill_v2": ("status=ok", "method=plain")},
    "success_thin": {
        "resolution_source": ("route=direct", "status=success"),
        "gap_fill_v2": ("rendered", "status=ok", "method=plain"),
    },
    "blocked_403": {
        "resolution_source": ("impersonate", "wayback", "url_context", "route=url_context", "status=blocked"),
        "gap_fill_v2": ("impersonate", "wayback", "status=blocked", "method=plain"),
    },
    "blocked_406": {
        "resolution_source": ("wayback", "url_context", "route=url_context", "status=blocked"),
        "gap_fill_v2": ("wayback", "status=blocked", "method=plain"),
    },
    "blocked_429": {
        "resolution_source": ("wayback", "url_context", "route=url_context", "status=blocked"),
        "gap_fill_v2": ("wayback", "status=blocked", "method=plain"),
    },
    "blocked_self_ref": {
        "resolution_source": ("wayback", "route=wayback", "status=blocked"),
        "gap_fill_v2": ("status=blocked", "method=plain"),
    },
    "not_found": {
        "resolution_source": ("wayback", "route=wayback", "status=not_found"),
        "gap_fill_v2": ("wayback", "status=error", "method=plain"),
    },
    "error_transport": {
        "resolution_source": ("wayback", "url_context", "route=url_context", "status=error"),
        "gap_fill_v2": ("wayback", "status=error", "method=plain"),
    },
    "error_5xx": {
        "resolution_source": ("wayback", "url_context", "route=url_context", "status=error"),
        "gap_fill_v2": ("wayback", "status=error", "method=plain"),
    },
    "js_wall": {
        "resolution_source": ("derived_api", "rendered", "url_context", "route=url_context", "status=js_wall"),
        "gap_fill_v2": ("derived_api", "rendered", "status=empty", "method=empty"),
    },
    "embed_shell": {
        "resolution_source": ("url_context", "route=url_context", "status=no_resolving_content"),
        "gap_fill_v2": ("derived_api", "rendered", "status=empty", "method=empty"),
    },
    "thin_page": {
        "resolution_source": (
            "derived_api",
            "rendered",
            "url_context",
            "route=url_context",
            "status=no_resolving_content",
        ),
        "gap_fill_v2": ("derived_api", "rendered", "status=empty", "method=empty"),
    },
    "no_matching_passage": {
        "resolution_source": ("route=direct", "status=no_resolving_content"),
        # The one row the switch moved, and only on a shape this verdict cannot produce.
        "gap_fill_v2": ("status=empty", "method=empty"),
    },
    "unsupported_type": {
        "resolution_source": ("route=direct", "status=unsupported_type"),
        "gap_fill_v2": ("wayback", "status=error", "method=plain"),
    },
    "unsupported_budget_skipped": {
        "resolution_source": ("route=direct", "status=unsupported_type"),
        "gap_fill_v2": ("wayback", "status=error", "method=plain"),
    },
    "unsupported_parse_contention": {
        "resolution_source": ("route=direct", "status=unsupported_type"),
        "gap_fill_v2": ("wayback", "status=error", "method=plain"),
    },
    "unreadable_no_text_layer": {
        "resolution_source": ("route=direct", "status=unreadable_document"),
        "gap_fill_v2": ("read_document", "status=ok", "method=document"),
    },
    "unreadable_encrypted": {
        "resolution_source": ("route=direct", "status=unreadable_document"),
        "gap_fill_v2": ("read_document", "status=ok", "method=document"),
    },
    "unreadable_malformed": {
        "resolution_source": ("route=direct", "status=unreadable_document"),
        "gap_fill_v2": ("read_document", "status=ok", "method=document"),
    },
    "empty_body": {
        "resolution_source": ("route=direct", "status=empty_body"),
        "gap_fill_v2": ("derived_api", "rendered", "status=empty", "method=empty"),
    },
    "throttled": {
        "resolution_source": ("route=direct", "status=throttled"),
        "gap_fill_v2": ("status=throttled", "method=throttled"),
    },
    "ssrf_blocked": {
        "resolution_source": ("route=direct", "status=ssrf_blocked"),
        "gap_fill_v2": ("status=blocked", "method=plain"),
    },
}


@pytest.mark.asyncio
@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
async def test_the_rung_order_holds_for_both_callers(case: _DirectCase, monkeypatch: pytest.MonkeyPatch) -> None:
    observed = {
        "resolution_source": await _observe_resolution_source(case, monkeypatch),
        "gap_fill_v2": await _observe_gap_fill(case, monkeypatch),
    }
    assert observed == _EXPECTED[case.name]
