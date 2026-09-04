"""The paid rung's timeout and attempt arithmetic, which nothing else in this package pins.

``_url_context_rung`` runs the Gemini reader in an ``asyncio.to_thread`` worker under an
``asyncio.wait_for``, and ``wait_for`` CANNOT CANCEL A THREAD: it cancels the coroutine waiting on
the worker and leaves the worker running. So two numbers, not one, decide what a stalled read
costs. The outer ``timeout`` decides how long the ladder waits. The client-side ``timeout_ms``
handed to the SDK is the only thing that ever returns the worker, and ``attempts`` decides how
many billed requests the SDK may dispatch inside it. A wrong pair strands a pooled worker past the
45 s provider wall and bills a call nobody reads.

The sibling gap-fill v2 reader has three dedicated pins on exactly this arithmetic
(``tests/test_agentic_tools.py``); this rung had none — ``timeout_ms`` and ``attempts`` appeared
nowhere in this package and the ``except TimeoutError`` branch was untested.

Pinned against the CONTRACT rather than the rung's line order: every test fixes
``FetchContext.rung_budget_s`` to a constant, so where in the rung the budget is read (before or
after the robots pre-check) does not change what these assert.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from metaculus_bot.constants import (
    RESOLUTION_SOURCE_RUNG_WALL_MARGIN_S,
    RESOLUTION_SOURCE_URL_CONTEXT_ATTEMPTS,
    RESOLUTION_SOURCE_URL_CONTEXT_MIN_BUDGET_S,
)
from metaculus_bot.research import resolution_source
from metaculus_bot.research.gemini_client_config import gemini_retry_sleep_allowance_s
from metaculus_bot.research.resolution_source import FetchContext, _fetch_one, _rung_counts
from tests.resolution_source_fakes import FakeResponse, FakeSession

_URL = "https://tracker.example.com/senate"

# Comfortably above the rung's floor, so the read runs and the kwargs are observable. Chosen
# distinct from every constant involved, so an assertion cannot pass by coincidence.
_ROOMY_BUDGET_S = 20.0

_ANSWER = "The table dated 2026-08-28 reports 12 major work stoppages beginning in 2026."


def _reader(*, raises: type[BaseException] | None = None):
    """A stand-in for ``run_url_context_read`` that records the kwargs the rung passes it."""
    calls: list[dict[str, Any]] = []

    def _read(url, ask, **kwargs):
        calls.append({"url": url, "ask": ask, **kwargs})
        if raises is not None:
            raise raises("the client ceiling fired")
        return (_ANSWER, 1, ["URL_RETRIEVAL_STATUS_SUCCESS"])

    return _read, calls


def _session() -> FakeSession:
    """A refused page plus an allowing robots.txt: every gate open except the ones under test."""
    return FakeSession(
        {
            _URL: FakeResponse(403, body=b"denied", content_type="text/html"),
            "https://tracker.example.com/robots.txt": FakeResponse(
                200, body=b"User-agent: *\nAllow: /\n", content_type="text/plain"
            ),
        }
    )


def _arm(monkeypatch, reader, *, budget_s: float) -> None:
    monkeypatch.setenv("RESOLUTION_SOURCE_URL_CONTEXT_ENABLED", "true")
    monkeypatch.setenv("GOOGLE_API_KEY", "key")
    monkeypatch.setattr(resolution_source, "run_url_context_read", reader)
    monkeypatch.setattr(FetchContext, "rung_budget_s", lambda self: budget_s)


class TestThePaidReadIsBoundedByTheRemainingWall:
    """What the rung tells the SDK, and whether the worst case it permits fits the wait."""

    async def test_the_client_ceiling_and_the_attempt_count_come_off_the_budget(self, monkeypatch):
        """The ceiling is the remaining budget less the margin the rung owes the outer wait, so
        the read returns its worker before the provider's own ``wait_for`` fires rather than
        after it — the margin is what makes "the rung returns first" true."""
        reader, calls = _reader()
        _arm(monkeypatch, reader, budget_s=_ROOMY_BUDGET_S)

        result = await _fetch_one(_session(), _URL, {}, FetchContext(query="How many stoppages?"))

        assert result.status == "success"
        assert calls[0]["attempts"] == RESOLUTION_SOURCE_URL_CONTEXT_ATTEMPTS
        assert calls[0]["timeout_ms"] == int((_ROOMY_BUDGET_S - RESOLUTION_SOURCE_RUNG_WALL_MARGIN_S) * 1000)

    async def test_the_read_is_waited_on_for_no_longer_than_the_rung_budget(self, monkeypatch):
        """Without this the arithmetic pin below would go inert: it measures the worst case
        against the budget, which is only the right denominator while the budget is what the
        rung actually waits. Recorded by wrapping ``wait_for`` rather than by reading the source,
        so a change to which value is passed shows up here."""
        waits: list[float | None] = []
        real_wait_for = asyncio.wait_for

        # ASYNC109 wants a caller-supplied deadline to be an `asyncio.timeout` block rather than a
        # `timeout` parameter. This is not a caller: it is a drop-in for `asyncio.wait_for`, whose
        # signature it must match exactly.
        async def _recording_wait_for(awaitable, *, timeout=None):  # noqa: ASYNC109
            waits.append(timeout)
            return await real_wait_for(awaitable, timeout=timeout)

        reader, _calls = _reader()
        _arm(monkeypatch, reader, budget_s=_ROOMY_BUDGET_S)
        monkeypatch.setattr(resolution_source.asyncio, "wait_for", _recording_wait_for)

        result = await _fetch_one(_session(), _URL, {}, FetchContext(query="ask"))

        assert result.status == "success"
        assert _ROOMY_BUDGET_S in waits

    async def test_the_worst_case_read_fits_the_wait_at_the_budget_floor(self, monkeypatch):
        """The invariant that costs money, on the worst budget the rung will still accept.

        ``wait_for`` stops waiting but does not stop the worker, so one billed call can outlive
        the wait — that is the accepted cost. What must never happen is the SDK DISPATCHING a
        further billed request after we stopped waiting, invisible to the ``GEMINI_USAGE`` line
        the read logs on return. So every permitted attempt plus every backoff sleep between them
        has to fit inside the wait.

        At the floor today: one attempt of 13 s inside a 15 s wait. Raising
        ``RESOLUTION_SOURCE_URL_CONTEXT_ATTEMPTS`` to 2 makes it 13 + 1 + 13 s against the same
        15 s, which is what this fails on rather than silently dispatching the second request.

        The per-attempt ceiling is read back off the call rather than recomputed here, so the
        formula stays in one place and this test asserts the invariant instead of the spelling.
        """
        reader, calls = _reader()
        _arm(monkeypatch, reader, budget_s=RESOLUTION_SOURCE_URL_CONTEXT_MIN_BUDGET_S)

        await _fetch_one(_session(), _URL, {}, FetchContext(query="ask"))

        attempts = calls[0]["attempts"]
        per_attempt_timeout_ms = calls[0]["timeout_ms"]
        # Both are integers in their own right: the SDK takes whole milliseconds and a whole
        # attempt count, and a float sneaking into either would be a defect of its own.
        assert isinstance(attempts, int)
        assert isinstance(per_attempt_timeout_ms, int)
        worst_case_ms = attempts * per_attempt_timeout_ms + 1000 * gemini_retry_sleep_allowance_s(attempts)
        assert worst_case_ms <= RESOLUTION_SOURCE_URL_CONTEXT_MIN_BUDGET_S * 1000


class TestATimedOutReadPublishesNothing:
    """The ``except TimeoutError`` branch: the one outcome where we paid and have no answer."""

    async def test_the_direct_result_stands_and_no_model_text_is_published(self, monkeypatch, caplog):
        """A timed-out read must leave the ladder exactly where the direct fetch left it. The
        alternative failure is the dangerous one: half an answer, or an empty ``success``, carrying
        the url_context disclosure under a section captioned primary grading evidence.

        The reader raises rather than the wait genuinely firing, because the floor is 15 s of real
        time and the branch cannot tell the two apart — ``wait_for`` re-raises an inner
        ``TimeoutError`` unchanged.
        """
        reader, calls = _reader(raises=TimeoutError)
        _arm(monkeypatch, reader, budget_s=_ROOMY_BUDGET_S)

        with caplog.at_level(logging.WARNING, logger="metaculus_bot.research.resolution_source"):
            result = await _fetch_one(_session(), _URL, {}, FetchContext(query="ask"))

        assert len(calls) == 1
        assert result.status == "blocked"
        assert "Read via Gemini url_context" not in result.text
        assert result.text == ""
        # The attempt is still recorded and still counted: the rung fired and Google may well
        # have billed it, so this is spend the telemetry has to carry.
        assert result.route == "url_context"
        assert _rung_counts([result])["url_context_reads"] == 1
        assert "url_context read timed out for tracker.example.com" in caplog.text
