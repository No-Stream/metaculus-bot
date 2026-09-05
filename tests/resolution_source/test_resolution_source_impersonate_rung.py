"""The impersonated-retry rung: a direct 403 re-dialed with a real browser's TLS fingerprint.

Measured 2026-09-04 from a GitHub Actions runner (`scripts/probes/fetch_diagnostic.py`): four
Akamai-fronted federal hosts answered the bot's own aiohttp client 403 and the same GET through
`curl_cffi` with Chrome impersonation 200, so the refusal was a fingerprint verdict and is
recoverable client-side. The transport (`research/impersonated_fetch.py`) is patched at the
import seam `resolution_source.fetch_impersonated` throughout; the suite's `_block_native_egress`
guard stays armed underneath, so a rung that reached the real transport would fail at teardown.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import UTC, datetime, timedelta

import pytest

from metaculus_bot.constants import (
    DOCUMENT_TEXT_PDF_MAX_BYTES,
    RESOLUTION_SOURCE_HTTP_TIMEOUT,
    RESOLUTION_SOURCE_IMPERSONATE_ENABLED_ENV,
    RESOLUTION_SOURCE_IMPERSONATE_MIN_BUDGET_S,
    RESOLUTION_SOURCE_MAX_RESPONSE_BYTES,
)
from metaculus_bot.research import impersonated_fetch, resolution_source
from metaculus_bot.research.impersonated_fetch import (
    IMPERSONATE_TRIGGER_STATUSES,
    ImpersonateBodyTooLarge,
    ImpersonateDeclined,
    ImpersonateHopRefused,
    ImpersonatePinNotHeld,
    ImpersonateRedirectLimit,
    ImpersonateTransportError,
    ImpersonateUnpinnable,
    impersonation_refused,
)
from metaculus_bot.research.resolution_fetch_result import ROUTE_CAVEATS, FetchResult, FetchStatus
from metaculus_bot.research.resolution_source import (
    _WAYBACK_TRIGGER_STATUSES,
    FetchContext,
    _fetch_one,
    _impersonate_rung_applies,
    _rung_counts,
    format_resolution_sections,
)
from metaculus_bot.research.wayback import wayback_snapshot_url
from scripts.telemetry.markers import parse_log_text
from tests.resolution_source_fakes import (
    _JS_SHELL,
    _RENDERED_PROSE,
    _ROBOTS_URL,
    _URL,
    ROBOTS_ALLOW_ALL,
    FakeResponse,
    FakeSession,
    _impersonated,
    _menu_tree_page,
    _prose_page,
    arm_paid_rung,
    fake_impersonated_fetch,
    paid_reader,
)
from tests.test_document_text import build_text_pdf

_NOW = datetime(2026, 9, 4, tzinfo=UTC)
_SECOND_URL = "https://tracker.example.com/house"
_LOGGER = "metaculus_bot.research.resolution_source"
_META = {
    "run_id": "999",
    "workflow": "tournament",
    "artifact": "research-999",
    "run_date": "2026-09-04T14:00:00Z",
    "log_file": "run.log",
}


def _direct(status: FetchStatus, http_status: int | None, *, url: str = _URL, reason=None) -> FetchResult:
    return FetchResult(
        url=url,
        status=status,
        text="body" if status == "success" else "",
        http_status=http_status,
        content_type="text/html",
        status_reason=reason,
    )


def _refused_page(*, server: str = "AkamaiGHost") -> FakeSession:
    return FakeSession({_URL: FakeResponse(403, body=b"denied", content_type="text/html", headers={"Server": server})})


@pytest.fixture(autouse=True)
def _arm_the_retry(monkeypatch):
    """Restore the transport's trigger set, which this package's conftest empties by default.

    The transport's constant OBJECT is restored rather than a copy, so the trigger population these
    tests assert on cannot drift from the one prod uses.
    """
    monkeypatch.setattr(impersonated_fetch, "IMPERSONATE_TRIGGER_STATUSES", IMPERSONATE_TRIGGER_STATUSES)


def _transport(monkeypatch, answer) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(resolution_source, "fetch_impersonated", fake_impersonated_fetch(answer, calls))
    return calls


def _escalation_lines(caplog) -> list[str]:
    return [message for message in caplog.messages if "RESOLUTION_SOURCE_ESCALATION" in message]


class TestImpersonateRungTrigger:
    """A direct-fetch 403, and nothing else: both halves of the predicate are load-bearing."""

    @pytest.mark.parametrize(
        ("status", "http_status", "reason", "fires"),
        [
            ("blocked", 403, None, True),
            # 406 is a content-negotiation refusal and 429 a throttle: neither is a fingerprint
            # verdict, and a retry against a host that just asked us to slow down could make our
            # position worse.
            ("blocked", 406, None, False),
            ("blocked", 429, None, False),
            # The Metaculus self-reference hop is `blocked` carrying the REDIRECT's status; handing
            # a URL this module refused to a second transport is the bypass the guard prevents.
            ("blocked", 301, None, False),
            ("blocked", 302, None, False),
            # 401 is an authentication requirement no fingerprint changes, and not even `blocked`.
            ("error", 401, None, False),
            ("error", None, None, False),
            ("js_wall", 200, None, False),
            ("no_resolving_content", 200, "thin_page", False),
            ("not_found", 404, None, False),
            ("stale_data", 403, None, False),
            ("empty_body", 200, None, False),
            ("unreadable_document", 200, "no_text_layer", False),
            ("unsupported_type", 200, None, False),
            ("success", 200, None, False),
        ],
    )
    def test_the_trigger_population(self, status, http_status, reason, fires):
        assert _impersonate_rung_applies(_direct(status, http_status, reason=reason)) is fires

    def test_the_trigger_is_read_off_the_transport_at_call_time(self, monkeypatch):
        """What the package conftest relies on to decline the rung for every other test, and what
        keeps gap-fill v2 on the same population: both fetchers read the transport's attribute."""
        monkeypatch.setattr(impersonated_fetch, "IMPERSONATE_TRIGGER_STATUSES", frozenset())

        assert _impersonate_rung_applies(_direct("blocked", 403)) is False

    def test_the_transport_owns_a_403_only_trigger(self):
        assert frozenset({403}) == IMPERSONATE_TRIGGER_STATUSES


class TestImpersonateRungRescue:
    async def test_a_403_page_is_served_from_the_impersonated_retry(self, monkeypatch):
        calls = _transport(monkeypatch, _impersonated(200, body=_prose_page(_RENDERED_PROSE)))
        host_sems: dict[str, asyncio.Semaphore] = {}

        result = await _fetch_one(_refused_page(), _URL, host_sems, FetchContext(now=_NOW))

        assert result.status == "success"
        assert result.route == "impersonate"
        assert "Nebraska Senate polling average" in result.text
        # The impersonated response's own 200, not the direct 403: the bytes came with a 200 and
        # the refusal lives on the escalation line's `from_status`, as a Wayback rescue does.
        assert result.http_status == 200
        assert result.failure_class is None
        assert result.server is None
        (attempt,) = result.rung_attempts
        assert (attempt.rung, attempt.from_status, attempt.outcome, attempt.skipped_reason) == (
            "impersonate",
            "blocked",
            "success",
            "",
        )
        assert attempt.wall_s is not None
        assert _rung_counts([result])["impersonate_attempts"] == 1
        # The transport was handed the direct path's own bounds and the caller's host map: both
        # body caps, so a declared PDF between them is read here as `_resolution_pdf_outcome`
        # would have read it rather than declined as oversized.
        (call,) = calls
        assert call["url"] == _URL
        assert call["host_sems"] is host_sems
        assert call["per_hop_timeout_s"] == RESOLUTION_SOURCE_HTTP_TIMEOUT
        assert call["max_bytes"] == RESOLUTION_SOURCE_MAX_RESPONSE_BYTES
        assert call["document_max_bytes"] == DOCUMENT_TEXT_PDF_MAX_BYTES

    async def test_the_retry_is_bounded_by_the_remaining_wall(self, monkeypatch):
        monkeypatch.setattr(FetchContext, "rung_budget_s", lambda self: 7.5)
        calls = _transport(monkeypatch, _impersonated(200, body=_prose_page(_RENDERED_PROSE)))
        before = resolution_source.time.monotonic()

        await _fetch_one(_refused_page(), _URL, {}, FetchContext(now=_NOW))

        (call,) = calls
        deadline = call["deadline_monotonic_s"]
        assert isinstance(deadline, float)
        assert before + 7.5 <= deadline <= resolution_source.time.monotonic() + 7.5

    async def test_the_rescue_renders_the_impersonate_caveat(self, monkeypatch):
        _transport(monkeypatch, _impersonated(200, body=_prose_page(_RENDERED_PROSE)))

        result = await _fetch_one(_refused_page(), _URL, {}, FetchContext(now=_NOW))
        rendered = format_resolution_sections([result], _NOW)

        assert ROUTE_CAVEATS["impersonate"] in rendered
        assert "Nebraska Senate polling average" in rendered

    async def test_the_rung_fires_with_the_flag_set_on_explicitly(self, monkeypatch):
        monkeypatch.setenv(RESOLUTION_SOURCE_IMPERSONATE_ENABLED_ENV, "true")
        calls = _transport(monkeypatch, _impersonated(200, body=_prose_page(_RENDERED_PROSE)))

        result = await _fetch_one(_refused_page(), _URL, {}, FetchContext(now=_NOW))

        assert len(calls) == 1
        assert result.route == "impersonate"

    async def test_it_runs_on_the_fast_path_with_no_fast_path_skip(self, monkeypatch):
        """One GET against a host that just answered us, exactly the cost of the meta-refresh hop;
        the fast-path gate is reserved for the browser and the paid read."""
        calls = _transport(monkeypatch, _impersonated(200, body=_prose_page(_RENDERED_PROSE)))

        result = await _fetch_one(_refused_page(), _URL, {}, FetchContext(now=_NOW, fast_path=True))

        assert len(calls) == 1
        assert result.route == "impersonate"
        assert result.status == "success"
        counts = _rung_counts([result])
        assert counts["fast_path_skips"] == 0
        assert counts["impersonate_attempts"] == 1


class TestImpersonateRungStillRefused:
    async def test_a_still_403_leaves_the_direct_result_standing_and_memoizes_the_host(self, monkeypatch):
        _transport(monkeypatch, _impersonated(403, body=b"denied", server="AkamaiGHost"))

        result = await _fetch_one(_refused_page(), _URL, {}, FetchContext(now=_NOW))

        # The DIRECT result, diagnostics intact, with the fired rung stamped onto it.
        assert result.status == "blocked"
        assert result.http_status == 403
        assert result.failure_class == "http_403"
        assert result.server == "akamaighost"
        assert result.route == "impersonate"
        assert [(a.rung, a.from_status, a.outcome) for a in result.rung_attempts] == [
            ("impersonate", "blocked", "blocked")
        ]
        assert impersonation_refused(_URL) is True

    @pytest.mark.parametrize("status", [406, 429])
    async def test_the_other_block_shapes_memoize_too(self, monkeypatch, status):
        _transport(monkeypatch, _impersonated(status, body=b""))

        result = await _fetch_one(_refused_page(), _URL, {}, FetchContext(now=_NOW))

        assert result.status == "blocked"
        assert [a.outcome for a in result.rung_attempts] == ["blocked"]
        assert impersonation_refused(_URL) is True

    async def test_the_memo_saves_the_second_url_on_the_host(self, monkeypatch, caplog):
        calls = _transport(monkeypatch, _impersonated(403, body=b"denied"))
        session = FakeSession(
            {
                _URL: FakeResponse(403, body=b"denied", content_type="text/html"),
                _SECOND_URL: FakeResponse(403, body=b"denied", content_type="text/html"),
            }
        )

        with caplog.at_level(logging.INFO, logger=_LOGGER):
            first = await _fetch_one(session, _URL, {}, FetchContext(now=_NOW))
            second = await _fetch_one(session, _SECOND_URL, {}, FetchContext(now=_NOW))
            resolution_source._log_fetch_outcome_markers(1, [second])

        assert len(calls) == 1
        assert first.route == "impersonate"
        assert second.route == "direct"
        assert [(a.rung, a.skipped_reason) for a in second.rung_attempts] == [
            ("impersonate", "impersonate_host_refused")
        ]
        assert _rung_counts([first, second])["impersonate_host_refused_skips"] == 1
        # A skip emits no escalation line: the memo hit rides the counts.
        assert _escalation_lines(caplog) == []

    async def test_a_block_answered_by_a_redirect_target_memoizes_both_hosts(self, monkeypatch, caplog):
        """The transport follows redirects itself, so the block can come from a later hop's netloc.
        Memoizing only the host DIALED would ban a host that never refused us and keep dialing the
        one that did, so the rung hands the transport both URLs and the memo covers both; the log
        line names the host that answered."""
        answered = "https://edge.example.net/denied"
        _transport(monkeypatch, _impersonated(403, body=b"denied", url=answered))

        with caplog.at_level(logging.INFO, logger=_LOGGER):
            result = await _fetch_one(_refused_page(), _URL, {}, FetchContext(now=_NOW))

        assert result.status == "blocked"
        assert [a.outcome for a in result.rung_attempts] == ["blocked"]
        assert impersonation_refused(answered) is True
        assert impersonation_refused(_URL) is True
        assert any("answered 403 by edge.example.net (blocked)" in message for message in caplog.messages)

    @pytest.mark.parametrize(("status", "outcome"), [(404, "not_found"), (410, "not_found"), (503, "error")])
    async def test_a_non_block_answer_stamps_its_own_outcome_and_does_not_memoize(self, monkeypatch, status, outcome):
        """`not_found` and `error` are in the rung's outcome domain too: `_NON_OK_FETCH_STATUS` maps
        404 and 410, and every other non-200 falls through to its `error` default. Neither says
        anything about the host's view of our fingerprint, so neither writes the memo."""
        _transport(monkeypatch, _impersonated(status, body=b""))

        result = await _fetch_one(_refused_page(), _URL, {}, FetchContext(now=_NOW))

        assert result.status == "blocked"
        assert [a.outcome for a in result.rung_attempts] == [outcome]
        assert impersonation_refused(_URL) is False

    async def test_a_404_under_impersonation_does_not_memoize(self, monkeypatch):
        """The path is gone, which says nothing about the host's view of our fingerprint, so the
        next cited URL on the host still earns its own dial (the contract the parametrized
        outcome test above does not cover)."""
        calls = _transport(monkeypatch, _impersonated(404, body=b""))
        session = FakeSession(
            {
                _URL: FakeResponse(403, body=b"denied", content_type="text/html"),
                _SECOND_URL: FakeResponse(403, body=b"denied", content_type="text/html"),
            }
        )

        first = await _fetch_one(session, _URL, {}, FetchContext(now=_NOW))
        second = await _fetch_one(session, _SECOND_URL, {}, FetchContext(now=_NOW))

        assert first.status == "blocked"
        assert [a.outcome for a in first.rung_attempts] == ["not_found"]
        assert impersonation_refused(_URL) is False
        assert len(calls) == 2
        assert second.route == "impersonate"

    async def test_a_200_that_classifies_as_unreadable_declines_and_leaves_the_paid_rung_reachable(self, monkeypatch):
        """Replacing `blocked` with `js_wall` on the page's record would change the fetch line
        without giving any later rung a way to act on it (the browser block keys on `direct`), so
        the rung stamps its own verdict on the attempt and leaves the direct result standing."""
        _transport(monkeypatch, _impersonated(200, body=_JS_SHELL))
        reader, reads = paid_reader()
        arm_paid_rung(monkeypatch, reader)
        session = FakeSession(
            {
                _URL: FakeResponse(403, body=b"denied", content_type="text/html"),
                _ROBOTS_URL: FakeResponse(200, body=ROBOTS_ALLOW_ALL, content_type="text/plain"),
            }
        )

        result = await _fetch_one(session, _URL, {}, FetchContext(now=_NOW, query="ask"))

        assert [call["url"] for call in reads] == [_URL]
        assert result.status == "success"
        assert result.route == "url_context"
        assert [(a.rung, a.from_status, a.outcome) for a in result.rung_attempts] == [
            ("impersonate", "blocked", "js_wall"),
            ("url_context", "blocked", "success"),
        ]
        assert impersonation_refused(_URL) is False

    async def test_with_nothing_after_it_the_unreadable_200_leaves_blocked_standing(self, monkeypatch):
        _transport(monkeypatch, _impersonated(200, body=_JS_SHELL))

        result = await _fetch_one(_refused_page(), _URL, {}, FetchContext(now=_NOW))

        assert result.status == "blocked"
        assert result.route == "impersonate"
        assert [a.outcome for a in result.rung_attempts] == ["js_wall"]

    async def test_a_withheld_impersonated_body_is_counted_on_the_direct_result(self, monkeypatch):
        """`chrome_metric_withholds` counts a withhold anywhere on the URL's ladder, and a 403 direct
        fetch carried no body for the metric to withhold, so the impersonated body's withhold is the
        only one this URL can produce. It is stamped on the direct result (as the rendered rung
        stamps its DOM's withhold), which `_fetch_one` carries onto whatever the ladder leaves
        standing, here the direct `blocked`; the discarded rung result would otherwise take the
        fact with it."""
        _transport(monkeypatch, _impersonated(200, body=_menu_tree_page()))

        result = await _fetch_one(_refused_page(), _URL, {}, FetchContext(now=_NOW))

        assert result.status == "blocked"
        assert result.route == "impersonate"
        assert [(a.outcome, a.skipped_reason) for a in result.rung_attempts] == [("no_resolving_content", "")]
        assert result.chrome_metric_withheld is True
        counts = _rung_counts([result])
        assert counts["chrome_metric_withholds"] == 1
        assert counts["chrome_metric_withholds_rescued"] == 0


class TestImpersonateRungSkips:
    async def test_the_kill_switch_declines_before_anything_is_dialed(self, monkeypatch, caplog):
        monkeypatch.setenv(RESOLUTION_SOURCE_IMPERSONATE_ENABLED_ENV, "false")
        calls = _transport(monkeypatch, _impersonated(200, body=_prose_page(_RENDERED_PROSE)))

        with caplog.at_level(logging.INFO, logger=_LOGGER):
            result = await _fetch_one(_refused_page(), _URL, {}, FetchContext(now=_NOW))
            resolution_source._log_fetch_outcome_markers(1, [result])

        assert calls == []
        assert result.status == "blocked"
        assert result.route == "direct"
        assert [(a.rung, a.skipped_reason) for a in result.rung_attempts] == [("impersonate", "impersonate_disabled")]
        counts = _rung_counts([result])
        assert counts["impersonate_disabled_skips"] == 1
        assert counts["impersonate_attempts"] == 0
        assert _escalation_lines(caplog) == []

    async def test_the_budget_floor_declines_before_anything_is_dialed(self, monkeypatch, caplog):
        monkeypatch.setattr(
            FetchContext, "rung_budget_s", lambda self: RESOLUTION_SOURCE_IMPERSONATE_MIN_BUDGET_S - 0.5
        )
        calls = _transport(monkeypatch, _impersonated(200, body=_prose_page(_RENDERED_PROSE)))

        with caplog.at_level(logging.INFO, logger=_LOGGER):
            result = await _fetch_one(_refused_page(), _URL, {}, FetchContext(now=_NOW))
            resolution_source._log_fetch_outcome_markers(1, [result])

        assert calls == []
        assert result.route == "direct"
        assert [(a.rung, a.skipped_reason) for a in result.rung_attempts] == [("impersonate", "wall_budget")]
        counts = _rung_counts([result])
        assert counts["rung_budget_skips"] == 1
        assert counts["impersonate_budget_skips"] == 1
        assert _escalation_lines(caplog) == []

    async def test_an_unpinnable_host_is_its_own_skip_on_the_attempt_already_started(self, monkeypatch, caplog):
        """Near-impossible in practice (the direct fetch resolved this host through the filtering
        resolver moments earlier), so a nonzero count means DNS disagreed with itself."""
        _transport(monkeypatch, ImpersonateUnpinnable("tracker.example.com will not pin"))

        with caplog.at_level(logging.INFO, logger=_LOGGER):
            result = await _fetch_one(_refused_page(), _URL, {}, FetchContext(now=_NOW))
            resolution_source._log_fetch_outcome_markers(1, [result])

        assert result.status == "blocked"
        assert result.route == "direct"
        assert [(a.rung, a.skipped_reason) for a in result.rung_attempts] == [("impersonate", "impersonate_unpinnable")]
        counts = _rung_counts([result])
        assert counts["impersonate_unpinnable_skips"] == 1
        assert counts["impersonate_attempts"] == 0
        assert _escalation_lines(caplog) == []

    async def test_a_transport_decline_is_a_fired_attempt_that_left_blocked_standing(self, monkeypatch, caplog):
        _transport(monkeypatch, ImpersonateTransportError(failure_class="tls", exc="SSLError"))

        with caplog.at_level(logging.INFO, logger=_LOGGER):
            result = await _fetch_one(_refused_page(), _URL, {}, FetchContext(now=_NOW))
            resolution_source._log_fetch_outcome_markers(1, [result])

        assert result.status == "blocked"
        assert result.route == "impersonate"
        assert [(a.rung, a.outcome, a.skipped_reason) for a in result.rung_attempts] == [("impersonate", "blocked", "")]
        assert _rung_counts([result])["impersonate_attempts"] == 1
        assert impersonation_refused(_URL) is False
        (line,) = _escalation_lines(caplog)
        assert "from_status=blocked rung=impersonate outcome=blocked" in line
        # A transport failure is a fact about the host, logged at INFO as the direct path's own are.
        (record,) = [r for r in caplog.records if "failed in transport" in r.getMessage()]
        assert record.levelno == logging.INFO
        assert "failure_class=tls" in record.getMessage()
        assert "exc=SSLError" in record.getMessage()

    @pytest.mark.parametrize(
        "decline",
        [
            ImpersonatePinNotHeld(_URL, expected_ip="93.184.216.34", actual_ip="10.0.0.8"),
            ImpersonateHopRefused("ssrf_blocked", hop_url="http://10.0.0.8/status", from_url=_URL),
            ImpersonateBodyTooLarge(_URL, bytes_read=6_000_000, max_bytes=RESOLUTION_SOURCE_MAX_RESPONSE_BYTES),
            ImpersonateRedirectLimit(_URL, final_url=f"{_URL}?hop=6"),
        ],
        ids=lambda decline: type(decline).__name__,
    )
    async def test_a_guard_decline_is_a_fired_attempt_logged_at_warning(self, monkeypatch, caplog, decline):
        """The generic `except ImpersonateDeclined` branch: a pin that did not hold, a refused
        redirect hop, an oversized body and a redirect chain past the cap are each a FIRED attempt
        that leaves the direct `blocked` standing and the host un-memoized (nothing was learned
        about the host's view of our fingerprint), logged at WARNING because each is a guard or a
        cap firing rather than the host's own behaviour."""
        assert isinstance(decline, ImpersonateDeclined)
        _transport(monkeypatch, decline)

        with caplog.at_level(logging.INFO, logger=_LOGGER):
            result = await _fetch_one(_refused_page(), _URL, {}, FetchContext(now=_NOW))
            resolution_source._log_fetch_outcome_markers(1, [result])

        assert result.status == "blocked"
        assert result.route == "impersonate"
        assert [(a.rung, a.outcome, a.skipped_reason) for a in result.rung_attempts] == [("impersonate", "blocked", "")]
        assert _rung_counts([result])["impersonate_attempts"] == 1
        assert impersonation_refused(_URL) is False
        (line,) = _escalation_lines(caplog)
        assert "from_status=blocked rung=impersonate outcome=blocked" in line
        (record,) = [r for r in caplog.records if "produced nothing" in r.getMessage()]
        assert record.levelno == logging.WARNING
        assert type(decline).__name__ in record.getMessage()


class TestImpersonateRungLadderPosition:
    """Between the direct fetch and the archive: a live page beats a stale capture, and a rescue
    saves the paid read on that URL entirely."""

    def _session(self, *, archive_serves: bool) -> FakeSession:
        snapshot = f"https://web.archive.org/web/{(_NOW - timedelta(days=2)).strftime('%Y%m%d%H%M%S')}id_/{_URL}"
        archive = (
            FakeResponse(302, headers={"Location": snapshot})
            if archive_serves
            else FakeResponse(404, body=b"", content_type="text/html")
        )
        return FakeSession(
            {
                _URL: FakeResponse(403, body=b"denied", content_type="text/html"),
                wayback_snapshot_url(_URL, now=_NOW): archive,
                snapshot: FakeResponse(200, body=_prose_page(_RENDERED_PROSE), content_type="text/html"),
                _ROBOTS_URL: FakeResponse(200, body=ROBOTS_ALLOW_ALL, content_type="text/plain"),
            }
        )

    @pytest.fixture(autouse=True)
    def _arm_the_rungs_behind_it(self, monkeypatch):
        monkeypatch.setattr(resolution_source, "_WAYBACK_TRIGGER_STATUSES", _WAYBACK_TRIGGER_STATUSES)

    async def test_a_rescue_reaches_neither_the_archive_nor_the_paid_reader(self, monkeypatch):
        _transport(monkeypatch, _impersonated(200, body=_prose_page(_RENDERED_PROSE)))
        reader, reads = paid_reader()
        arm_paid_rung(monkeypatch, reader)
        session = self._session(archive_serves=True)

        result = await _fetch_one(session, _URL, {}, FetchContext(now=_NOW, query="ask"))

        assert result.route == "impersonate"
        assert [a.rung for a in result.rung_attempts] == ["impersonate"]
        assert not any(request.startswith("https://web.archive.org/") for request in session.requested)
        assert reads == []

    async def test_a_failed_retry_still_reaches_both(self, monkeypatch):
        _transport(monkeypatch, _impersonated(403, body=b"denied"))
        reader, reads = paid_reader()
        arm_paid_rung(monkeypatch, reader)
        session = self._session(archive_serves=False)

        result = await _fetch_one(session, _URL, {}, FetchContext(now=_NOW, query="ask"))

        assert [a.rung for a in result.rung_attempts] == ["impersonate", "wayback", "url_context"]
        assert any(request.startswith("https://web.archive.org/") for request in session.requested)
        assert [call["url"] for call in reads] == [_URL]
        assert result.route == "url_context"

    async def test_a_failed_retry_lets_a_fresh_capture_serve_the_page(self, monkeypatch):
        _transport(monkeypatch, _impersonated(403, body=b"denied"))
        session = self._session(archive_serves=True)

        result = await _fetch_one(session, _URL, {}, FetchContext(now=_NOW))

        assert result.route == "wayback"
        assert result.status == "success"
        assert [(a.rung, a.outcome) for a in result.rung_attempts] == [
            ("impersonate", "blocked"),
            ("wayback", "success"),
        ]


class TestImpersonateRungDialsTheLandingUrl:
    """The retry dials the hop that ANSWERED 403 (`direct.url`), which is the URL the host actually
    refused, while the attempt stays keyed on the cited URL the escalation line names."""

    _FINAL = "https://www.tracker.example.com/senate"

    def _redirected_session(self) -> FakeSession:
        return FakeSession(
            {
                _URL: FakeResponse(302, headers={"Location": self._FINAL}),
                self._FINAL: FakeResponse(403, body=b"denied", content_type="text/html"),
            }
        )

    async def test_a_redirected_direct_fetch_hands_the_transport_its_final_url(self, monkeypatch):
        calls = _transport(monkeypatch, _impersonated(200, body=_prose_page(_RENDERED_PROSE), url=self._FINAL))

        result = await _fetch_one(self._redirected_session(), _URL, {}, FetchContext(now=_NOW))

        assert [call["url"] for call in calls] == [self._FINAL]
        assert result.status == "success"
        assert result.route == "impersonate"
        # The classifier's base is the URL the response was read from, as the direct path's last
        # hop already is, so the published section names the real document.
        assert result.url == self._FINAL
        (attempt,) = result.rung_attempts
        assert attempt.url == _URL

    @pytest.mark.parametrize(
        "landed",
        ["https://www.metaculus.com/questions/999/", "http://10.0.0.8/status"],
    )
    async def test_a_landing_the_guard_refuses_is_a_decline_with_no_attempt(self, monkeypatch, caplog, landed):
        """Unreachable through `_fetch_direct`, whose every hop is vetted, so it is driven with a
        hand-built direct result: the URL the transport dials is decided here, so the refusal
        lives here too, and it declines before any attempt is opened."""
        calls = _transport(monkeypatch, _impersonated(200, body=_prose_page(_RENDERED_PROSE)))
        direct = FetchResult(url=landed, status="blocked", text="", http_status=403, content_type="text/html")
        ctx = FetchContext(now=_NOW)

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            result = await resolution_source._impersonate_rung(_URL, direct, host_sems={}, ctx=ctx)

        assert result is None
        assert calls == []
        assert ctx.rungs == []
        assert [message for message in caplog.messages if "not re-dialing" in message]


class TestImpersonateRungBodyClassification:
    """A rescued body goes through the SAME bytes-level classification as a directly-fetched one,
    on every content-type branch: HTML, a document, and the raw text family."""

    async def test_a_pdf_rescue_keeps_the_pdf_local_route_and_caveat(self, monkeypatch, caplog):
        """The bls.gov `wkstp.pdf` case: one of the four measured recoverable URLs is a document.
        `route` is the last rung that FIRED and the local read is what the text came from, so the
        result reads `pdf_local` (the accounting a meta-refresh hop onto a PDF already produces)
        and the forecaster sees the passage-selection caveat; both rungs stay on the record."""
        _transport(
            monkeypatch,
            _impersonated(
                200,
                body=build_text_pdf([["Hospitalizations reported: 922", "Deaths reported: 2 as of August 24"]]),
                content_type="application/pdf",
            ),
        )

        with caplog.at_level(logging.INFO, logger=_LOGGER):
            result = await _fetch_one(
                _refused_page(), _URL, {}, FetchContext(now=_NOW, query="hospitalizations reported")
            )
            resolution_source._log_fetch_outcome_markers(1, [result])

        assert result.status == "success"
        assert result.route == "pdf_local"
        assert "922" in result.text
        assert [(a.rung, a.from_status, a.outcome) for a in result.rung_attempts] == [
            ("impersonate", "blocked", "success"),
            ("pdf_local", "unsupported_type", "success"),
        ]
        counts = _rung_counts([result])
        assert counts["impersonate_attempts"] == 1
        assert counts["pdf_documents_read"] == 1
        rendered = format_resolution_sections([result], _NOW)
        assert ROUTE_CAVEATS["pdf_local"] in rendered
        assert ROUTE_CAVEATS["impersonate"] not in rendered
        assert [rung for line in _escalation_lines(caplog) for rung in re.findall(r"rung=(\S+)", line)] == [
            "impersonate",
            "pdf_local",
        ]

    async def test_a_body_declared_pdf_that_is_not_one_is_unsupported_and_declines(self, monkeypatch):
        _transport(monkeypatch, _impersonated(200, body=b"<html>not a document</html>", content_type="application/pdf"))

        result = await _fetch_one(_refused_page(), _URL, {}, FetchContext(now=_NOW))

        assert result.status == "blocked"
        assert [a.outcome for a in result.rung_attempts] == ["unsupported_type"]

    @pytest.mark.parametrize(
        ("content_type", "body", "expected"),
        [
            ("application/json", b'{"stoppages": 12, "as_of": "2026-08-28"}', '"stoppages": 12'),
            ("text/csv", b'date,count\n2026-08-01,<a href="/x">11</a>\n2026-08-02,12\n', "2026-08-02,12"),
            ("text/plain", b"Major work stoppages beginning in 2026: 12 through August.", "12 through August"),
        ],
    )
    async def test_a_raw_body_rescue_goes_through_the_text_path(self, monkeypatch, content_type, body, expected):
        _transport(monkeypatch, _impersonated(200, body=body, content_type=content_type))

        result = await _fetch_one(_refused_page(), _URL, {}, FetchContext(now=_NOW))

        assert result.status == "success"
        assert result.route == "impersonate"
        assert expected in result.text
        # The CSV branch strips the allow-listed markup exactly as the direct path does.
        assert "<a href" not in result.text

    async def test_a_vacuous_raw_body_is_refused_and_declines(self, monkeypatch):
        _transport(monkeypatch, _impersonated(200, body=b"   \n", content_type="application/json"))

        result = await _fetch_one(_refused_page(), _URL, {}, FetchContext(now=_NOW))

        assert result.status == "blocked"
        assert [a.outcome for a in result.rung_attempts] == ["empty_body"]

    async def test_an_empty_content_type_is_sniffed_like_the_direct_path(self, monkeypatch):
        """No header at all routes through the document branch and its `%PDF-` check."""
        _transport(monkeypatch, _impersonated(200, body=b"\x89PNG\r\n\x1a\nbinary", content_type=""))

        result = await _fetch_one(_refused_page(), _URL, {}, FetchContext(now=_NOW))

        assert result.status == "blocked"
        assert [a.outcome for a in result.rung_attempts] == ["unsupported_type"]

    @pytest.mark.parametrize(
        ("content_type", "body"),
        [
            ("text/html; charset=utf-8", _prose_page(_RENDERED_PROSE)),
            ("text/html", _JS_SHELL),
            ("text/html", _menu_tree_page()),
            ("application/json", b'{"stoppages": 12, "as_of": "2026-08-28"}'),
            ("application/json", b"   \n"),
            ("text/csv", b'date,count\n2026-08-01,<a href="/x">11</a>\n2026-08-02,12\n'),
            ("text/plain", b"Major work stoppages beginning in 2026: 12 through August."),
            ("image/png", b"\x89PNG\r\n\x1a\nbinary"),
            ("application/octet-stream", b"\x89PNG\r\n\x1a\nbinary"),
            ("", b"\x89PNG\r\n\x1a\nbinary"),
            ("application/pdf", build_text_pdf([["Hospitalizations reported: 922", "Deaths reported: 2"]])),
            ("application/pdf", b"<html>not a document</html>"),
            ("application/octet-stream", build_text_pdf([["Hospitalizations reported: 922"]])),
        ],
        ids=lambda value: value if isinstance(value, str) else f"{len(value)}B",
    )
    async def test_the_two_routers_agree_on_every_body_shape(self, content_type, body):
        """The three-way content-type router is stated twice, in `_resolution_response_outcome` for
        the aiohttp path and in `_impersonated_body_outcome` for this rung, off the same vocabularies
        in the same order. Adding a token to a shared vocabulary propagates; adding or reordering a
        BRANCH does not, so the two are pinned equal on every body shape, the pending-document case
        resolved through `_finish_document` on both sides, down to the `pdf_local` attempt each
        opens on its own context. Equal on every field a classification decides (`_shape`); the
        attempts are compared by rung and trigger because their wall times cannot be."""
        direct_ctx = FetchContext(now=_NOW, query="hospitalizations reported")
        impersonated_ctx = FetchContext(now=_NOW, query="hospitalizations reported")

        via_direct = await resolution_source._resolution_response_outcome(
            FakeResponse(200, body=body, content_type=content_type), _URL, direct_ctx
        )
        if isinstance(via_direct, resolution_source._PendingDocument):
            via_direct = await resolution_source._finish_document(via_direct, direct_ctx)
        via_impersonated = await resolution_source._impersonated_body_outcome(
            _impersonated(200, body=body, content_type=content_type), impersonated_ctx
        )

        assert isinstance(via_direct, FetchResult), "no fixture here carries a meta-refresh hop"
        assert self._shape(via_direct) == self._shape(via_impersonated)
        assert [(a.rung, a.from_status) for a in direct_ctx.rungs] == [
            (a.rung, a.from_status) for a in impersonated_ctx.rungs
        ]

    @staticmethod
    def _shape(result: FetchResult) -> tuple[object, ...]:
        return (
            result.status,
            result.http_status,
            result.status_reason,
            result.text,
            result.content_type,
            result.chrome_metric_withheld,
            result.precision_rescued,
            result.unreadable_embeds,
            result.datawrapper_charts,
            result.server,
            result.failure_class,
            result.exc,
        )


class TestImpersonateRungMarkerLines:
    """The two existing marker lines carry the whole record; no new spec was added."""

    def _harvest(self, caplog) -> tuple[dict, list[dict]]:
        text = "\n".join(f"2026-09-04 14:00:00,000 - {_LOGGER} - INFO - {message}" for message in caplog.messages)
        harvested = parse_log_text(text + "\n", **_META)
        (fetch,) = harvested["resolution_source_fetch"]
        return fetch, harvested["resolution_source_escalation"]

    async def test_a_rescue_is_one_fetch_line_and_one_escalation_line(self, monkeypatch, caplog):
        _transport(monkeypatch, _impersonated(200, body=_prose_page(_RENDERED_PROSE)))

        with caplog.at_level(logging.INFO, logger=_LOGGER):
            result = await _fetch_one(_refused_page(), _URL, {}, FetchContext(now=_NOW))
            resolution_source._log_fetch_outcome_markers(44211, [result])

        assert (
            f"RESOLUTION_SOURCE_FETCH: question=44211 url={_URL} status=ok http=200 embeds=none route=impersonate"
            in caplog.messages
        )
        (line,) = _escalation_lines(caplog)
        assert re.fullmatch(
            rf"RESOLUTION_SOURCE_ESCALATION: question=44211 url={re.escape(_URL)} "
            r"from_status=blocked rung=impersonate outcome=success wall_s=\d+\.\d\d",
            line,
        )
        fetch, escalations = self._harvest(caplog)
        assert (fetch["status"], fetch["http"], fetch["route"], fetch["failure_class"]) == (
            "ok",
            200,
            "impersonate",
            None,
        )
        (escalation,) = escalations
        assert (escalation["from_status"], escalation["rung"], escalation["outcome"]) == (
            "blocked",
            "impersonate",
            "success",
        )
        assert isinstance(escalation["wall_s"], float)

    async def test_a_failed_attempt_keeps_the_direct_diagnostics_on_the_fetch_line(self, monkeypatch, caplog):
        _transport(monkeypatch, _impersonated(403, body=b"denied", server="AkamaiGHost"))

        with caplog.at_level(logging.INFO, logger=_LOGGER):
            result = await _fetch_one(_refused_page(), _URL, {}, FetchContext(now=_NOW))
            resolution_source._log_fetch_outcome_markers(44211, [result])

        assert (
            f"RESOLUTION_SOURCE_FETCH: question=44211 url={_URL} status=blocked http=403 embeds=none "
            "route=impersonate failure_class=http_403 server=akamaighost"
        ) in caplog.messages
        fetch, escalations = self._harvest(caplog)
        assert (fetch["status"], fetch["http"], fetch["route"]) == ("blocked", 403, "impersonate")
        assert (fetch["failure_class"], fetch["server"]) == ("http_403", "akamaighost")
        (escalation,) = escalations
        assert (escalation["from_status"], escalation["rung"], escalation["outcome"]) == (
            "blocked",
            "impersonate",
            "blocked",
        )
