"""Neither third-party rung may be handed a URL our own SSRF guard refused.

Wayback and Gemini ``url_context`` are the two rungs whose EGRESS IS NOT OURS: one asks
web.archive.org to fetch the URL, the other asks Google to. ``_fetch_one`` routes every
non-success direct outcome through ``_escalate_unresolved`` — ``ssrf_blocked`` included, since
that function branches only on ``success`` — so the single thing standing between a URL we
refused and a third-party fetch of it is the absence of that status from the two rungs' trigger
sets. Both sets say so in a comment ("WE refused that URL, and handing it to a third-party
fetcher is exactly the bypass the guard exists to prevent") and neither said so in a test.

What that costs if the sets are ever widened was measured rather than reasoned about: adding
``ssrf_blocked`` to ``_WAYBACK_TRIGGER_STATUSES`` makes the ladder issue
``GET https://web.archive.org/web/2026id_/http://169.254.169.254/latest/meta-data/`` — the bot
asking a third party to read the cloud instance-metadata endpoint on its behalf, which is the
whole point of the guard.

Its own module because the invariant is one security property shared by two rungs, rather than a
fact about how either rung behaves; the rungs' own behaviour stays in
``test_resolution_source_wayback_rung.py`` and ``test_resolution_source_url_context_rung.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import get_args

import pytest

from metaculus_bot.research import impersonated_fetch, resolution_source
from metaculus_bot.research.impersonated_fetch import IMPERSONATE_TRIGGER_STATUSES
from metaculus_bot.research.resolution_fetch_result import FetchResult, FetchStatus
from metaculus_bot.research.resolution_source import (
    _URL_CONTEXT_TRIGGER_STATUSES,
    _WAYBACK_TRIGGER_STATUSES,
    FetchContext,
    _fetch_one,
    _impersonate_rung_applies,
)
from tests.resolution_source_fakes import (
    FakeResponse,
    FakeSession,
    _impersonated,
    _prose_page,
    arm_paid_rung,
    fake_impersonated_fetch,
    paid_reader,
)

_NOW = datetime(2026, 9, 4, tzinfo=UTC)

# The two shapes the preflight refuses, both of which reach `_escalate_unresolved` as
# `ssrf_blocked`. The IP literal is rejected without DNS (`_ip_is_disallowed` on a link-local
# address); the hostname is rejected on RESOLUTION, which is the shape a rebinding server or a
# split-horizon internal DNS entry produces and the one a reader is likeliest to think is safe.
_IMDS_URL = "http://169.254.169.254/latest/meta-data/"
_PRIVATE_HOSTNAME_URL = "https://intranet.example.com/dashboard"
_REFUSED_URLS = [_IMDS_URL, _PRIVATE_HOSTNAME_URL]


@pytest.fixture
def _resolve_the_intranet_host_privately(monkeypatch):
    """Override the package's public-DNS stub for the one hostname that must resolve privately.

    The package conftest answers every ``*.example.com`` lookup with 8.8.8.8 so ordinary fixtures
    are not all ``ssrf_blocked``; this test needs exactly that outcome, so it patches
    ``getaddrinfo`` again inside the test (the later patch wins, which the conftest documents).
    """

    def _sync_ainfo(host, port, *args, **kwargs):
        del port, args, kwargs
        address = "10.0.0.7" if host == "intranet.example.com" else "8.8.8.8"
        return [(0, 0, 0, "", (address, 0))]

    monkeypatch.setattr(resolution_source.socket, "getaddrinfo", _sync_ainfo)


@pytest.mark.usefixtures("_resolve_the_intranet_host_privately")
class TestTheArchiveNeverFetchesAUrlWeRefused:
    """The archive rung, armed exactly as production runs it, on a URL the preflight refused."""

    @pytest.fixture(autouse=True)
    def _arm_the_archive(self, monkeypatch):
        """Restore the rung's own trigger set, which this package's conftest empties by default.

        Without this the test proves nothing: an emptied set declines every status, so the
        exclusion under test would hold for a reason production does not have. The module's
        constant OBJECT is restored rather than a copy, so the population asserted on here cannot
        drift from the one prod uses.
        """
        monkeypatch.setattr(resolution_source, "_WAYBACK_TRIGGER_STATUSES", _WAYBACK_TRIGGER_STATUSES)

    def _session(self) -> FakeSession:
        """A session that WOULD serve the archive, keyed by prefix so any snapshot URL matches.

        The refused URL itself gets no handler: it must never be requested, and ``FakeSession``
        raises on an unregistered URL, so a regression that fetched it would fail loudly. The
        archive handler exists so a widened trigger set fails on the assertions below (status,
        request log) rather than on a "no handler" error whose message reads like a broken test.
        """
        return FakeSession(
            {
                "https://web.archive.org/": FakeResponse(
                    200, body=_prose_page("Archived body."), content_type="text/html"
                )
            }
        )

    @pytest.mark.parametrize("url", _REFUSED_URLS)
    async def test_a_refused_url_never_reaches_the_archive(self, url):
        session = self._session()

        result = await _fetch_one(session, url, {}, FetchContext(now=_NOW))

        assert result.status == "ssrf_blocked"
        assert result.route == "direct"
        # No attempt at all, not a skip: the rung declined on the trigger population, which is
        # one gate earlier than any budget or cap check and the only one that cannot be relaxed.
        assert result.rung_attempts == []
        # Not one request of any kind, which subsumes "no web.archive.org request": the refusal
        # lands before the direct GET, so the archive is the only thing that could have fetched.
        assert session.requested == []


@pytest.mark.usefixtures("_resolve_the_intranet_host_privately")
class TestThePaidReaderNeverSeesAUrlWeRefused:
    """The paid rung, armed and funded, on a URL the preflight refused.

    Handing a refused URL to Gemini is the same bypass as handing it to the archive and costs
    money on the way through. Every other gate here is deliberately open (flag on, key present,
    full budget) so the only thing that can decline is the trigger population.
    """

    @pytest.fixture
    def armed_reader(self, monkeypatch) -> list[dict[str, object]]:
        """Flag on, key present, reader faked. Returns the call log the assertion reads.

        Shared arming (``tests/resolution_source_fakes.py``), so "armed" means here exactly what
        it means in the three other modules that drive this rung.
        """
        reader, reads = paid_reader(text="Whatever the model would have said.")
        arm_paid_rung(monkeypatch, reader)
        return reads

    @pytest.mark.parametrize("url", _REFUSED_URLS)
    async def test_a_refused_url_is_never_read_for_money(self, url, armed_reader):
        # No handlers: neither the page nor its robots.txt may be requested, and the robots
        # pre-check goes through the same guarded fetch, so both are refused before any GET.
        session = FakeSession({})

        result = await _fetch_one(session, url, {}, FetchContext(now=_NOW, query="ask"))

        assert result.status == "ssrf_blocked"
        assert result.route == "direct"
        assert result.rung_attempts == []
        assert armed_reader == []
        assert session.requested == []


@pytest.mark.usefixtures("_resolve_the_intranet_host_privately")
class TestTheImpersonatedRetryNeverDialsAUrlWeRefused:
    """The impersonated retry, armed exactly as production runs it, on a URL the preflight refused.

    Its egress IS ours, but its transport is libcurl rather than aiohttp, so the connect-time
    ``FilteringResolver`` never sees it: handing it a URL this module refused would re-dial that
    URL through a client with none of the direct path's guards. The trigger is the one gate that
    cannot be relaxed, and it is a 403 on a ``blocked`` result, which ``ssrf_blocked`` never is.
    """

    @pytest.fixture(autouse=True)
    def _arm_the_retry(self, monkeypatch):
        monkeypatch.setattr(impersonated_fetch, "IMPERSONATE_TRIGGER_STATUSES", IMPERSONATE_TRIGGER_STATUSES)

    @pytest.mark.parametrize("url", _REFUSED_URLS)
    async def test_a_refused_url_is_never_retried_under_impersonation(self, url, monkeypatch):
        calls: list[dict[str, object]] = []
        monkeypatch.setattr(
            resolution_source,
            "fetch_impersonated",
            fake_impersonated_fetch(_impersonated(200, body=_prose_page("Whatever the host served.")), calls),
        )
        session = FakeSession({})

        result = await _fetch_one(session, url, {}, FetchContext(now=_NOW))

        assert result.status == "ssrf_blocked"
        assert result.route == "direct"
        assert result.rung_attempts == []
        assert calls == []
        assert session.requested == []


class TestBothTriggerSetsExcludeOurOwnRefusal:
    """The frozenset populations themselves, so a widening is a test failure rather than a probe.

    Membership rather than whole-set equality: the point is one status that must never be in
    either set, and the sets' positive populations are pinned by the rungs' own behaviour tests.
    """

    def test_ssrf_blocked_is_still_the_spelling_the_guard_emits(self):
        """Both assertions below go inert if the status is ever renamed — they would then be true
        of a token nothing produces, while the renamed status sat unexcluded in both sets."""
        assert "ssrf_blocked" in get_args(FetchStatus)

    def test_the_archive_trigger_set_excludes_it(self):
        assert "ssrf_blocked" not in _WAYBACK_TRIGGER_STATUSES

    def test_the_paid_reader_trigger_set_excludes_it(self):
        assert "ssrf_blocked" not in _URL_CONTEXT_TRIGGER_STATUSES

    @pytest.mark.parametrize("http_status", [None, 301, 403])
    def test_the_impersonate_trigger_excludes_it(self, http_status):
        """Through the predicate rather than a set: the trigger keys on the STATUS being `blocked`
        as well as on the 403, so a refusal carrying a 403 (which nothing produces today) would
        still not fire it."""
        refused = FetchResult(url=_IMDS_URL, status="ssrf_blocked", text="", http_status=http_status, content_type=None)

        assert _impersonate_rung_applies(refused) is False

    @pytest.mark.usefixtures("_resolve_the_intranet_host_privately")
    @pytest.mark.parametrize("url", _REFUSED_URLS)
    async def test_the_preflight_still_refuses_both_probe_urls(self, url):
        """The two rung tests above are only meaningful while these URLs are refused. A preflight
        that started allowing link-local literals, or a DNS stub that stopped answering privately,
        would make both of them pass by never producing ``ssrf_blocked`` at all."""
        assert await resolution_source.is_public_http_url(url) is False
