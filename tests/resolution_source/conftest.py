"""Directory-scoped fixtures for the resolution-source tests.

The autouse DNS stub is why these live in a conftest rather than being imported into each
module: every hostname in these tests is under ``example.com`` and the SSRF guard resolves
it for real, so a module that forgot the stub would see every fetch turn into
``ssrf_blocked`` instead of failing loudly. The plain page builders and the fake aiohttp
session are ordinary imports from ``tests/resolution_source_fakes.py``.
"""

from __future__ import annotations

import asyncio

import pytest

from metaculus_bot.research import resolution_source
from metaculus_bot.research.http_fetch import reset_host_semaphores, reset_pdf_parse_semaphore
from metaculus_bot.research.robots_policy import reset_robots_cache
from tests.resolution_source_fakes import _INFOGRAM_EMBED_MARKUP, _embed_shell_page


@pytest.fixture(autouse=True)
def _reset_shared_gates():
    """Drop the process-wide gates and caches around every test in this package.

    All three deliberately outlive one provider call. The per-host semaphore map and the
    PDF-parse gate are what make politeness and the parse bound hold across concurrent
    questions, so without a reset a permit one test left held would gate another and a
    serialization assertion would pass or fail on test order. They self-heal across event
    loops too, but resetting is cheap and states the intent.

    The robots.txt cache is the one whose leak is silent rather than merely order-dependent:
    it is keyed by HOST, and every test in this package fetches ``tracker.example.com``, so a
    test that registers a ``Disallow`` for Google-Extended would otherwise decline the paid
    rung in every later test on that host — a rung that never ran, asserted as a rung that
    declined. Reset here rather than in each class's setup, which is where it used to live and
    where a new module has no way to know it was needed.
    """
    reset_host_semaphores()
    reset_pdf_parse_semaphore()
    reset_robots_cache()
    yield
    reset_host_semaphores()
    reset_pdf_parse_semaphore()
    reset_robots_cache()


@pytest.fixture(autouse=True)
def _decline_the_browser_rung(monkeypatch):
    """Decline the headless-Chromium escalation for every test in this package by default.

    Nine fixtures here classify as ``js_wall`` or ``thin_page``, which is exactly what sends the
    ladder to ``render_page``. The network guarantee itself is the suite's ``_block_native_egress``
    (``tests/conftest.py``), which refuses ``BrowserType.launch`` for every test; this fixture is
    what keeps that refusal from being the OUTCOME. Without it each of those nine tests spawns
    Playwright's node driver, reaches the refused launch, has ``render_page`` swallow the refusal
    into its ``None`` "declined" signal, and then fails at teardown on the attempt the guard
    recorded (measured 2026-09-03: 344 passed, 9 teardown errors). Declining one call earlier
    keeps them deterministic and driverless.

    ``None`` is the transport's own signal, so the ladder degrades exactly as it does on a runner
    where the ``continue-on-error`` Chromium install failed — which is also what keeps every
    pre-ladder expectation in this package intact, minus the one ``renderer_unavailable`` skip
    the attempt now records. Tests that exercise the rung monkeypatch
    ``resolution_source.render_page`` again in the test body; that later patch wins.
    """

    async def _declined(url: str, *, host_gate, goto_timeout_ms: int = 0, harvest_json: bool = False) -> None:
        """The transport's declined signal. Returns None implicitly; ruff owns that spelling."""
        del url, host_gate, goto_timeout_ms, harvest_json
        # A real yield point, so the stub schedules like the browser rung it stands in for.
        await asyncio.sleep(0)

    monkeypatch.setattr(resolution_source, "render_page", _declined)


@pytest.fixture(autouse=True)
def _decline_the_wayback_rung(monkeypatch):
    """Empty the archive rung's trigger set for every test in this package by default.

    A convenience rather than a containment, unlike the browser fixture above: nothing here can
    reach the network in the first place. The rung fires on ``blocked`` / ``error`` /
    ``not_found``, the outcome dozens of these tests deliberately produce, and every one of them
    drives a ``FakeSession``, so an unwanted fire issues no request at all — the archive URL has
    no handler and ``FakeSession.get`` raises ``AssertionError: no handler for URL
    https://web.archive.org/...`` mid-ladder. That error is outside the hop's
    ``(TimeoutError, aiohttp.ClientError)`` catch, so it surfaces as a loud failure of an
    assertion the test never meant to make, which is the reason to decline rather than a network
    risk. Emptying the trigger set declines before the rung looks at anything, so it records no
    attempt, claims no ``route`` and issues no request, keeping every pre-ladder expectation in
    this package intact without adding an archive handler to dozens of tests.

    Tests that exercise the rung restore the module's OWN constant object (imported, so the two
    cannot drift), which reads as "these statuses trigger the archive". The trigger population is
    pinned in ``TestWaybackRung``, and its ``ssrf_blocked`` EXCLUSION — which this fixture would
    otherwise hide from every test in the package — in
    ``test_resolution_source_third_party_rung_ssrf.py``.
    """
    monkeypatch.setattr(resolution_source, "_WAYBACK_TRIGGER_STATUSES", frozenset())


@pytest.fixture(autouse=True)
def _stub_public_dns(monkeypatch):
    """Every test hostname in this package uses ``*.example.com``, an RFC-2606
    reserved TLD with no real DNS. Without a stub, the SSRF guard's
    ``getaddrinfo`` call raises ``gaierror`` and every fetch becomes
    ``ssrf_blocked``. Return a public IP by default; tests that need private-
    IP behavior monkeypatch ``getaddrinfo`` again inside the test body (that
    later patch wins).
    """

    def _sync_ainfo(host, port, *args, **kwargs):
        del host, port, args, kwargs
        return [(0, 0, 0, "", ("8.8.8.8", 0))]

    monkeypatch.setattr(resolution_source.socket, "getaddrinfo", _sync_ainfo)


@pytest.fixture
def article_html() -> bytes:
    """A ~2 KB article-shaped HTML fixture. Trafilatura should extract the
    <article> body while dropping nav/footer chrome.
    """
    return (
        b"<!doctype html><html><head><title>Sample Report</title></head><body>"
        b"<nav>Home | About | Contact</nav>"
        b"<article><h1>Latest CPI Reading</h1>"
        b"<p>The Bureau of Labor Statistics reported a 3.2 percent annual "
        b"increase in the Consumer Price Index for the twelve months ending "
        b"in September 2026. Core CPI, which excludes food and energy, rose "
        b"3.4 percent over the same period. Housing costs contributed the "
        b"largest share of the monthly increase, while used-car prices fell "
        b"slightly.</p>"
        b"<p>Analysts had projected a 3.3 percent headline reading. The "
        b"lower-than-expected result was welcomed by markets and reinforced "
        b"expectations that the Federal Reserve would hold rates steady at "
        b"its next meeting.</p></article>"
        b"<footer>&copy; 2026 Example News</footer></body></html>"
    )


@pytest.fixture
def infogram_shell_html() -> bytes:
    return _embed_shell_page(_INFOGRAM_EMBED_MARKUP)


@pytest.fixture
def tracker_with_infogram_html() -> bytes:
    """The 44554 shape: real forecast prose (581 extracted chars) around the embed
    that holds the resolving polling average. The prose is worth keeping; what it
    does NOT contain is any polling number."""
    return (
        "<!doctype html><html><head><title>The 2026 Senate Forecast</title></head><body>"
        "<article><h1>The 2026 Senate Forecast</h1>"
        "<p>The forecast predicts the outcome of every Senate race in 2026 using a data-driven "
        "model that factors in the latest polling, historic trends, candidate quality, and "
        "fundraising. Every day, we simulate the election 50,000 times to get the best projection "
        "we can on how likely each party is to win the majority.</p>"
        f"{_INFOGRAM_EMBED_MARKUP}"
        "<p>Background: after a successful 2024 cycle, Republicans hold a 53-47 advantage, and "
        "Democrats need to flip four seats to take a 51-49 majority. Their best offensive "
        "opportunities are Maine and North Carolina, with Ohio and Alaska also competitive.</p>"
        "</article></body></html>"
    ).encode()
