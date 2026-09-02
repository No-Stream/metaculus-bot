"""Directory-scoped fixtures for the resolution-source tests.

The autouse DNS stub is why these live in a conftest rather than being imported into each
module: every hostname in these tests is under ``example.com`` and the SSRF guard resolves
it for real, so a module that forgot the stub would see every fetch turn into
``ssrf_blocked`` instead of failing loudly. The plain page builders and the fake aiohttp
session are ordinary imports from ``tests/resolution_source_fakes.py``.
"""

from __future__ import annotations

import pytest

from metaculus_bot.research import resolution_source
from tests.resolution_source_fakes import _INFOGRAM_EMBED_MARKUP, _embed_shell_page


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
