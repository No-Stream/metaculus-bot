"""The shared robots.txt cache: one read per host per run, whoever asks and however many ask at once.

The rule-matching half (``google_extended_disallows``) is pinned in ``tests/test_agentic_tools.py``
beside the reader that first used it; what is pinned here is the CACHE's behaviour under the
concurrency both paid readers actually produce. Two same-host URLs escalate in one question, or
two questions reach one host at once, and each caller used to fetch robots.txt for itself — so a
later failed fetch could overwrite the verdict an earlier one had cached, after which the paid
read was spent on a host that refuses it.
"""

from __future__ import annotations

import asyncio

import pytest

from metaculus_bot.research import robots_policy

_HOST_URL_A = "https://report.example.org/chapters/1/"
_HOST_URL_B = "https://report.example.org/chapters/2/"
_DISALLOW_GOOGLE_EXTENDED = "User-agent: Google-Extended\nDisallow: /\n"


@pytest.fixture(autouse=True)
def _fresh_cache():
    robots_policy.reset_robots_cache()
    yield
    robots_policy.reset_robots_cache()


class _Fetcher:
    """A robots.txt fetch the test releases by hand, so the interleaving is the test's to choose."""

    def __init__(self, robots_txt: str | None) -> None:
        self.robots_txt = robots_txt
        self.calls: list[str] = []
        self.release = asyncio.Event()

    async def __call__(self, robots_url: str) -> str | None:
        self.calls.append(robots_url)
        await self.release.wait()
        return self.robots_txt


class TestSingleFlight:
    async def test_concurrent_callers_on_one_host_share_one_read(self):
        fetcher = _Fetcher(_DISALLOW_GOOGLE_EXTENDED)

        first = asyncio.create_task(robots_policy.google_extended_blocks_url(_HOST_URL_A, fetch_text=fetcher))
        second = asyncio.create_task(robots_policy.google_extended_blocks_url(_HOST_URL_B, fetch_text=fetcher))
        await asyncio.sleep(0)
        assert fetcher.calls == ["https://report.example.org/robots.txt"], "the second caller started its own read"
        fetcher.release.set()

        assert await asyncio.gather(first, second) == [True, True]
        assert fetcher.calls == ["https://report.example.org/robots.txt"]

    async def test_a_waiters_cancellation_leaves_the_shared_read_running(self):
        """Awaiting the leader's future directly would let a cancelled waiter cancel the read
        everyone else is waiting on."""
        fetcher = _Fetcher(_DISALLOW_GOOGLE_EXTENDED)

        leader = asyncio.create_task(robots_policy.google_extended_blocks_url(_HOST_URL_A, fetch_text=fetcher))
        waiter = asyncio.create_task(robots_policy.google_extended_blocks_url(_HOST_URL_B, fetch_text=fetcher))
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        fetcher.release.set()

        assert await leader is True
        assert await robots_policy.google_extended_blocks_url(_HOST_URL_B, fetch_text=fetcher) is True
        assert len(fetcher.calls) == 1

    async def test_a_cancelled_leader_leaves_nothing_cached_and_waiters_proceed_toward_paying(self):
        """The leader's cancellation is its own caller's business (a provider deadline); the
        waiters get the same answer an unreadable robots.txt gives — pay — and the next caller
        reads again, because nothing was learned."""
        fetcher = _Fetcher(_DISALLOW_GOOGLE_EXTENDED)

        leader = asyncio.create_task(robots_policy.google_extended_blocks_url(_HOST_URL_A, fetch_text=fetcher))
        waiter = asyncio.create_task(robots_policy.google_extended_blocks_url(_HOST_URL_B, fetch_text=fetcher))
        await asyncio.sleep(0)
        leader.cancel()
        with pytest.raises(asyncio.CancelledError):
            await leader

        assert await waiter is False
        assert "report.example.org" not in robots_policy._ROBOTS_TXT_CACHE
        fetcher.release.set()
        assert await robots_policy.google_extended_blocks_url(_HOST_URL_A, fetch_text=fetcher) is True
        assert len(fetcher.calls) == 2


class TestCachedVerdicts:
    async def test_a_disallow_is_cached_and_never_re_read(self):
        fetcher = _Fetcher(_DISALLOW_GOOGLE_EXTENDED)
        fetcher.release.set()

        assert await robots_policy.google_extended_blocks_url(_HOST_URL_A, fetch_text=fetcher) is True
        assert await robots_policy.google_extended_blocks_url(_HOST_URL_B, fetch_text=fetcher) is True
        assert len(fetcher.calls) == 1

    async def test_an_unreadable_robots_txt_is_remembered_as_unreadable(self):
        """Fails toward paying, once: a host whose robots.txt we could not read is not re-asked
        on every cited URL."""
        fetcher = _Fetcher(None)
        fetcher.release.set()

        assert await robots_policy.google_extended_blocks_url(_HOST_URL_A, fetch_text=fetcher) is False
        assert await robots_policy.google_extended_blocks_url(_HOST_URL_B, fetch_text=fetcher) is False
        assert len(fetcher.calls) == 1

    async def test_a_verdict_once_cached_is_not_overwritten_by_a_later_failed_read(self):
        """The write path is guarded as well as single-flighted: a None never replaces text."""
        robots_policy._remember_robots_txt("report.example.org", _DISALLOW_GOOGLE_EXTENDED)

        robots_policy._remember_robots_txt("report.example.org", None)

        assert robots_policy._ROBOTS_TXT_CACHE["report.example.org"] == _DISALLOW_GOOGLE_EXTENDED
