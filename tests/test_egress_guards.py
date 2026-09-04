"""The suite's network-egress guards, tripped on purpose so a refactor cannot drop one silently.

Every other test relies on the two autouse fixtures in ``tests/conftest.py`` without ever calling
into them: ``_block_network_egress`` (Python sockets) and ``_block_native_egress`` (headless
Chromium through Playwright, libcurl through curl_cffi). A guard that stopped patching would fail
no test by itself; the suite would just start making real requests again. So this module calls
each chokepoint the way its real callers do, asserts the refusal, and for the native guard clears
the attempt it recorded so that guard's own teardown check does not fail the test for the trip it
was asked to cause.

Nothing here reaches a host: every guarded call raises before a socket, a browser or a curl handle
is touched, and the one real connect is to a listener on loopback.
"""

from __future__ import annotations

import asyncio
import socket

import pytest
import yfinance
from curl_cffi import requests as curl_requests
from playwright._impl._browser_type import BrowserType as ImplBrowserType
from playwright.async_api import BrowserType as AsyncBrowserType

# example.com's address. Never dialed: the socket guard raises before the connect is attempted.
_PUBLIC_ADDRESS = ("93.184.216.34", 80)
_NATIVE_REFUSAL = "blocked in tests by _block_native_egress"


class TestSocketGuard:
    def test_connect_to_a_public_host_is_refused(self) -> None:
        with socket.socket() as sock, pytest.raises(RuntimeError, match="Network access blocked in tests"):
            sock.connect(_PUBLIC_ADDRESS)

    def test_connect_ex_to_a_public_host_is_refused(self) -> None:
        with socket.socket() as sock, pytest.raises(RuntimeError, match="Network access blocked in tests"):
            sock.connect_ex(_PUBLIC_ADDRESS)

    def test_loopback_passes_through_to_the_real_connect(self) -> None:
        with socket.socket() as listener, socket.socket() as client:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            client.connect(listener.getsockname())


class TestBrowserGuard:
    """Playwright's impl-layer ``BrowserType`` is the one class both public APIs delegate to."""

    @pytest.mark.parametrize("entry_point", ["launch", "launch_persistent_context", "connect", "connect_over_cdp"])
    async def test_every_way_to_obtain_a_browser_is_refused(
        self, entry_point: str, native_egress_attempts: list[str]
    ) -> None:
        # Bypasses __init__ on purpose: a real instance needs a running driver connection, and the
        # guard has to raise before the driver is consulted at all.
        browser_type = object.__new__(ImplBrowserType)
        with pytest.raises(RuntimeError, match=_NATIVE_REFUSAL):
            await getattr(browser_type, entry_point)()
        assert native_egress_attempts == [f"playwright BrowserType.{entry_point}"]
        native_egress_attempts.clear()

    async def test_the_public_async_launch_delegates_into_the_guard(self, native_egress_attempts: list[str]) -> None:
        """``playwright.chromium.launch(...)`` in ``rendered_fetch`` is this wrapper method."""
        impl = object.__new__(ImplBrowserType)
        impl._loop = asyncio.get_running_loop()  # the only impl attribute the async wrapper's constructor reads
        with pytest.raises(RuntimeError, match=_NATIVE_REFUSAL):
            await AsyncBrowserType(impl).launch(headless=True, args=["--host-resolver-rules=MAP example.com 8.8.8.8"])
        assert native_egress_attempts == ["playwright BrowserType.launch"]
        native_egress_attempts.clear()


class TestCurlGuard:
    """Every curl_cffi verb helper lands in ``Session.request`` / ``AsyncSession.request``."""

    def test_sync_session_get_is_refused(self, native_egress_attempts: list[str]) -> None:
        with curl_requests.Session() as session, pytest.raises(RuntimeError, match=_NATIVE_REFUSAL):
            session.get("https://example.com/sync")
        assert native_egress_attempts == ["curl_cffi GET https://example.com/sync"]
        native_egress_attempts.clear()

    async def test_async_session_get_is_refused(self, native_egress_attempts: list[str]) -> None:
        async with curl_requests.AsyncSession() as session:
            with pytest.raises(RuntimeError, match=_NATIVE_REFUSAL):
                await session.get("https://example.com/async")
        assert native_egress_attempts == ["curl_cffi GET https://example.com/async"]
        native_egress_attempts.clear()

    def test_module_level_get_is_refused(self, native_egress_attempts: list[str]) -> None:
        """The shape ``scripts/probes/fetch_diagnostic.py`` uses: a one-shot Session under the hood."""
        with pytest.raises(RuntimeError, match=_NATIVE_REFUSAL):
            curl_requests.get("https://example.com/module")
        assert native_egress_attempts == ["curl_cffi GET https://example.com/module"]
        native_egress_attempts.clear()

    def test_yfinance_reaches_libcurl_through_the_guard(self, native_egress_attempts: list[str]) -> None:
        """yfinance builds its own impersonating ``Session`` and calls ``get`` on it; it also swallows
        the refusal into an empty frame, which is exactly why the guard records the attempt."""
        history = yfinance.Ticker("AAPL").history(period="5d", raise_errors=False)
        assert history.empty
        assert native_egress_attempts, "yfinance made no curl_cffi request, so the guard saw nothing"
        assert all(attempt.startswith("curl_cffi GET https://") for attempt in native_egress_attempts)
        native_egress_attempts.clear()
