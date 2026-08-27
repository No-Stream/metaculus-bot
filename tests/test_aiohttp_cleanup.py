"""Behavior pins for the aiohttp session autoclose monkey-patch.

The module wraps ``aiohttp.ClientSession.__init__`` process-wide and registers an
``atexit`` handler, so every test here restores both via ``monkeypatch`` and
captures the registered handler instead of letting the real interpreter shutdown
run it. Assertions go through the captured handler rather than a private name so
they stay valid if the closure is hoisted to a module-level helper.
"""

from __future__ import annotations

import asyncio
import atexit
from collections.abc import Callable
from typing import Any

import aiohttp
import pytest

from metaculus_bot.aiohttp_cleanup import _SENTINEL, enable_aiohttp_session_autoclose


def _enable_and_capture_handlers(monkeypatch: pytest.MonkeyPatch, *, calls: int = 1) -> list[Callable[[], None]]:
    """Enable the autoclose patch ``calls`` times, returning the atexit handlers registered.

    Restores ``ClientSession.__init__`` and the idempotency sentinel on teardown so
    the global patch never leaks into sibling tests.
    """
    registered: list[Callable[[], None]] = []

    def _fake_register(func: Callable[..., Any], *args: Any) -> Callable[..., Any]:
        registered.append(lambda: func(*args))
        return func

    monkeypatch.setattr(atexit, "register", _fake_register)
    # Recording the current values makes monkeypatch undo the module's own reassignment.
    monkeypatch.setattr(aiohttp.ClientSession, "__init__", aiohttp.ClientSession.__init__)
    monkeypatch.setattr(aiohttp.ClientSession, _SENTINEL, False, raising=False)

    for _ in range(calls):
        enable_aiohttp_session_autoclose()
    return registered


class TestEnableAutoclose:
    def test_wraps_init_and_registers_one_exit_handler(self, monkeypatch: pytest.MonkeyPatch) -> None:
        original_init = aiohttp.ClientSession.__init__

        handlers = _enable_and_capture_handlers(monkeypatch)

        assert len(handlers) == 1
        assert aiohttp.ClientSession.__init__ is not original_init
        assert getattr(aiohttp.ClientSession, _SENTINEL) is True

    def test_repeat_calls_do_not_stack_wrappers_or_handlers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        handlers = _enable_and_capture_handlers(monkeypatch, calls=3)

        assert len(handlers) == 1

    def test_handler_is_a_noop_when_nothing_was_tracked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        (handler,) = _enable_and_capture_handlers(monkeypatch)

        # No sessions constructed, so the handler must return without touching a loop.
        handler()


class TestExitHandlerClosesSessions:
    @pytest.mark.asyncio
    async def test_open_session_is_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        (handler,) = _enable_and_capture_handlers(monkeypatch)

        session = aiohttp.ClientSession()
        try:
            assert not session.closed

            # Called from inside a running loop: the handler schedules the close.
            handler()
            await asyncio.sleep(0.05)

            assert session.closed
        finally:
            if not session.closed:
                await session.close()

    @pytest.mark.asyncio
    async def test_already_closed_session_is_left_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        (handler,) = _enable_and_capture_handlers(monkeypatch)

        session = aiohttp.ClientSession()
        await session.close()
        assert session.closed

        handler()
        await asyncio.sleep(0.05)

        assert session.closed

    @pytest.mark.asyncio
    async def test_close_failure_on_one_session_does_not_strand_the_rest(self, monkeypatch: pytest.MonkeyPatch) -> None:
        (handler,) = _enable_and_capture_handlers(monkeypatch)

        failing = aiohttp.ClientSession()
        healthy = aiohttp.ClientSession()

        async def _boom() -> None:
            await asyncio.sleep(0)
            raise RuntimeError("close blew up")

        monkeypatch.setattr(failing, "close", _boom)

        try:
            handler()
            await asyncio.sleep(0.05)

            # The broad catch is the point: one bad session must not skip its siblings.
            assert healthy.closed
        finally:
            monkeypatch.undo()
            for session in (failing, healthy):
                if not session.closed:
                    await session.close()
