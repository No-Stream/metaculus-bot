"""Auto-close lingering aiohttp sessions at process exit.

Mitigates occasional "Unclosed client session" warnings from aiohttp when
using EXA search or other providers under high concurrency.  Tracks sessions
via a WeakSet and closes them in an atexit handler.
"""

import asyncio
import atexit
import logging
import weakref

import aiohttp

logger: logging.Logger = logging.getLogger(__name__)

_SENTINEL = "_metaculus_autoclose_wrapped"


async def _close_all(to_close: list[aiohttp.ClientSession]) -> None:
    """Close every session, letting one failure never strand the rest."""
    for s in to_close:
        try:
            await s.close()
        # Boundary: best-effort cleanup during interpreter shutdown. A raise here would
        # both skip the remaining sessions and surface as a noisy atexit traceback.
        except Exception as e:  # noqa: BLE001  # HARNESS-SCAN-EXEMPT-broad-except  # pragma: no cover
            logger.debug(f"Error closing aiohttp session at exit: {e}")


def _drain_on_any_available_loop(to_close: list[aiohttp.ClientSession]) -> None:
    """Run ``_close_all`` on the running loop, else on a throwaway loop of our own."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        loop.create_task(_close_all(to_close))
        return

    try:
        asyncio.run(_close_all(to_close))
    except RuntimeError:
        new_loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(new_loop)
            new_loop.run_until_complete(_close_all(to_close))
        finally:
            new_loop.close()


def _close_open_sessions(open_sessions: weakref.WeakSet[aiohttp.ClientSession]) -> None:
    """atexit handler: close whichever tracked sessions are still open."""
    to_close = [s for s in list(open_sessions) if not s.closed]
    if not to_close:
        return
    logger.debug(f"Closing {len(to_close)} lingering aiohttp sessions at exit")
    _drain_on_any_available_loop(to_close)


def enable_aiohttp_session_autoclose() -> None:
    """Monkey-patch ``aiohttp.ClientSession.__init__`` to track open sessions
    and register an ``atexit`` handler that closes any still open at shutdown.

    Idempotent: a sentinel on ``ClientSession`` guards against re-wrapping
    ``__init__`` and re-registering the atexit handler on repeat calls (which
    would otherwise stack a fresh WeakSet + atexit closure each time).
    """
    if getattr(aiohttp.ClientSession, _SENTINEL, False):
        return

    open_sessions: weakref.WeakSet[aiohttp.ClientSession] = weakref.WeakSet()
    original_init = aiohttp.ClientSession.__init__

    def tracking_init(self: aiohttp.ClientSession, *args, **kwargs):  # type: ignore[no-untyped-def]
        original_init(self, *args, **kwargs)
        open_sessions.add(self)

    aiohttp.ClientSession.__init__ = tracking_init  # type: ignore[assignment]

    atexit.register(_close_open_sessions, open_sessions)
    setattr(aiohttp.ClientSession, _SENTINEL, True)
