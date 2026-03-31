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


def enable_aiohttp_session_autoclose() -> None:
    """Monkey-patch ``aiohttp.ClientSession.__init__`` to track open sessions
    and register an ``atexit`` handler that closes any still open at shutdown.
    """
    open_sessions: weakref.WeakSet[aiohttp.ClientSession] = weakref.WeakSet()
    original_init = aiohttp.ClientSession.__init__

    def tracking_init(self: aiohttp.ClientSession, *args, **kwargs):  # type: ignore[no-untyped-def]
        original_init(self, *args, **kwargs)
        open_sessions.add(self)

    aiohttp.ClientSession.__init__ = tracking_init  # type: ignore[assignment]

    def _close_open_sessions() -> None:
        to_close = [s for s in list(open_sessions) if not s.closed]
        if not to_close:
            return
        logger.debug(f"Closing {len(to_close)} lingering aiohttp sessions at exit")

        async def _close_all() -> None:
            for s in to_close:
                try:
                    await s.close()
                except Exception as e:  # pragma: no cover - best-effort cleanup
                    logger.debug(f"Error closing aiohttp session at exit: {e}")

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            loop.create_task(_close_all())
        else:
            try:
                asyncio.run(_close_all())
            except RuntimeError:
                new_loop = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(new_loop)
                    new_loop.run_until_complete(_close_all())
                finally:
                    new_loop.close()

    atexit.register(_close_open_sessions)
