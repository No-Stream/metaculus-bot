"""Environment configuration helpers for the Metaculus bot."""

from __future__ import annotations

import logging
from threading import Lock

from dotenv import load_dotenv

_logger = logging.getLogger(__name__)
_LOCK = Lock()
_ENV_LOADED = False


def load_environment() -> None:
    """Load environment variables from standard .env files exactly once.

    The function is safe to call multiple times across modules; only the first
    invocation triggers calls into python-dotenv. Subsequent calls are no-ops.
    """

    global _ENV_LOADED

    if _ENV_LOADED:
        return

    with _LOCK:
        if _ENV_LOADED:
            return

        try:
            load_dotenv()
            load_dotenv(".env.local", override=True)
        except Exception as exc:  # pragma: no cover - defensive logging only
            _logger.warning("Failed to load environment files: %s", exc)
        finally:
            _ENV_LOADED = True


__all__ = ["load_environment"]
