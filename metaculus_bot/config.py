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

    global _ENV_LOADED  # noqa: PLW0603  # one-shot process-wide init flag, guarded by _LOCK

    if _ENV_LOADED:
        return

    with _LOCK:
        if _ENV_LOADED:
            return

        try:
            load_dotenv()
            load_dotenv(".env.local", override=True)
        # Boundary: .env files are optional (in CI the env comes from Actions secrets), so a
        # dotenv read failure must never block process startup.
        except Exception as exc:  # noqa: BLE001  # HARNESS-SCAN-EXEMPT-broad-except  # pragma: no cover
            _logger.warning("Failed to load environment files: %s", exc)
        finally:
            _ENV_LOADED = True


__all__ = ["load_environment"]
