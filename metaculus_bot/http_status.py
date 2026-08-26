"""Shared reader for the HTTP status forecasting-tools hides behind a re-raise.

ft's ``raise_for_status_with_additional_info`` re-raises a fresh, message-only
``requests.HTTPError`` (no ``response`` attached) and chains the original via
``raise ... from e``. Both MetaculusClient-patching modules — ``fetch_hardening``
and ``publish_hardening`` — need the real status to make their (deliberately
different) retry decisions, so the chain walk lives here ONCE; a correction to it
cannot land on one surface only.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

# ft's message-only HTTPError embeds the status as literal text ("Status code:
# 405"). Anchored on that phrase so an unrelated 3-digit number in a server
# message can't be misread as a status.
_STATUS_IN_MESSAGE = re.compile(r"Status code:\s*(?P<status>\d{3})\b")

# ft wraps at most once, so a short walk suffices; the bound also guards against
# a pathological self-referencing chain.
_CAUSE_CHAIN_DEPTH = 4


def iter_cause_chain(exc: BaseException, max_depth: int = _CAUSE_CHAIN_DEPTH) -> Iterator[BaseException]:
    """``exc`` plus its ``__cause__`` links, bounded and cycle-safe."""
    seen: set[int] = set()
    current: BaseException | None = exc
    for _ in range(max_depth):
        if current is None or id(current) in seen:
            return
        seen.add(id(current))
        yield current
        current = current.__cause__


def http_status_from_exception(exc: BaseException) -> int | None:
    """Best-effort HTTP status for a requests-shaped exception, or None.

    Prefers a real ``response.status_code`` anywhere in the cause chain, then
    falls back to ft's own message text.
    """
    for cause in iter_cause_chain(exc):
        status = getattr(getattr(cause, "response", None), "status_code", None)
        if isinstance(status, int):
            return status
    match = _STATUS_IN_MESSAGE.search(str(exc))
    return int(match.group("status")) if match else None
