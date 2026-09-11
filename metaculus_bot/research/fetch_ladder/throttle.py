"""Recognition of short host rate-limit interstitials shared by both callers."""

from __future__ import annotations

FETCH_THROTTLE_PAGE_MAX_CHARS = 1200
FETCH_THROTTLE_PHRASES: tuple[str, ...] = (
    "rate limit",
    "rate-limit",
    "ratelimit",
    "limit exceeded",
    "too many requests",
    "query per",
    "queries per",
    "requests per",
    "per second per ip",
    "retry after",
    "try again later",
    "please slow down",
)


def matched_throttle_phrase(text: str) -> str | None:
    """Return the phrase when a short body is a host's rate-limit interstitial."""
    stripped = text.strip()
    if not stripped or len(stripped) > FETCH_THROTTLE_PAGE_MAX_CHARS:
        return None
    lowered = stripped.lower()
    return next((phrase for phrase in FETCH_THROTTLE_PHRASES if phrase in lowered), None)
