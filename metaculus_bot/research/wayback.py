"""Reading a Wayback Machine snapshot URL: the route in, the timestamp out, the disclosure.

The archive is the one free route whose EGRESS IS NOT OURS. That is the whole reason it earns a
rung: measured 2026-09-03, the dominant cause of Tier-1 fetch failures is not fetch technique
but where the request comes from — the same client with the same headers gets 403 from a GitHub
Actions runner and 200 from a residential address on bls.gov, cdc.gov and fsis.usda.gov alike.
web.archive.org dials the host from an address the host does not associate with us, and it
rescued 3 of the 22 URLs that still failed from the laptop.

What it does NOT rescue is a JavaScript wall: the archive faithfully stores the unrendered
shell, so bsky.jazco.dev and dcas.dmdc.osd.mil both extract 0 chars from snapshots 2-3 days old.
That is why the caller triggers this rung on ``blocked`` / ``error`` / ``not_found`` and never on
``js_wall``, where the browser rung rescued 6 of 8 instead.

Route facts, live-verified 2026-09-03 against ``https://www.bls.gov/wsp/``:

- ``https://web.archive.org/web/2026id_/<url>`` answers ``302`` with
  ``Location: https://web.archive.org/web/20260828221347id_/https://www.bls.gov/wsp/`` and an
  ``x-archive-redirect-reason: found capture at 20260828221347`` header. So the 14-digit capture
  timestamp arrives in the FINAL URL, which is why the caller reads it off the finished hop
  rather than asking for it.
- ``https`` is required: port 80 refuses (it reads like throttling).
- The availability API is NOT used — it 429s under any concurrency.
- ``id_`` asks for the raw stored bytes rather than the archive's rewritten page, so the
  extraction sees what the host served, not the toolbar the archive injects.

A snapshot is admissible as primary grading evidence ONLY with its age stated (operator
decision, 2026-09-03), which is what :func:`wayback_lead` renders, and only inside the age
bound the caller enforces on :func:`snapshot_age_days`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

WAYBACK_HOST = "web.archive.org"

# `2026id_` is a 4-digit "as near to 2026 as you have" request plus the raw-bytes modifier; the
# archive redirects it to the nearest capture and puts that capture's full 14-digit timestamp in
# the Location. Asking for a bare year rather than a specific date is deliberate: we want the
# freshest capture the archive holds, and the freshness guard is what decides whether it is
# fresh enough.
_SNAPSHOT_REQUEST_TEMPLATE = "https://web.archive.org/web/2026id_/{url}"

# The final URL's shape: 14-digit timestamp, an optional two-letter modifier (`id_` raw bytes,
# `if_` framed, `im_` image), then the original URL verbatim. Anchored on the archive's own host
# so nothing else can be mistaken for a snapshot URL.
_SNAPSHOT_URL_RE = re.compile(r"^https?://web\.archive\.org/web/(\d{14})(?:[a-z]{2}_)?/(.+)$", re.IGNORECASE)

# How far ahead of our clock a capture timestamp may sit before it is unusable rather than
# maximally fresh. Same value and same reason as the Datawrapper freshness guard's: this
# tolerates ordinary clock skew and nothing more, because past that a future date means a broken
# clock or a misparse — and the lead it authorizes asserts a capture date to forecasters.
_CLOCK_SKEW_TOLERANCE = timedelta(hours=6)


@dataclass(frozen=True, slots=True)
class WaybackSnapshot:
    """A parsed snapshot URL: when the capture was taken, and what it is a capture OF."""

    captured_at: datetime
    inner_url: str


def wayback_snapshot_url(url: str) -> str:
    """The archive URL to fetch for ``url``'s freshest stored capture.

    ``url`` is interpolated verbatim rather than percent-encoded: the archive's path format
    carries the original URL as-is (scheme, slashes and query included), and encoding it makes
    the archive treat it as a different resource.
    """
    return _SNAPSHOT_REQUEST_TEMPLATE.format(url=url)


def parse_snapshot_url(final_url: str) -> WaybackSnapshot | None:
    """The capture timestamp and inner URL of a finished snapshot fetch, or None.

    None means the fetch never landed on a datable capture — the archive answered our
    four-digit request directly, or redirected somewhere that is not a snapshot URL at all. The
    caller treats that exactly as it treats a too-old capture, because an undatable copy cannot
    carry the age disclosure that makes a snapshot admissible.
    """
    match = _SNAPSHOT_URL_RE.match(final_url)
    if match is None:
        return None
    try:
        captured_at = datetime.strptime(match.group(1), "%Y%m%d%H%M%S").replace(tzinfo=UTC)
    except ValueError:
        return None
    inner_url = match.group(2)
    if not inner_url:
        return None
    return WaybackSnapshot(captured_at=captured_at, inner_url=inner_url)


def innermost_url(url: str) -> str:
    """The URL a snapshot is ultimately a capture OF, unwrapping nested captures; ``url`` itself
    when it is not a snapshot URL.

    Repeated rather than single-level because the archive stores captures of its own capture
    URLs, and a capture OF a capture presents ``web.archive.org`` as its inner host — which clears
    a hostname-keyed self-reference check and a public-URL check at one level of unwrapping.
    Terminates because every pass strictly shortens the string.
    """
    while (snapshot := parse_snapshot_url(url)) is not None:
        url = snapshot.inner_url
    return url


def snapshot_age_days(snapshot: WaybackSnapshot, now: datetime) -> float | None:
    """How many days before ``now`` the capture was taken, or None when that is unusable.

    None on a capture dated implausibly far in the FUTURE, which is a broken clock or a misparse
    on one side rather than the freshest possible copy — the same two-sided rule the Datawrapper
    freshness guard applies, and for the same reason: the disclosure asserts a date.
    """
    delta = now - snapshot.captured_at
    if -delta > _CLOCK_SKEW_TOLERANCE:
        return None
    return max(0.0, delta.total_seconds() / 86400.0)


def wayback_lead(snapshot: WaybackSnapshot, age_days: float, live_status: str) -> str:
    """The MANDATORY disclosure an archived copy carries.

    Every clause is load-bearing. It names the archive, so a forecaster knows the bytes did not
    come from the host now. It gives the capture date AND the age in days, because the section
    around it is captioned "primary grading evidence" and a question about a quantity that moves
    daily is graded on a reading this copy may not carry. And it says what happened to the live
    page, since "the tracker was down" and "the tracker has no reading" are different evidence.
    """
    return (
        f"[Archived copy from the Wayback Machine, captured {snapshot.captured_at.date().isoformat()}, "
        f"{int(age_days)} days before this forecast; the live page could not be fetched ({live_status}).]"
    )
