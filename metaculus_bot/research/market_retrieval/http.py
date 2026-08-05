"""Everything between the socket and a typed row: the bounded GET, the body caps, the
per-query failure accounting, and the scalar coercions the venue parsers apply to
untrusted JSON.

Split out of ``venues.py`` because none of it is venue-specific — the four venue paths sit
on top of this one transport layer — and because the coercions have the widest reuse in the
package. Built on ``research/http_fetch.py``, which owns the session factory and the
streaming body reads this module's caps are enforced with.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

import aiohttp

from metaculus_bot.research.http_fetch import read_body_capped, read_body_snippet
from metaculus_bot.research.market_retrieval.types import MarketMatch, SettlementSource, _FetchTally

logger = logging.getLogger(__name__)

# Per-HTTP-call cap. The snapshot-level wall sits outside it.
PLATFORM_HTTP_TIMEOUT = 10.0

# One bounded backoff between retries. Two attempts is the measured sufficiency: the venue
# fan-out already issues several queries per venue, so a persistent outage is visible
# without a third attempt and a transient one is covered by the second.
HTTP_RETRY_BACKOFF_SECS = 0.5

# Hard cap on one buffered response body (the search endpoints don't paginate).
MAX_RESPONSE_BYTES = 10 * 1024 * 1024


# ---------------------------------------------------------------------------
# Scalar coercions over untrusted JSON
# ---------------------------------------------------------------------------


def safe_float(value: Any) -> float | None:
    # The isfinite guard is the same one `safe_int` needs and for a sharper reason: json.loads
    # accepts bare NaN/Infinity literals, and NaN defeats every comparison in `_liquidity_label`,
    # so a row whose volume arrived as NaN falls through to the strongest label and renders
    # `signal=deep` — presenting missing data to a forecaster as the best possible liquidity
    # evidence. None renders `no-liquidity-data`, which is what a missing figure means.
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def safe_int(value: Any) -> int | None:
    number = safe_float(value)
    if number is None:
        return None
    return int(number)


def parse_iso(value: Any) -> datetime | None:
    """An ISO timestamp as a UTC-AWARE datetime, or None.

    Normalized HERE, at the one boundary all four venue parsers pass through, so no comparison
    site downstream has to remember: ``fromisoformat`` returns an aware datetime for the ``Z``
    form and a NAIVE one for a bare timestamp or a date-only string, and mixing the two makes
    ``max(closes)`` over an event's nested markets raise ``TypeError: can't compare offset-naive
    and offset-aware datetimes``. That runs inside `to_thread` with no guard, so it reaches the
    snapshot-level net and zeroes ALL FOUR venues for the question — and the offending event
    stays in the 6h catalogue cache, so every later question repeats it.

    A naive value is TREATED as UTC — the same assumption ``assemble_pool``'s ``as_of`` makes.
    Not an assertion that every venue publishes UTC: PredictIt's ``dateEnd`` is historically
    Eastern, and only the date-granular rendering makes that <=5h skew immaterial. Behaviour is
    unchanged for rendering, since attaching a tzinfo does not shift the wall clock.
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def parse_iso_guarded(value: Any) -> datetime | None:
    """``parse_iso`` behind a ``YYYY-MM-DD`` prefix guard, for a field with sentinels.

    ``fromisoformat`` rejects both shapes PredictIt actually ships in ``dateEnd``: the
    literal ``"NA"`` and ``"N/A"`` no-close sentinels, and 7-digit fractional seconds. The
    guard admits only strings that begin with a real calendar date, and falls back to that
    date alone when the rest of the timestamp is unparseable — a close DATE is all the
    ranker needs to tell same-window from different-window, so truncating to it loses
    nothing and never raises.
    """
    text = str(value or "").strip()
    if len(text) < 10 or text[4] != "-" or text[7] != "-":
        return None
    # Calendar-date field slices, not data sampling.
    if not (text[:4].isdigit() and text[5:7].isdigit() and text[8:10].isdigit()):  # noqa: HARNESS-SCAN-EXEMPT-subsampling
        return None
    return parse_iso(text) or parse_iso(text[:10])  # noqa: HARNESS-SCAN-EXEMPT-subsampling


def settlement_sources(raw: Any) -> tuple[SettlementSource, ...]:
    """A venue's ``[{name, url}, ...]`` block, verbatim, as typed rows.

    Only the two fields, so the raw-research archive's shape cannot silently pick up
    whatever else the venue starts shipping in that block.
    """
    if not isinstance(raw, list):
        return ()
    out: list[SettlementSource] = []
    for source in raw:
        if not isinstance(source, dict):
            continue
        name = str(source.get("name") or "").strip()
        url = str(source.get("url") or "").strip()
        if name or url:
            out.append(SettlementSource(name=name, url=url))
    return tuple(out)


# ---------------------------------------------------------------------------
# Bounded GET
# ---------------------------------------------------------------------------


async def read_json_capped(resp: Any, label: str) -> Any | None:
    """Parse a body as JSON, rejecting anything over MAX_RESPONSE_BYTES.

    Test stubs that only implement ``.json()`` take the fallback path. Returns None on a
    decode failure or an oversized body; the caller logs.
    """
    read_method = getattr(resp, "read", None)
    if read_method is not None and callable(read_method):
        raw = await read_body_capped(resp, max_bytes=MAX_RESPONSE_BYTES, label=label)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError, UnicodeDecodeError) as exc:
            logger.warning(f"{label} JSON decode failed: {exc}")
            return None
    try:
        return await resp.json()
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning(f"{label} JSON decode failed: {exc}")
        return None  # noqa: ASYNC910


async def http_get_with_backoff(
    session: Any,
    url: str,
    params: dict[str, str],
    *,
    max_attempts: int,
    retryable_statuses: Iterable[int] | None = None,
    label: str,
) -> Any | None:
    """GET ``url`` with ``max_attempts`` and one bounded backoff between retries.

    Returns the parsed JSON body on 200, or None on retry exhaustion / a non-200. Caps the
    body at MAX_RESPONSE_BYTES so a runaway upstream cannot blow up memory, and caps
    cumulative sleep so retries cannot exceed the per-platform budget.

    ``retryable_statuses`` defaults to (403, 429, 500, 502, 503, 504); anything >= 500 is
    retryable regardless.
    """
    retryable: set[int] = set(retryable_statuses or (403, 429, 500, 502, 503, 504))
    cumulative_sleep = 0.0
    timeout = aiohttp.ClientTimeout(total=PLATFORM_HTTP_TIMEOUT, sock_read=PLATFORM_HTTP_TIMEOUT)

    for attempt in range(max_attempts):
        try:
            async with session.get(url, params=params, timeout=timeout) as resp:
                status = resp.status
                if status in retryable or status >= 500:
                    if attempt + 1 >= max_attempts:
                        logger.warning(f"{label} HTTP {status} after {attempt + 1} attempts; giving up")
                        return None
                    sleep_for = HTTP_RETRY_BACKOFF_SECS
                    if cumulative_sleep + sleep_for + PLATFORM_HTTP_TIMEOUT > PLATFORM_HTTP_TIMEOUT * max_attempts:
                        logger.warning(f"{label} HTTP {status}: sleep budget exhausted; giving up")
                        return None
                    logger.warning(f"{label} HTTP {status}; retry {attempt + 2}/{max_attempts} after {sleep_for:.2f}s")
                    await asyncio.sleep(sleep_for)
                    cumulative_sleep += sleep_for
                    continue
                if status != 200:
                    snippet = await read_body_snippet(resp)
                    logger.warning(f"{label} HTTP {status} non-retryable: {snippet}")
                    return None
                return await read_json_capped(resp, label)
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            if attempt + 1 >= max_attempts:
                logger.warning(f"{label} transient error after {attempt + 1} attempts: {exc}")
                return None  # noqa: ASYNC910
            sleep_for = HTTP_RETRY_BACKOFF_SECS
            logger.warning(f"{label} transient error: {exc}; retry {attempt + 2}/{max_attempts} after {sleep_for:.2f}s")
            await asyncio.sleep(sleep_for)
            cumulative_sleep += sleep_for
    return None  # noqa: ASYNC910


def flatten_results(results: Sequence[Any], platform: str) -> tuple[list[MarketMatch], _FetchTally]:
    """Flatten per-query search results and tally how many sub-queries were lost.

    A leaf search signals upstream failure with ``None`` and a successful-but-empty parse
    with ``[]``, so anything that is not a list — a raised task or a ``None`` — counts as a
    lost sub-query rather than being logged and discarded. One venue's 403 on one long
    query therefore degrades THAT QUERY and nothing else: not the venue's other queries,
    and not the question.
    """
    out: list[MarketMatch] = []
    queries_ok = 0
    queries_failed = 0
    for result in results:
        if isinstance(result, list):
            out.extend(result)
            queries_ok += 1
        elif isinstance(result, Exception):
            logger.warning(f"{platform} query task raised: {result}")
            queries_failed += 1
        else:
            queries_failed += 1
    return out, _FetchTally(queries_ok, queries_failed)
