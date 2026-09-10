"""The two thin adapters from a :class:`KnownApiResult` to the two callers of rung 0.

One returns the gap-fill loop's ``ToolOutcome`` (method ``known_api``); one returns the
resolution-source fetcher's ``FetchResult`` (a success carries the rendered text, route
``known_api``). The wiring step adds ``known_api`` to ``provenance._METHOD_TO_TIER`` and to the
``FetchRoute`` literal; until it does, the route is a parameter so this compiles against the
unedited types. Detail: docs/research.md "Known-API registry".
"""

from __future__ import annotations

from typing import cast

from metaculus_bot.research.agentic.types import ToolOutcome
from metaculus_bot.research.known_api.result import KnownApiResult
from metaculus_bot.research.resolution_fetch_result import FetchResult, FetchRoute, FetchStatus

KNOWN_API_METHOD = "known_api"
KNOWN_API_ROUTE = "known_api"

# A known-API status maps onto the fetcher's existing vocabulary; only ``ok`` carries text.
_STATUS_TO_FETCH_STATUS: dict[str, str] = {
    "ok": "success",
    "empty": "no_resolving_content",
    "not_found": "not_found",
    "error": "error",
}


def to_tool_outcome(result: KnownApiResult) -> ToolOutcome:
    """The gap-fill loop's outcome for a known-API result; ``method=known_api``, status verbatim."""
    return ToolOutcome(
        content_markdown=result.content_markdown,
        links=list(result.links),
        method=KNOWN_API_METHOD,
        status=result.status,
    )


def to_fetch_result(result: KnownApiResult, *, url: str, route: str = KNOWN_API_ROUTE) -> FetchResult:
    """The resolution-source fetcher's result for a known-API answer; a success carries the text.

    ``route`` is a parameter because the ``FetchRoute`` literal does not yet carry ``known_api``
    (the wiring step adds it); the default is the intended value and the cast keeps this honest
    against the unedited type.
    """
    status = cast(FetchStatus, _STATUS_TO_FETCH_STATUS[result.status])
    return FetchResult(
        url=url or result.source_url,
        status=status,
        text=result.content_markdown if result.status == "ok" else "",
        http_status=None,
        content_type=None,
        route=cast(FetchRoute, route),
    )
