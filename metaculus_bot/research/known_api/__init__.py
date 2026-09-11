"""The known-API registry: FRED, Yahoo Finance, Kalshi and SEC EDGAR URLs as deterministic API calls.

A URL whose host this package recognises is answered by that host's public API -- no LLM, no
paid key, the same client code the research providers already use -- rather than fetched as a
web page. It fills rung 0 of the shared fetch ladder (the ``policy.known_api`` seat) and gives
the gap-fill v2 driver three explicit tools for a windowed read the URL forms cannot express.
Detail: docs/research.md "Known-API registry", docs/agentic_gap_fill.md "The known-API tools".

Only the light layer is re-exported here (``translate`` and its data types, plus the URL
parsers the extraction seams share). The backends, the tool specs and the two adapters pull
fredapi / yfinance / aiohttp, so they are imported from their submodules
(``known_api.backends``, ``known_api.tools``, ``known_api.adapters``) by the wiring step and
their tests, keeping this package importable from the stdlib-only parse seams without dragging
those dependencies in.
"""

from __future__ import annotations

from metaculus_bot.research.known_api import parse
from metaculus_bot.research.known_api.result import KNOWN_API_STATUSES, KnownApiResult
from metaculus_bot.research.known_api.translate import KnownApiCall, translate

__all__ = [
    "KNOWN_API_STATUSES",
    "KnownApiCall",
    "KnownApiResult",
    "parse",
    "translate",
]
