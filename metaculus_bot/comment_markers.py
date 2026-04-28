"""Shared constants + regex for the STACKED=true/false HTML-comment marker.

The marker is injected into each published Metaculus comment by the bot's
``_create_unified_explanation`` override (see ``main.py``), and parsed back out
by the residual-analysis collector via
``metaculus_bot.performance_analysis.parsing.parse_stacked_marker``.

Keeping the literals + regex in one module avoids silent producer/consumer
drift if either side changes the comment shape.
"""

from __future__ import annotations

import re

STACKED_MARKER_TRUE: str = "<!-- STACKED=true -->"
STACKED_MARKER_FALSE: str = "<!-- STACKED=false -->"

# Tolerant of surrounding whitespace and casing so accidental reformatting of
# already-published comments (e.g. a markdown editor normalizing whitespace)
# doesn't silently desync the collector.
STACKED_MARKER_RE: re.Pattern[str] = re.compile(
    r"<!--\s*STACKED=(true|false)\s*-->",
    re.IGNORECASE,
)

# Section headers emitted by ``metaculus_bot.stacking.combine_stacker_and_base_reasoning``
# inside the single R1 body for stacked questions. Shared with
# ``metaculus_bot.performance_analysis.parsing`` which splits the body on
# ``STACKED_BASE_REASONING_HEADER`` to recover per-base-model attribution.
STACKER_META_ANALYSIS_HEADER: str = "## Stacker Meta-Analysis"
STACKED_BASE_REASONING_HEADER: str = "## Base Model Reasoning (inputs to stacker)"

# Sanity-check: the regex must match both literals. If a future edit breaks
# this invariant, fail at import time rather than silently when the collector
# runs weeks later against real comments.
assert STACKED_MARKER_RE.search(STACKED_MARKER_TRUE) is not None, (
    f"STACKED_MARKER_RE does not match STACKED_MARKER_TRUE={STACKED_MARKER_TRUE!r}"
)
assert STACKED_MARKER_RE.search(STACKED_MARKER_FALSE) is not None, (
    f"STACKED_MARKER_RE does not match STACKED_MARKER_FALSE={STACKED_MARKER_FALSE!r}"
)
assert STACKER_META_ANALYSIS_HEADER.startswith("## "), (
    f"STACKER_META_ANALYSIS_HEADER must be a markdown H2 header, got {STACKER_META_ANALYSIS_HEADER!r}"
)
assert STACKED_BASE_REASONING_HEADER.startswith("## "), (
    f"STACKED_BASE_REASONING_HEADER must be a markdown H2 header, got {STACKED_BASE_REASONING_HEADER!r}"
)

__all__ = [
    "STACKED_MARKER_TRUE",
    "STACKED_MARKER_FALSE",
    "STACKED_MARKER_RE",
    "STACKER_META_ANALYSIS_HEADER",
    "STACKED_BASE_REASONING_HEADER",
]
