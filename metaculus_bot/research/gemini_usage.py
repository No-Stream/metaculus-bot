"""Token-usage telemetry for the two native google-genai call sites.

Both native Gemini paths — grounded search (``research/gemini_search.py``) and gap-fill
v2's ``read_document`` (``research/agentic/tool_backends.py``) — bill the operator's
personal Google AI Studio key, and neither recorded what it spent. Google-side spend was
therefore invisible in the run logs: the 2026-09 reconstruction of it only worked because
the raw research archive happened to store whole SDK responses, which is an accident of a
different feature rather than a measurement. One INFO line per response makes it a query,
and the thinking split is the field worth having — 71% of grounded-search output tokens
measured as thinking, which is what the explicit thinking levels were set from.

The marker is a CONTRACT with ``scripts/telemetry/markers.py``: field order and spelling
are what the archive parser reads, and ``question`` is last because it is optional (the
document reader has no question id to carry).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_NOT_AVAILABLE = "n/a"

# (marker field, SDK ``GenerateContentResponseUsageMetadata`` attribute). Every one of
# these is Optional on the SDK model, so any of them can legitimately read ``n/a``.
_TOKEN_FIELDS: tuple[tuple[str, str], ...] = (
    ("prompt_tokens", "prompt_token_count"),
    ("tool_use_prompt_tokens", "tool_use_prompt_token_count"),
    ("candidates_tokens", "candidates_token_count"),
    ("thoughts_tokens", "thoughts_token_count"),
    ("total_tokens", "total_token_count"),
)


def _render_count(value: object) -> str:
    """Render one token count, or ``n/a`` when the API reported nothing for it.

    A missing count must never render as 0: ``thoughts_tokens=0`` is a real and meaningful
    reading (the model did not think) and an absent field would be indistinguishable from
    it, which would turn a gap in Google's reporting into a measurement we never made.
    ``bool`` is excluded because it is an ``int`` subclass and a True would render as 1.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return _NOT_AVAILABLE
    return str(value)


def _reported_model(model_version: object, configured_model: str) -> str:
    """The served model id, falling back to the configured one.

    Anything that is not a non-empty string falls back: the marker's fields are
    whitespace-delimited, and a stray object's repr (a test double's, or a shape a future
    SDK returns) carries spaces and would silently split into fields the parser then reads
    as a token count.
    """
    if isinstance(model_version, str) and model_version:
        return model_version
    return configured_model


def _render_token_counts(response: Any) -> list[tuple[str, str]]:
    """The five token counts as ``(field, rendered)`` pairs, all ``n/a`` if unreadable."""
    try:
        usage = getattr(response, "usage_metadata", None)
        return [(field, _render_count(getattr(usage, attribute, None))) for field, attribute in _TOKEN_FIELDS]
    except (AttributeError, TypeError, ValueError) as exc:
        logger.warning(f"GeminiUsage: could not read usage metadata ({type(exc).__name__}): {exc}")
        return [(field, _NOT_AVAILABLE) for field, _attribute in _TOKEN_FIELDS]


def _render_search_queries(response: Any) -> str:
    """Total ``web_search_queries`` across every candidate's grounding metadata.

    Zero rather than ``n/a`` when nothing is there: the SDK omits the list when the search
    tool issued no queries, so an absent list IS a count of none, which is also the honest
    reading for a url_context-only document read. Summed over all candidates because the
    field is per-candidate, even though today's calls return one.
    """
    try:
        total = 0
        for candidate in getattr(response, "candidates", None) or ():
            metadata = getattr(candidate, "grounding_metadata", None)
            if metadata is not None:
                total += len(metadata.web_search_queries or ())
        return str(total)
    except (AttributeError, TypeError, ValueError) as exc:
        logger.warning(f"GeminiUsage: could not count search queries ({type(exc).__name__}): {exc}")
        return _NOT_AVAILABLE


def log_gemini_usage(response: Any, *, role: str, model: str, question: str | None = None) -> str:
    """Log and return one ``GEMINI_USAGE`` line for a google-genai response.

    ``model`` is the configured id, used only as the fallback: the line prefers the
    response's own ``model_version``, which is what actually served the request and so what
    a spend reconciliation has to key on when an alias resolves elsewhere.

    Never raises. This is pure observation attached to a path that bills money and returns
    research, so a telemetry read must not be able to take that path down; a payload it
    cannot read reports ``n/a`` rather than a fabricated zero. The token counts and the
    query count are guarded SEPARATELY because the tokens are the reason the marker exists:
    a response whose grounding metadata we cannot walk must not cost us the spend figures.
    """
    counts = _render_token_counts(response)
    search_queries = _render_search_queries(response)
    reported_model = _reported_model(getattr(response, "model_version", None), model)

    fields = " ".join(f"{field}={value}" for field, value in counts)
    line = f"GEMINI_USAGE: role={role} model={reported_model} {fields} search_queries={search_queries}"
    if question is not None:
        line += f" question={question}"
    logger.info(line)
    return line
