"""Gemini grounded search research provider.

Uses the `google-genai` SDK directly (NOT via OpenRouter) so we get real
first-party Google Search grounding rather than OpenRouter's Exa-backed web
plugin. This adds a genuinely distinct search index to the ensemble — the
Metaculus Fall 2025 writeup identified research breadth as the single
strongest predictor of winning bots.

Mirrors `_native_search_provider` in `research_providers.py` for consistency.
"""

import asyncio
import functools
import logging
import os
from typing import Any

from forecasting_tools.data_models.questions import MetaculusQuestion
from google import genai
from google.genai import types as genai_types

from metaculus_bot.constants import (
    GEMINI_SEARCH_DEFAULT_MODEL,
    GEMINI_SEARCH_MODEL_ENV,
    GEMINI_SEARCH_TIMEOUT,
    GOOGLE_API_KEY_ENV,
)
from metaculus_bot.prompts import web_research_prompt
from metaculus_bot.research.providers import ResearchCallable

logger = logging.getLogger(__name__)

__all__ = [
    "_extract_url_context_telemetry",
    "build_gemini_client",
    "gemini_search_provider",
    "invoke_gemini_grounded",
]

# Header-only initializer for the sources list. Checking `len(sources_lines) > _SOURCES_HEADER_LEN`
# against this named constant keeps the sources-present gate tied to the init block.
_SOURCES_HEADER_LEN = 3


@functools.lru_cache(maxsize=1)
def _cached_client_for_key(api_key: str) -> genai.Client:
    """Process-global cached genai.Client keyed on API key.

    SDK clients are designed to be long-lived; keeping one across a backtest
    lets TLS connections and HTTP/2 multiplexing be reused across the ~thousands
    of calls the Gemini provider + gap-fill make per round. Keyed on api_key so
    a rotated key (rare) produces a fresh client.
    """
    return genai.Client(api_key=api_key)


def build_gemini_client() -> genai.Client:
    """Return the cached google-genai Client for the operator's personal Gemini key.

    Reads GOOGLE_API_KEY (the operator's personal Google AI Studio key — in CI
    populated from ``secrets.GEMINI_API_KEY``). There is no Metaculus-donated
    Gemini key on the google-genai side; the donated path only exists for
    OpenRouter-routed Gemini models. Raises ValueError if the key is missing
    so misconfiguration is loud.
    """
    api_key = os.getenv(GOOGLE_API_KEY_ENV)
    if not api_key:
        raise ValueError(f"{GOOGLE_API_KEY_ENV} must be set to use the Gemini search provider")
    return _cached_client_for_key(api_key)


def _resolve_model(model_slug: str | None) -> str:
    return model_slug or os.getenv(GEMINI_SEARCH_MODEL_ENV, GEMINI_SEARCH_DEFAULT_MODEL)


_URL_RETRIEVAL_SUCCESS = "URL_RETRIEVAL_STATUS_SUCCESS"
_URL_CONTEXT_NONE_MARKER = "_url_context: none_"
_URL_CONTEXT_HEADER = "### URL Context Fetches"


def _coerce_status_name(status: object) -> str:
    """Coerce a url_retrieval_status (enum-with-.name, plain string, or None) to a string name."""
    name = getattr(status, "name", None)
    if isinstance(name, str):
        return name
    if isinstance(status, str):
        return status
    return str(status)


def _extract_url_context_telemetry(
    response: genai_types.GenerateContentResponse,
) -> tuple[bool, int, int, list[tuple[str, str]]]:
    """Pull url_context retrieval telemetry off a grounded-search response.

    Returns ``(reported, n_total, n_success, [(status_name, retrieved_url), ...])``. ``reported``
    is whether the SDK attached a url_context_metadata object at all — the tool ran and reported
    back, even with an empty fetch list — which lets callers keep 'fired but fetched nothing'
    greppably distinct from 'no url_context signal'. ``n_total`` is the url_metadata entry count
    and ``n_success`` those with status URL_RETRIEVAL_STATUS_SUCCESS.

    Every field read here is a declared field on the typed SDK pydantic models
    (``url_context_metadata``, ``url_metadata``, and each ``UrlMetadata``'s
    ``url_retrieval_status`` / ``retrieved_url``), so they're accessed directly: a
    future SDK rename should fail loudly here rather than silently report zero
    forever, matching the direct ``.grounding_metadata`` access below. A
    None-valued (but present) entry still coerces gracefully —
    ``_coerce_status_name(None)`` maps to ``"None"`` and ``None or ""`` to ``""`` —
    so telemetry never takes down the research path it only observes.
    """
    candidates = response.candidates
    if not candidates:
        return (False, 0, 0, [])

    url_context_metadata = candidates[0].url_context_metadata
    if url_context_metadata is None:
        return (False, 0, 0, [])

    url_metadata = url_context_metadata.url_metadata
    if not url_metadata:
        return (True, 0, 0, [])

    entries: list[tuple[str, str]] = []
    n_success = 0
    for meta in url_metadata:
        status_name = _coerce_status_name(meta.url_retrieval_status)
        retrieved_url = meta.retrieved_url or ""
        if status_name == _URL_RETRIEVAL_SUCCESS:
            n_success += 1
        entries.append((status_name, retrieved_url))

    return (True, len(url_metadata), n_success, entries)


def _format_url_context_marker(reported: bool, entries: list[tuple[str, str]]) -> str:
    """Build the greppable url_context telemetry block appended to persisted research.

    Only SUCCESSFUL fetches are listed inline (under ``### URL Context Fetches``) — those URLs were
    genuinely read, so they are real research context. Any other reported state (fired but fetched
    nothing, or every retrieval failed) collapses to the terse ``_url_context: none_`` marker, so a
    'did nothing useful' run never pushes failed/dead URLs at the forecaster. No url_context signal
    at all → empty string (no marker). Failed-fetch URLs are still captured in the INFO logs for
    auditing, just not in the forecaster-facing research blob.
    """
    successes = [(status, url) for status, url in entries if status == _URL_RETRIEVAL_SUCCESS and url]
    if successes:
        lines = ["", "", _URL_CONTEXT_HEADER]
        lines.extend(f"{status} — {url}" for status, url in successes)
        return "\n".join(lines)
    if reported:
        return f"\n\n{_URL_CONTEXT_NONE_MARKER}"
    return ""


def _format_grounded_response(response: genai_types.GenerateContentResponse) -> str:
    """Stitch response text with inline citations from grounding metadata.

    Output format:
        <response text>

        ### Sources
        [1] <title> — <uri>
        [2] <title> — <uri>
        ...

    Inline citation markers are inserted per-segment using
    grounding_metadata.grounding_supports, iterating in reverse end_index order
    so index offsets stay valid while we mutate the string. Falls back to a
    plain-text + sources-list if supports are missing.
    """
    text = response.text or ""
    if not text:
        return ""

    candidates = response.candidates
    if not candidates:
        return text

    metadata = candidates[0].grounding_metadata
    if metadata is None:
        # Grounding didn't fire — return plain text.
        return text

    chunks = metadata.grounding_chunks
    supports = metadata.grounding_supports

    if not chunks:
        # Grounding enabled but model didn't cite anything — return the text as-is.
        return text

    # Insert inline citation markers based on supports, iterating right-to-left
    # so earlier offsets remain stable as we mutate the string.
    annotated = text
    if supports:
        try:
            sorted_supports = sorted(
                supports,
                key=lambda s: s.segment.end_index if s.segment and s.segment.end_index is not None else 0,
                reverse=True,
            )
            for support in sorted_supports:
                segment = support.segment
                if segment is None or segment.end_index is None:
                    continue
                chunk_indices = support.grounding_chunk_indices
                if not chunk_indices:
                    continue
                # Convert to 1-indexed markers, dedup, sort for readability.
                markers = sorted({int(i) + 1 for i in chunk_indices})
                marker_str = "[" + ", ".join(str(m) for m in markers) + "]"
                end_index = segment.end_index
                annotated = annotated[:end_index] + marker_str + annotated[end_index:]
        except (AttributeError, TypeError, ValueError) as exc:
            # Malformed supports shouldn't kill the whole response.
            logger.warning(f"GeminiSearch: could not splice inline citations ({type(exc).__name__}): {exc}")
            annotated = text

    # Append a Sources section
    sources_lines = ["", "", "### Sources"]
    for idx, chunk in enumerate(chunks, start=1):
        web = chunk.web
        if web and web.uri:
            if web.title:
                sources_lines.append(f"[{idx}] {web.title} — {web.uri}")
            else:
                sources_lines.append(f"[{idx}] {web.uri}")

    return annotated + "\n".join(sources_lines) if len(sources_lines) > _SOURCES_HEADER_LEN else annotated


async def invoke_gemini_grounded(
    prompt: str,
    *,
    model_slug: str | None = None,
    include_url_context: bool = True,
) -> str:
    """Invoke Gemini with Google Search grounding and return formatted text.

    Used by the first-pass Gemini search provider (and the ablation harness);
    gap-fill uses OpenAI native search, not this google-genai grounded path.
    Enables the URL context tool alongside Google Search by default so the model
    can directly read specific URLs (e.g., resolution sources named in question
    fine print).

    Raises on SDK errors — callers decide whether to fail hard or soft.
    """
    client = build_gemini_client()
    model = _resolve_model(model_slug)

    tools: list[Any] = [{"google_search": {}}]
    if include_url_context:
        tools.append({"url_context": {}})

    config = genai_types.GenerateContentConfig(tools=tools)

    logger.info(f"GeminiSearch: calling {model} with grounding")
    try:
        response = await asyncio.wait_for(
            client.aio.models.generate_content(model=model, contents=prompt, config=config),
            timeout=GEMINI_SEARCH_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning(f"GeminiSearch: {model} timed out after {GEMINI_SEARCH_TIMEOUT}s")
        raise

    formatted = _format_grounded_response(response)
    n_chunks = 0
    candidates = response.candidates
    if candidates:
        metadata = candidates[0].grounding_metadata
        if metadata is not None and metadata.grounding_chunks:
            n_chunks = len(metadata.grounding_chunks)

    reported, n_url_total, n_url_success, url_entries = _extract_url_context_telemetry(response)
    logger.info(
        f"GeminiSearch: got {len(formatted)} chars, {n_chunks} grounding chunks, "
        f"{n_url_success}/{n_url_total} url_context fetches from {model}"
    )
    if url_entries:
        for status, url in url_entries:
            logger.info(f"GeminiSearch: url_context {status} — {url}")

    # Only annotate non-empty research; an empty result must stay empty so callers can soft-fail.
    if formatted:
        formatted += _format_url_context_marker(reported, url_entries)
    return formatted


def gemini_search_provider(
    model_slug: str | None = None,
    is_benchmarking: bool = False,
) -> ResearchCallable:
    """Research provider using Gemini with Google Search grounding.

    Mirrors the `_native_search_provider` contract (`MetaculusQuestion -> str`).
    """

    async def _fetch(question: MetaculusQuestion) -> str:  # noqa: D401
        prompt = web_research_prompt(
            question.question_text,
            is_benchmarking=is_benchmarking,
            citation_style="auto_annotated",
            allow_resolution_source_reading=True,
        )
        return await invoke_gemini_grounded(prompt, model_slug=model_slug)

    return _fetch
