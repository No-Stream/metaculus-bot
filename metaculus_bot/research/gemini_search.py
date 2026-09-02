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
import re
from collections.abc import Sequence
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
from metaculus_bot.research.bracket_groups import (
    BRACKET_GROUP_RE,
    iter_group_items,
    join_group_items,
    rebuild_group,
)
from metaculus_bot.research.gemini_attribution import rewrite_unsupported_attributions
from metaculus_bot.research.provider_diagnostics import record_provider_detail
from metaculus_bot.research.providers import ResearchCallable
from metaculus_bot.research.raw_log import record_raw_research
from metaculus_bot.research.url_context_telemetry import (
    URL_RETRIEVAL_SUCCESS,
    extract_url_context_telemetry,
)

logger = logging.getLogger(__name__)

__all__ = [
    "build_gemini_client",
    "extract_url_context_telemetry",
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


_URL_CONTEXT_NONE_MARKER = "_url_context: none_"
_URL_CONTEXT_HEADER = "### URL Context Fetches"


def _format_url_context_marker(reported: bool, entries: list[tuple[str, str]]) -> str:
    """Build the greppable url_context telemetry block appended to persisted research.

    Only SUCCESSFUL fetches are listed inline (under ``### URL Context Fetches``) — those URLs were
    genuinely read, so they are real research context. Any other reported state (fired but fetched
    nothing, or every retrieval failed) collapses to the terse ``_url_context: none_`` marker, so a
    'did nothing useful' run never pushes failed/dead URLs at the forecaster. No url_context signal
    at all → empty string (no marker). Failed-fetch URLs are still captured in the INFO logs for
    auditing, just not in the forecaster-facing research blob.
    """
    successes = [(status, url) for status, url in entries if status == URL_RETRIEVAL_SUCCESS and url]
    if successes:
        lines = ["", "", _URL_CONTEXT_HEADER]
        lines.extend(f"{status} — {url}" for status, url in successes)
        return "\n".join(lines)
    if reported:
        return f"\n\n{_URL_CONTEXT_NONE_MARKER}"
    return ""


def _format_source_label(web: object) -> str:
    """Render a grounding chunk's web source as ``title — domain`` (no redirect URL).

    The SDK's ``chunk.web.uri`` is an opaque vertexaisearch grounding-api-redirect
    blob (~250 chars) that a text-only forecaster cannot resolve. ``chunk.web.domain``
    carries the real source domain (e.g. ``aljazeera.com``), so we render that plus
    the title and drop the blob entirely. Falls back to the title alone when the
    domain is absent (or vice versa); returns ``""`` when neither is present, so a
    label-less chunk contributes no line rather than leaking the redirect URL.
    """
    if web is None:
        return ""
    domain = (getattr(web, "domain", None) or "").strip()
    title = (getattr(web, "title", None) or "").strip()
    if title and domain and title != domain:
        return f"{title} — {domain}"
    return title or domain


def _splice_inline_citations(text: str, supports: Sequence[Any] | None) -> str:
    """Insert ``[N]`` citation markers into ``text`` at each support's segment boundary.

    Google's ``segment.end_index`` is a UTF-8 BYTE offset into the response text, so
    we splice on the encoded bytes rather than the Python str (which is indexed by
    codepoint). Indexing the str by a byte offset shifts every marker left by the
    count of multi-byte chars (em-dashes, smart quotes) before it, landing markers
    mid-word ("civilization. T[1]hese" instead of "civilization.[1] These").
    Iterating right-to-left keeps earlier byte offsets valid as we mutate the buffer.

    Returns ``text`` unchanged when there are no supports, and falls back to the
    ORIGINAL text (never a half-spliced buffer) if the splice fails.
    """
    if not supports:
        return text
    try:
        sorted_supports = sorted(
            supports,
            key=lambda s: s.segment.end_index if s.segment and s.segment.end_index is not None else 0,
            reverse=True,
        )
        annotated_bytes = text.encode("utf-8")
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
            annotated_bytes = annotated_bytes[:end_index] + marker_str.encode("utf-8") + annotated_bytes[end_index:]
        return annotated_bytes.decode("utf-8")
    except (AttributeError, TypeError, ValueError, UnicodeDecodeError) as exc:
        # Malformed supports (or a byte offset that lands mid-codepoint) shouldn't kill the response.
        logger.warning(f"GeminiSearch: could not splice inline citations ({type(exc).__name__}): {exc}")
        return text


# Gemini writes its OWN hierarchical citation indices — ``[2.4.1]``, ``[1.1.1, 1.1.2]``,
# ``[A: NASA, 1.1.2]`` — indexing a source list that does not exist on our side, alongside the
# resolvable ``[N]`` markers ``_splice_inline_citations`` puts in from real grounding metadata.
# 173 of 323 archived sections carry them and 163 carry BOTH families, so half the corpus hands a
# forecaster a bracket field where some brackets resolve and some are decoration, with nothing to
# tell them apart (scratch/residual_2026-08-31/gemini_search_audit/cutB_pattern.md §3.1).
#
# A dotted run only counts as an index when it is DELIMITED the way a citation is — sitting at a
# group edge or against whitespace/``,``/``;``/``:`` — and when every component is at most two
# digits. Both bounds are measured, not guessed: across the 2,609 archived bracket groups that
# hold a dotted token, first components run 1..6 and the largest component anywhere is 39. The
# two-digit bound is therefore comfortably above real indices while excluding the content classes
# that would otherwise match — a year (``[2026.08]``), an IP octet (``[192.168.1.1]``) — and the
# delimiter rule excludes a quantity (``[3.8%]``, ``[$1.5]``, ``[1.5 million]``) and a version
# (``[v2.1.3]``). Zero of those 2,609 groups is anything but a citation index (validation:
# scratch/next_season_bundle_2026-09/item3_citation_strip/VALIDATION.md).
#
# What a bracket group IS, where its items split, and how a rewritten one is put back
# together all come from ``research/bracket_groups.py`` — the same grammar the attribution
# check reads the same text through immediately after this pass, so the two cannot come to
# disagree about one string. Only the index token itself is this pass's own.
_CITATION_INDEX_RE = re.compile(r"(?<![^\s,;:])\d{1,2}(?:\.\d{1,2})+(?=\s*(?:[,;:]|$))")


def _tidy_group_item(item: str) -> str:
    """Normalize one comma/semicolon item of a bracket group after index removal.

    Drops the separator a removed index left behind (``2.1.4: A`` -> ``A``,
    ``A: official 2.4.1`` -> ``A: official``) and collapses the whitespace it opened up.
    """
    return re.sub(r"\s+", " ", item).strip().strip(":").strip()


def _strip_model_citation_indices(text: str) -> str:
    """Remove Gemini's self-authored hierarchical citation indices from bracket groups.

    MUST run AFTER ``_splice_inline_citations``: that function indexes the ORIGINAL response
    text by grounding-support BYTE offsets, so editing the text first would shift every offset
    and land our real ``[N]`` markers mid-word. Our markers are plain integers, so they survive
    this pass untouched; only dotted runs go.

    A group emptied of everything but punctuation is removed along with one preceding space, so
    ``"office [1.1.1, 1.1.2]. He"`` reads ``"office. He"`` rather than ``"office . He"``.
    Idempotent: a second pass finds no qualifying token.
    """

    def replace(match: re.Match[str]) -> str:
        inner = match.group("inner")
        stripped_inner = _CITATION_INDEX_RE.sub("", inner)
        if stripped_inner == inner:
            return match.group(0)
        # An item that is nothing but punctuation once its index is gone said only the
        # index; dropping it (rather than emptying it) is this pass's own filter, which is
        # why ``iter_group_items`` hands items over raw.
        kept: list[tuple[str, str]] = []
        for separator, item in iter_group_items(stripped_inner):
            tidied = _tidy_group_item(item)
            if any(char.isalnum() for char in tidied):
                kept.append((separator, tidied))
        if not kept:
            return ""
        return rebuild_group(match, join_group_items(kept))

    return BRACKET_GROUP_RE.sub(replace, text)


def _grounded_source_labels(chunks: Sequence[Any]) -> list[tuple[int, str]]:
    """``(1-based chunk index, rendered label)`` for every chunk that carries a label.

    The single derivation of "what our grounding record says", read by both the
    ``### Sources`` block and the unsupported-attribution check — so the check can never
    judge an attribution against a source list different from the one the forecaster is
    shown. The index is the CHUNK's, not the surviving entry's, because the spliced inline
    ``[N]`` markers point at chunk positions; renumbering would misaim them.
    """
    labels = []
    for idx, chunk in enumerate(chunks, start=1):
        label = _format_source_label(chunk.web)
        if label:
            labels.append((idx, label))
    return labels


def _render_sources_section(chunks: Sequence[Any]) -> str:
    """Render the trailing ``### Sources`` block, or ``""`` when no chunk carries a label.

    Renders the real source domain, NOT the opaque
    vertexaisearch.cloud.google.com/grounding-api-redirect/<~250-char blob> URI. The
    domain carries all the signal a text-only forecaster can use, and the redirect
    blobs were ~5% of the whole research bundle. Entries stay 1:1 with the grounding
    chunks so the inline [N] markers keep pointing at the right source (deduping
    would misalign them).
    """
    sources_lines = ["", "", "### Sources"]
    sources_lines.extend(f"[{idx}] {label}" for idx, label in _grounded_source_labels(chunks))
    return "\n".join(sources_lines) if len(sources_lines) > _SOURCES_HEADER_LEN else ""


def _check_attributions(text: str, chunks: Sequence[Any], *, qid: int | None) -> str:
    """Mark the tier-tag attributions this response's own grounding record cannot back.

    Runs AFTER the citation-index strip, on the annotated body only (the ``### Sources``
    block is appended afterwards and never passes through), and only where we have
    renderable grounded labels to compare against — an empty label list is a measurement
    failure rather than a verdict, so it leaves every tag standing and records nothing.
    That absence is the signal: on a schema-v2 record, no ``unsupported_attributions``
    count means the check had no evidence base (or the record predates it), while a
    recorded 0 means it ran and found nothing.

    Deliberately NOT alertable and nothing keys on the count: 70% of the archived corpus's
    outlet-named tier tags are unsupported, so this is the model's habitual embellishment
    rather than a bot defect, and an absent outlet does not make the FACT wrong.
    """
    labels = [label for _idx, label in _grounded_source_labels(chunks)]
    if not labels:
        return text
    checked = rewrite_unsupported_attributions(text, labels)
    # ``tier_tags`` rides alongside because the count this check exists to report is not
    # readable without it: the marker below is gated on ``unsupported``, so a response that
    # carried no tier tags at all and one whose every tag was backed both archive as
    # ``unsupported_attributions=0`` and log nothing. It counts OUTLET-NAMED tier items only
    # (generic tier words like "official" — 307 of the corpus's 790 items — are excluded
    # before matching), so a 0 reads as "no outlet-named tags", not "no tier tags"; the
    # definitive check for the latter is a grep for "[A: " over the archived section.
    record_provider_detail(
        qid,
        "gemini_search",
        {"counts": {"tier_tags": checked.tagged, "unsupported_attributions": checked.unsupported}},
    )
    if checked.unsupported:
        # ``labels`` rides the line because the same count reads completely differently
        # against it: q38195 named 21 outlets over ONE grounded domain.
        logger.info(
            f"GEMINI_UNSUPPORTED_ATTRIBUTION: question={qid} tagged={checked.tagged} "
            f"unsupported={checked.unsupported} groups={checked.groups_rewritten} labels={len(labels)}"
        )
    return checked.text


def _format_grounded_response(
    response: genai_types.GenerateContentResponse,
    *,
    qid: int | None = None,
    model: str | None = None,
) -> str:
    """Stitch response text with inline citations from grounding metadata.

    Output format:
        <response text>

        ### Sources
        [1] <title> — <domain>
        [2] <title> — <domain>
        ...

    Inline citation markers are inserted per-segment using
    grounding_metadata.grounding_supports, iterating in reverse end_index order
    so index offsets stay valid while we mutate the string. Falls back to a
    plain-text + sources-list if supports are missing. The model's own hierarchical
    ``[2.4.1]`` indices are then stripped (``_strip_model_citation_indices``) so every
    bracket left in the body resolves against the rendered ``### Sources`` list, and the
    surviving source-tier tags are checked against that same list (``_check_attributions``)
    so an outlet our own grounding record cannot back reads as
    ``[unverified attribution]``; the sources block itself is appended after both passes
    and never goes through either.

    Grounded-chunk floor: a response with no grounding evidence at all — zero
    google_search chunks AND no successful url_context read — is suppressed
    (returns ``""``) rather than passed through, because the whole premise of this
    provider is grounded retrieval and ungrounded Gemini text is a demonstrated
    fabrication vector (Q38195, 2026-07-19: 30 search queries, 0 grounding chunks,
    a confident fabricated contract table with fake ``[primary]`` tags reached
    forecasters). ``qid`` / ``model`` are threaded in only to make that
    suppression WARN greppable. "No grounding evidence" includes a response with no
    candidates at all: there is no path around the floor that returns text.
    """
    text = response.text or ""
    if not text:
        return ""

    # No candidates means no grounding metadata at all, which IS the ungrounded case the
    # floor below exists to refuse — the old early `return text` here bypassed the Q38195
    # guard entirely. Unreachable on today's SDK (``response.text`` is derived from a
    # candidate, so text-without-candidates cannot happen), but a hole in a fabrication
    # guard should not depend on an SDK invariant it doesn't own.
    candidates = response.candidates
    metadata = candidates[0].grounding_metadata if candidates else None
    if metadata is None or not metadata.grounding_chunks:
        # Grounded-chunk floor. No google_search grounding chunks reached us:
        # either the search tool never fired (metadata is None) or it fired and
        # grounded nothing (chunks empty, the Q38195 case). The only reason to
        # still pass the text through is a successful url_context read — that is
        # genuine retrieval, so it counts as grounding. Absent both, the text is
        # ungrounded parametric output; suppress the section (the orchestrator
        # then omits it) and leave a greppable WARN.
        _, _, n_url_success, _ = extract_url_context_telemetry(response)
        if n_url_success == 0:
            n_queries = len(metadata.web_search_queries or []) if metadata is not None else 0
            logger.warning(f"GEMINI_UNGROUNDED_SUPPRESSED: question={qid} model={model} queries={n_queries}")
            # Record the loss so the Provider Diagnostics line and the schema-v2 archive
            # carry a `lost=grounding:...` token. Without it, the "" this returns maps to
            # ProviderResult status `empty` — byte-identical to a healthy Gemini call that
            # legitimately found nothing, since the provider didn't raise and so no counter
            # moves. Mirrors _degraded_to_raw_articles, which solved the same shape for the
            # AskNews summarizer. Deliberately NOT an alertable counter: folding a new term
            # into alertable_count changes what CI treats as red, which is the operator's
            # call, not a side effect of adding visibility.
            record_provider_detail(qid, "gemini_search", {"sources": {"grounding": "error(ungrounded_suppressed)"}})
            return ""
        # url_context grounded the text but google_search produced no chunks: keep
        # the text as-is (no citation markers to splice, no Sources block). The
        # caller appends the url_context fetch marker. No GEMINI_GROUNDING_DENSITY here:
        # the marker measures google_search support density, and a response with neither
        # chunks nor supports has an undefined density rather than a zero one.
        return _strip_model_citation_indices(text)

    supports = metadata.grounding_supports or ()
    # Grounding DENSITY, as telemetry and never as a gate: post-floor the median response
    # carries one support per ~872 chars and 41% of passing responses have <=3 supports,
    # which is the floor-immune surface where the ~33% embellishment rate lives. Not a gate
    # because q44944's decisive, true, later-verified ICE figure came out of a 1-support
    # response (gemini_search_audit/VERDICT.md §2-3). ``chars`` is the RAW model text — the
    # denominator the audit measured — not the annotated or sources-appended length.
    logger.info(
        f"GEMINI_GROUNDING_DENSITY: question={qid} chunks={len(metadata.grounding_chunks)} "
        f"supports={len(supports)} chars={len(text)}"
    )
    annotated = _strip_model_citation_indices(_splice_inline_citations(text, supports))
    return _check_attributions(annotated, metadata.grounding_chunks, qid=qid) + _render_sources_section(
        metadata.grounding_chunks
    )


async def invoke_gemini_grounded(
    prompt: str,
    *,
    model_slug: str | None = None,
    include_url_context: bool = True,
    qid: int | None = None,
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
    except TimeoutError:
        logger.warning(f"GeminiSearch: {model} timed out after {GEMINI_SEARCH_TIMEOUT}s")
        raise

    # Capture the raw SDK response (text + grounding metadata: the actual Google
    # queries and sources) before formatting drops most of it.
    record_raw_research(qid=qid, provider="gemini_search", payload=response)

    formatted = _format_grounded_response(response, qid=qid, model=model)
    n_chunks = 0
    candidates = response.candidates
    if candidates:
        metadata = candidates[0].grounding_metadata
        if metadata is not None and metadata.grounding_chunks:
            n_chunks = len(metadata.grounding_chunks)

    reported, n_url_total, n_url_success, url_entries = extract_url_context_telemetry(response)
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

    async def _fetch(question: MetaculusQuestion) -> str:
        prompt = web_research_prompt(
            question.question_text,
            # The MC ballot (None on other types): grounded search can only query candidate
            # names it has been shown (q44952 — zero retrieval on the eventual winner).
            options=getattr(question, "options", None),
            is_benchmarking=is_benchmarking,
            citation_style="auto_annotated",
            allow_resolution_source_reading=True,
        )
        return await invoke_gemini_grounded(
            prompt, model_slug=model_slug, qid=getattr(question, "id_of_question", None)
        )

    return _fetch
