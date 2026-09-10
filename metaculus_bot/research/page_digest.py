"""An LLM-extractive digest of a long web page, grounded literally against the page text.

A cited page over the per-URL character cap used to be read from the top, so its tail was
unreachable; the gap-fill loop's deterministic BM25 digest reached the tail but the operator does
not trust a lexical ranker as the primary mechanism (decision of 2026-09-09, recorded in
``scratch_docs_and_planning/fetch_ladder_unification_plan_2026-09-09.md``). So a cheap model reads
the page and returns the passages that bear on the caller's query, verbatim and ranked, and a
literal grounding check drops anything it did not copy. BM25 keeps two jobs: the free pre-filter
that cuts a very long page to a few thousand tokens before the model reads it, and the fallback
when the call fails, returns nothing grounded, or cannot fit inside the caller's remaining wall.
The fallback is exactly the digest both callers shipped before, which is what makes this a
strictly-safer change on the fetch wall.

One entry point, :func:`digest_page`, takes the page text, the caller's query and the remaining
wall in seconds and returns a :class:`PageDigest`. Every call bills the ``page_digest_extractor``
role on the ``CREDIT_ROLE_SPEND`` ledger through the same builder every support role uses. The
three counters the digest carries ride the callers' ``RESOLUTION_SOURCE_FETCH`` marker as optional
tail fields (``passages_returned``, ``passages_grounded``, ``fallback_used``); this module emits no
marker of its own. Detail: docs/research.md "Page digest".
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import openai
from forecasting_tools import GeneralLlm
from pydantic import BaseModel, ValidationError

from metaculus_bot.constants import (
    DOCUMENT_DIGEST_TOP_K,
    DOCUMENT_DIGEST_WINDOW_CHARS,
    PAGE_DIGEST_EXTRACTOR_EFFORT,
    PAGE_DIGEST_EXTRACTOR_MODEL,
    PAGE_DIGEST_EXTRACTOR_TIMEOUT_S,
    PAGE_DIGEST_MIN_CALL_BUDGET_S,
    PAGE_DIGEST_PREFILTER_MAX_CHARS,
    PAGE_DIGEST_WALL_MARGIN_S,
)
from metaculus_bot.fallback_openrouter import build_llm_with_openrouter_fallback
from metaculus_bot.research.document_text import select_passages

logger = logging.getLogger(__name__)

# The CREDIT_ROLE_SPEND row every extractor call books under (docs/operations.md "Per-role spend").
PAGE_DIGEST_ROLE = "page_digest_extractor"

# ``PageDigest.method`` values: which mechanism chose the passages a reader sees.
DIGEST_METHOD_LLM_EXTRACTIVE = "llm_extractive"
DIGEST_METHOD_BM25 = "bm25"

# Marks a cut between non-adjacent pre-filter windows so the model cannot read across our splice.
_WINDOW_SEPARATOR = "\n[...]\n"

# Each clause names a shape the grounding check drops unread (paraphrase, summary, splice); docs/research.md "Page digest".
PAGE_DIGEST_EXTRACTOR_PROMPT = """You are extracting passages from a web page for a forecaster who cannot read the whole page.

Query (what the forecaster needs from this page):
{query}

Return the passages of the page that bear on the query, most relevant first.
- Copy each passage VERBATIM from the page text below, character for character. Do not paraphrase, summarize, translate, correct, or join non-adjacent text: a passage that is not an exact copy is discarded unread.
- A passage is one contiguous span: a sentence, a few sentences, or a block of table rows. Keep every number with its label, date and unit.
- Return at most {max_passages} passages, and an empty list if nothing on the page bears on the query.

Page text:
{page_text}"""

# Each is an outcome a healthy deployment produces, and each has one remedy: the free BM25 digest.
_EXPECTED_CALL_FAILURES: tuple[type[BaseException], ...] = (
    TimeoutError,  # asyncio.wait_for at the call budget, so the caller's wall never slips
    openai.APIError,  # the root of every litellm provider and API error (llm_retry.llm_status_code)
    ValidationError,  # the model answered off-schema, and a second paid attempt cannot fit inside the wall
)


class PageDigestPassages(BaseModel):
    """The extractor's structured answer: the verbatim passages, most relevant first."""

    passages: list[str]


@dataclass(frozen=True)
class PageDigest:
    """What a reader sees of a long page, plus the three counters the fetch marker carries.

    ``passages`` opens with the page's first window whenever the page has one, so a reader still
    sees what the page is, then the accepted passages in rank order. ``passages_returned`` is the
    raw count the model answered with and ``passages_grounded`` how many of those were literal
    substrings of the page, so the difference is the model's fabrication count; both are zero when
    no call was made. ``fallback_used`` is True whenever the passages after the opening came from
    BM25 rather than the model, and ``method`` names which.
    """

    passages: list[str]
    passages_returned: int
    passages_grounded: int
    fallback_used: bool
    method: str


async def digest_page(text: str, query: str, *, budget_seconds: float) -> PageDigest:
    """The passages of ``text`` that bear on ``query``, inside ``budget_seconds`` of the caller's wall.

    The one paid call is bounded by ``min(PAGE_DIGEST_EXTRACTOR_TIMEOUT_S, budget_seconds -
    PAGE_DIGEST_WALL_MARGIN_S)`` and is not attempted at all below ``PAGE_DIGEST_MIN_CALL_BUDGET_S``
    or on an empty page, so a caller can always hand over its remaining wall verbatim. Every path
    that does not return the model's grounded passages returns the BM25 digest with
    ``fallback_used=True``.
    """
    opening = _opening_passage(text)
    call_budget = min(PAGE_DIGEST_EXTRACTOR_TIMEOUT_S, budget_seconds - PAGE_DIGEST_WALL_MARGIN_S)
    if not text.strip() or call_budget < PAGE_DIGEST_MIN_CALL_BUDGET_S:
        return _bm25_digest(text, query, opening=opening, passages_returned=0, passages_grounded=0)

    prompt = PAGE_DIGEST_EXTRACTOR_PROMPT.format(
        query=query, max_passages=DOCUMENT_DIGEST_TOP_K, page_text=_prefiltered_text(text, query)
    )
    try:
        raw = await asyncio.wait_for(_build_extractor().invoke(prompt), timeout=call_budget)
        returned = PageDigestPassages.model_validate_json(raw).passages
    except _EXPECTED_CALL_FAILURES as exc:
        logger.warning(f"PAGE_DIGEST extractor call failed ({type(exc).__name__}); serving the BM25 digest")
        return _bm25_digest(text, query, opening=opening, passages_returned=0, passages_grounded=0)

    grounded, kept = _ground(returned, text, opening)
    if not kept:
        return _bm25_digest(text, query, opening=opening, passages_returned=len(returned), passages_grounded=grounded)
    return PageDigest(
        passages=_with_opening(opening, kept),
        passages_returned=len(returned),
        passages_grounded=grounded,
        fallback_used=False,
        method=DIGEST_METHOD_LLM_EXTRACTIVE,
    )


def _build_extractor() -> GeneralLlm:
    return build_llm_with_openrouter_fallback(
        model=PAGE_DIGEST_EXTRACTOR_MODEL,
        role=PAGE_DIGEST_ROLE,
        reasoning={"effort": PAGE_DIGEST_EXTRACTOR_EFFORT},
        timeout=PAGE_DIGEST_EXTRACTOR_TIMEOUT_S,
        allowed_tries=1,  # the BM25 fallback is the retry; a second paid attempt cannot fit inside the wall
        response_format=PageDigestPassages,
        extra_body={"provider": {"require_parameters": True}},  # OpenRouter rejects, rather than drops, the schema
    )


def _prefiltered_text(text: str, query: str) -> str:
    """What the model reads: the whole page under the cap, else its best BM25 windows in page order."""
    if len(text) <= PAGE_DIGEST_PREFILTER_MAX_CHARS:
        return text
    windows = select_passages(text, query, top_k=PAGE_DIGEST_PREFILTER_MAX_CHARS // DOCUMENT_DIGEST_WINDOW_CHARS)
    if not windows:  # no query token anywhere on the page: the head is what a reader saw before
        return text[:PAGE_DIGEST_PREFILTER_MAX_CHARS]
    return _WINDOW_SEPARATOR.join(window.text for window in sorted(windows, key=lambda window: window.start))


def _ground(returned: list[str], text: str, opening: str) -> tuple[int, list[str]]:
    """``(grounded count, kept passages)``: literal substrings of the page, minus repeats and the opening."""
    page = _normalize(text)
    opening_norm = _normalize(opening)
    grounded = 0
    kept: list[str] = []
    seen: set[str] = set()
    for passage in returned:
        candidate = _normalize(passage)
        if not candidate or candidate not in page:
            continue
        grounded += 1
        if candidate in opening_norm or candidate in seen:
            continue
        seen.add(candidate)
        kept.append(passage.strip())
    return grounded, kept


def _bm25_digest(text: str, query: str, *, opening: str, passages_returned: int, passages_grounded: int) -> PageDigest:
    """Today's deterministic digest, with the model's counters carried through for the marker."""
    opening_norm = _normalize(opening)
    ranked = [p.text for p in select_passages(text, query, top_k=DOCUMENT_DIGEST_TOP_K)]
    kept = [passage for passage in ranked if _normalize(passage) not in opening_norm]
    return PageDigest(
        passages=_with_opening(opening, kept),
        passages_returned=passages_returned,
        passages_grounded=passages_grounded,
        fallback_used=True,
        method=DIGEST_METHOD_BM25,
    )


def _opening_passage(text: str) -> str:
    """The page's first digest window, cut on a word boundary so it never ends mid-token."""
    head = text[:DOCUMENT_DIGEST_WINDOW_CHARS]
    if len(text) > DOCUMENT_DIGEST_WINDOW_CHARS and " " in head:
        head = head.rsplit(" ", 1)[0]
    return head.strip()


def _with_opening(opening: str, passages: list[str]) -> list[str]:
    return [opening, *passages] if opening else list(passages)


def _normalize(text: str) -> str:
    return " ".join(text.split())
