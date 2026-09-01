"""Tests for the Gemini grounded search research provider.

These tests mock the google-genai SDK at the module level; no live API calls.
Patterns mirror ``tests/test_native_search_provider.py``.
"""

from collections.abc import Sequence
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai import types as genai_types

from metaculus_bot.research.provider_diagnostics import _is_lost_source, pop_provider_detail


def _make_q(text: str) -> MagicMock:
    """Build a minimal MetaculusQuestion-shaped mock for the new ResearchCallable
    contract. Tests only care about question_text on this path."""
    q = MagicMock()
    q.question_text = text
    return q


# ---------------------------------------------------------------------------
# Canned response helpers (for grounding metadata tests)
# ---------------------------------------------------------------------------


class CannedWebChunk:
    def __init__(self, uri: str, title: str | None, domain: str | None = None) -> None:
        self.web = SimpleNamespace(uri=uri, title=title, domain=domain)


class CannedSegment:
    def __init__(self, end_index: int, text: str) -> None:
        self.end_index = end_index
        self.text = text


class CannedSupport:
    def __init__(self, seg: CannedSegment, indices: list[int]) -> None:
        self.segment = seg
        self.grounding_chunk_indices = indices


class CannedStatus:
    """A url_retrieval_status enum stand-in exposing ``.name`` like the real SDK enum."""

    def __init__(self, name: str) -> None:
        self.name = name


class CannedUrlMeta:
    """Mirror of ``google.genai.types.UrlMetadata`` (retrieved_url + url_retrieval_status)."""

    def __init__(self, retrieved_url: str | None, url_retrieval_status: object) -> None:
        self.retrieved_url = retrieved_url
        self.url_retrieval_status = url_retrieval_status


def _make_response(
    text: str,
    chunks: list[CannedWebChunk] | None = None,
    supports: list[CannedSupport] | None = None,
    url_metadata: Sequence[object] | None = None,
    web_search_queries: list[str] | None = None,
) -> SimpleNamespace:
    # ``web_search_queries`` is a declared field on the real SDK GroundingMetadata
    # (Optional[list[str]]); the zero-chunk floor reads it to size its WARN, so the
    # attribute must always be present here (defaults to None) to mirror the SDK.
    metadata = SimpleNamespace(
        grounding_chunks=chunks,
        grounding_supports=supports,
        web_search_queries=web_search_queries,
    )
    url_context_metadata = SimpleNamespace(url_metadata=url_metadata) if url_metadata is not None else None
    candidate = SimpleNamespace(
        grounding_metadata=metadata,
        url_context_metadata=url_context_metadata,
    )
    return SimpleNamespace(text=text, candidates=[candidate])


def _make_client_with_response(response: object) -> MagicMock:
    """Build a MagicMock Client whose aio.models.generate_content awaits to ``response``."""
    client = MagicMock()
    client.aio = MagicMock()
    client.aio.models = MagicMock()
    client.aio.models.generate_content = AsyncMock(return_value=response)
    return client


# ---------------------------------------------------------------------------
# build_gemini_client
# ---------------------------------------------------------------------------


def test_builder_raises_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing GOOGLE_API_KEY should raise. The grounded-search side has no
    donated/shared key path — Google AI Studio doesn't offer one — so this is
    the only key gate to test here.
    """
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    from metaculus_bot.research.gemini_search import build_gemini_client

    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        build_gemini_client()


# ---------------------------------------------------------------------------
# gemini_search_provider: model selection & tool wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_provider_uses_default_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no GEMINI_SEARCH_MODEL env set, default slug is used."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    monkeypatch.delenv("GEMINI_SEARCH_MODEL", raising=False)

    response = _make_response("some research text")
    fake_client = _make_client_with_response(response)

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        from metaculus_bot.research.gemini_search import gemini_search_provider

        provider = gemini_search_provider()
        await provider(_make_q("Will X happen?"))

    assert fake_client.aio.models.generate_content.await_count == 1
    call_kwargs = fake_client.aio.models.generate_content.await_args.kwargs
    assert call_kwargs["model"] == "gemini-3-flash-preview"
    # The question_text must actually reach the SDK (guard against broken f-string interpolation).
    assert "Will X happen?" in call_kwargs["contents"]


@pytest.mark.asyncio
async def test_provider_uses_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """GEMINI_SEARCH_MODEL env var overrides the default."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    monkeypatch.setenv("GEMINI_SEARCH_MODEL", "gemini-2.5-flash")

    response = _make_response("research text")
    fake_client = _make_client_with_response(response)

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        from metaculus_bot.research.gemini_search import gemini_search_provider

        provider = gemini_search_provider()
        await provider(_make_q("Will X happen?"))

    call_kwargs = fake_client.aio.models.generate_content.await_args.kwargs
    assert call_kwargs["model"] == "gemini-2.5-flash"


@pytest.mark.asyncio
async def test_provider_uses_explicit_slug(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit ``model_slug=`` param takes precedence over env var."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    monkeypatch.setenv("GEMINI_SEARCH_MODEL", "gemini-2.5-flash")

    response = _make_response("research text")
    fake_client = _make_client_with_response(response)

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        from metaculus_bot.research.gemini_search import gemini_search_provider

        provider = gemini_search_provider(model_slug="gemini-explicit-override")
        await provider(_make_q("Will X happen?"))

    call_kwargs = fake_client.aio.models.generate_content.await_args.kwargs
    assert call_kwargs["model"] == "gemini-explicit-override"


@pytest.mark.asyncio
async def test_provider_attaches_google_search_and_url_context_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The generate_content config must include both the GoogleSearch and url_context tools."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    response = _make_response("research text")
    fake_client = _make_client_with_response(response)

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        from metaculus_bot.research.gemini_search import gemini_search_provider

        provider = gemini_search_provider()
        await provider(_make_q("Will X happen?"))

    call_kwargs = fake_client.aio.models.generate_content.await_args.kwargs
    config = call_kwargs["config"]
    tools = list(config.tools)
    assert len(tools) == 2
    # The SDK normalizes the {"google_search": {}} / {"url_context": {}} dicts into
    # pydantic Tool objects with the corresponding attribute populated.
    google_search_configured = any(getattr(t, "google_search", None) is not None for t in tools)
    url_context_configured = any(getattr(t, "url_context", None) is not None for t in tools)
    assert google_search_configured
    assert url_context_configured


# ---------------------------------------------------------------------------
# gemini_search_provider: prompt content
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_benchmarking_carve_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """is_benchmarking=True: prompt contains 'benchmarking run' and no market/crowd-odds ask."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    response = _make_response("research text")
    fake_client = _make_client_with_response(response)

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        from metaculus_bot.research.gemini_search import gemini_search_provider

        provider = gemini_search_provider(is_benchmarking=True)
        await provider(_make_q("Will X happen?"))

    call_kwargs = fake_client.aio.models.generate_content.await_args.kwargs
    prompt = call_kwargs["contents"]
    assert "benchmarking run" in prompt
    assert "Market-implied or crowd odds" not in prompt


@pytest.mark.asyncio
async def test_non_benchmarking_includes_prediction_markets(monkeypatch: pytest.MonkeyPatch) -> None:
    """is_benchmarking=False: prompt includes the market/crowd-odds bullet."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    response = _make_response("research text")
    fake_client = _make_client_with_response(response)

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        from metaculus_bot.research.gemini_search import gemini_search_provider

        provider = gemini_search_provider(is_benchmarking=False)
        await provider(_make_q("Will X happen?"))

    call_kwargs = fake_client.aio.models.generate_content.await_args.kwargs
    prompt = call_kwargs["contents"]
    assert "Market-implied or crowd odds" in prompt
    assert "benchmarking run" not in prompt


@pytest.mark.asyncio
async def test_prompt_carries_the_mc_ballot(monkeypatch: pytest.MonkeyPatch) -> None:
    """An MC question's option list must reach the grounded-search prompt: a searching model
    can only query candidate names it has been shown (the q44952 gap — no research stage ever
    saw the ballot)."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    response = _make_response("research text")
    fake_client = _make_client_with_response(response)

    question = _make_q("Who will win the World Yo-Yo Contest?")
    question.options = ["Mir Kim", "Hunter Feuerstein", "Other"]

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        from metaculus_bot.research.gemini_search import gemini_search_provider

        provider = gemini_search_provider(is_benchmarking=False)
        await provider(question)

    prompt = fake_client.aio.models.generate_content.await_args.kwargs["contents"]
    assert "Options (in resolution order): Mir Kim | Hunter Feuerstein | Other" in prompt


# ---------------------------------------------------------------------------
# _format_grounded_response behavior (via invoke_gemini_grounded)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_citations_appended_to_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """Response with grounding chunks ends with a '### Sources' block citing title + domain."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    blob = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbC123"
    chunks = [
        CannedWebChunk(uri=blob, title="Example One", domain="example.com"),
        CannedWebChunk(uri=blob, title="Example Two", domain="two.example.org"),
    ]
    response = _make_response("body text", chunks=chunks, supports=None)
    fake_client = _make_client_with_response(response)

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        from metaculus_bot.research.gemini_search import invoke_gemini_grounded

        out = await invoke_gemini_grounded("prompt")

    assert "### Sources" in out
    assert "Example One — example.com" in out
    assert "Example Two — two.example.org" in out
    # The opaque vertexaisearch redirect blob must never reach the forecaster.
    assert "vertexaisearch" not in out
    assert "grounding-api-redirect" not in out
    # Sources comes after body text
    assert out.index("body text") < out.index("### Sources")


@pytest.mark.asyncio
async def test_sources_render_domain_not_redirect_blob(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sources cite title + domain, drop the redirect blob, and stay 1:1 with chunks.

    Covers the title+domain case, the domain-only fallback (no title), the
    title-only fallback (no domain), and a duplicate domain — which keeps its own
    index so the inline [N] markers remain valid (we do not renumber/dedup).
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    blob = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/" + ("X" * 200)
    chunks = [
        CannedWebChunk(uri=blob, title="Al Jazeera", domain="aljazeera.com"),
        CannedWebChunk(uri=blob, title=None, domain="senate.gov"),  # no title -> domain only
        CannedWebChunk(uri=blob, title="Reuters", domain=None),  # no domain -> title only
        CannedWebChunk(uri=blob, title="Al Jazeera", domain="aljazeera.com"),  # duplicate keeps its index
    ]
    response = _make_response("body text", chunks=chunks, supports=None)
    fake_client = _make_client_with_response(response)

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        from metaculus_bot.research.gemini_search import invoke_gemini_grounded

        out = await invoke_gemini_grounded("prompt")

    assert "vertexaisearch" not in out
    assert "grounding-api-redirect" not in out
    assert "[1] Al Jazeera — aljazeera.com" in out
    assert "[2] senate.gov" in out
    assert "[3] Reuters" in out
    assert "[4] Al Jazeera — aljazeera.com" in out


@pytest.mark.asyncio
async def test_inline_citation_markers_inserted(monkeypatch: pytest.MonkeyPatch) -> None:
    """A support mapping a segment end_index to chunk 0 produces a ``[1]`` marker after that offset.

    With multiple supports, reverse-iteration must preserve earlier offsets.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    text = "Alpha fact. Beta fact."
    end_alpha = text.index("Alpha fact.") + len("Alpha fact.")
    end_beta = text.index("Beta fact.") + len("Beta fact.")

    blob = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/zzz"
    chunks = [
        CannedWebChunk(uri=blob, title="A", domain="a.example.com"),
        CannedWebChunk(uri=blob, title="B", domain="b.example.com"),
    ]
    supports = [
        CannedSupport(seg=CannedSegment(end_index=end_alpha, text="Alpha fact."), indices=[0]),
        CannedSupport(seg=CannedSegment(end_index=end_beta, text="Beta fact."), indices=[1]),
    ]
    response = _make_response(text, chunks=chunks, supports=supports)
    fake_client = _make_client_with_response(response)

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        from metaculus_bot.research.gemini_search import invoke_gemini_grounded

        out = await invoke_gemini_grounded("prompt")

    assert "Alpha fact.[1]" in out
    assert "Beta fact.[2]" in out
    # The sources block must also be appended whenever chunks are present — this is
    # the common production path (chunks + supports together), not chunks-only.
    assert "### Sources" in out
    assert "a.example.com" in out
    assert "vertexaisearch" not in out


@pytest.mark.asyncio
async def test_inline_citation_markers_respect_utf8_byte_offsets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: Google's segment.end_index is a UTF-8 BYTE offset, not a codepoint index.

    Derived from the real Q578 gemini_search record
    (backtests/research_archive/raw/29718821482.jsonl). The model text contains an
    em-dash (``—``, 3 bytes / 1 codepoint) at codepoint 194, so at the first
    support's byte offset 291 the byte cursor runs 2 ahead of the codepoint cursor.
    The pre-fix code sliced the Python str at codepoint 291 and produced the
    observed mid-word corruption ``"...civilization. T[1]hese estimates"``. The
    correct byte-space splice lands the marker at the sentence boundary:
    ``"...civilization.[1] These estimates"``.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    # Verbatim opening of the Q578 raw payload text, trimmed to the minimal
    # reproducing snippet (through the first two sentences).
    text = (
        "There is no scientific consensus that humans will go extinct before 2100, "
        "but researchers and specialized forecasters have produced a wide range of "
        'probabilistic estimates for "existential risk"—defined as events that '
        "could cause human extinction or the permanent collapse of civilization. "
        "These estimates vary significantly."
    )
    # Sanity-anchor the fixture to the real payload's geometry: em-dash at
    # codepoint 194, first support end_index at byte 291.
    assert text.index("—") == 194
    byte_end_index = 291
    # The byte offset points at the space after "civilization." — one past the period.
    assert text.encode("utf-8")[:byte_end_index].decode("utf-8").endswith("collapse of civilization.")

    blob = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbC"
    chunks = [CannedWebChunk(uri=blob, title="RAND", domain="rand.org")]
    supports = [CannedSupport(seg=CannedSegment(end_index=byte_end_index, text=""), indices=[0])]
    response = _make_response(text, chunks=chunks, supports=supports)
    fake_client = _make_client_with_response(response)

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        from metaculus_bot.research.gemini_search import invoke_gemini_grounded

        out = await invoke_gemini_grounded("prompt")

    # The marker lands cleanly at the sentence boundary...
    assert "collapse of civilization.[1] These estimates" in out
    # ...and the pre-fix mid-word corruption is gone.
    assert "T[1]hese" not in out
    assert "civilization. T[1]" not in out


@pytest.mark.asyncio
async def test_no_candidates_is_suppressed_by_the_grounded_chunk_floor(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Text with no candidates has no grounding evidence, so the floor must refuse it.

    This test used to assert the opposite ("returns its plain text") — an early return that
    walked straight past the Q38195 fabrication guard. The branch is unreachable on today's
    SDK (``response.text`` derives from a candidate), but a hole in a fabrication guard
    shouldn't rest on an SDK invariant we don't own.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    response = SimpleNamespace(text="plain ungrounded body", candidates=[])
    fake_client = _make_client_with_response(response)

    with (
        patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client),
        caplog.at_level("WARNING"),
    ):
        from metaculus_bot.research.gemini_search import invoke_gemini_grounded

        out = await invoke_gemini_grounded("prompt")

    assert out == ""
    assert "GEMINI_UNGROUNDED_SUPPRESSED" in caplog.text
    assert "queries=0" in caplog.text


@pytest.mark.asyncio
async def test_empty_response_text_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """response.text == '' short-circuits to empty, even when candidates + grounding metadata are populated.

    Isolating the ``not text`` guard: if that early-return regresses (e.g. moved below the candidates
    check), the chunks/supports path would produce a non-empty "\\n\\n### Sources\\n[1] ..." string and
    this assertion would fail.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    chunks = [CannedWebChunk(uri="https://example.com/1", title="Example One")]
    supports = [CannedSupport(seg=CannedSegment(end_index=0, text=""), indices=[0])]
    response = _make_response("", chunks=chunks, supports=supports)
    fake_client = _make_client_with_response(response)

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        from metaculus_bot.research.gemini_search import invoke_gemini_grounded

        out = await invoke_gemini_grounded("prompt")

    assert out == ""


@pytest.mark.asyncio
async def test_zero_grounding_chunks_suppresses_ungrounded_text(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Regression (Q38195): grounding metadata present, chunks empty, many search queries.

    On the real Q38195 record Gemini issued 30 web-search queries and Google
    returned ZERO grounding chunks, yet the model emitted a confident, fabricated
    contract table with fake ``[primary]`` tags. Pre-fix, the formatter passed that
    ungrounded parametric text through verbatim (the fabrication reached
    forecasters). Post-fix, a response whose grounding fired-but-returned-no-chunks
    must be suppressed entirely (``""``) so the section is omitted upstream, and a
    greppable WARN must fire.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    fabricated = (
        "Generative AI has emerged as a primary driver of labor tension. "
        "Key contract expirations: Boeing IAM 837 (July 22, 2029) [primary]."
    )
    # grounding_metadata present (grounding fired), grounding_chunks empty, and
    # a populated web_search_queries list — the exact Q38195 shape.
    response = _make_response(
        fabricated,
        chunks=None,
        supports=None,
        web_search_queries=[f"query {i}" for i in range(30)],
    )
    fake_client = _make_client_with_response(response)

    with (
        patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client),
        caplog.at_level("WARNING"),
    ):
        from metaculus_bot.research.gemini_search import invoke_gemini_grounded

        out = await invoke_gemini_grounded("prompt", qid=38195)

    # The ungrounded fabrication must NOT reach the forecaster.
    assert out == ""
    assert "[primary]" not in out
    assert "Boeing" not in out
    # A greppable WARN with the query count must fire.
    assert "GEMINI_UNGROUNDED_SUPPRESSED" in caplog.text
    assert "queries=30" in caplog.text


@pytest.mark.asyncio
async def test_ungrounded_suppression_records_a_provider_loss_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """The suppression must be visible where degradation is read, not only in the WARN.

    Suppressing correctly still costs the whole Google research leg, and the loss used to
    be invisible in all three places the diagnostics convention teaches you to look: no
    counter moves (the provider didn't raise, so ``provider_failure_count`` stays 0 and
    ``alertable_count`` stays flat), no ``record_provider_detail`` call, and the resulting
    ``""`` maps to ProviderResult status ``empty`` — byte-identical to a healthy Gemini
    call that legitimately found nothing. If grounding degrades persistently (AI Studio
    prepaid exhaustion, an SDK contract shift), every forecast in the season loses the
    Google leg behind a normal-looking diagnostics line. So the suppression now records a
    per-source loss token, mirroring ``_degraded_to_raw_articles`` for the summarizer.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    response = _make_response("Ungrounded prose.", chunks=None, supports=None, web_search_queries=["q"])
    fake_client = _make_client_with_response(response)

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        from metaculus_bot.research.gemini_search import invoke_gemini_grounded

        out = await invoke_gemini_grounded("prompt", qid=38195)

    assert out == ""
    detail = pop_provider_detail(38195, "gemini_search")
    token = detail["sources"]["grounding"]
    # _is_lost_source treats anything not starting with "ok"/"none" as a loss, which is
    # what renders the `lost=grounding:...` segment on the diagnostics line and in the
    # schema-v2 archive.
    assert _is_lost_source(token), f"the recorded token must read as a LOST source; got {token!r}"
    assert "ungrounded" in token


@pytest.mark.asyncio
async def test_url_context_read_survives_zero_search_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful url_context read IS grounding, so the text passes the chunk floor.

    Google's search tool grounded nothing (no chunks), but url_context retrieved a
    page, so this is genuine retrieval rather than the Q38195 parametric case. The
    text comes back unmodified — there are no chunks to cite, so no inline markers
    and no ``### Sources`` block.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    response = _make_response(
        "Read straight off the resolving page.",
        chunks=None,
        supports=None,
        url_metadata=[CannedUrlMeta("https://gov.example/report", CannedStatus("URL_RETRIEVAL_STATUS_SUCCESS"))],
    )
    fake_client = _make_client_with_response(response)

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        from metaculus_bot.research.gemini_search import invoke_gemini_grounded

        out = await invoke_gemini_grounded("prompt", qid=1)

    assert "Read straight off the resolving page." in out
    assert "### Sources" not in out


@pytest.mark.asyncio
async def test_malformed_supports_fall_back_to_unspliced_text(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A support whose end_index is not an integer must not cost us the response.

    The splice raises TypeError partway through mutating the byte buffer; the
    formatter falls back to the ORIGINAL text (never a half-spliced buffer), still
    renders the Sources block, and leaves a greppable WARN.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    blob = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbC123"
    bad_support = CannedSupport(cast(CannedSegment, SimpleNamespace(end_index="not-an-int", text="x")), [0])
    response = _make_response(
        "body text",
        chunks=[CannedWebChunk(uri=blob, title="Example One", domain="example.com")],
        supports=[bad_support],
    )
    fake_client = _make_client_with_response(response)

    with (
        patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client),
        caplog.at_level("WARNING"),
    ):
        from metaculus_bot.research.gemini_search import invoke_gemini_grounded

        out = await invoke_gemini_grounded("prompt")

    assert out.startswith("body text")
    assert "[1] Example One — example.com" in out
    assert "could not splice inline citations" in caplog.text


# ---------------------------------------------------------------------------
# url_context telemetry: extract_url_context_telemetry
# ---------------------------------------------------------------------------


def test_url_context_telemetry_empty_when_metadata_absent() -> None:
    """No candidates / no url_context_metadata yield reported=False; an empty url_metadata list
    yields reported=True (the tool fired but fetched nothing). All cases give zero counts + empty list."""
    from metaculus_bot.research.url_context_telemetry import extract_url_context_telemetry

    def extract(response: object) -> tuple[bool, int, int, list[tuple[str, str]]]:
        return extract_url_context_telemetry(cast(genai_types.GenerateContentResponse, response))

    # No url_context signal at all → reported=False.
    assert extract(SimpleNamespace(text="x", candidates=[])) == (False, 0, 0, [])
    assert extract(SimpleNamespace(text="x", candidates=None)) == (False, 0, 0, [])
    # candidate present but url_context_metadata is None → still no signal.
    assert extract(_make_response("x", url_metadata=None)) == (False, 0, 0, [])
    # url_context_metadata present but url_metadata list is empty → fired-but-empty (reported=True).
    assert extract(_make_response("x", url_metadata=[])) == (True, 0, 0, [])


def test_url_context_telemetry_parses_success_and_error() -> None:
    """Two url_metadata entries (1 SUCCESS via enum-like .name, 1 ERROR via plain string).

    Proves both counts and the (status_name, url) list, and that the status is coerced
    defensively whether it arrives as an enum-with-.name or a plain string.
    """
    from metaculus_bot.research.url_context_telemetry import extract_url_context_telemetry

    url_metadata = [
        CannedUrlMeta(
            retrieved_url="https://example.com/ok",
            url_retrieval_status=CannedStatus("URL_RETRIEVAL_STATUS_SUCCESS"),
        ),
        CannedUrlMeta(
            retrieved_url="https://example.com/bad",
            url_retrieval_status="URL_RETRIEVAL_STATUS_ERROR",
        ),
    ]
    response = cast(genai_types.GenerateContentResponse, _make_response("body", url_metadata=url_metadata))

    reported, n_total, n_success, entries = extract_url_context_telemetry(response)

    assert reported is True
    assert n_total == 2
    assert n_success == 1
    assert entries == [
        ("URL_RETRIEVAL_STATUS_SUCCESS", "https://example.com/ok"),
        ("URL_RETRIEVAL_STATUS_ERROR", "https://example.com/bad"),
    ]


def test_url_context_telemetry_coerces_none_valued_fields() -> None:
    """A real ``UrlMetadata`` with both fields present-but-None coerces to ``("None", "")`` rather
    than raising — telemetry must never break the research path. ``UrlMetadata`` declares both fields
    as Optional (extra='forbid'), so a None-valued entry is the genuine degenerate case the SDK can
    emit; it is counted, not silently skipped.
    """
    from metaculus_bot.research.url_context_telemetry import extract_url_context_telemetry

    degenerate = genai_types.UrlMetadata(retrieved_url=None, url_retrieval_status=None)
    good = CannedUrlMeta(
        retrieved_url="https://example.com/ok",
        url_retrieval_status=CannedStatus("URL_RETRIEVAL_STATUS_SUCCESS"),
    )
    response = cast(genai_types.GenerateContentResponse, _make_response("body", url_metadata=[degenerate, good]))

    reported, n_total, n_success, entries = extract_url_context_telemetry(response)

    assert reported is True
    assert n_total == 2
    assert n_success == 1
    assert len(entries) == 2
    assert ("None", "") in entries
    assert ("URL_RETRIEVAL_STATUS_SUCCESS", "https://example.com/ok") in entries


# ---------------------------------------------------------------------------
# url_context telemetry: marker persisted into invoke_gemini_grounded output
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_url_context_fetches_marker_in_returned_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """When url_context fetched URLs, the returned text carries a greppable subsection — but ONLY the
    SUCCESSFUL fetches are listed inline (those URLs were actually read, so they are real research
    context). A co-occurring failed fetch must stay out of the forecaster-facing text (it is captured
    in the INFO audit log instead).
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    url_metadata = [
        CannedUrlMeta(
            retrieved_url="https://example.com/ok",
            url_retrieval_status=CannedStatus("URL_RETRIEVAL_STATUS_SUCCESS"),
        ),
        CannedUrlMeta(
            retrieved_url="https://example.com/bad",
            url_retrieval_status="URL_RETRIEVAL_STATUS_ERROR",
        ),
    ]
    response = _make_response("body text", url_metadata=url_metadata)
    fake_client = _make_client_with_response(response)

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        from metaculus_bot.research.gemini_search import invoke_gemini_grounded

        out = await invoke_gemini_grounded("prompt")

    assert "### URL Context Fetches" in out
    assert "URL_RETRIEVAL_STATUS_SUCCESS — https://example.com/ok" in out
    # Failed fetches must NOT appear inline — only successfully-read URLs are research context.
    assert "URL_RETRIEVAL_STATUS_ERROR" not in out
    assert "https://example.com/bad" not in out
    # The terse "none" marker must NOT appear when at least one fetch succeeded.
    assert "_url_context: none_" not in out
    # The fetch subsection must stay out of the grounding-only Sources block.
    assert "### Sources" not in out


@pytest.mark.asyncio
async def test_url_context_none_marker_when_no_fetches(monkeypatch: pytest.MonkeyPatch) -> None:
    """When url_context fired but fetched nothing (empty url_metadata), a terse greppable
    ``_url_context: none_`` marker is appended — distinguishing 'fired but empty' from
    'we don't capture it' — and NO fetch list is emitted.

    The response carries a grounding chunk so it clears the grounded-chunk floor; this
    isolates the url_context marker layer, which is orthogonal to grounding.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    chunks = [CannedWebChunk(uri="https://vertexaisearch/redirect", title="Src", domain="src.example.com")]
    response = _make_response("body text", chunks=chunks, url_metadata=[])
    fake_client = _make_client_with_response(response)

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        from metaculus_bot.research.gemini_search import invoke_gemini_grounded

        out = await invoke_gemini_grounded("prompt")

    assert "_url_context: none_" in out
    assert "### URL Context Fetches" not in out
    assert out.startswith("body text")


@pytest.mark.asyncio
async def test_no_url_context_marker_when_tool_did_not_fire(monkeypatch: pytest.MonkeyPatch) -> None:
    """When url_context produced no signal at all (no url_context_metadata on the candidate),
    the returned text must carry no url_context marker of any kind — no fetch list AND no terse
    'none' marker. This pins the requirement that an inert url_context tool never pollutes
    forecaster-facing research.

    The response carries a grounding chunk so it clears the grounded-chunk floor; the Sources block
    it produces is expected and orthogonal to the url_context marker layer under test.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    # url_metadata=None makes _make_response build url_context_metadata=None on the candidate.
    chunks = [CannedWebChunk(uri="https://vertexaisearch/redirect", title="Src", domain="src.example.com")]
    response = _make_response("body text", chunks=chunks, url_metadata=None)
    fake_client = _make_client_with_response(response)

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        from metaculus_bot.research.gemini_search import invoke_gemini_grounded

        out = await invoke_gemini_grounded("prompt")

    assert out.startswith("body text")
    assert "### URL Context Fetches" not in out
    assert "_url_context: none_" not in out


@pytest.mark.asyncio
async def test_url_context_none_marker_when_all_fetches_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    """When url_context fired but every retrieval FAILED, the forecaster-facing text collapses to the
    terse ``_url_context: none_`` marker — the failed URLs must NOT be listed as research context
    (only successful fetches were actually read). The failed URL is still in the INFO audit log.

    The response carries a grounding chunk so google_search grounding clears the floor; the failed
    url_context fetch is orthogonal.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    chunks = [CannedWebChunk(uri="https://vertexaisearch/redirect", title="Src", domain="src.example.com")]
    url_metadata = [
        CannedUrlMeta(
            retrieved_url="https://example.com/bad",
            url_retrieval_status=CannedStatus("URL_RETRIEVAL_STATUS_ERROR"),
        ),
    ]
    response = _make_response("body text", chunks=chunks, url_metadata=url_metadata)
    fake_client = _make_client_with_response(response)

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        from metaculus_bot.research.gemini_search import invoke_gemini_grounded

        out = await invoke_gemini_grounded("prompt")

    assert "_url_context: none_" in out
    assert "### URL Context Fetches" not in out
    assert "https://example.com/bad" not in out
    assert out.startswith("body text")


# ---------------------------------------------------------------------------
# Parallel provider selection in main.py
# ---------------------------------------------------------------------------


class TestParallelProviderSelectionGemini:
    """Tests for Gemini gating via GEMINI_SEARCH_ENABLED in ``_select_research_providers``."""

    def test_select_research_providers_includes_gemini_when_enabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("GEMINI_SEARCH_ENABLED", "true")
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
        monkeypatch.delenv("NATIVE_SEARCH_ENABLED", raising=False)
        monkeypatch.delenv("FINANCIAL_DATA_ENABLED", raising=False)
        monkeypatch.setenv("ASKNEWS_CLIENT_ID", "id")
        monkeypatch.setenv("ASKNEWS_SECRET", "secret")

        from forecasting_tools import GeneralLlm

        from metaculus_bot.research.orchestrator import ResearchOrchestrator

        mock_llm = GeneralLlm(model="test/model", temperature=0.0)
        orch = ResearchOrchestrator(default_llm=mock_llm, summarizer_llm=mock_llm)
        mock_provider = AsyncMock(return_value="primary research")

        with patch.object(orch, "_select_research_provider", return_value=(mock_provider, "asknews")):
            providers = orch._select_research_providers()

        provider_names = [name for _, name in providers]
        assert "gemini_search" in provider_names

    def test_select_research_providers_excludes_gemini_when_disabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("GEMINI_SEARCH_ENABLED", "false")
        monkeypatch.delenv("NATIVE_SEARCH_ENABLED", raising=False)
        monkeypatch.delenv("FINANCIAL_DATA_ENABLED", raising=False)
        monkeypatch.setenv("ASKNEWS_CLIENT_ID", "id")
        monkeypatch.setenv("ASKNEWS_SECRET", "secret")

        from forecasting_tools import GeneralLlm

        from metaculus_bot.research.orchestrator import ResearchOrchestrator

        mock_llm = GeneralLlm(model="test/model", temperature=0.0)
        orch = ResearchOrchestrator(default_llm=mock_llm, summarizer_llm=mock_llm)
        mock_provider = AsyncMock(return_value="primary research")

        with patch.object(orch, "_select_research_provider", return_value=(mock_provider, "asknews")):
            providers = orch._select_research_providers()

        provider_names = [name for _, name in providers]
        assert "gemini_search" not in provider_names
