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
    def __init__(self, uri: str, title: str | None) -> None:
        self.web = SimpleNamespace(uri=uri, title=title)


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
) -> SimpleNamespace:
    metadata = SimpleNamespace(
        grounding_chunks=chunks,
        grounding_supports=supports,
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
    """is_benchmarking=True: prompt contains 'benchmarking run' and NOT 'Prediction market'."""
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
    assert "Prediction market" not in prompt


@pytest.mark.asyncio
async def test_non_benchmarking_includes_prediction_markets(monkeypatch: pytest.MonkeyPatch) -> None:
    """is_benchmarking=False: prompt includes 'Prediction market' line."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    response = _make_response("research text")
    fake_client = _make_client_with_response(response)

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        from metaculus_bot.research.gemini_search import gemini_search_provider

        provider = gemini_search_provider(is_benchmarking=False)
        await provider(_make_q("Will X happen?"))

    call_kwargs = fake_client.aio.models.generate_content.await_args.kwargs
    prompt = call_kwargs["contents"]
    assert "Prediction market" in prompt
    assert "benchmarking run" not in prompt


# ---------------------------------------------------------------------------
# _format_grounded_response behavior (via invoke_gemini_grounded)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_citations_appended_to_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """Response with grounding chunks ends with a '### Sources' block listing both URIs/titles."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    chunks = [
        CannedWebChunk(uri="https://example.com/1", title="Example One"),
        CannedWebChunk(uri="https://example.com/2", title="Example Two"),
    ]
    response = _make_response("body text", chunks=chunks, supports=None)
    fake_client = _make_client_with_response(response)

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        from metaculus_bot.research.gemini_search import invoke_gemini_grounded

        out = await invoke_gemini_grounded("prompt")

    assert "### Sources" in out
    assert "https://example.com/1" in out
    assert "Example One" in out
    assert "https://example.com/2" in out
    assert "Example Two" in out
    # Sources comes after body text
    assert out.index("body text") < out.index("### Sources")


@pytest.mark.asyncio
async def test_inline_citation_markers_inserted(monkeypatch: pytest.MonkeyPatch) -> None:
    """A support mapping a segment end_index to chunk 0 produces a ``[1]`` marker after that offset.

    With multiple supports, reverse-iteration must preserve earlier offsets.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    text = "Alpha fact. Beta fact."
    end_alpha = text.index("Alpha fact.") + len("Alpha fact.")
    end_beta = text.index("Beta fact.") + len("Beta fact.")

    chunks = [
        CannedWebChunk(uri="https://example.com/a", title="A"),
        CannedWebChunk(uri="https://example.com/b", title="B"),
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
    assert "https://example.com/a" in out


@pytest.mark.asyncio
async def test_missing_grounding_metadata_returns_plain_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """A response with no grounding metadata still returns its plain text."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    response = SimpleNamespace(text="plain response body", candidates=[])
    fake_client = _make_client_with_response(response)

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        from metaculus_bot.research.gemini_search import invoke_gemini_grounded

        out = await invoke_gemini_grounded("prompt")

    assert out == "plain response body"


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


# ---------------------------------------------------------------------------
# url_context telemetry: _extract_url_context_telemetry
# ---------------------------------------------------------------------------


def test_url_context_telemetry_empty_when_metadata_absent() -> None:
    """No candidates / no url_context_metadata yield reported=False; an empty url_metadata list
    yields reported=True (the tool fired but fetched nothing). All cases give zero counts + empty list."""
    from metaculus_bot.research.gemini_search import _extract_url_context_telemetry

    def extract(response: object) -> tuple[bool, int, int, list[tuple[str, str]]]:
        return _extract_url_context_telemetry(cast(genai_types.GenerateContentResponse, response))

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
    from metaculus_bot.research.gemini_search import _extract_url_context_telemetry

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

    reported, n_total, n_success, entries = _extract_url_context_telemetry(response)

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
    from metaculus_bot.research.gemini_search import _extract_url_context_telemetry

    degenerate = genai_types.UrlMetadata(retrieved_url=None, url_retrieval_status=None)
    good = CannedUrlMeta(
        retrieved_url="https://example.com/ok",
        url_retrieval_status=CannedStatus("URL_RETRIEVAL_STATUS_SUCCESS"),
    )
    response = cast(genai_types.GenerateContentResponse, _make_response("body", url_metadata=[degenerate, good]))

    reported, n_total, n_success, entries = _extract_url_context_telemetry(response)

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
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    response = _make_response("body text", url_metadata=[])
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
    the returned text must be the grounded body only — no fetch list AND no terse 'none' marker.
    This pins the requirement that an inert url_context tool never pollutes forecaster-facing research.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    # url_metadata=None makes _make_response build url_context_metadata=None on the candidate.
    response = _make_response("body text", url_metadata=None)
    fake_client = _make_client_with_response(response)

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        from metaculus_bot.research.gemini_search import invoke_gemini_grounded

        out = await invoke_gemini_grounded("prompt")

    assert out == "body text"
    assert "### URL Context Fetches" not in out
    assert "_url_context: none_" not in out


@pytest.mark.asyncio
async def test_url_context_none_marker_when_all_fetches_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    """When url_context fired but every retrieval FAILED, the forecaster-facing text collapses to the
    terse ``_url_context: none_`` marker — the failed URLs must NOT be listed as research context
    (only successful fetches were actually read). The failed URL is still in the INFO audit log.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    url_metadata = [
        CannedUrlMeta(
            retrieved_url="https://example.com/bad",
            url_retrieval_status=CannedStatus("URL_RETRIEVAL_STATUS_ERROR"),
        ),
    ]
    response = _make_response("body text", url_metadata=url_metadata)
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
