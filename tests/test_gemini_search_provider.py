"""Tests for the Gemini grounded search research provider.

These tests mock the google-genai SDK at the module level; no live API calls.
Patterns mirror ``tests/test_native_search_provider.py``.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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


def _make_response(
    text: str,
    chunks: list[CannedWebChunk] | None = None,
    supports: list[CannedSupport] | None = None,
) -> SimpleNamespace:
    metadata = SimpleNamespace(
        grounding_chunks=chunks,
        grounding_supports=supports,
    )
    candidate = SimpleNamespace(grounding_metadata=metadata)
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
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    from metaculus_bot.gemini_search_provider import build_gemini_client

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

    with patch("metaculus_bot.gemini_search_provider.genai.Client", return_value=fake_client):
        from metaculus_bot.gemini_search_provider import gemini_search_provider

        provider = gemini_search_provider()
        await provider("Will X happen?")

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

    with patch("metaculus_bot.gemini_search_provider.genai.Client", return_value=fake_client):
        from metaculus_bot.gemini_search_provider import gemini_search_provider

        provider = gemini_search_provider()
        await provider("Will X happen?")

    call_kwargs = fake_client.aio.models.generate_content.await_args.kwargs
    assert call_kwargs["model"] == "gemini-2.5-flash"


@pytest.mark.asyncio
async def test_provider_uses_explicit_slug(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit ``model_slug=`` param takes precedence over env var."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    monkeypatch.setenv("GEMINI_SEARCH_MODEL", "gemini-2.5-flash")

    response = _make_response("research text")
    fake_client = _make_client_with_response(response)

    with patch("metaculus_bot.gemini_search_provider.genai.Client", return_value=fake_client):
        from metaculus_bot.gemini_search_provider import gemini_search_provider

        provider = gemini_search_provider(model_slug="gemini-explicit-override")
        await provider("Will X happen?")

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

    with patch("metaculus_bot.gemini_search_provider.genai.Client", return_value=fake_client):
        from metaculus_bot.gemini_search_provider import gemini_search_provider

        provider = gemini_search_provider()
        await provider("Will X happen?")

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

    with patch("metaculus_bot.gemini_search_provider.genai.Client", return_value=fake_client):
        from metaculus_bot.gemini_search_provider import gemini_search_provider

        provider = gemini_search_provider(is_benchmarking=True)
        await provider("Will X happen?")

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

    with patch("metaculus_bot.gemini_search_provider.genai.Client", return_value=fake_client):
        from metaculus_bot.gemini_search_provider import gemini_search_provider

        provider = gemini_search_provider(is_benchmarking=False)
        await provider("Will X happen?")

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

    with patch("metaculus_bot.gemini_search_provider.genai.Client", return_value=fake_client):
        from metaculus_bot.gemini_search_provider import invoke_gemini_grounded

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

    with patch("metaculus_bot.gemini_search_provider.genai.Client", return_value=fake_client):
        from metaculus_bot.gemini_search_provider import invoke_gemini_grounded

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

    with patch("metaculus_bot.gemini_search_provider.genai.Client", return_value=fake_client):
        from metaculus_bot.gemini_search_provider import invoke_gemini_grounded

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

    with patch("metaculus_bot.gemini_search_provider.genai.Client", return_value=fake_client):
        from metaculus_bot.gemini_search_provider import invoke_gemini_grounded

        out = await invoke_gemini_grounded("prompt")

    assert out == ""


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

        from main import TemplateForecaster

        with patch.object(TemplateForecaster, "__init__", lambda self: None):
            bot = TemplateForecaster.__new__(TemplateForecaster)
            bot._custom_research_provider = None
            bot.is_benchmarking = False
            bot.allow_research_fallback = True

            async def mock_provider(q: str) -> str:
                return "primary research"

            with patch.object(bot, "_select_research_provider", return_value=(mock_provider, "asknews")):
                providers = bot._select_research_providers()

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

        from main import TemplateForecaster

        with patch.object(TemplateForecaster, "__init__", lambda self: None):
            bot = TemplateForecaster.__new__(TemplateForecaster)
            bot._custom_research_provider = None
            bot.is_benchmarking = False
            bot.allow_research_fallback = True

            async def mock_provider(q: str) -> str:
                return "primary research"

            with patch.object(bot, "_select_research_provider", return_value=(mock_provider, "asknews")):
                providers = bot._select_research_providers()

        provider_names = [name for _, name in providers]
        assert "gemini_search" not in provider_names
