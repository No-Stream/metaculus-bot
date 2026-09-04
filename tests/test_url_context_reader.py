"""The one Gemini ``url_context`` read both paid surfaces share, driven against a fake SDK client.

No network: ``google.genai.Client`` is replaced at the seam the reader imports it through, and the
fake response carries only what the reader and its two observers (``log_gemini_usage``,
``extract_url_context_telemetry``) read. What is pinned is the reader's CONTRACT on the response:
the text comes off the SDK's documented ``.text`` property (``None`` when the answer has no text
part, which must read as an empty answer rather than crash or fabricate), and the retrieval
count and statuses come off the url_context metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from metaculus_bot.research import url_context_reader


@dataclass
class _UrlMetadata:
    url_retrieval_status: str
    retrieved_url: str


@dataclass
class _UrlContextMetadata:
    url_metadata: list[_UrlMetadata]


@dataclass
class _Candidate:
    url_context_metadata: _UrlContextMetadata | None
    grounding_metadata: None = None


@dataclass
class _Response:
    text: str | None
    candidates: list[_Candidate] = field(default_factory=list)
    usage_metadata: None = None
    model_version: str = "gemini-test"


class _FakeModels:
    def __init__(self, response: Any, calls: list[dict[str, Any]]) -> None:
        self._response = response
        self._calls = calls

    def generate_content(self, *, model: str, contents: str, config: Any) -> Any:
        self._calls.append({"model": model, "contents": contents, "config": config})
        return self._response


def _install_fake_client(monkeypatch: pytest.MonkeyPatch, response: Any) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    class _FakeClient:
        def __init__(self, *, api_key: str, http_options: Any) -> None:
            del api_key, http_options
            self.models = _FakeModels(response, calls)

    monkeypatch.setattr("google.genai.Client", _FakeClient)
    return calls


def _read(**overrides: Any) -> tuple[str, int, list[str]]:
    kwargs: dict[str, Any] = {
        "api_key": "key",
        "role": "resolution_source",
        "model": "gemini-test",
        "thinking_level": "low",
        "timeout_ms": 10_000,
        "attempts": 1,
    }
    kwargs.update(overrides)
    return url_context_reader.run_url_context_read("https://x.example.gov/report", "what does it say?", **kwargs)


class TestRunUrlContextRead:
    def test_a_grounded_answer_comes_back_with_its_retrieval_count_and_statuses(self, monkeypatch):
        response = _Response(
            text="The report says 12.",
            candidates=[
                _Candidate(
                    _UrlContextMetadata(
                        [
                            _UrlMetadata("URL_RETRIEVAL_STATUS_SUCCESS", "https://x.example.gov/report"),
                            _UrlMetadata("URL_RETRIEVAL_STATUS_ERROR", "https://x.example.gov/other"),
                        ]
                    )
                )
            ],
        )
        calls = _install_fake_client(monkeypatch, response)

        text, n_success, statuses = _read()

        assert text == "The report says 12."
        assert n_success == 1
        assert statuses == ["URL_RETRIEVAL_STATUS_SUCCESS", "URL_RETRIEVAL_STATUS_ERROR"]
        assert calls[0]["contents"].endswith("URL: https://x.example.gov/report")
        assert "verbatim quotes" in calls[0]["contents"]

    def test_a_response_with_no_text_part_reads_as_an_empty_answer(self, monkeypatch):
        """The SDK's `.text` is None when no candidate carries a text part. Both callers withhold
        on an empty answer, so this must come back as "" and not as a crash or a fabricated
        string."""
        _install_fake_client(monkeypatch, _Response(text=None, candidates=[_Candidate(None)]))

        text, n_success, statuses = _read()

        assert text == ""
        assert n_success == 0
        assert statuses == []

    def test_a_response_without_the_text_property_fails_loudly(self, monkeypatch):
        """Read directly rather than through `getattr(..., "")`: an SDK rename must surface as an
        error on the paid call, not be laundered into "the model answered nothing"."""

        @dataclass
        class _Renamed:
            candidates: list[Any] = field(default_factory=list)
            usage_metadata: None = None
            model_version: str = "gemini-test"

        _install_fake_client(monkeypatch, _Renamed())

        with pytest.raises(AttributeError):
            _read()
