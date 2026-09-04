"""The one Gemini ``url_context`` read both paid surfaces share, and the prompt wrapped around it.

``url_context_reader`` is the one call both paying surfaces share: gap-fill v2's ``read_document``
tool and the Tier-1 resolution-source ladder's last rung. Neither renders raw retrieval output;
both render the MODEL'S ANSWER about a document, under a caption that calls the section primary
grading evidence, so the three instructions in ``build_document_prompt`` are what keep the answer
checkable at all: quotes instead of paraphrase, the document's own dates, and an explicit "this
does not address the ask" instead of a fluent answer assembled out of recall (the Q38195 failure:
30 search queries, 0 grounding chunks, a fabricated contract table in front of forecasters).

Each instruction gets its own test, pinned on the sentence that carries the rule rather than on
the whole prompt string, so rewording the surrounding wrapper is free and dropping a rule is not.

No network anywhere here: ``google.genai.Client`` is replaced at the seam the reader imports it
through, and the fake responses carry only what the reader and its two observers
(``log_gemini_usage``, ``extract_url_context_telemetry``) read. The contract pinned on the response
is that the text comes off the SDK's documented ``.text`` property (``None`` when the answer has no
text part, which must read as an empty answer rather than crash or fabricate), and that the
retrieval count and statuses come off the url_context metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from metaculus_bot.research import url_context_reader
from metaculus_bot.research.url_context_reader import build_document_prompt, run_url_context_read

_ASK = "How many major work stoppages began in 2026, and as of what date?"


class TestTheDocumentPromptKeepsAReadCheckable:
    """The three instructions, one test each."""

    def test_the_callers_ask_is_carried_through(self):
        """The wrapper adds instructions; the ask is what the read is FOR."""
        assert _ASK in build_document_prompt(_ASK)

    def test_the_answer_must_quote_the_document(self):
        """A paraphrase of a document we cannot see is unverifiable, and v2's quote check has
        nothing to match against without verbatim spans."""
        assert "Answer using verbatim quotes from the document whenever possible." in build_document_prompt(_ASK)

    def test_the_documents_own_dates_must_come_back(self):
        """Both callers render the answer into evidence whose AGE decides what it is worth: the
        ladder's disclosure line and v2's findings artifact are both dated claims."""
        assert "Include the document's stated dates." in build_document_prompt(_ASK)

    def test_a_document_that_does_not_answer_must_say_so_plainly(self):
        """Without an explicit out, the fluent answer from parametric memory is the likeliest
        thing to come back, which is the failure this whole path is guarded against."""
        assert (
            "If the document does not address the ask, begin your reply with NOT_ADDRESSED and say so plainly."
            in build_document_prompt(_ASK)
        )

    def test_the_non_answer_opens_with_a_machine_readable_sentinel(self):
        """The plain "this does not address the ask" is the DESIGNED answer for a retrieved page
        that does not discuss the question, and rendered under the primary-grading-evidence
        caption it is prose standing in for an absent section. The sentinel is what lets the
        ladder withhold it instead of publishing it; the constant is what the ladder matches on,
        so the prompt has to carry that exact spelling."""
        assert url_context_reader.NOT_ADDRESSED_SENTINEL == "NOT_ADDRESSED"
        assert url_context_reader.NOT_ADDRESSED_SENTINEL in build_document_prompt(_ASK)


def _document_response(text: str) -> Any:
    """A fake Gemini response shaped like the typed SDK models the reader's telemetry reads."""
    url_metadata = [
        SimpleNamespace(url_retrieval_status="URL_RETRIEVAL_STATUS_SUCCESS", retrieved_url="https://bls.gov/wsp/")
    ]
    candidate = SimpleNamespace(url_context_metadata=SimpleNamespace(url_metadata=url_metadata))
    return SimpleNamespace(text=text, candidates=[candidate])


class TestTheReadActuallySendsThatPrompt:
    """The instructions are only worth pinning while the read still carries them.

    ``run_url_context_read`` builds ``contents`` itself, so the prompt builder could be bypassed
    (or its result dropped) with every test above still green.
    """

    @pytest.fixture
    def sent_contents(self, monkeypatch) -> list[str]:
        """Capture the ``contents`` string each ``generate_content`` call receives."""
        contents: list[str] = []

        def _fake_client(**_kwargs: Any) -> Any:
            def _generate_content(**kwargs: Any) -> Any:
                contents.append(kwargs["contents"])
                return _document_response("Twelve, per the table dated 2026-08-28.")

            return SimpleNamespace(models=SimpleNamespace(generate_content=_generate_content))

        monkeypatch.setattr("google.genai.Client", _fake_client)
        return contents

    def test_the_prompt_and_the_url_both_reach_the_model(self, sent_contents):
        text, n_retrievals, statuses = run_url_context_read(
            "https://bls.gov/wsp/",
            _ASK,
            api_key="key",
            role="resolution_source",
            model="gemini-test",
            thinking_level="low",
            timeout_ms=13_000,
            attempts=1,
        )

        assert sent_contents == [f"{build_document_prompt(_ASK)}\n\nURL: https://bls.gov/wsp/"]
        assert text == "Twelve, per the table dated 2026-08-28."
        assert n_retrievals == 1
        assert statuses == ["URL_RETRIEVAL_STATUS_SUCCESS"]


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
