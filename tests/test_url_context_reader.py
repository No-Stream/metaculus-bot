"""The prompt wrapped around every paid Gemini ``url_context`` read.

``url_context_reader`` is the one call both paying surfaces share: gap-fill v2's ``read_document``
tool and the Tier-1 resolution-source ladder's last rung. Neither renders raw retrieval output —
both render the MODEL'S ANSWER about a document, under a caption that calls the section primary
grading evidence — so the three instructions in ``build_document_prompt`` are what keep the answer
checkable at all: quotes instead of paraphrase, the document's own dates, and an explicit "this
does not address the ask" instead of a fluent answer assembled out of recall (the Q38195 failure:
30 search queries, 0 grounding chunks, a fabricated contract table in front of forecasters).

Each instruction gets its own test, pinned on the sentence that carries the rule rather than on
the whole prompt string, so rewording the surrounding wrapper is free and dropping a rule is not.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

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
        thing to come back — which is the failure this whole path is guarded against."""
        assert "If the document does not address the ask, say that plainly." in build_document_prompt(_ASK)


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
