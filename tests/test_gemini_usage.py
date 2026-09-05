"""Tests for the GEMINI_USAGE token-spend marker (``research/gemini_usage.py``).

The line shape is a contract with ``scripts/telemetry/markers.py``, so the field order and
spelling are pinned literally here rather than field-by-field: a reordering that still
carries every field would break the archive parser silently.
"""

import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from google.genai import types as genai_types

from metaculus_bot.research.gemini_usage import _TOKEN_FIELDS, log_gemini_usage


def _usage(**counts: object) -> SimpleNamespace:
    """A usage_metadata stand-in; every field is Optional on the real SDK model.

    Values are typed ``object`` so a case can hand over a non-int and prove the marker
    refuses to render it as a count.
    """
    fields: dict[str, object] = {
        "prompt_token_count": None,
        "tool_use_prompt_token_count": None,
        "candidates_token_count": None,
        "thoughts_token_count": None,
        "total_token_count": None,
    }
    fields.update(counts)
    return SimpleNamespace(**fields)


def _response(
    usage: object = None,
    *,
    model_version: str | None = None,
    search_queries: list[list[str] | None] | None = None,
) -> SimpleNamespace:
    """A GenerateContentResponse stand-in; ``search_queries`` is one entry per candidate."""
    candidates = [
        SimpleNamespace(grounding_metadata=SimpleNamespace(web_search_queries=queries))
        for queries in (search_queries or [])
    ]
    return SimpleNamespace(usage_metadata=usage, model_version=model_version, candidates=candidates)


class TestGeminiUsageMarker:
    def test_full_usage_renders_every_field_in_contract_order(self, caplog: pytest.LogCaptureFixture) -> None:
        response = _response(
            _usage(
                prompt_token_count=1200,
                tool_use_prompt_token_count=340,
                candidates_token_count=900,
                thoughts_token_count=2600,
                total_token_count=5040,
            ),
            model_version="gemini-3-flash-preview-002",
            search_queries=[["who won", "when"]],
        )

        with caplog.at_level(logging.INFO, logger="metaculus_bot.research.gemini_usage"):
            line = log_gemini_usage(response, role="grounded_search", model="gemini-3-flash-preview", question="44944")

        assert line == (
            "GEMINI_USAGE: role=grounded_search model=gemini-3-flash-preview-002 prompt_tokens=1200 "
            "tool_use_prompt_tokens=340 candidates_tokens=900 thoughts_tokens=2600 total_tokens=5040 "
            "search_queries=2 question=44944"
        )
        assert line in caplog.text, "the line must be logged, not merely returned"

    def test_absent_counts_read_n_a_rather_than_zero(self) -> None:
        # A gap in Google's reporting must not become a measurement: thoughts_tokens=0 is a
        # real reading (the model did not think) and has to stay distinguishable.
        line = log_gemini_usage(
            _response(_usage(prompt_token_count=10, total_token_count=10)),
            role="read_document",
            model="gemini-3.5-flash",
        )

        assert "prompt_tokens=10" in line
        assert "thoughts_tokens=n/a" in line
        assert "candidates_tokens=n/a" in line
        assert "tool_use_prompt_tokens=n/a" in line

    def test_zero_counts_render_as_zero(self) -> None:
        line = log_gemini_usage(
            _response(_usage(thoughts_token_count=0, prompt_token_count=0)),
            role="read_document",
            model="gemini-3.5-flash",
        )

        assert "thoughts_tokens=0" in line
        assert "prompt_tokens=0" in line

    def test_question_is_omitted_when_not_supplied(self) -> None:
        line = log_gemini_usage(_response(_usage()), role="read_document", model="gemini-3.5-flash")

        assert "question=" not in line
        assert line.endswith("search_queries=0"), "search_queries stays the last field when no question rides along"

    def test_configured_model_is_the_fallback_when_the_response_names_none(self) -> None:
        line = log_gemini_usage(_response(_usage()), role="grounded_search", model="gemini-3-flash-preview")

        assert "model=gemini-3-flash-preview " in line

    def test_search_queries_sum_across_candidates_and_default_to_zero(self) -> None:
        several = log_gemini_usage(
            _response(_usage(), search_queries=[["a", "b"], None, ["c"]]),
            role="grounded_search",
            model="m",
        )
        assert "search_queries=3" in several

        # An absent query list IS a count of none (the SDK omits it when the tool issued no
        # queries), so this one is a real 0 rather than an n/a.
        none_reported = log_gemini_usage(_response(_usage()), role="grounded_search", model="m")
        assert "search_queries=0" in none_reported


class TestGeminiUsageNeverRaises:
    """Pure observation on a path that bills money and returns research: it must not be
    able to take that path down, and a payload it cannot read must not invent numbers."""

    @pytest.mark.parametrize(
        "response",
        [
            SimpleNamespace(),  # nothing at all (an older SDK shape, or a test double)
            SimpleNamespace(usage_metadata=None, model_version=None, candidates=None),
            SimpleNamespace(usage_metadata=_usage(), model_version=None, candidates="not-a-list"),
            SimpleNamespace(usage_metadata="wrong-type", model_version=None, candidates=[]),
            SimpleNamespace(usage_metadata=_usage(prompt_token_count="lots"), model_version=None, candidates=[]),
        ],
        ids=["empty", "all-none", "candidates-not-iterable", "usage-wrong-type", "count-not-an-int"],
    )
    def test_malformed_responses_report_n_a(self, response: Any) -> None:
        line = log_gemini_usage(response, role="grounded_search", model="gemini-3-flash-preview")

        assert line.startswith("GEMINI_USAGE: role=grounded_search model=gemini-3-flash-preview ")
        assert "prompt_tokens=n/a" in line

    def test_unreadable_grounding_metadata_does_not_cost_the_token_counts(self) -> None:
        """The tokens are the reason the marker exists, so the two reads stay independent.

        This is the real ``read_document`` shape at one remove: its candidates carry
        url_context metadata and the grounding fields the query count walks are absent, and
        a single guard around both reads blanked the spend figures over it.
        """
        candidate = SimpleNamespace(url_context_metadata=SimpleNamespace(url_metadata=[]))
        response = SimpleNamespace(
            usage_metadata=_usage(prompt_token_count=8000, total_token_count=8420),
            model_version=None,
            candidates=[candidate],
        )

        line = log_gemini_usage(response, role="read_document", model="gemini-3.5-flash")

        assert "prompt_tokens=8000" in line
        assert "total_tokens=8420" in line
        assert "search_queries=0" in line, "a document read issues no search queries; that is a real 0"

    def test_a_mock_response_does_not_leak_a_repr_into_the_fields(self) -> None:
        # A MagicMock answers every attribute, so the model id would come back as a repr
        # carrying spaces and split into bogus fields. Only a real string is trusted.
        line = log_gemini_usage(MagicMock(), role="read_document", model="gemini-3.5-flash")

        assert "model=gemini-3.5-flash " in line
        assert "MagicMock" not in line


class TestTheNamesThisModuleReadsAreDeclaredSdkFields:
    """Every name read off a response is a declared field on the installed google-genai models.

    This is the price of degrading to ``n/a`` instead of crashing. The reads are ``getattr``
    with defaults, so an SDK rename cannot raise here — it would report ``n/a`` for the rest of
    the season on a marker whose whole purpose is answering how much Google spend a run made.
    The sibling ``url_context_telemetry`` can afford direct field reads because it decides
    whether research is TRUSTED and so must fail loudly; this one records spend on a path that
    also returns the research, so the loud failure belongs in CI instead.
    """

    def test_the_token_counts_are_usage_metadata_fields(self) -> None:
        declared = set(genai_types.GenerateContentResponseUsageMetadata.model_fields)
        read = {attribute for _field, attribute in _TOKEN_FIELDS}

        assert read <= declared, f"renamed or removed on the SDK: {sorted(read - declared)}"

    def test_the_response_and_grounding_names_are_declared_fields(self) -> None:
        assert {"usage_metadata", "model_version", "candidates"} <= set(
            genai_types.GenerateContentResponse.model_fields
        )
        assert "grounding_metadata" in genai_types.Candidate.model_fields
        assert "web_search_queries" in genai_types.GroundingMetadata.model_fields
