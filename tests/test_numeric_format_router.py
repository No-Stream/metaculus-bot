"""Tests for ``metaculus_bot.numeric_format_router``.

After F8 (2026-07-08) the router is a thin fallback shim:

1. Passes parser-extracted percentiles through unchanged when available.
2. F5 fallback: lifts ``declared_percentiles`` from the JSON block when the
   parser returned None.
3. Raises ``ValueError`` when neither source has percentiles.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from forecasting_tools.data_models.numeric_report import Percentile

from metaculus_bot.numeric.pipeline import build_numeric_distribution, sanitize_percentiles
from metaculus_bot.numeric_format_router import route_numeric_output
from tests.conftest import make_mock_numeric_question as _make_numeric_question


def _wrap_json_block(payload: dict[str, Any]) -> str:
    return f"Some analysis prose.\n\n```json\n{json.dumps(payload)}\n```\n\nEnd."


_VALID_PERCENTILES_PAYLOAD: dict[str, Any] = {
    "question_type": "numeric",
    "declared_percentiles": {"0.1": 10.0, "0.5": 50.0, "0.9": 90.0},
}


def _percentiles_from_payload(payload: dict[str, Any]) -> list[Percentile]:
    """Convert declared_percentiles dict to a list[Percentile] like main.py would."""
    pct_dict = payload["declared_percentiles"]
    return [
        Percentile(percentile=float(k), value=float(v))
        for k, v in sorted(pct_dict.items(), key=lambda kv: float(kv[0]))
    ]


class TestRouteNumericOutput:
    def test_parser_percentiles_pass_through_unchanged(self) -> None:
        """When parser returns percentiles, they are passed through untouched
        even if the JSON block has a conflicting set."""
        rationale = _wrap_json_block(_VALID_PERCENTILES_PAYLOAD)
        declared = _percentiles_from_payload(_VALID_PERCENTILES_PAYLOAD)

        result = route_numeric_output(rationale=rationale, declared_percentiles=declared)

        assert result == declared

    def test_neither_source_raises_value_error(self) -> None:
        rationale = "Just prose, no JSON, no percentiles."
        with pytest.raises(ValueError):
            route_numeric_output(rationale=rationale, declared_percentiles=None)

    def test_parser_percentiles_with_no_block_pass_through(self) -> None:
        """Parser percentiles arrive even when there's no JSON block at all."""
        rationale = "Just prose."
        declared = _percentiles_from_payload(_VALID_PERCENTILES_PAYLOAD)

        result = route_numeric_output(rationale=rationale, declared_percentiles=declared)

        assert result == declared

    def test_f5_fallback_lifts_block_percentiles_when_parser_none(self) -> None:
        """F5: parser returns None but the JSON block carries declared_percentiles."""
        rationale = _wrap_json_block(_VALID_PERCENTILES_PAYLOAD)

        result = route_numeric_output(rationale=rationale, declared_percentiles=None)

        sorted_pcts = sorted(result, key=lambda p: p.percentile)
        assert len(sorted_pcts) == 3
        assert sorted_pcts[0].percentile == pytest.approx(0.1)
        assert sorted_pcts[0].value == pytest.approx(10.0)
        assert sorted_pcts[-1].percentile == pytest.approx(0.9)
        assert sorted_pcts[-1].value == pytest.approx(90.0)

    def test_f5_fallback_with_13_percentiles_drives_full_pipeline(self) -> None:
        """F5 fallback with the full 13 declared percentiles runs end-to-end
        through sanitize + build_numeric_distribution."""

        thirteen_percentiles = {
            "0.01": 3.0,
            "0.025": 5.0,
            "0.05": 8.0,
            "0.10": 12.0,
            "0.20": 20.0,
            "0.40": 35.0,
            "0.50": 50.0,
            "0.60": 60.0,
            "0.80": 75.0,
            "0.90": 85.0,
            "0.95": 92.0,
            "0.975": 96.0,
            "0.99": 98.0,
        }
        payload: dict[str, Any] = {
            "question_type": "numeric",
            "declared_percentiles": thirteen_percentiles,
        }
        rationale = _wrap_json_block(payload)
        question = _make_numeric_question()

        result = route_numeric_output(rationale=rationale, declared_percentiles=None)

        assert len(result) == 13

        sanitized, zero_point = sanitize_percentiles(result, question)
        prediction = build_numeric_distribution(sanitized, question, zero_point)
        assert prediction is not None
        assert len(prediction.declared_percentiles) >= 13

    def test_malformed_block_with_none_parser_raises(self) -> None:
        """Malformed JSON + no parser percentiles → ValueError (no silent
        fallback to an empty list)."""
        rationale = "Prose.\n\n```json\n{this is not valid JSON}\n```\nDone."
        with pytest.raises(ValueError):
            route_numeric_output(rationale=rationale, declared_percentiles=None)
