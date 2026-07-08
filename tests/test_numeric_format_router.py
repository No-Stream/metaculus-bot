"""Tests for ``metaculus_bot.numeric_format_router``.

After Workstream C1 the router is percentile-only. Its job is:
1. Pass through parser-extracted percentiles when available.
2. F5 fallback: lift declared_percentiles from the JSON block when the parser fails.
3. Raise ValueError when neither source has percentiles.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from forecasting_tools.data_models.numeric_report import Percentile

from metaculus_bot.numeric_format_router import (
    detect_numeric_format,
    route_numeric_output,
)
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


# ---------------------------------------------------------------------------
# detect_numeric_format
# ---------------------------------------------------------------------------


class TestDetectNumericFormat:
    def test_pure_percentiles_returns_percentiles(self) -> None:
        rationale = _wrap_json_block(_VALID_PERCENTILES_PAYLOAD)
        assert detect_numeric_format(rationale) == "percentiles"

    def test_no_json_block_returns_none(self) -> None:
        rationale = "Just prose, no fenced JSON.\nPercentile 50: 42"
        assert detect_numeric_format(rationale) is None

    def test_malformed_json_block_returns_none(self) -> None:
        rationale = "Prose.\n\n```json\n{this is not valid JSON}\n```\nDone."
        assert detect_numeric_format(rationale) is None

    def test_empty_percentiles_returns_none(self) -> None:
        payload = {"question_type": "numeric", "declared_percentiles": {}}
        rationale = _wrap_json_block(payload)
        assert detect_numeric_format(rationale) is None


# ---------------------------------------------------------------------------
# route_numeric_output
# ---------------------------------------------------------------------------


class TestRouteNumericOutput:
    def test_percentiles_only_rationale_returns_percentiles_format(self) -> None:
        rationale = _wrap_json_block(_VALID_PERCENTILES_PAYLOAD)
        declared = _percentiles_from_payload(_VALID_PERCENTILES_PAYLOAD)
        question = _make_numeric_question()

        result = route_numeric_output(
            rationale=rationale,
            declared_percentiles=declared,
            question=question,
        )

        assert result.format == "percentiles"
        assert result.declared_percentiles == declared
        assert result.cdf_percentiles == declared

    def test_neither_format_raises_value_error(self) -> None:
        rationale = "Just prose, no JSON, no percentiles."
        question = _make_numeric_question()
        with pytest.raises(ValueError):
            route_numeric_output(
                rationale=rationale,
                declared_percentiles=None,
                question=question,
            )

    def test_no_json_with_declared_percentiles_uses_percentile_path(self) -> None:
        rationale = "Just prose."
        declared = _percentiles_from_payload(_VALID_PERCENTILES_PAYLOAD)
        question = _make_numeric_question()

        result = route_numeric_output(
            rationale=rationale,
            declared_percentiles=declared,
            question=question,
        )

        assert result.format == "percentiles"
        assert result.declared_percentiles == declared

    def test_structured_block_percentiles_fallback_when_arg_none(self) -> None:
        """F5: parser fails but JSON block has declared_percentiles."""
        rationale = _wrap_json_block(_VALID_PERCENTILES_PAYLOAD)
        question = _make_numeric_question()

        result = route_numeric_output(
            rationale=rationale,
            declared_percentiles=None,
            question=question,
        )

        assert result.format == "percentiles"
        assert result.declared_percentiles is not None
        assert len(result.declared_percentiles) == 3
        sorted_pcts = sorted(result.declared_percentiles, key=lambda p: p.percentile)
        assert sorted_pcts[0].percentile == pytest.approx(0.1)
        assert sorted_pcts[0].value == pytest.approx(10.0)

    def test_f5_fallback_with_13_percentiles_drives_full_pipeline(self) -> None:
        """F5 fallback with full 13 percentiles succeeds end-to-end."""
        from metaculus_bot.numeric.pipeline import (  # noqa: PLC0415
            build_numeric_distribution,
            sanitize_percentiles,
        )

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

        result = route_numeric_output(
            rationale=rationale,
            declared_percentiles=None,
            question=question,
        )

        assert result.format == "percentiles"
        assert result.declared_percentiles is not None
        assert len(result.declared_percentiles) == 13

        sanitized, zero_point = sanitize_percentiles(result.declared_percentiles, question)
        prediction = build_numeric_distribution(sanitized, question, zero_point)
        assert prediction is not None
        assert len(prediction.declared_percentiles) >= 13
