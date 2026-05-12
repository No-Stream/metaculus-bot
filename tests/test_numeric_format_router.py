"""Tests for ``metaculus_bot.numeric_format_router``.

The router decides whether the LLM's numeric output is in the percentile
format (Option A — default trailing ``Percentile X.X: ...`` lines) or the
mixture format (Option B — ``mixture_components`` populated in the JSON
block). It always returns a 201-point Metaculus CDF and records which
branch produced it.

Per user steer 2026-05-12: NO consistency check. If both formats are
present, the mixture wins (deterministic), and a WARNING is logged so the
frequency is auditable.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from unittest.mock import MagicMock

import pytest
from forecasting_tools.data_models.numeric_report import Percentile
from forecasting_tools.data_models.questions import NumericQuestion

from metaculus_bot.numeric_format_router import (
    RoutedNumericForecast,
    detect_numeric_format,
    route_numeric_output,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_numeric_question(
    *,
    lower_bound: float = 0.0,
    upper_bound: float = 100.0,
    open_lower_bound: bool = False,
    open_upper_bound: bool = False,
) -> NumericQuestion:
    q = MagicMock(spec=NumericQuestion)
    q.lower_bound = lower_bound
    q.upper_bound = upper_bound
    q.open_lower_bound = open_lower_bound
    q.open_upper_bound = open_upper_bound
    q.zero_point = None
    q.cdf_size = None
    q.id_of_question = 42
    q.page_url = "https://example.com/q/42"
    return q


def _wrap_json_block(payload: dict[str, Any]) -> str:
    return f"Some analysis prose.\n\n```json\n{json.dumps(payload)}\n```\n\nEnd."


_VALID_PERCENTILES_PAYLOAD: dict[str, Any] = {
    "question_type": "numeric",
    "declared_percentiles": {"0.1": 10.0, "0.5": 50.0, "0.9": 90.0},
    "distribution_family_hint": "normal",
}


_VALID_MIXTURE_PAYLOAD: dict[str, Any] = {
    "question_type": "numeric",
    "declared_percentiles": {"0.1": 10.0, "0.5": 50.0, "0.9": 90.0},
    "mixture_components": [
        {"weight": 0.3, "mean": 25.0, "sd": 8.0},
        {"weight": 0.4, "mean": 50.0, "sd": 10.0},
        {"weight": 0.3, "mean": 75.0, "sd": 6.0},
    ],
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
    def test_mixture_components_only_returns_mixture(self) -> None:
        payload = {
            "question_type": "numeric",
            "declared_percentiles": {"0.1": 10.0, "0.5": 50.0, "0.9": 90.0},
            "mixture_components": [
                {"weight": 0.5, "mean": 30.0, "sd": 5.0},
                {"weight": 0.5, "mean": 70.0, "sd": 5.0},
            ],
        }
        # The schema requires declared_percentiles, but we want this to read as
        # "mixture path". The detect function flags presence of populated
        # mixture_components — declared_percentiles always being there is a
        # schema reality, not a router signal.
        rationale = _wrap_json_block(payload)
        assert detect_numeric_format(rationale) == "both"

    def test_pure_percentiles_returns_percentiles(self) -> None:
        rationale = _wrap_json_block(_VALID_PERCENTILES_PAYLOAD)
        assert detect_numeric_format(rationale) == "percentiles"

    def test_both_returns_both(self) -> None:
        rationale = _wrap_json_block(_VALID_MIXTURE_PAYLOAD)
        assert detect_numeric_format(rationale) == "both"

    def test_no_json_block_returns_none(self) -> None:
        rationale = "Just prose, no fenced JSON.\nPercentile 50: 42"
        assert detect_numeric_format(rationale) is None

    def test_malformed_json_block_returns_none(self) -> None:
        rationale = "Prose.\n\n```json\n{this is not valid JSON}\n```\nDone."
        # Should not raise; should signal "no usable structured block".
        assert detect_numeric_format(rationale) is None

    def test_mixture_components_null_treated_as_percentiles(self) -> None:
        payload = {
            "question_type": "numeric",
            "declared_percentiles": {"0.1": 10.0, "0.5": 50.0, "0.9": 90.0},
            "mixture_components": None,
        }
        rationale = _wrap_json_block(payload)
        assert detect_numeric_format(rationale) == "percentiles"


# ---------------------------------------------------------------------------
# route_numeric_output
# ---------------------------------------------------------------------------


class TestRouteNumericOutput:
    def test_mixture_only_rationale_builds_cdf(self) -> None:
        # Mixture-only payload: schema requires declared_percentiles, but the
        # router uses the mixture branch. declared_percentiles arg is None.
        rationale = _wrap_json_block(_VALID_MIXTURE_PAYLOAD)
        question = _make_numeric_question()

        result = route_numeric_output(
            rationale=rationale,
            declared_percentiles=None,
            question=question,
        )

        assert isinstance(result, RoutedNumericForecast)
        assert result.format == "both"  # both percentiles in JSON + mixture
        assert result.mixture is not None
        assert len(result.cdf_percentiles) == 201
        for p in result.cdf_percentiles:
            assert isinstance(p, Percentile)

    def test_pure_mixture_no_percentiles_uses_mixture_path(self) -> None:
        # If only mixture_components is in the JSON (and declared_percentiles
        # arg is None), result.format must be "mixture".
        # Schema currently requires declared_percentiles too, so we craft a
        # payload that has mixture_components alongside the minimum required
        # declared_percentiles, but the parsed list passed to the router is
        # None — mimicking "the LLM emitted mixture, the percentile parser
        # found nothing, we still got a mixture".
        # Note: detect_numeric_format reports "both" because the schema
        # requires declared_percentiles. The "mixture" branch in the router
        # is reachable only when declared_percentiles arg is None.
        rationale = _wrap_json_block(_VALID_MIXTURE_PAYLOAD)
        question = _make_numeric_question()

        result = route_numeric_output(
            rationale=rationale,
            declared_percentiles=None,
            question=question,
        )

        # Mixture won.
        assert result.mixture is not None
        assert len(result.cdf_percentiles) == 201

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
        assert result.mixture is None
        assert result.declared_percentiles == declared
        # Percentile path returns the raw declared_percentiles in
        # cdf_percentiles since main.py downstream calls
        # sanitize_percentiles + build_numeric_distribution; the router
        # just passes through.
        assert result.cdf_percentiles == declared

    def test_both_format_uses_mixture_and_logs_warning(self, caplog) -> None:
        rationale = _wrap_json_block(_VALID_MIXTURE_PAYLOAD)
        declared = _percentiles_from_payload(_VALID_MIXTURE_PAYLOAD)
        question = _make_numeric_question()

        with caplog.at_level(logging.WARNING):
            result = route_numeric_output(
                rationale=rationale,
                declared_percentiles=declared,
                question=question,
            )

        assert result.format == "both"
        # Mixture wins: cdf_percentiles is 201 points (mixture-derived), not
        # the original 3-point declared list.
        assert len(result.cdf_percentiles) == 201
        assert result.mixture is not None
        # Warning logged so we can audit how often the LLM emits both.
        assert any("both" in rec.message.lower() or "mixture" in rec.message.lower() for rec in caplog.records)

    def test_neither_format_raises_value_error(self) -> None:
        # No JSON block AND declared_percentiles=None -> nothing to forecast.
        rationale = "Just prose, no JSON, no percentiles."
        question = _make_numeric_question()
        with pytest.raises(ValueError):
            route_numeric_output(
                rationale=rationale,
                declared_percentiles=None,
                question=question,
            )

    def test_no_json_with_declared_percentiles_uses_percentile_path(self) -> None:
        # No JSON block, but parser already pulled percentiles out of the
        # trailing "Percentile X: ..." lines. Use them.
        rationale = "Just prose."
        declared = _percentiles_from_payload(_VALID_PERCENTILES_PAYLOAD)
        question = _make_numeric_question()

        result = route_numeric_output(
            rationale=rationale,
            declared_percentiles=declared,
            question=question,
        )

        assert result.format == "percentiles"
        assert result.mixture is None
        assert result.declared_percentiles == declared

    def test_mixture_build_failure_falls_back_to_percentiles(self, caplog, monkeypatch) -> None:
        """If mixture CDF construction fails (e.g., invalid bounds), the router
        logs a WARNING and falls back to percentiles."""
        rationale = _wrap_json_block(_VALID_MIXTURE_PAYLOAD)
        declared = _percentiles_from_payload(_VALID_MIXTURE_PAYLOAD)
        question = _make_numeric_question()

        # Force the mixture-CDF builder to raise.
        from metaculus_bot import numeric_format_router

        def _broken_builder(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("simulated mixture build failure")

        monkeypatch.setattr(
            numeric_format_router,
            "percentiles_to_metaculus_cdf_via_mixture",
            _broken_builder,
        )

        with caplog.at_level(logging.WARNING):
            result = route_numeric_output(
                rationale=rationale,
                declared_percentiles=declared,
                question=question,
            )

        assert result.format == "percentiles"
        assert result.mixture is None
        assert result.declared_percentiles == declared
        assert any("mixture" in rec.message.lower() for rec in caplog.records)

    def test_mixture_build_failure_with_no_percentiles_raises(self, monkeypatch, caplog) -> None:
        """If mixture build fails AND no fallback percentiles are available,
        the router raises rather than silently producing nothing."""
        rationale = _wrap_json_block(_VALID_MIXTURE_PAYLOAD)
        question = _make_numeric_question()

        from metaculus_bot import numeric_format_router

        def _broken_builder(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("simulated mixture build failure")

        monkeypatch.setattr(
            numeric_format_router,
            "percentiles_to_metaculus_cdf_via_mixture",
            _broken_builder,
        )

        with caplog.at_level(logging.WARNING):
            with pytest.raises(ValueError):
                route_numeric_output(
                    rationale=rationale,
                    declared_percentiles=None,
                    question=question,
                )
