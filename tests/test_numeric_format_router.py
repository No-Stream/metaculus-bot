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

import pytest
from forecasting_tools.data_models.numeric_report import Percentile

from metaculus_bot.numeric_format_router import (
    RoutedNumericForecast,
    detect_numeric_format,
    route_numeric_output,
)
from tests.conftest import make_mock_numeric_question as _make_numeric_question


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

    def test_pure_mixture_with_percentiles_in_json_returns_both_format(self) -> None:
        # F2: clarify the contract. The current v1 schema requires
        # declared_percentiles even when mixture_components is present, so a
        # payload-with-mixture always carries percentiles in the JSON. Even
        # when the function-arg declared_percentiles is None, the router
        # detects percentiles inside the structured block and reports
        # ``format == "both"``, not ``"mixture"``. The literal "mixture"
        # branch is exercised in the next test via parser bypass.
        rationale = _wrap_json_block(_VALID_MIXTURE_PAYLOAD)
        question = _make_numeric_question()

        result = route_numeric_output(
            rationale=rationale,
            declared_percentiles=None,
            question=question,
        )

        assert result.format == "both"
        assert result.mixture is not None
        assert len(result.cdf_percentiles) == 201

    def test_mixture_only_no_percentiles_in_json(self, monkeypatch) -> None:
        # F2: literal "mixture" branch — only reachable when the parser
        # returns a NumericStructured with empty declared_percentiles,
        # which the v1 schema currently disallows but we defend against.
        # Monkeypatch parse_structured_block so it returns a stand-in with
        # mixture_components populated and declared_percentiles empty.
        from metaculus_bot import numeric_format_router
        from metaculus_bot.structured_output_schema import (
            MixtureComponentDeclaration,
            NumericStructured,
        )

        # NumericStructured.model_validate would reject empty declared_percentiles,
        # so build the instance via model_construct (skips validation) — this
        # mimics a future schema where mixture-only payloads are legal.
        fake_structured = NumericStructured.model_construct(
            question_type="numeric",
            declared_percentiles={},
            mixture_components=[
                MixtureComponentDeclaration(weight=0.5, mean=30.0, sd=5.0),
                MixtureComponentDeclaration(weight=0.5, mean=70.0, sd=5.0),
            ],
            distribution_family_hint=None,
            student_t_df=None,
            tails=None,
            scenarios=[],
            prior=None,
        )
        monkeypatch.setattr(
            numeric_format_router,
            "parse_structured_block",
            lambda rationale, qtype: fake_structured,
        )
        question = _make_numeric_question()

        result = route_numeric_output(
            rationale="ignored",
            declared_percentiles=None,
            question=question,
        )

        assert result.format == "mixture"
        assert result.mixture is not None
        assert result.declared_percentiles is None
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
        the router raises rather than silently producing nothing.

        The structured block in _VALID_MIXTURE_PAYLOAD ships percentiles, so
        we must monkeypatch parse_structured_block to return mixture-only
        for this test (otherwise F5's structured-block fallback kicks in)."""
        from metaculus_bot import numeric_format_router
        from metaculus_bot.structured_output_schema import (
            MixtureComponentDeclaration,
            NumericStructured,
        )

        question = _make_numeric_question()

        mixture_only = NumericStructured.model_construct(
            question_type="numeric",
            declared_percentiles={},
            mixture_components=[
                MixtureComponentDeclaration(weight=0.5, mean=30.0, sd=5.0),
                MixtureComponentDeclaration(weight=0.5, mean=70.0, sd=5.0),
            ],
            distribution_family_hint=None,
            student_t_df=None,
            tails=None,
            scenarios=[],
            prior=None,
        )
        monkeypatch.setattr(
            numeric_format_router,
            "parse_structured_block",
            lambda rationale, qtype: mixture_only,
        )

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
                    rationale="ignored",
                    declared_percentiles=None,
                    question=question,
                )

    def test_structured_block_percentiles_fallback_when_arg_none(self) -> None:
        # F5: parser-LLM may fail to extract trailing "Percentile X.X" lines
        # (declared_percentiles arg is None), but the rationale's JSON block
        # carries valid declared_percentiles. The router should pick those up
        # as fallback rather than raising.
        rationale = _wrap_json_block(_VALID_PERCENTILES_PAYLOAD)
        question = _make_numeric_question()

        result = route_numeric_output(
            rationale=rationale,
            declared_percentiles=None,
            question=question,
        )

        assert result.format == "percentiles"
        assert result.mixture is None
        assert result.declared_percentiles is not None
        assert len(result.declared_percentiles) == 3
        # Round-trip: 0.1 → 10.0, 0.5 → 50.0, 0.9 → 90.0.
        sorted_pcts = sorted(result.declared_percentiles, key=lambda p: p.percentile)
        assert sorted_pcts[0].percentile == pytest.approx(0.1)
        assert sorted_pcts[0].value == pytest.approx(10.0)
        assert sorted_pcts[1].percentile == pytest.approx(0.5)
        assert sorted_pcts[1].value == pytest.approx(50.0)
        assert sorted_pcts[2].percentile == pytest.approx(0.9)
        assert sorted_pcts[2].value == pytest.approx(90.0)

    def test_structured_block_percentiles_fallback_on_mixture_failure(self, monkeypatch, caplog) -> None:
        # F5: when the mixture builder fails AND the function-arg declared
        # percentiles are None, the router should fall back to the
        # structured-block declared_percentiles instead of raising.
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
            result = route_numeric_output(
                rationale=rationale,
                declared_percentiles=None,
                question=question,
            )

        assert result.format == "percentiles"
        assert result.mixture is None
        assert result.declared_percentiles is not None
        assert len(result.declared_percentiles) == 3

    def test_f5_fallback_with_11_percentiles_drives_full_pipeline(self) -> None:
        # F22 (b)/(c): F5 fallback path uses unsanitized percentiles from the
        # structured block. Verify that when the JSON block carries the full
        # 11 standard percentiles, the downstream pipeline (sanitize_percentiles
        # + build_numeric_distribution) still succeeds end-to-end. Earlier the
        # test only checked router output; this test extends coverage to the
        # full numeric pipeline that main.py runs after the router returns.
        eleven_percentiles = {
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
        }
        payload: dict[str, Any] = {
            "question_type": "numeric",
            "declared_percentiles": eleven_percentiles,
            "distribution_family_hint": "normal",
        }
        rationale = _wrap_json_block(payload)
        question = _make_numeric_question()

        result = route_numeric_output(
            rationale=rationale,
            declared_percentiles=None,
            question=question,
        )

        assert result.format == "percentiles"
        assert result.mixture is None
        assert result.declared_percentiles is not None
        assert len(result.declared_percentiles) == 11

        # Drive the downstream pipeline that main.py runs on the percentile
        # branch. Both calls must succeed without raising.
        from metaculus_bot.numeric.pipeline import (
            build_numeric_distribution,
            sanitize_percentiles,
        )

        sanitized, zero_point = sanitize_percentiles(result.declared_percentiles, question)
        prediction = build_numeric_distribution(sanitized, question, zero_point)
        assert prediction is not None
        assert len(prediction.declared_percentiles) >= 11
