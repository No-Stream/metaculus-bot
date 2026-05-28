"""Tests for the ARM_PDF structured-math aggregation runner.

ARM_PDF replaces the LLM stacker with deterministic probability-math. For each
surviving forecaster, it extracts the structured JSON block, computes a
prediction from the declared math (hazard, Bayes, mixture, fitted distribution,
option_probs), and aggregates via pointwise median — apples-to-apples with
ARM_MEDIAN except inputs are structured-math-derived rather than declared answers.

Forecasters that lack a parseable structured-math block are DROPPED (no fallback
to declared answers). If fewer than ABLATION_MIN_FORECASTERS survive, the arm
returns success=False.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

from forecasting_tools import BinaryQuestion, MultipleChoiceQuestion, NumericQuestion

from metaculus_bot.ablation.cache import AblationCache

_OPEN = datetime(2026, 1, 1)
_RESOLVE = datetime(2026, 5, 1)


# ---------------------------------------------------------------------------
# Question factories
# ---------------------------------------------------------------------------


def _make_binary_q(qid: int = 1) -> BinaryQuestion:
    q = MagicMock(spec=BinaryQuestion)
    q.id_of_question = qid
    q.question_text = "Will it rain?"
    q.background_info = "bg"
    q.resolution_criteria = "rc"
    q.fine_print = ""
    q.page_url = f"https://example.com/q/{qid}"
    q.open_time = _OPEN
    q.scheduled_resolution_time = _RESOLVE
    return q


def _make_mc_q(qid: int = 2) -> MultipleChoiceQuestion:
    q = MagicMock(spec=MultipleChoiceQuestion)
    q.id_of_question = qid
    q.question_text = "Which color?"
    q.options = ["Red", "Blue", "Green"]
    q.background_info = "bg"
    q.resolution_criteria = "rc"
    q.fine_print = ""
    q.page_url = f"https://example.com/q/{qid}"
    q.open_time = _OPEN
    q.scheduled_resolution_time = _RESOLVE
    return q


def _make_numeric_q(qid: int = 3) -> NumericQuestion:
    q = MagicMock(spec=NumericQuestion)
    q.id_of_question = qid
    q.question_text = "What will X be?"
    q.background_info = "bg"
    q.resolution_criteria = "rc"
    q.fine_print = ""
    q.page_url = f"https://example.com/q/{qid}"
    q.open_time = _OPEN
    q.scheduled_resolution_time = _RESOLVE
    q.lower_bound = 0.0
    q.upper_bound = 100.0
    q.open_lower_bound = True
    q.open_upper_bound = True
    q.zero_point = None
    return q


# ---------------------------------------------------------------------------
# Structured block helpers — embed in reasoning text
# ---------------------------------------------------------------------------


def _binary_hazard_reasoning(rate: float = 0.1, elapsed: float = 0.3, remaining: float = 0.7) -> str:
    block = json.dumps(
        {
            "question_type": "binary",
            "hazard": {
                "rate_per_unit": rate,
                "unit": "month",
                "window_duration_units": 6.0,
                "elapsed_fraction": elapsed,
                "remaining_fraction": remaining,
            },
            "evidence": [],
            "posterior_prob": 0.5,
        }
    )
    return f"Some reasoning text\n\n```json\n{block}\n```"


def _binary_bayes_reasoning(k: int = 3, n: int = 10, evidence_direction: str = "up") -> str:
    block = json.dumps(
        {
            "question_type": "binary",
            "base_rate": {"k": k, "n": n, "ref_class": "similar events"},
            "evidence": [
                {"summary": "Evidence 1", "direction": evidence_direction, "strength": "moderate"},
            ],
            "posterior_prob": 0.5,
        }
    )
    return f"Analysis text\n\n```json\n{block}\n```"


def _binary_prior_blend_reasoning(prior_prob: float = 0.4) -> str:
    block = json.dumps(
        {
            "question_type": "binary",
            "prior": {"prob": prior_prob, "source": "historical data"},
            "evidence": [
                {"summary": "Recent increase", "direction": "up", "strength": "weak"},
            ],
            "posterior_prob": 0.5,
        }
    )
    return f"Prior-based reasoning\n\n```json\n{block}\n```"


def _binary_no_structure_reasoning() -> str:
    """A reasoning without any structured math — should be dropped by pdf arm."""
    return "I think the probability is about 60% based on my analysis."


def _mc_reasoning(option_probs: dict[str, float] | None = None) -> str:
    if option_probs is None:
        option_probs = {"Red": 0.5, "Blue": 0.3, "Green": 0.2}
    block = json.dumps(
        {
            "question_type": "multiple_choice",
            "option_probs": option_probs,
        }
    )
    return f"MC reasoning\n\n```json\n{block}\n```"


def _numeric_mixture_reasoning(mean_shift: float = 0.0) -> str:
    block = json.dumps(
        {
            "question_type": "numeric",
            "declared_percentiles": {"0.1": 25.0 + mean_shift, "0.5": 50.0 + mean_shift, "0.9": 75.0 + mean_shift},
            "mixture_components": [
                {"weight": 0.6, "mean": 40.0 + mean_shift, "sd": 12.0},
                {"weight": 0.4, "mean": 60.0 + mean_shift, "sd": 10.0},
            ],
        }
    )
    return f"Numeric reasoning\n\n```json\n{block}\n```"


def _numeric_percentile_reasoning(percentiles: dict[str, float] | None = None) -> str:
    if percentiles is None:
        percentiles = {"0.1": 20.0, "0.5": 50.0, "0.9": 80.0}
    block = json.dumps(
        {
            "question_type": "numeric",
            "declared_percentiles": percentiles,
            "distribution_family_hint": "normal",
        }
    )
    return f"Numeric reasoning\n\n```json\n{block}\n```"


# ---------------------------------------------------------------------------
# Forecaster payload factory
# ---------------------------------------------------------------------------


def _make_forecaster_payload(reasoning: str, prediction_value: dict | None = None) -> dict:
    """Build a forecaster payload with the given reasoning and optional prediction_value."""
    return {
        "model": "test-model/v1",
        "reasoning": reasoning,
        "prediction_value": prediction_value or {"type": "binary", "prob": 0.5},
        "errors": [],
    }


# ---------------------------------------------------------------------------
# Tests: Binary — hazard cascade
# ---------------------------------------------------------------------------


class TestBinaryHazardCascade:
    def test_hazard_produces_valid_probability(self, tmp_path: Any) -> None:
        """Hazard-based structured math produces a prob in [0, 1]."""
        from metaculus_bot.ablation.run_pdf import run_pdf_for_qid

        cache = AblationCache(str(tmp_path))
        question = _make_binary_q(qid=100)
        payloads = {
            "model_a": _make_forecaster_payload(_binary_hazard_reasoning(rate=0.1)),
            "model_b": _make_forecaster_payload(_binary_hazard_reasoning(rate=0.2)),
            "model_c": _make_forecaster_payload(_binary_hazard_reasoning(rate=0.15)),
        }
        result = asyncio.run(
            run_pdf_for_qid(
                qid=100,
                question=question,
                forecaster_payloads=payloads,
                cache=cache,
                force=False,
            )
        )
        assert result["success"] is True
        assert result["arm"] == "pdf"
        pred = result["stacker_prediction"]
        assert pred["type"] == "binary"
        assert 0.0 < pred["prob"] < 1.0

    def test_hazard_takes_priority_over_bayes(self, tmp_path: Any) -> None:
        """When both hazard and base_rate are present, hazard wins (cascade priority)."""
        from metaculus_bot.ablation.run_pdf import run_pdf_for_qid

        # Build a block with both hazard AND base_rate
        block = json.dumps(
            {
                "question_type": "binary",
                "hazard": {
                    "rate_per_unit": 0.5,
                    "unit": "month",
                    "window_duration_units": 12.0,
                    "elapsed_fraction": 0.0,
                    "remaining_fraction": 1.0,
                },
                "base_rate": {"k": 1, "n": 100, "ref_class": "test"},
                "evidence": [],
                "posterior_prob": 0.5,
            }
        )
        reasoning = f"text\n\n```json\n{block}\n```"
        cache = AblationCache(str(tmp_path))
        question = _make_binary_q(qid=101)
        payloads = {
            "model_a": _make_forecaster_payload(reasoning),
            "model_b": _make_forecaster_payload(reasoning),
        }
        result = asyncio.run(
            run_pdf_for_qid(
                qid=101,
                question=question,
                forecaster_payloads=payloads,
                cache=cache,
                force=False,
            )
        )
        assert result["success"] is True
        # Hazard with rate=0.5 over 12 months gives ~1-exp(-6) ≈ 0.9975
        # Base rate 1/100 = 0.01 would be very different
        prob = result["stacker_prediction"]["prob"]
        assert prob > 0.9, f"Expected high prob from hazard cascade, got {prob}"


# ---------------------------------------------------------------------------
# Tests: Binary — Bayes cascade
# ---------------------------------------------------------------------------


class TestBinaryBayesCascade:
    def test_bayes_update_with_evidence(self, tmp_path: Any) -> None:
        """Beta-binomial update + evidence strength adjusts the probability."""
        from metaculus_bot.ablation.run_pdf import run_pdf_for_qid

        cache = AblationCache(str(tmp_path))
        question = _make_binary_q(qid=200)
        payloads = {
            "model_a": _make_forecaster_payload(_binary_bayes_reasoning(k=3, n=10, evidence_direction="up")),
            "model_b": _make_forecaster_payload(_binary_bayes_reasoning(k=5, n=10, evidence_direction="up")),
        }
        result = asyncio.run(
            run_pdf_for_qid(
                qid=200,
                question=question,
                forecaster_payloads=payloads,
                cache=cache,
                force=False,
            )
        )
        assert result["success"] is True
        prob = result["stacker_prediction"]["prob"]
        # Both have "up" evidence so should be above raw base rate
        assert 0.02 < prob < 0.98


# ---------------------------------------------------------------------------
# Tests: Binary — prior blend cascade
# ---------------------------------------------------------------------------


class TestBinaryPriorBlend:
    def test_prior_blend_with_evidence(self, tmp_path: Any) -> None:
        """When only prior + evidence set (no base_rate, no hazard), blend works."""
        from metaculus_bot.ablation.run_pdf import run_pdf_for_qid

        cache = AblationCache(str(tmp_path))
        question = _make_binary_q(qid=300)
        payloads = {
            "model_a": _make_forecaster_payload(_binary_prior_blend_reasoning(prior_prob=0.3)),
            "model_b": _make_forecaster_payload(_binary_prior_blend_reasoning(prior_prob=0.5)),
        }
        result = asyncio.run(
            run_pdf_for_qid(
                qid=300,
                question=question,
                forecaster_payloads=payloads,
                cache=cache,
                force=False,
            )
        )
        assert result["success"] is True
        prob = result["stacker_prediction"]["prob"]
        assert 0.02 < prob < 0.98


# ---------------------------------------------------------------------------
# Tests: Binary — drop when no structured math
# ---------------------------------------------------------------------------


class TestBinaryDropNoStructure:
    def test_drops_forecasters_without_structure(self, tmp_path: Any) -> None:
        """Forecasters without parseable structured blocks are excluded from pdf arm."""
        from metaculus_bot.ablation.run_pdf import run_pdf_for_qid

        cache = AblationCache(str(tmp_path))
        question = _make_binary_q(qid=400)
        # 2 forecasters with structure, 1 without
        payloads = {
            "model_a": _make_forecaster_payload(_binary_hazard_reasoning(rate=0.1)),
            "model_b": _make_forecaster_payload(_binary_hazard_reasoning(rate=0.2)),
            "model_c": _make_forecaster_payload(_binary_no_structure_reasoning()),
        }
        result = asyncio.run(
            run_pdf_for_qid(
                qid=400,
                question=question,
                forecaster_payloads=payloads,
                cache=cache,
                force=False,
            )
        )
        assert result["success"] is True
        # Only 2 forecasters should contribute
        assert result["n_forecasters_used"] == 2

    def test_insufficient_after_drop_returns_failure(self, tmp_path: Any) -> None:
        """When structured-math drops leave <2 forecasters, return success=False."""
        from metaculus_bot.ablation.run_pdf import run_pdf_for_qid

        cache = AblationCache(str(tmp_path))
        question = _make_binary_q(qid=401)
        # Only 1 forecaster with structure
        payloads = {
            "model_a": _make_forecaster_payload(_binary_hazard_reasoning(rate=0.1)),
            "model_b": _make_forecaster_payload(_binary_no_structure_reasoning()),
            "model_c": _make_forecaster_payload(_binary_no_structure_reasoning()),
        }
        result = asyncio.run(
            run_pdf_for_qid(
                qid=401,
                question=question,
                forecaster_payloads=payloads,
                cache=cache,
                force=False,
            )
        )
        assert result["success"] is False
        assert result["reason"] == "insufficient_structured_forecasters"


# ---------------------------------------------------------------------------
# Tests: Multiple choice
# ---------------------------------------------------------------------------


class TestMultipleChoice:
    def test_mc_option_probs_produce_valid_prediction(self, tmp_path: Any) -> None:
        """Well-formed option_probs produce a valid MC prediction."""
        from metaculus_bot.ablation.run_pdf import run_pdf_for_qid

        cache = AblationCache(str(tmp_path))
        question = _make_mc_q(qid=500)
        payloads = {
            "model_a": _make_forecaster_payload(
                _mc_reasoning({"Red": 0.5, "Blue": 0.3, "Green": 0.2}),
                prediction_value={
                    "type": "multiple_choice",
                    "options": [
                        {"option_name": "Red", "probability": 0.5},
                        {"option_name": "Blue", "probability": 0.3},
                        {"option_name": "Green", "probability": 0.2},
                    ],
                },
            ),
            "model_b": _make_forecaster_payload(
                _mc_reasoning({"Red": 0.4, "Blue": 0.4, "Green": 0.2}),
                prediction_value={
                    "type": "multiple_choice",
                    "options": [
                        {"option_name": "Red", "probability": 0.4},
                        {"option_name": "Blue", "probability": 0.4},
                        {"option_name": "Green", "probability": 0.2},
                    ],
                },
            ),
        }
        result = asyncio.run(
            run_pdf_for_qid(
                qid=500,
                question=question,
                forecaster_payloads=payloads,
                cache=cache,
                force=False,
            )
        )
        assert result["success"] is True
        pred = result["stacker_prediction"]
        assert pred["type"] == "multiple_choice"
        total_prob = sum(opt["probability"] for opt in pred["options"])
        assert abs(total_prob - 1.0) < 0.01

    def test_mc_drops_malformed_blocks(self, tmp_path: Any) -> None:
        """MC forecasters with malformed option_probs are dropped."""
        from metaculus_bot.ablation.run_pdf import run_pdf_for_qid

        cache = AblationCache(str(tmp_path))
        question = _make_mc_q(qid=501)
        # One good, one with bad sum (doesn't sum to 1)
        bad_block = json.dumps(
            {
                "question_type": "multiple_choice",
                "option_probs": {"Red": 0.5, "Blue": 0.1, "Green": 0.1},  # sums to 0.7
            }
        )
        bad_reasoning = f"text\n\n```json\n{bad_block}\n```"
        payloads = {
            "model_a": _make_forecaster_payload(
                _mc_reasoning({"Red": 0.5, "Blue": 0.3, "Green": 0.2}),
                prediction_value={
                    "type": "multiple_choice",
                    "options": [
                        {"option_name": "Red", "probability": 0.5},
                        {"option_name": "Blue", "probability": 0.3},
                        {"option_name": "Green", "probability": 0.2},
                    ],
                },
            ),
            "model_b": _make_forecaster_payload(
                bad_reasoning,
                prediction_value={
                    "type": "multiple_choice",
                    "options": [
                        {"option_name": "Red", "probability": 0.5},
                        {"option_name": "Blue", "probability": 0.3},
                        {"option_name": "Green", "probability": 0.2},
                    ],
                },
            ),
            "model_c": _make_forecaster_payload(
                _mc_reasoning({"Red": 0.3, "Blue": 0.5, "Green": 0.2}),
                prediction_value={
                    "type": "multiple_choice",
                    "options": [
                        {"option_name": "Red", "probability": 0.3},
                        {"option_name": "Blue", "probability": 0.5},
                        {"option_name": "Green", "probability": 0.2},
                    ],
                },
            ),
        }
        result = asyncio.run(
            run_pdf_for_qid(
                qid=501,
                question=question,
                forecaster_payloads=payloads,
                cache=cache,
                force=False,
            )
        )
        assert result["success"] is True
        assert result["n_forecasters_used"] == 2


# ---------------------------------------------------------------------------
# Tests: Numeric — mixture components path
# ---------------------------------------------------------------------------


class TestNumericMixture:
    def test_mixture_produces_201_point_cdf(self, tmp_path: Any) -> None:
        """Mixture components path produces a valid 201-point CDF."""
        from metaculus_bot.ablation.run_pdf import run_pdf_for_qid

        cache = AblationCache(str(tmp_path))
        question = _make_numeric_q(qid=600)
        payloads = {
            "model_a": _make_forecaster_payload(
                _numeric_mixture_reasoning(mean_shift=0.0),
                prediction_value={"type": "numeric", "cdf_probabilities": [i / 200.0 for i in range(201)]},
            ),
            "model_b": _make_forecaster_payload(
                _numeric_mixture_reasoning(mean_shift=5.0),
                prediction_value={"type": "numeric", "cdf_probabilities": [i / 200.0 for i in range(201)]},
            ),
        }
        result = asyncio.run(
            run_pdf_for_qid(
                qid=600,
                question=question,
                forecaster_payloads=payloads,
                cache=cache,
                force=False,
            )
        )
        assert result["success"] is True
        pred = result["stacker_prediction"]
        assert pred["type"] == "numeric"
        assert len(pred["cdf_probabilities"]) == 201


# ---------------------------------------------------------------------------
# Tests: Numeric — declared percentiles fitting path
# ---------------------------------------------------------------------------


class TestNumericPercentileFit:
    def test_normal_fit_produces_valid_cdf(self, tmp_path: Any) -> None:
        """Declared percentiles with normal hint produce a valid CDF."""
        from metaculus_bot.ablation.run_pdf import run_pdf_for_qid

        cache = AblationCache(str(tmp_path))
        question = _make_numeric_q(qid=700)
        payloads = {
            "model_a": _make_forecaster_payload(
                _numeric_percentile_reasoning({"0.1": 30.0, "0.5": 50.0, "0.9": 70.0}),
                prediction_value={"type": "numeric", "cdf_probabilities": [i / 200.0 for i in range(201)]},
            ),
            "model_b": _make_forecaster_payload(
                _numeric_percentile_reasoning({"0.1": 25.0, "0.5": 48.0, "0.9": 75.0}),
                prediction_value={"type": "numeric", "cdf_probabilities": [i / 200.0 for i in range(201)]},
            ),
        }
        result = asyncio.run(
            run_pdf_for_qid(
                qid=700,
                question=question,
                forecaster_payloads=payloads,
                cache=cache,
                force=False,
            )
        )
        assert result["success"] is True
        cdf = result["stacker_prediction"]["cdf_probabilities"]
        assert len(cdf) == 201
        # CDF should be non-decreasing
        for i in range(1, len(cdf)):
            assert cdf[i] >= cdf[i - 1] - 1e-9


# ---------------------------------------------------------------------------
# Tests: Min-forecasters guard
# ---------------------------------------------------------------------------


class TestMinForecastersGuard:
    def test_below_threshold_returns_failure(self, tmp_path: Any) -> None:
        """Fewer than ABLATION_MIN_FORECASTERS structured forecasters => success=False."""
        from metaculus_bot.ablation.run_pdf import run_pdf_for_qid

        cache = AblationCache(str(tmp_path))
        question = _make_binary_q(qid=800)
        # All forecasters have no structured block
        payloads = {
            "model_a": _make_forecaster_payload(_binary_no_structure_reasoning()),
            "model_b": _make_forecaster_payload(_binary_no_structure_reasoning()),
        }
        result = asyncio.run(
            run_pdf_for_qid(
                qid=800,
                question=question,
                forecaster_payloads=payloads,
                cache=cache,
                force=False,
            )
        )
        assert result["success"] is False
        assert "insufficient" in result["reason"]


# ---------------------------------------------------------------------------
# Tests: Output schema parity
# ---------------------------------------------------------------------------


class TestOutputSchema:
    def test_payload_has_required_keys(self, tmp_path: Any) -> None:
        """Output payload has the same top-level keys as arm_stack_aug payloads."""
        from metaculus_bot.ablation.run_pdf import run_pdf_for_qid

        cache = AblationCache(str(tmp_path))
        question = _make_binary_q(qid=900)
        payloads = {
            "model_a": _make_forecaster_payload(_binary_hazard_reasoning(rate=0.1)),
            "model_b": _make_forecaster_payload(_binary_hazard_reasoning(rate=0.2)),
        }
        result = asyncio.run(
            run_pdf_for_qid(
                qid=900,
                question=question,
                forecaster_payloads=payloads,
                cache=cache,
                force=False,
            )
        )
        required_keys = {
            "success",
            "arm",
            "stacker_prediction",
            "stacker_meta_reasoning",
            "computed_quantities",
            "cross_model_aggregation",
            "stacker_model_used",
            "n_forecasters_used",
            "ran_at",
            "tools_enabled_at_runtime",
            "errors",
        }
        assert required_keys.issubset(set(result.keys()))
        assert result["stacker_model_used"] == "structured_math"
        assert result["tools_enabled_at_runtime"] is False
        assert result["cross_model_aggregation"] is None or result["cross_model_aggregation"] == ""


# ---------------------------------------------------------------------------
# Tests: Cache behavior
# ---------------------------------------------------------------------------


class TestCacheBehavior:
    def test_cache_hit_returns_without_recompute(self, tmp_path: Any) -> None:
        """Second call with force=False returns cached result."""
        from metaculus_bot.ablation.run_pdf import run_pdf_for_qid

        cache = AblationCache(str(tmp_path))
        question = _make_binary_q(qid=1000)
        payloads = {
            "model_a": _make_forecaster_payload(_binary_hazard_reasoning(rate=0.1)),
            "model_b": _make_forecaster_payload(_binary_hazard_reasoning(rate=0.2)),
        }
        result1 = asyncio.run(
            run_pdf_for_qid(
                qid=1000,
                question=question,
                forecaster_payloads=payloads,
                cache=cache,
                force=False,
            )
        )
        result2 = asyncio.run(
            run_pdf_for_qid(
                qid=1000,
                question=question,
                forecaster_payloads=payloads,
                cache=cache,
                force=False,
            )
        )
        # Cache may add metadata keys (e.g. cache_schema_version); compare core fields
        assert result1["success"] == result2["success"]
        assert result1["stacker_prediction"] == result2["stacker_prediction"]
        assert result1["n_forecasters_used"] == result2["n_forecasters_used"]

    def test_force_recomputes(self, tmp_path: Any) -> None:
        """force=True bypasses cache and recomputes."""
        from metaculus_bot.ablation.run_pdf import run_pdf_for_qid

        cache = AblationCache(str(tmp_path))
        question = _make_binary_q(qid=1001)
        payloads = {
            "model_a": _make_forecaster_payload(_binary_hazard_reasoning(rate=0.1)),
            "model_b": _make_forecaster_payload(_binary_hazard_reasoning(rate=0.2)),
        }
        result1 = asyncio.run(
            run_pdf_for_qid(
                qid=1001,
                question=question,
                forecaster_payloads=payloads,
                cache=cache,
                force=False,
            )
        )
        # Force recompute — timestamps should differ
        result2 = asyncio.run(
            run_pdf_for_qid(
                qid=1001,
                question=question,
                forecaster_payloads=payloads,
                cache=cache,
                force=True,
            )
        )
        # Both succeed with same prediction value
        assert result1["success"] is True
        assert result2["success"] is True
        assert result1["stacker_prediction"] == result2["stacker_prediction"]


# ---------------------------------------------------------------------------
# Tests: min_forecasters kwarg
# ---------------------------------------------------------------------------


class TestMinForecastersKwarg:
    def test_min_forecasters_1_succeeds_with_single_structured_forecaster(self, tmp_path: Any) -> None:
        """With min_forecasters=1, a qid with exactly 1 structured forecaster should succeed."""
        from metaculus_bot.ablation.run_pdf import run_pdf_for_qid

        cache = AblationCache(str(tmp_path))
        question = _make_binary_q(qid=1200)
        payloads = {
            "model_a": _make_forecaster_payload(_binary_hazard_reasoning(rate=0.1)),
            "model_b": _make_forecaster_payload(_binary_no_structure_reasoning()),
            "model_c": _make_forecaster_payload(_binary_no_structure_reasoning()),
        }
        result = asyncio.run(
            run_pdf_for_qid(
                qid=1200,
                question=question,
                forecaster_payloads=payloads,
                cache=cache,
                force=False,
                min_forecasters=1,
            )
        )
        assert result["success"] is True
        assert result["n_forecasters_used"] == 1

    def test_min_forecasters_2_fails_with_single_structured_forecaster(self, tmp_path: Any) -> None:
        """With min_forecasters=2, a qid with only 1 structured forecaster should fail."""
        from metaculus_bot.ablation.run_pdf import run_pdf_for_qid

        cache = AblationCache(str(tmp_path))
        question = _make_binary_q(qid=1201)
        payloads = {
            "model_a": _make_forecaster_payload(_binary_hazard_reasoning(rate=0.1)),
            "model_b": _make_forecaster_payload(_binary_no_structure_reasoning()),
            "model_c": _make_forecaster_payload(_binary_no_structure_reasoning()),
        }
        result = asyncio.run(
            run_pdf_for_qid(
                qid=1201,
                question=question,
                forecaster_payloads=payloads,
                cache=cache,
                force=False,
                min_forecasters=2,
            )
        )
        assert result["success"] is False
        assert result["reason"] == "insufficient_structured_forecasters"

    def test_default_min_forecasters_is_2(self, tmp_path: Any) -> None:
        """Default behavior (no kwarg) should be min_forecasters=2."""
        from metaculus_bot.ablation.run_pdf import run_pdf_for_qid

        cache = AblationCache(str(tmp_path))
        question = _make_binary_q(qid=1202)
        # Only 1 structured — should fail with default min_forecasters=2
        payloads = {
            "model_a": _make_forecaster_payload(_binary_hazard_reasoning(rate=0.1)),
            "model_b": _make_forecaster_payload(_binary_no_structure_reasoning()),
        }
        result = asyncio.run(
            run_pdf_for_qid(
                qid=1202,
                question=question,
                forecaster_payloads=payloads,
                cache=cache,
                force=False,
            )
        )
        assert result["success"] is False
        assert result["reason"] == "insufficient_structured_forecasters"

    def test_arm_label_reflects_min_forecasters(self, tmp_path: Any) -> None:
        """The arm label in the payload should reflect which pdf variant was run."""
        from metaculus_bot.ablation.run_pdf import ARM_PDF_MIN1, ARM_PDF_MIN2, run_pdf_for_qid

        cache = AblationCache(str(tmp_path))
        question = _make_binary_q(qid=1203)
        payloads = {
            "model_a": _make_forecaster_payload(_binary_hazard_reasoning(rate=0.1)),
            "model_b": _make_forecaster_payload(_binary_hazard_reasoning(rate=0.2)),
        }
        result_min1 = asyncio.run(
            run_pdf_for_qid(
                qid=1203,
                question=question,
                forecaster_payloads=payloads,
                cache=cache,
                force=True,
                min_forecasters=1,
                arm_label=ARM_PDF_MIN1,
            )
        )
        result_min2 = asyncio.run(
            run_pdf_for_qid(
                qid=1203,
                question=question,
                forecaster_payloads=payloads,
                cache=cache,
                force=True,
                min_forecasters=2,
                arm_label=ARM_PDF_MIN2,
            )
        )
        assert result_min1["arm"] == "pdf_min1"
        assert result_min2["arm"] == "pdf_min2"


# ---------------------------------------------------------------------------
# Tests: Surviving forecaster filter (pre-existing errors/None)
# ---------------------------------------------------------------------------


class TestSurvivingFilter:
    def test_drops_forecasters_with_errors(self, tmp_path: Any) -> None:
        """Forecasters with non-empty errors field are excluded before structured-math."""
        from metaculus_bot.ablation.run_pdf import run_pdf_for_qid

        cache = AblationCache(str(tmp_path))
        question = _make_binary_q(qid=1100)
        payloads = {
            "model_a": _make_forecaster_payload(_binary_hazard_reasoning(rate=0.1)),
            "model_b": _make_forecaster_payload(_binary_hazard_reasoning(rate=0.2)),
            "model_c": {
                "model": "bad-model",
                "reasoning": _binary_hazard_reasoning(rate=0.3),
                "prediction_value": {"type": "binary", "prob": 0.5},
                "errors": ["parse error"],
            },
        }
        result = asyncio.run(
            run_pdf_for_qid(
                qid=1100,
                question=question,
                forecaster_payloads=payloads,
                cache=cache,
                force=False,
            )
        )
        assert result["success"] is True
        assert result["n_forecasters_used"] == 2

    def test_drops_forecasters_with_none_prediction(self, tmp_path: Any) -> None:
        """Forecasters with prediction_value=None are excluded."""
        from metaculus_bot.ablation.run_pdf import run_pdf_for_qid

        cache = AblationCache(str(tmp_path))
        question = _make_binary_q(qid=1101)
        payloads = {
            "model_a": _make_forecaster_payload(_binary_hazard_reasoning(rate=0.1)),
            "model_b": _make_forecaster_payload(_binary_hazard_reasoning(rate=0.2)),
            "model_c": {
                "model": "dead-model",
                "reasoning": "",
                "prediction_value": None,
                "errors": [],
            },
        }
        result = asyncio.run(
            run_pdf_for_qid(
                qid=1101,
                question=question,
                forecaster_payloads=payloads,
                cache=cache,
                force=False,
            )
        )
        assert result["success"] is True
        assert result["n_forecasters_used"] == 2
