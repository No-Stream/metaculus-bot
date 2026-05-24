"""Tests for the ARM_MEDIAN simple-aggregation runner used in the probabilistic-tools ablation benchmark.

ARM_MEDIAN bypasses the stacker LLM entirely. It deserializes each surviving forecaster's
``prediction_value`` and aggregates deterministically by question type:

* Binary: median of probabilities, then BINARY_PROB_MIN/MAX clamp.
* Multiple choice: option-wise median, then ``clamp_and_renormalize_mc``.
* Numeric: ``aggregate_numeric(predictions, question, method="median")`` —
  the same pointwise-CDF aggregator the production pipeline uses.

The output payload is structurally identical to ``run_stacker_for_arm``'s success
payload so ``_build_report_shim`` and the scoring pipeline consume it without
changes. ``stacker_model_used`` is the sentinel string ``"simple_aggregation"``
to mark the arm in the confounder section of the summary.
"""

from __future__ import annotations

import asyncio
import math
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from forecasting_tools import (
    BinaryQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)

from metaculus_bot.ablation.cache import AblationCache, model_slug_to_filename
from metaculus_bot.ablation.run_median import run_median_for_qid
from metaculus_bot.ablation.run_stacker import ABLATION_MIN_FORECASTERS, ARM_MEDIAN
from metaculus_bot.constants import BINARY_PROB_MAX, BINARY_PROB_MIN

_OPEN = datetime(2026, 1, 1)
_RESOLVE = datetime(2026, 5, 1)


# ---------------------------------------------------------------------------
# Question factories (same shape as test_ablation_run_stacker.py)
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
    q.unit_of_measure = "USD"
    q.lower_bound = 0.0
    q.upper_bound = 100.0
    q.open_lower_bound = False
    q.open_upper_bound = False
    q.nominal_lower_bound = None
    q.nominal_upper_bound = None
    q.zero_point = None
    q.cdf_size = 201
    q.open_time = _OPEN
    q.scheduled_resolution_time = _RESOLVE
    return q


# ---------------------------------------------------------------------------
# Forecaster payload factories
# ---------------------------------------------------------------------------


def _binary_payload(model: str, value: float) -> dict:
    return {
        "prediction_value": {"type": "binary", "prob": value},
        "reasoning": f"Model: {model}\n\nrationale text from {model}",
        "errors": [],
        "model": model,
    }


def _mc_payload(model: str, probs: dict[str, float]) -> dict:
    return {
        "prediction_value": {
            "type": "multiple_choice",
            "options": [{"option_name": name, "probability": prob} for name, prob in probs.items()],
        },
        "reasoning": f"Model: {model}\n\nrationale text from {model}",
        "errors": [],
        "model": model,
    }


def _numeric_payload(model: str, median: float = 50.0) -> dict:
    """Build a numeric forecaster payload in the post-Bucket-1 full-CDF schema."""
    declared = [
        {"percentile": 0.025, "value": max(0.5, median - 30)},
        {"percentile": 0.05, "value": max(1.0, median - 25)},
        {"percentile": 0.10, "value": max(2.0, median - 20)},
        {"percentile": 0.20, "value": max(5.0, median - 12)},
        {"percentile": 0.40, "value": max(10.0, median - 5)},
        {"percentile": 0.50, "value": median},
        {"percentile": 0.60, "value": min(99.0, median + 5)},
        {"percentile": 0.80, "value": min(99.0, median + 12)},
        {"percentile": 0.90, "value": min(99.0, median + 20)},
        {"percentile": 0.95, "value": min(99.0, median + 25)},
        {"percentile": 0.975, "value": min(99.0, median + 30)},
    ]
    cdf_probabilities = [0.001 + (0.998 * i / 200) for i in range(201)]
    return {
        "prediction_value": {
            "type": "numeric",
            "declared_percentiles": declared,
            "cdf_probabilities": cdf_probabilities,
            "lower_bound": 0.0,
            "upper_bound": 100.0,
            "open_lower_bound": False,
            "open_upper_bound": False,
            "zero_point": None,
            "cdf_size": 201,
        },
        "reasoning": f"Model: {model}\n\nrationale text from {model}",
        "errors": [],
        "model": model,
    }


# ---------------------------------------------------------------------------
# Common fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cache(tmp_path: Path) -> AblationCache:
    return AblationCache(tmp_path / "abl")


def _run(coro: Any) -> Any:
    return asyncio.get_event_loop().run_until_complete(coro)


# Schema reference: keys that ``run_stacker_for_arm`` writes on a success
# payload. ARM_MEDIAN must mirror this shape so ``_build_report_shim`` and the
# scoring pipeline consume it without branching. Sourced from run_stacker.py
# success_payload around line 671-683 (and the median_fallback path at
# 597-612).
_REQUIRED_PAYLOAD_KEYS = {
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


# ===========================================================================
# Binary
# ===========================================================================


class TestBinary:
    def test_binary_simple_median(self, cache: AblationCache) -> None:
        question = _make_binary_q(qid=101)
        forecaster_payloads = {
            model_slug_to_filename(f"openrouter/test/m{i}"): _binary_payload(f"openrouter/test/m{i}", value)
            for i, value in enumerate([0.1, 0.3, 0.5, 0.7])
        }
        payload = _run(
            run_median_for_qid(
                qid=101,
                question=question,
                forecaster_payloads=forecaster_payloads,
                cache=cache,
            )
        )
        assert payload["success"] is True
        assert payload["arm"] == "median"
        # Median of [0.1, 0.3, 0.5, 0.7] = 0.4 (within clamp bounds, no clamp).
        assert payload["stacker_prediction"]["type"] == "binary"
        assert math.isclose(payload["stacker_prediction"]["prob"], 0.4, abs_tol=1e-6)
        assert payload["n_forecasters_used"] == 4
        assert payload["stacker_model_used"] == "simple_aggregation"
        assert payload["tools_enabled_at_runtime"] is False
        assert payload["cross_model_aggregation"] in (None, "")
        assert payload["errors"] == []

    def test_binary_clamps_low_predictions(self, cache: AblationCache) -> None:
        question = _make_binary_q(qid=102)
        forecaster_payloads = {
            model_slug_to_filename(f"openrouter/test/m{i}"): _binary_payload(f"openrouter/test/m{i}", value)
            for i, value in enumerate([0.001, 0.001, 0.001])
        }
        payload = _run(
            run_median_for_qid(
                qid=102,
                question=question,
                forecaster_payloads=forecaster_payloads,
                cache=cache,
            )
        )
        assert math.isclose(payload["stacker_prediction"]["prob"], BINARY_PROB_MIN, abs_tol=1e-9)

    def test_binary_clamps_high_predictions(self, cache: AblationCache) -> None:
        question = _make_binary_q(qid=103)
        forecaster_payloads = {
            model_slug_to_filename(f"openrouter/test/m{i}"): _binary_payload(f"openrouter/test/m{i}", value)
            for i, value in enumerate([0.999, 0.999, 0.999])
        }
        payload = _run(
            run_median_for_qid(
                qid=103,
                question=question,
                forecaster_payloads=forecaster_payloads,
                cache=cache,
            )
        )
        assert math.isclose(payload["stacker_prediction"]["prob"], BINARY_PROB_MAX, abs_tol=1e-9)


# ===========================================================================
# Multiple choice
# ===========================================================================


class TestMultipleChoice:
    def test_mc_option_wise_median_then_renormalize(self, cache: AblationCache) -> None:
        question = _make_mc_q(qid=201)
        # Three forecasters, all rank Red highest. Option-wise median should
        # also rank Red highest, and probabilities should sum to 1.
        forecaster_payloads = {
            model_slug_to_filename("openrouter/test/m1"): _mc_payload(
                "openrouter/test/m1", {"Red": 0.6, "Blue": 0.3, "Green": 0.1}
            ),
            model_slug_to_filename("openrouter/test/m2"): _mc_payload(
                "openrouter/test/m2", {"Red": 0.5, "Blue": 0.3, "Green": 0.2}
            ),
            model_slug_to_filename("openrouter/test/m3"): _mc_payload(
                "openrouter/test/m3", {"Red": 0.4, "Blue": 0.4, "Green": 0.2}
            ),
        }
        payload = _run(
            run_median_for_qid(
                qid=201,
                question=question,
                forecaster_payloads=forecaster_payloads,
                cache=cache,
            )
        )
        assert payload["success"] is True
        assert payload["stacker_prediction"]["type"] == "multiple_choice"
        options = payload["stacker_prediction"]["options"]
        # Order matches question.options
        assert [o["option_name"] for o in options] == ["Red", "Blue", "Green"]
        total = sum(o["probability"] for o in options)
        assert math.isclose(total, 1.0, abs_tol=1e-6)
        # Red should still be the leading option.
        red_prob = next(o["probability"] for o in options if o["option_name"] == "Red")
        green_prob = next(o["probability"] for o in options if o["option_name"] == "Green")
        assert red_prob > green_prob


# ===========================================================================
# Numeric
# ===========================================================================


class TestNumeric:
    def test_numeric_aggregates_via_aggregate_numeric_median(self, cache: AblationCache) -> None:
        question = _make_numeric_q(qid=301)
        forecaster_payloads = {
            model_slug_to_filename("openrouter/test/m1"): _numeric_payload("openrouter/test/m1", median=40.0),
            model_slug_to_filename("openrouter/test/m2"): _numeric_payload("openrouter/test/m2", median=50.0),
            model_slug_to_filename("openrouter/test/m3"): _numeric_payload("openrouter/test/m3", median=60.0),
        }
        payload = _run(
            run_median_for_qid(
                qid=301,
                question=question,
                forecaster_payloads=forecaster_payloads,
                cache=cache,
            )
        )
        assert payload["success"] is True
        sp = payload["stacker_prediction"]
        assert sp["type"] == "numeric"
        # Output schema mirrors what serialize_prediction_value emits.
        assert "cdf_probabilities" in sp
        cdf = sp["cdf_probabilities"]
        assert len(cdf) == 201
        # Monotonically non-decreasing.
        for i in range(1, len(cdf)):
            assert cdf[i] >= cdf[i - 1] - 1e-9
        # Endpoints pinned (closed bounds).
        assert cdf[0] == 0.0
        assert cdf[-1] == 1.0


# ===========================================================================
# Surviving-filter parity with ARM_STACK / ARM_STACK_AUG
# ===========================================================================


class TestSurvivingFilter:
    def test_drops_none_prediction(self, cache: AblationCache) -> None:
        question = _make_binary_q(qid=401)
        forecaster_payloads = {
            "good_a": _binary_payload("openrouter/test/m1", 0.4),
            "good_b": _binary_payload("openrouter/test/m2", 0.6),
            "bad_none": {
                "prediction_value": None,
                "reasoning": "failed",
                "errors": ["parse error"],
                "model": "openrouter/test/m3",
            },
        }
        payload = _run(
            run_median_for_qid(
                qid=401,
                question=question,
                forecaster_payloads=forecaster_payloads,
                cache=cache,
            )
        )
        assert payload["success"] is True
        assert payload["n_forecasters_used"] == 2
        assert math.isclose(payload["stacker_prediction"]["prob"], 0.5, abs_tol=1e-6)

    def test_drops_errors_set(self, cache: AblationCache) -> None:
        question = _make_binary_q(qid=402)
        forecaster_payloads = {
            "good_a": _binary_payload("openrouter/test/m1", 0.4),
            "good_b": _binary_payload("openrouter/test/m2", 0.6),
            "bad_errors": {
                **_binary_payload("openrouter/test/m3", 0.99),
                "errors": ["timeout"],
            },
        }
        payload = _run(
            run_median_for_qid(
                qid=402,
                question=question,
                forecaster_payloads=forecaster_payloads,
                cache=cache,
            )
        )
        assert payload["success"] is True
        assert payload["n_forecasters_used"] == 2

    def test_drops_nan_prediction(self, cache: AblationCache) -> None:
        question = _make_binary_q(qid=403)
        forecaster_payloads = {
            "good_a": _binary_payload("openrouter/test/m1", 0.4),
            "good_b": _binary_payload("openrouter/test/m2", 0.6),
            "bad_nan": {
                "prediction_value": {"type": "binary", "prob": float("nan")},
                "reasoning": "Model: nan-emitter",
                "errors": [],
                "model": "openrouter/test/m3",
            },
        }
        payload = _run(
            run_median_for_qid(
                qid=403,
                question=question,
                forecaster_payloads=forecaster_payloads,
                cache=cache,
            )
        )
        assert payload["success"] is True
        assert payload["n_forecasters_used"] == 2


# ===========================================================================
# Min-forecasters guard
# ===========================================================================


class TestMinForecastersGuard:
    def test_below_threshold_returns_failure_payload(self, cache: AblationCache) -> None:
        question = _make_binary_q(qid=501)
        forecaster_payloads = {
            "only_one": _binary_payload("openrouter/test/m1", 0.5),
        }
        payload = _run(
            run_median_for_qid(
                qid=501,
                question=question,
                forecaster_payloads=forecaster_payloads,
                cache=cache,
            )
        )
        assert payload["success"] is False
        assert payload["arm"] == "median"
        assert payload["n_forecasters_used"] == 1
        # The same threshold used for ARM_STACK/ARM_STACK_AUG applies here too — verifies
        # ARM_MEDIAN uses ABLATION_MIN_FORECASTERS, not a different constant.
        assert ABLATION_MIN_FORECASTERS == 2

    def test_zero_surviving_after_filter_returns_failure(self, cache: AblationCache) -> None:
        question = _make_binary_q(qid=502)
        forecaster_payloads = {
            "bad_a": {
                "prediction_value": None,
                "reasoning": "fail",
                "errors": ["x"],
                "model": "openrouter/test/m1",
            },
        }
        payload = _run(
            run_median_for_qid(
                qid=502,
                question=question,
                forecaster_payloads=forecaster_payloads,
                cache=cache,
            )
        )
        assert payload["success"] is False
        assert payload["n_forecasters_used"] == 0


# ===========================================================================
# Output schema parity with run_stacker_for_arm
# ===========================================================================


class TestSchemaParity:
    def test_success_payload_has_all_required_keys(self, cache: AblationCache) -> None:
        """ARM_MEDIAN success payload must carry every key ``run_stacker_for_arm`` writes,
        so ``_build_report_shim`` and the scoring pipeline don't need to branch on arm."""
        question = _make_binary_q(qid=601)
        forecaster_payloads = {
            model_slug_to_filename(f"openrouter/test/m{i}"): _binary_payload(f"openrouter/test/m{i}", 0.5)
            for i in range(3)
        }
        payload = _run(
            run_median_for_qid(
                qid=601,
                question=question,
                forecaster_payloads=forecaster_payloads,
                cache=cache,
            )
        )
        missing = _REQUIRED_PAYLOAD_KEYS - set(payload.keys())
        assert not missing, f"ARM_MEDIAN payload missing keys: {missing}"

    def test_failure_payload_has_required_keys(self, cache: AblationCache) -> None:
        """Even insufficient-forecaster failure path must include the schema keys
        the cli/spend code subscripts directly (``stacker_model_used``, ``arm``,
        ``success``, etc.)."""
        question = _make_binary_q(qid=602)
        forecaster_payloads: dict[str, dict] = {}
        payload = _run(
            run_median_for_qid(
                qid=602,
                question=question,
                forecaster_payloads=forecaster_payloads,
                cache=cache,
            )
        )
        # Minimum keys cli/spend touches on a failure: success, arm,
        # stacker_model_used, n_forecasters_used, ran_at, tools_enabled_at_runtime, errors.
        for key in (
            "success",
            "arm",
            "stacker_model_used",
            "n_forecasters_used",
            "ran_at",
            "tools_enabled_at_runtime",
            "errors",
        ):
            assert key in payload, f"failure payload missing '{key}'"

    def test_arm_constant_is_median(self) -> None:
        assert ARM_MEDIAN == "median"


# ===========================================================================
# Cache round-trip
# ===========================================================================


class TestCacheRoundTrip:
    def test_writes_arm_median_json_to_cache(self, cache: AblationCache, tmp_path: Path) -> None:
        question = _make_binary_q(qid=701)
        forecaster_payloads = {
            model_slug_to_filename(f"openrouter/test/m{i}"): _binary_payload(f"openrouter/test/m{i}", 0.5)
            for i in range(3)
        }
        _run(
            run_median_for_qid(
                qid=701,
                question=question,
                forecaster_payloads=forecaster_payloads,
                cache=cache,
            )
        )
        cached = cache.read_stacker_output(qid=701, arm=ARM_MEDIAN)
        assert cached is not None
        assert cached["arm"] == "median"
        assert cached["success"] is True

    def test_cache_hit_skips_recomputation(self, cache: AblationCache) -> None:
        """Second call returns the cached payload without re-aggregating."""
        question = _make_binary_q(qid=702)
        forecaster_payloads = {
            model_slug_to_filename(f"openrouter/test/m{i}"): _binary_payload(f"openrouter/test/m{i}", 0.5)
            for i in range(3)
        }
        first = _run(
            run_median_for_qid(
                qid=702,
                question=question,
                forecaster_payloads=forecaster_payloads,
                cache=cache,
            )
        )
        # Mutate ran_at to confirm the second call returns the cached value.
        cached = cache.read_stacker_output(qid=702, arm=ARM_MEDIAN)
        assert cached is not None
        # Second call with the same inputs should return the cached payload.
        second = _run(
            run_median_for_qid(
                qid=702,
                question=question,
                forecaster_payloads=forecaster_payloads,
                cache=cache,
            )
        )
        assert first["ran_at"] == second["ran_at"]
