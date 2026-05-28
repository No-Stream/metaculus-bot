"""Tests for the ARM_MEAN and ARM_MEDIAN simple-aggregation runners.

Both bypass the stacker LLM entirely and aggregate deterministically by question type:

* Binary: mean/median of probabilities, then BINARY_PROB_MIN/MAX clamp.
* Multiple choice: option-wise mean/median, then ``clamp_and_renormalize_mc``.
* Numeric: ``aggregate_numeric(predictions, question, method=...)`` —
  the same pointwise-CDF aggregator the production pipeline uses.

Tests are parametrized over (method_label, runner_fn, arm_constant) so shared
behavior is verified for both arms in a single test class. Median-only tests
(surviving-filter, failure-payload schema) remain non-parametrized.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

import pytest

from metaculus_bot.ablation.cache import AblationCache, model_slug_to_filename
from metaculus_bot.ablation.run_simple_agg import run_mean_for_qid, run_median_for_qid
from metaculus_bot.ablation.run_stacker import ABLATION_MIN_FORECASTERS, ARM_MEAN, ARM_MEDIAN
from metaculus_bot.constants import BINARY_PROB_MAX, BINARY_PROB_MIN
from tests.conftest import make_mock_binary_question, make_mock_mc_question, make_mock_numeric_question

# Type alias for the runner functions
RunnerFn = Callable[..., Coroutine[Any, Any, dict[str, Any]]]


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

# Parametrize IDs for readability in pytest output
_ARMS = pytest.mark.parametrize(
    ("method_label", "runner_fn", "arm_constant"),
    [
        ("mean", run_mean_for_qid, ARM_MEAN),
        ("median", run_median_for_qid, ARM_MEDIAN),
    ],
    ids=["mean", "median"],
)


# ===========================================================================
# Binary — parametrized over mean/median
# ===========================================================================


@_ARMS
class TestBinary:
    def test_aggregation_value(
        self, cache: AblationCache, method_label: str, runner_fn: RunnerFn, arm_constant: str
    ) -> None:
        question = make_mock_binary_question(qid=101)
        forecaster_payloads = {
            model_slug_to_filename(f"openrouter/test/m{i}"): _binary_payload(f"openrouter/test/m{i}", value)
            for i, value in enumerate([0.1, 0.3, 0.5, 0.7])
        }
        payload = asyncio.run(
            runner_fn(qid=101, question=question, forecaster_payloads=forecaster_payloads, cache=cache)
        )
        assert payload["success"] is True
        assert payload["arm"] == method_label
        # Both mean and median of [0.1, 0.3, 0.5, 0.7] = 0.4
        assert payload["stacker_prediction"]["type"] == "binary"
        assert math.isclose(payload["stacker_prediction"]["prob"], 0.4, abs_tol=1e-6)
        assert payload["n_forecasters_used"] == 4
        assert payload["stacker_model_used"] == "simple_aggregation"
        assert payload["tools_enabled_at_runtime"] is False
        assert payload["cross_model_aggregation"] in (None, "")
        assert payload["errors"] == []

    def test_clamps_low_predictions(
        self, cache: AblationCache, method_label: str, runner_fn: RunnerFn, arm_constant: str
    ) -> None:
        question = make_mock_binary_question(qid=102)
        forecaster_payloads = {
            model_slug_to_filename(f"openrouter/test/m{i}"): _binary_payload(f"openrouter/test/m{i}", value)
            for i, value in enumerate([0.001, 0.001, 0.001])
        }
        payload = asyncio.run(
            runner_fn(qid=102, question=question, forecaster_payloads=forecaster_payloads, cache=cache)
        )
        assert math.isclose(payload["stacker_prediction"]["prob"], BINARY_PROB_MIN, abs_tol=1e-9)

    def test_clamps_high_predictions(
        self, cache: AblationCache, method_label: str, runner_fn: RunnerFn, arm_constant: str
    ) -> None:
        question = make_mock_binary_question(qid=103)
        forecaster_payloads = {
            model_slug_to_filename(f"openrouter/test/m{i}"): _binary_payload(f"openrouter/test/m{i}", value)
            for i, value in enumerate([0.999, 0.999, 0.999])
        }
        payload = asyncio.run(
            runner_fn(qid=103, question=question, forecaster_payloads=forecaster_payloads, cache=cache)
        )
        assert math.isclose(payload["stacker_prediction"]["prob"], BINARY_PROB_MAX, abs_tol=1e-9)


# ===========================================================================
# Mean-only: divergence from median on skewed inputs
# ===========================================================================


class TestBinaryMeanSpecific:
    def test_mean_differs_from_median_on_skewed(self, cache: AblationCache) -> None:
        """Mean and median diverge on asymmetric inputs."""
        question = make_mock_binary_question(qid=104)
        forecaster_payloads = {
            model_slug_to_filename(f"openrouter/test/m{i}"): _binary_payload(f"openrouter/test/m{i}", value)
            for i, value in enumerate([0.1, 0.1, 0.9])
        }
        payload = asyncio.run(
            run_mean_for_qid(qid=104, question=question, forecaster_payloads=forecaster_payloads, cache=cache)
        )
        # Mean of [0.1, 0.1, 0.9] = 0.3667; median would be 0.1
        expected_mean = (0.1 + 0.1 + 0.9) / 3.0
        assert math.isclose(payload["stacker_prediction"]["prob"], expected_mean, abs_tol=1e-4)


# ===========================================================================
# Multiple choice — parametrized
# ===========================================================================


@_ARMS
class TestMultipleChoice:
    def test_option_wise_aggregation_then_renormalize(
        self, cache: AblationCache, method_label: str, runner_fn: RunnerFn, arm_constant: str
    ) -> None:
        question = make_mock_mc_question(qid=201)
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
        payload = asyncio.run(
            runner_fn(qid=201, question=question, forecaster_payloads=forecaster_payloads, cache=cache)
        )
        assert payload["success"] is True
        assert payload["stacker_prediction"]["type"] == "multiple_choice"
        options = payload["stacker_prediction"]["options"]
        assert [o["option_name"] for o in options] == ["Red", "Blue", "Green"]
        total = sum(o["probability"] for o in options)
        assert math.isclose(total, 1.0, abs_tol=1e-6)
        red_prob = next(o["probability"] for o in options if o["option_name"] == "Red")
        green_prob = next(o["probability"] for o in options if o["option_name"] == "Green")
        assert red_prob > green_prob


# ===========================================================================
# Numeric — parametrized
# ===========================================================================


@_ARMS
class TestNumeric:
    def test_numeric_aggregates_via_aggregate_numeric(
        self, cache: AblationCache, method_label: str, runner_fn: RunnerFn, arm_constant: str
    ) -> None:
        question = make_mock_numeric_question(id_of_question=301, with_open_resolve_times=True)
        forecaster_payloads = {
            model_slug_to_filename("openrouter/test/m1"): _numeric_payload("openrouter/test/m1", median=40.0),
            model_slug_to_filename("openrouter/test/m2"): _numeric_payload("openrouter/test/m2", median=50.0),
            model_slug_to_filename("openrouter/test/m3"): _numeric_payload("openrouter/test/m3", median=60.0),
        }
        payload = asyncio.run(
            runner_fn(qid=301, question=question, forecaster_payloads=forecaster_payloads, cache=cache)
        )
        assert payload["success"] is True
        sp = payload["stacker_prediction"]
        assert sp["type"] == "numeric"
        assert "cdf_probabilities" in sp
        cdf = sp["cdf_probabilities"]
        assert len(cdf) == 201
        for i in range(1, len(cdf)):
            assert cdf[i] >= cdf[i - 1] - 1e-9
        assert cdf[0] == 0.0
        assert cdf[-1] == 1.0


# ===========================================================================
# Min-forecasters guard — parametrized
# ===========================================================================


@_ARMS
class TestMinForecastersGuard:
    def test_below_threshold_returns_failure_payload(
        self, cache: AblationCache, method_label: str, runner_fn: RunnerFn, arm_constant: str
    ) -> None:
        question = make_mock_binary_question(qid=501)
        forecaster_payloads = {
            "only_one": _binary_payload("openrouter/test/m1", 0.5),
        }
        payload = asyncio.run(
            runner_fn(qid=501, question=question, forecaster_payloads=forecaster_payloads, cache=cache)
        )
        assert payload["success"] is False
        assert payload["arm"] == method_label
        assert payload["n_forecasters_used"] == 1
        assert ABLATION_MIN_FORECASTERS == 2


# ===========================================================================
# Output schema parity — parametrized
# ===========================================================================


@_ARMS
class TestSchemaParity:
    def test_success_payload_has_all_required_keys(
        self, cache: AblationCache, method_label: str, runner_fn: RunnerFn, arm_constant: str
    ) -> None:
        question = make_mock_binary_question(qid=601)
        forecaster_payloads = {
            model_slug_to_filename(f"openrouter/test/m{i}"): _binary_payload(f"openrouter/test/m{i}", 0.5)
            for i in range(3)
        }
        payload = asyncio.run(
            runner_fn(qid=601, question=question, forecaster_payloads=forecaster_payloads, cache=cache)
        )
        missing = _REQUIRED_PAYLOAD_KEYS - set(payload.keys())
        assert not missing, f"ARM_{method_label.upper()} payload missing keys: {missing}"

    def test_arm_constant_value(self, method_label: str, runner_fn: RunnerFn, arm_constant: str) -> None:
        assert arm_constant == method_label


# ===========================================================================
# Cache round-trip — parametrized
# ===========================================================================


@_ARMS
class TestCacheRoundTrip:
    def test_writes_arm_json_to_cache(
        self, cache: AblationCache, method_label: str, runner_fn: RunnerFn, arm_constant: str
    ) -> None:
        question = make_mock_binary_question(qid=701)
        forecaster_payloads = {
            model_slug_to_filename(f"openrouter/test/m{i}"): _binary_payload(f"openrouter/test/m{i}", 0.5)
            for i in range(3)
        }
        asyncio.run(runner_fn(qid=701, question=question, forecaster_payloads=forecaster_payloads, cache=cache))
        cached = cache.read_stacker_output(qid=701, arm=arm_constant)
        assert cached is not None
        assert cached["arm"] == method_label
        assert cached["success"] is True

    def test_cache_hit_skips_recomputation(
        self, cache: AblationCache, method_label: str, runner_fn: RunnerFn, arm_constant: str
    ) -> None:
        question = make_mock_binary_question(qid=702)
        forecaster_payloads = {
            model_slug_to_filename(f"openrouter/test/m{i}"): _binary_payload(f"openrouter/test/m{i}", 0.5)
            for i in range(3)
        }
        first = asyncio.run(runner_fn(qid=702, question=question, forecaster_payloads=forecaster_payloads, cache=cache))
        second = asyncio.run(
            runner_fn(qid=702, question=question, forecaster_payloads=forecaster_payloads, cache=cache)
        )
        assert first["ran_at"] == second["ran_at"]


# ===========================================================================
# Median-only: surviving-filter parity with ARM_STACK / ARM_STACK_AUG
# ===========================================================================


class TestSurvivingFilter:
    def test_drops_none_prediction(self, cache: AblationCache) -> None:
        question = make_mock_binary_question(qid=401)
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
        payload = asyncio.run(
            run_median_for_qid(qid=401, question=question, forecaster_payloads=forecaster_payloads, cache=cache)
        )
        assert payload["success"] is True
        assert payload["n_forecasters_used"] == 2
        assert math.isclose(payload["stacker_prediction"]["prob"], 0.5, abs_tol=1e-6)

    def test_drops_errors_set(self, cache: AblationCache) -> None:
        question = make_mock_binary_question(qid=402)
        forecaster_payloads = {
            "good_a": _binary_payload("openrouter/test/m1", 0.4),
            "good_b": _binary_payload("openrouter/test/m2", 0.6),
            "bad_errors": {
                **_binary_payload("openrouter/test/m3", 0.99),
                "errors": ["timeout"],
            },
        }
        payload = asyncio.run(
            run_median_for_qid(qid=402, question=question, forecaster_payloads=forecaster_payloads, cache=cache)
        )
        assert payload["success"] is True
        assert payload["n_forecasters_used"] == 2

    def test_drops_nan_prediction(self, cache: AblationCache) -> None:
        question = make_mock_binary_question(qid=403)
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
        payload = asyncio.run(
            run_median_for_qid(qid=403, question=question, forecaster_payloads=forecaster_payloads, cache=cache)
        )
        assert payload["success"] is True
        assert payload["n_forecasters_used"] == 2


# ===========================================================================
# Median-only: failure payload schema + zero-surviving edge case
# ===========================================================================


class TestMedianFailurePayload:
    def test_failure_payload_has_required_keys(self, cache: AblationCache) -> None:
        """Even insufficient-forecaster failure path must include the schema keys
        the cli/spend code subscripts directly."""
        question = make_mock_binary_question(qid=602)
        forecaster_payloads: dict[str, dict] = {}
        payload = asyncio.run(
            run_median_for_qid(qid=602, question=question, forecaster_payloads=forecaster_payloads, cache=cache)
        )
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

    def test_zero_surviving_after_filter_returns_failure(self, cache: AblationCache) -> None:
        question = make_mock_binary_question(qid=502)
        forecaster_payloads = {
            "bad_a": {
                "prediction_value": None,
                "reasoning": "fail",
                "errors": ["x"],
                "model": "openrouter/test/m1",
            },
        }
        payload = asyncio.run(
            run_median_for_qid(qid=502, question=question, forecaster_payloads=forecaster_payloads, cache=cache)
        )
        assert payload["success"] is False
        assert payload["n_forecasters_used"] == 0
