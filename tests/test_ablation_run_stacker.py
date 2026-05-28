"""Tests for the per-arm stacker runner used in the probabilistic-tools ablation benchmark.

Treatment in our A/B test is *only* visible to the stacker. Forecasters run once and never
see tool output. Arm A (tools off) and arm B (tools on) differ only by the
``PROBABILISTIC_TOOLS_ENABLED`` env-var state at the moment we call:

* ``tool_runner.run_tools_for_forecaster`` (per-rationale "Computed quantities" markdown)
* ``tool_runner.build_cross_model_aggregation`` (single deterministic-math block)

Both runners are env-gated internally: when the flag is unset they return ``""``. So the
runner toggles the env-var in-process per arm via the ``probabilistic_tools_enabled``
context manager, and otherwise calls the same code path on each arm.

These tests heavily mock the LLM-invoking primitives (``stacking.run_stacking_*``,
``tool_runner.*``) so they're fast and deterministic. The integration with real LLMs is
out of scope here — we're verifying the per-arm orchestration contract.

Note on AsyncMock side_effect functions: ``AsyncMock(side_effect=fn)`` calls ``fn`` with
the same args as the mock invocation, then awaits its return value. We use *sync* helper
functions here (not ``async def``) to avoid flake8-async ASYNC124 warnings — the helpers
have no actual ``await`` calls so making them async would just be noise.
"""

from __future__ import annotations

import asyncio
import math
import os
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from forecasting_tools import (
    BinaryQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
    PredictedOptionList,
)
from forecasting_tools.data_models.multiple_choice_report import PredictedOption
from forecasting_tools.data_models.numeric_report import Percentile

from metaculus_bot.ablation.cache import AblationCache, model_slug_to_filename
from metaculus_bot.ablation.run_stacker import (
    ARM_STACK,
    ARM_STACK_AUG,
    probabilistic_tools_enabled,
    run_stacker_batch,
    run_stacker_for_arm,
)

FEATURE_FLAG = "PROBABILISTIC_TOOLS_ENABLED"

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
    q.options = ["Red", "Blue"]
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


def _binary_payload(model: str = "openrouter/test/m1", value: float = 0.6) -> dict:
    return {
        "prediction_value": {"type": "binary", "prob": value},
        "reasoning": f"Model: {model}\n\nrationale text from {model}",
        "errors": [],
        "model": model,
    }


def _numeric_payload(model: str = "openrouter/test/m1", median: float = 50.0) -> dict:
    """Build a numeric forecaster payload in the post-Bucket-1 full-CDF schema.

    Schema is what ``serialize_prediction_value`` emits for a real
    ``NumericDistribution``: declared_percentiles + the constraint-enforced
    201-point CDF + bounds + zero_point + cdf_size. Tests assemble payloads
    directly here instead of running the serializer, so we synthesize a
    monotone linear CDF that spans the bounds.
    """
    declared = [
        {"percentile": 0.025, "value": median - 30},
        {"percentile": 0.05, "value": median - 25},
        {"percentile": 0.10, "value": median - 20},
        {"percentile": 0.20, "value": median - 12},
        {"percentile": 0.40, "value": median - 5},
        {"percentile": 0.50, "value": median},
        {"percentile": 0.60, "value": median + 5},
        {"percentile": 0.80, "value": median + 12},
        {"percentile": 0.90, "value": median + 20},
        {"percentile": 0.95, "value": median + 25},
        {"percentile": 0.975, "value": median + 30},
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


def _mc_payload(model: str = "openrouter/test/m1") -> dict:
    return {
        "prediction_value": {
            "type": "multiple_choice",
            "options": [
                {"option_name": "Red", "probability": 0.6},
                {"option_name": "Blue", "probability": 0.4},
            ],
        },
        "reasoning": f"Model: {model}\n\nrationale text from {model}",
        "errors": [],
        "model": model,
    }


def _three_binary_forecasters() -> dict[str, dict]:
    """Pre-built dict of three valid binary forecaster payloads."""
    return {
        model_slug_to_filename("openrouter/test/m1"): _binary_payload("openrouter/test/m1", 0.6),
        model_slug_to_filename("openrouter/test/m2"): _binary_payload("openrouter/test/m2", 0.5),
        model_slug_to_filename("openrouter/test/m3"): _binary_payload("openrouter/test/m3", 0.4),
    }


def _three_numeric_forecasters() -> dict[str, dict]:
    return {
        model_slug_to_filename("openrouter/test/m1"): _numeric_payload("openrouter/test/m1", 50.0),
        model_slug_to_filename("openrouter/test/m2"): _numeric_payload("openrouter/test/m2", 55.0),
        model_slug_to_filename("openrouter/test/m3"): _numeric_payload("openrouter/test/m3", 60.0),
    }


def _three_mc_forecasters() -> dict[str, dict]:
    return {
        model_slug_to_filename("openrouter/test/m1"): _mc_payload("openrouter/test/m1"),
        model_slug_to_filename("openrouter/test/m2"): _mc_payload("openrouter/test/m2"),
        model_slug_to_filename("openrouter/test/m3"): _mc_payload("openrouter/test/m3"),
    }


def _capture_base_texts(args: tuple[Any, ...], kwargs: dict[str, Any]) -> list[str]:
    """Pull the base_texts argument from a captured stacker call (positional or kw)."""
    if len(args) > 4:
        base_texts = args[4]
    else:
        base_texts = kwargs.get("base_texts", [])
    assert base_texts is not None
    return list(base_texts)


# ---------------------------------------------------------------------------
# Common fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cache(tmp_path: Path) -> AblationCache:
    return AblationCache(tmp_path / "abl")


@pytest.fixture
def stacker_llm() -> MagicMock:
    return MagicMock(name="stacker_llm")


@pytest.fixture
def fallback_stacker_llm() -> MagicMock:
    return MagicMock(name="fallback_stacker_llm")


@pytest.fixture
def parser_llm() -> MagicMock:
    return MagicMock(name="parser_llm")


@pytest.fixture(autouse=True)
def _ensure_flag_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every test start with the flag explicitly unset so no leakage from earlier tests."""
    monkeypatch.delenv(FEATURE_FLAG, raising=False)


# ===========================================================================
# probabilistic_tools_enabled context manager
# ===========================================================================


class TestProbabilisticToolsEnabled:
    def test_true_sets_env_var_to_one_during_block(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(FEATURE_FLAG, raising=False)
        assert FEATURE_FLAG not in os.environ
        with probabilistic_tools_enabled(True):
            assert os.environ[FEATURE_FLAG] == "1"
        assert FEATURE_FLAG not in os.environ

    def test_false_unsets_env_var_during_block(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(FEATURE_FLAG, "1")
        with probabilistic_tools_enabled(False):
            assert FEATURE_FLAG not in os.environ
        # Restored
        assert os.environ[FEATURE_FLAG] == "1"

    def test_env_var_restored_to_previous_value_on_exit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(FEATURE_FLAG, "foo")
        with probabilistic_tools_enabled(True):
            assert os.environ[FEATURE_FLAG] == "1"
        assert os.environ[FEATURE_FLAG] == "foo"

    def test_env_var_restored_on_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(FEATURE_FLAG, raising=False)
        with pytest.raises(RuntimeError, match="boom"):
            with probabilistic_tools_enabled(True):
                assert os.environ[FEATURE_FLAG] == "1"
                raise RuntimeError("boom")
        assert FEATURE_FLAG not in os.environ

    def test_env_var_restored_on_exception_when_previously_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(FEATURE_FLAG, "preset")
        with pytest.raises(ValueError):
            with probabilistic_tools_enabled(False):
                assert FEATURE_FLAG not in os.environ
                raise ValueError("boom")
        assert os.environ[FEATURE_FLAG] == "preset"


# ===========================================================================
# Helper: run an async coroutine in a sync test
# ===========================================================================


def _run(coro: Any) -> Any:
    return asyncio.get_event_loop().run_until_complete(coro)


# ===========================================================================
# run_stacker_for_arm — env-var visibility / arm semantics
# ===========================================================================


class TestArmEnvVarSemantics:
    def test_arm_a_runs_with_flag_unset(
        self,
        cache: AblationCache,
        stacker_llm: MagicMock,
        fallback_stacker_llm: MagicMock,
        parser_llm: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Arm A: tool_runner.run_tools_for_forecaster sees env-var unset at call time."""
        monkeypatch.delenv(FEATURE_FLAG, raising=False)
        seen_flag_states: list[str | None] = []

        def _record_flag_state(*_args: Any, **_kwargs: Any) -> str:
            seen_flag_states.append(os.environ.get(FEATURE_FLAG))
            return ""

        def _fake_stacker(*_args: Any, **_kwargs: Any) -> tuple[float, str]:
            return 0.5, "meta"

        with (
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.run_tools_for_forecaster",
                side_effect=_record_flag_state,
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.build_cross_model_aggregation",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.stacking.run_stacking_binary",
                new=AsyncMock(side_effect=_fake_stacker),
            ),
        ):
            payload = _run(
                run_stacker_for_arm(
                    question=_make_binary_q(),
                    research_blob="research",
                    forecaster_payloads=_three_binary_forecasters(),
                    arm=ARM_STACK,
                    cache=cache,
                    stacker_llm=stacker_llm,
                    fallback_stacker_llm=fallback_stacker_llm,
                    parser_llm=parser_llm,
                )
            )
        assert payload["success"] is True
        assert payload["arm"] == ARM_STACK
        assert payload["tools_enabled_at_runtime"] is False
        # The flag was unset for every per-forecaster tool-runner call
        assert seen_flag_states  # at least one call recorded
        for state in seen_flag_states:
            assert state is None, f"Arm A should not have FEATURE_FLAG set; got {state!r}"

    def test_arm_b_runs_with_flag_set_to_one(
        self,
        cache: AblationCache,
        stacker_llm: MagicMock,
        fallback_stacker_llm: MagicMock,
        parser_llm: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv(FEATURE_FLAG, raising=False)
        seen_flag_states: list[str | None] = []

        def _record_flag_state(*_args: Any, **_kwargs: Any) -> str:
            seen_flag_states.append(os.environ.get(FEATURE_FLAG))
            return ""

        def _fake_stacker(*_args: Any, **_kwargs: Any) -> tuple[float, str]:
            return 0.5, "meta"

        with (
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.run_tools_for_forecaster",
                side_effect=_record_flag_state,
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.build_cross_model_aggregation",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.stacking.run_stacking_binary",
                new=AsyncMock(side_effect=_fake_stacker),
            ),
        ):
            payload = _run(
                run_stacker_for_arm(
                    question=_make_binary_q(),
                    research_blob="research",
                    forecaster_payloads=_three_binary_forecasters(),
                    arm=ARM_STACK_AUG,
                    cache=cache,
                    stacker_llm=stacker_llm,
                    fallback_stacker_llm=fallback_stacker_llm,
                    parser_llm=parser_llm,
                )
            )
        assert payload["success"] is True
        assert payload["arm"] == ARM_STACK_AUG
        assert payload["tools_enabled_at_runtime"] is True
        assert seen_flag_states
        for state in seen_flag_states:
            assert state == "1", f"Arm B should have FEATURE_FLAG=1; got {state!r}"


# ===========================================================================
# run_stacker_for_arm — aggregated_tool_output passing
# ===========================================================================


class TestAggregatedToolOutputPassing:
    def test_arm_b_passes_aggregated_tool_output_to_stacker(
        self,
        cache: AblationCache,
        stacker_llm: MagicMock,
        fallback_stacker_llm: MagicMock,
        parser_llm: MagicMock,
    ) -> None:
        captured_kwargs: dict[str, Any] = {}

        def _fake_stacker(*_args: Any, **kwargs: Any) -> tuple[float, str]:
            captured_kwargs.update(kwargs)
            return 0.5, "meta"

        with (
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.run_tools_for_forecaster",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.build_cross_model_aggregation",
                return_value="FAKE AGG",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.stacking.run_stacking_binary",
                new=AsyncMock(side_effect=_fake_stacker),
            ),
        ):
            payload = _run(
                run_stacker_for_arm(
                    question=_make_binary_q(),
                    research_blob="R",
                    forecaster_payloads=_three_binary_forecasters(),
                    arm=ARM_STACK_AUG,
                    cache=cache,
                    stacker_llm=stacker_llm,
                    fallback_stacker_llm=fallback_stacker_llm,
                    parser_llm=parser_llm,
                )
            )
        assert captured_kwargs.get("aggregated_tool_output") == "FAKE AGG"
        assert payload["cross_model_aggregation"] == "FAKE AGG"

    def test_arm_a_passes_none_aggregated_tool_output_to_stacker(
        self,
        cache: AblationCache,
        stacker_llm: MagicMock,
        fallback_stacker_llm: MagicMock,
        parser_llm: MagicMock,
    ) -> None:
        captured_kwargs: dict[str, Any] = {}

        def _fake_stacker(*_args: Any, **kwargs: Any) -> tuple[float, str]:
            captured_kwargs.update(kwargs)
            return 0.5, "meta"

        with (
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.run_tools_for_forecaster",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.build_cross_model_aggregation",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.stacking.run_stacking_binary",
                new=AsyncMock(side_effect=_fake_stacker),
            ),
        ):
            payload = _run(
                run_stacker_for_arm(
                    question=_make_binary_q(),
                    research_blob="R",
                    forecaster_payloads=_three_binary_forecasters(),
                    arm=ARM_STACK,
                    cache=cache,
                    stacker_llm=stacker_llm,
                    fallback_stacker_llm=fallback_stacker_llm,
                    parser_llm=parser_llm,
                )
            )
        # Match production main.py:797-803 — `... or None` truthy check.
        assert captured_kwargs.get("aggregated_tool_output") is None
        assert payload["cross_model_aggregation"] == ""


# ===========================================================================
# run_stacker_for_arm — per-forecaster Computed Quantities augmentation
# ===========================================================================


class TestPerForecasterComputedQuantities:
    def test_arm_b_appends_computed_quantities_to_each_rationale(
        self,
        cache: AblationCache,
        stacker_llm: MagicMock,
        fallback_stacker_llm: MagicMock,
        parser_llm: MagicMock,
    ) -> None:
        captured_base_texts: list[list[str]] = []

        def _fake_stacker(*args: Any, **kwargs: Any) -> tuple[float, str]:
            captured_base_texts.append(_capture_base_texts(args, kwargs))
            return 0.5, "meta"

        with (
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.run_tools_for_forecaster",
                return_value="TOOLMD",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.build_cross_model_aggregation",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.stacking.run_stacking_binary",
                new=AsyncMock(side_effect=_fake_stacker),
            ),
        ):
            _run(
                run_stacker_for_arm(
                    question=_make_binary_q(),
                    research_blob="R",
                    forecaster_payloads=_three_binary_forecasters(),
                    arm=ARM_STACK_AUG,
                    cache=cache,
                    stacker_llm=stacker_llm,
                    fallback_stacker_llm=fallback_stacker_llm,
                    parser_llm=parser_llm,
                )
            )
        assert captured_base_texts
        for base_text in captured_base_texts[0]:
            assert "## Computed quantities\nTOOLMD" in base_text

    def test_no_augmentation_when_runner_returns_empty(
        self,
        cache: AblationCache,
        stacker_llm: MagicMock,
        fallback_stacker_llm: MagicMock,
        parser_llm: MagicMock,
    ) -> None:
        captured_base_texts: list[list[str]] = []

        def _fake_stacker(*args: Any, **kwargs: Any) -> tuple[float, str]:
            captured_base_texts.append(_capture_base_texts(args, kwargs))
            return 0.5, "meta"

        with (
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.run_tools_for_forecaster",
                return_value="",  # nothing to append
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.build_cross_model_aggregation",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.stacking.run_stacking_binary",
                new=AsyncMock(side_effect=_fake_stacker),
            ),
        ):
            _run(
                run_stacker_for_arm(
                    question=_make_binary_q(),
                    research_blob="R",
                    forecaster_payloads=_three_binary_forecasters(),
                    arm=ARM_STACK_AUG,  # even in arm B
                    cache=cache,
                    stacker_llm=stacker_llm,
                    fallback_stacker_llm=fallback_stacker_llm,
                    parser_llm=parser_llm,
                )
            )
        assert captured_base_texts
        for base_text in captured_base_texts[0]:
            assert "Computed quantities" not in base_text

    def test_strips_model_tag_before_passing_to_stacker(
        self,
        cache: AblationCache,
        stacker_llm: MagicMock,
        fallback_stacker_llm: MagicMock,
        parser_llm: MagicMock,
    ) -> None:
        """Per spec: use stacking.strip_model_tag to remove the "Model: ...\n\n" prefix."""
        captured_base_texts: list[list[str]] = []

        def _fake_stacker(*args: Any, **kwargs: Any) -> tuple[float, str]:
            captured_base_texts.append(_capture_base_texts(args, kwargs))
            return 0.5, "meta"

        with (
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.run_tools_for_forecaster",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.build_cross_model_aggregation",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.stacking.run_stacking_binary",
                new=AsyncMock(side_effect=_fake_stacker),
            ),
        ):
            _run(
                run_stacker_for_arm(
                    question=_make_binary_q(),
                    research_blob="R",
                    forecaster_payloads=_three_binary_forecasters(),
                    arm=ARM_STACK,
                    cache=cache,
                    stacker_llm=stacker_llm,
                    fallback_stacker_llm=fallback_stacker_llm,
                    parser_llm=parser_llm,
                )
            )
        # rationale starts with "Model: openrouter/test/m1\n\n"; after strip, must not start with "Model:"
        assert captured_base_texts
        for base_text in captured_base_texts[0]:
            assert not base_text.startswith("Model: "), f"Expected stripped, got: {base_text[:60]!r}"

    def test_per_forecaster_computed_quantities_recorded_in_payload(
        self,
        cache: AblationCache,
        stacker_llm: MagicMock,
        fallback_stacker_llm: MagicMock,
        parser_llm: MagicMock,
    ) -> None:
        def _fake_stacker(*_args: Any, **_kwargs: Any) -> tuple[float, str]:
            return 0.5, "meta"

        with (
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.run_tools_for_forecaster",
                return_value="TOOLMD",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.build_cross_model_aggregation",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.stacking.run_stacking_binary",
                new=AsyncMock(side_effect=_fake_stacker),
            ),
        ):
            payload = _run(
                run_stacker_for_arm(
                    question=_make_binary_q(),
                    research_blob="R",
                    forecaster_payloads=_three_binary_forecasters(),
                    arm=ARM_STACK_AUG,
                    cache=cache,
                    stacker_llm=stacker_llm,
                    fallback_stacker_llm=fallback_stacker_llm,
                    parser_llm=parser_llm,
                )
            )
        # 3 forecasters, all augmented
        assert len(payload["computed_quantities"]) == 3
        for v in payload["computed_quantities"].values():
            assert v == "TOOLMD"

    def test_arm_a_payload_has_empty_computed_quantities(
        self,
        cache: AblationCache,
        stacker_llm: MagicMock,
        fallback_stacker_llm: MagicMock,
        parser_llm: MagicMock,
    ) -> None:
        """Arm A: tool_runner returns "" because env-var is unset; payload's computed_quantities is empty."""

        def _fake_stacker(*_args: Any, **_kwargs: Any) -> tuple[float, str]:
            return 0.5, "meta"

        with (
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.run_tools_for_forecaster",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.build_cross_model_aggregation",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.stacking.run_stacking_binary",
                new=AsyncMock(side_effect=_fake_stacker),
            ),
        ):
            payload = _run(
                run_stacker_for_arm(
                    question=_make_binary_q(),
                    research_blob="R",
                    forecaster_payloads=_three_binary_forecasters(),
                    arm=ARM_STACK,
                    cache=cache,
                    stacker_llm=stacker_llm,
                    fallback_stacker_llm=fallback_stacker_llm,
                    parser_llm=parser_llm,
                )
            )
        assert payload["computed_quantities"] == {}


# ===========================================================================
# Cache short-circuit / force semantics
# ===========================================================================


class TestCacheBehavior:
    def test_cache_hit_short_circuits_no_llm_or_tool_calls(
        self,
        cache: AblationCache,
        stacker_llm: MagicMock,
        fallback_stacker_llm: MagicMock,
        parser_llm: MagicMock,
    ) -> None:
        # Pre-write a stacker output for (qid=1, arm=A)
        cached_payload = {
            "success": True,
            "arm": ARM_STACK,
            "stacker_prediction": 0.42,
            "stacker_meta_reasoning": "cached",
            "computed_quantities": {},
            "cross_model_aggregation": "",
            "stacker_model_used": "primary",
            "n_forecasters_used": 3,
            "ran_at": "2026-05-13T00:00:00",
            "tools_enabled_at_runtime": False,
            "errors": [],
        }
        cache.write_stacker_output(qid=1, arm=ARM_STACK, payload=cached_payload)

        runner_mock = MagicMock(return_value="")
        agg_mock = MagicMock(return_value="")
        stacker_mock = AsyncMock()

        with (
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.run_tools_for_forecaster",
                new=runner_mock,
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.build_cross_model_aggregation",
                new=agg_mock,
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.stacking.run_stacking_binary",
                new=stacker_mock,
            ),
        ):
            payload = _run(
                run_stacker_for_arm(
                    question=_make_binary_q(qid=1),
                    research_blob="R",
                    forecaster_payloads=_three_binary_forecasters(),
                    arm=ARM_STACK,
                    cache=cache,
                    stacker_llm=stacker_llm,
                    fallback_stacker_llm=fallback_stacker_llm,
                    parser_llm=parser_llm,
                )
            )
        assert payload == {**cached_payload, "cache_schema_version": 1}
        runner_mock.assert_not_called()
        agg_mock.assert_not_called()
        stacker_mock.assert_not_called()

    def test_force_true_bypasses_cache(
        self,
        cache: AblationCache,
        stacker_llm: MagicMock,
        fallback_stacker_llm: MagicMock,
        parser_llm: MagicMock,
    ) -> None:
        cached_payload = {
            "success": True,
            "arm": ARM_STACK,
            "stacker_prediction": 0.42,
            "stacker_meta_reasoning": "cached",
            "computed_quantities": {},
            "cross_model_aggregation": "",
            "stacker_model_used": "primary",
            "n_forecasters_used": 3,
            "ran_at": "2026-05-13T00:00:00",
            "tools_enabled_at_runtime": False,
            "errors": [],
        }
        cache.write_stacker_output(qid=1, arm=ARM_STACK, payload=cached_payload)

        def _fake_stacker(*_args: Any, **_kwargs: Any) -> tuple[float, str]:
            return 0.99, "fresh meta"

        with (
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.run_tools_for_forecaster",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.build_cross_model_aggregation",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.stacking.run_stacking_binary",
                new=AsyncMock(side_effect=_fake_stacker),
            ),
        ):
            payload = _run(
                run_stacker_for_arm(
                    question=_make_binary_q(qid=1),
                    research_blob="R",
                    forecaster_payloads=_three_binary_forecasters(),
                    arm=ARM_STACK,
                    cache=cache,
                    stacker_llm=stacker_llm,
                    fallback_stacker_llm=fallback_stacker_llm,
                    parser_llm=parser_llm,
                    force=True,
                )
            )
        assert payload["stacker_prediction"] == {"type": "binary", "prob": 0.99}
        assert payload["stacker_meta_reasoning"] == "fresh meta"


# ===========================================================================
# Insufficient forecasters
# ===========================================================================


class TestInsufficientForecasters:
    def test_one_valid_forecaster_caches_error_payload(
        self,
        cache: AblationCache,
        stacker_llm: MagicMock,
        fallback_stacker_llm: MagicMock,
        parser_llm: MagicMock,
    ) -> None:
        # Only one valid; one with prediction_value=None gets filtered out.
        forecasters = {
            model_slug_to_filename("openrouter/test/m1"): _binary_payload("openrouter/test/m1", 0.6),
            model_slug_to_filename("openrouter/test/m2"): {
                **_binary_payload("openrouter/test/m2"),
                "prediction_value": None,
                "errors": ["model failed"],
            },
        }

        runner_mock = MagicMock(return_value="")
        stacker_mock = AsyncMock()

        with (
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.run_tools_for_forecaster",
                new=runner_mock,
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.stacking.run_stacking_binary",
                new=stacker_mock,
            ),
        ):
            payload = _run(
                run_stacker_for_arm(
                    question=_make_binary_q(qid=1),
                    research_blob="R",
                    forecaster_payloads=forecasters,
                    arm=ARM_STACK,
                    cache=cache,
                    stacker_llm=stacker_llm,
                    fallback_stacker_llm=fallback_stacker_llm,
                    parser_llm=parser_llm,
                )
            )
        assert payload["success"] is False
        assert payload["reason"] == "insufficient_forecasters"
        # Cached
        on_disk = cache.read_stacker_output(qid=1, arm=ARM_STACK)
        assert on_disk is not None
        assert on_disk["success"] is False
        # Stacker never invoked
        stacker_mock.assert_not_called()

    def test_filters_out_none_values_and_errors(
        self,
        cache: AblationCache,
        stacker_llm: MagicMock,
        fallback_stacker_llm: MagicMock,
        parser_llm: MagicMock,
    ) -> None:
        """2 valid, 1 None — proceeds with 2."""
        forecasters = {
            model_slug_to_filename("openrouter/test/m1"): _binary_payload("openrouter/test/m1", 0.6),
            model_slug_to_filename("openrouter/test/m2"): _binary_payload("openrouter/test/m2", 0.4),
            model_slug_to_filename("openrouter/test/m3"): {
                **_binary_payload("openrouter/test/m3"),
                "prediction_value": None,
                "errors": ["fail"],
            },
        }

        def _fake_stacker(*_args: Any, **_kwargs: Any) -> tuple[float, str]:
            return 0.5, "meta"

        with (
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.run_tools_for_forecaster",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.build_cross_model_aggregation",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.stacking.run_stacking_binary",
                new=AsyncMock(side_effect=_fake_stacker),
            ),
        ):
            payload = _run(
                run_stacker_for_arm(
                    question=_make_binary_q(qid=1),
                    research_blob="R",
                    forecaster_payloads=forecasters,
                    arm=ARM_STACK,
                    cache=cache,
                    stacker_llm=stacker_llm,
                    fallback_stacker_llm=fallback_stacker_llm,
                    parser_llm=parser_llm,
                )
            )
        assert payload["success"] is True
        assert payload["n_forecasters_used"] == 2


# ===========================================================================
# Primary -> fallback chain
# ===========================================================================


class TestPrimaryFallbackChain:
    def test_primary_failure_falls_back_to_fallback_llm(
        self,
        cache: AblationCache,
        stacker_llm: MagicMock,
        fallback_stacker_llm: MagicMock,
        parser_llm: MagicMock,
    ) -> None:
        call_log: list[str] = []

        def _fake_stacker(*args: Any, **_kwargs: Any) -> tuple[float, str]:
            stacker = args[0]
            if stacker is stacker_llm:
                call_log.append("primary")
                raise RuntimeError("primary boom")
            if stacker is fallback_stacker_llm:
                call_log.append("fallback")
                return 0.7, "fallback meta"
            raise AssertionError(f"unexpected stacker llm: {stacker}")

        with (
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.run_tools_for_forecaster",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.build_cross_model_aggregation",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.stacking.run_stacking_binary",
                new=AsyncMock(side_effect=_fake_stacker),
            ),
        ):
            payload = _run(
                run_stacker_for_arm(
                    question=_make_binary_q(qid=1),
                    research_blob="R",
                    forecaster_payloads=_three_binary_forecasters(),
                    arm=ARM_STACK,
                    cache=cache,
                    stacker_llm=stacker_llm,
                    fallback_stacker_llm=fallback_stacker_llm,
                    parser_llm=parser_llm,
                )
            )
        assert payload["success"] is True
        assert payload["stacker_model_used"] == "fallback"
        assert payload["stacker_prediction"] == {"type": "binary", "prob": 0.7}
        assert call_log == ["primary", "fallback"]
        assert any("primary boom" in e for e in payload["errors"])

    def test_both_stackers_fail_engages_median_fallback(
        self,
        cache: AblationCache,
        stacker_llm: MagicMock,
        fallback_stacker_llm: MagicMock,
        parser_llm: MagicMock,
    ) -> None:
        """Both stackers fail -> M3 tertiary MEDIAN fallback engages.

        Previously this path cached success=False; with M3 the question
        gets a degraded-but-publishable median forecast tagged
        ``stacker_model_used="median_fallback"``.
        """

        def _fake_stacker(*args: Any, **_kwargs: Any) -> tuple[float, str]:
            stacker = args[0]
            if stacker is stacker_llm:
                raise RuntimeError("primary boom")
            raise RuntimeError("fallback boom")

        with (
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.run_tools_for_forecaster",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.build_cross_model_aggregation",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.stacking.run_stacking_binary",
                new=AsyncMock(side_effect=_fake_stacker),
            ),
        ):
            payload = _run(
                run_stacker_for_arm(
                    question=_make_binary_q(qid=99),
                    research_blob="R",
                    forecaster_payloads=_three_binary_forecasters(),
                    arm=ARM_STACK,
                    cache=cache,
                    stacker_llm=stacker_llm,
                    fallback_stacker_llm=fallback_stacker_llm,
                    parser_llm=parser_llm,
                )
            )
        assert payload["success"] is True
        assert payload["stacker_model_used"] == "median_fallback"
        # Both upstream errors recorded so audit can correlate to provider outages
        assert "primary boom" in str(payload["errors"])
        assert "fallback boom" in str(payload["errors"])
        # Cached payload visible for downstream confounder analysis
        on_disk = cache.read_stacker_output(qid=99, arm=ARM_STACK)
        assert on_disk is not None
        assert on_disk["success"] is True
        assert on_disk["stacker_model_used"] == "median_fallback"


# ===========================================================================
# C1 — Soft deadlines on stacker calls
#
# Production wraps each stacker dispatch in
# ``asyncio.wait_for(... , timeout=STACKER_SOFT_DEADLINE)`` (main.py:1243,
# 1271). Without that wrapper, a stuck stacker can hold a question for the
# entire litellm timeout(480) when allowed_tries=1, and once concurrent
# stacker calls share the global window-patch lock, every other question
# waits behind the stalled one. The soft deadline bounds each call and
# lets the primary→fallback chain make progress.
# ===========================================================================


class TestSoftDeadline:
    def test_primary_stacker_timeout_falls_back_to_fallback_llm(
        self,
        cache: AblationCache,
        stacker_llm: MagicMock,
        fallback_stacker_llm: MagicMock,
        parser_llm: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A stalled primary stacker is killed by STACKER_SOFT_DEADLINE.

        Mocks the primary stacker to sleep for >> deadline; the runner must
        timeout, fall back to the fallback LLM, succeed, and record the
        timeout in the payload's ``errors``.
        """
        from metaculus_bot.ablation import run_stacker as run_stacker_module

        monkeypatch.setattr(run_stacker_module, "STACKER_SOFT_DEADLINE", 1)

        async def _slow_or_fast(*args: Any, **_kwargs: Any) -> tuple[float, str]:
            stacker = args[0]
            if stacker is stacker_llm:
                await asyncio.sleep(5)
                return 0.99, "should never reach"
            if stacker is fallback_stacker_llm:
                await asyncio.sleep(0)
                return 0.7, "fallback meta"
            raise AssertionError(f"unexpected stacker llm: {stacker}")

        with (
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.run_tools_for_forecaster",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.build_cross_model_aggregation",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.stacking.run_stacking_binary",
                new=AsyncMock(side_effect=_slow_or_fast),
            ),
        ):
            start = asyncio.get_event_loop().time()
            payload = _run(
                run_stacker_for_arm(
                    question=_make_binary_q(qid=801),
                    research_blob="R",
                    forecaster_payloads=_three_binary_forecasters(),
                    arm=ARM_STACK,
                    cache=cache,
                    stacker_llm=stacker_llm,
                    fallback_stacker_llm=fallback_stacker_llm,
                    parser_llm=parser_llm,
                )
            )
            elapsed = asyncio.get_event_loop().time() - start

        assert elapsed < 3.0, f"runner did not honor soft deadline; elapsed={elapsed:.1f}s"
        assert payload["success"] is True
        assert payload["stacker_model_used"] == "fallback"
        assert any("TimeoutError" in e for e in payload["errors"]), payload["errors"]

    def test_both_stackers_timeout_records_failure(
        self,
        cache: AblationCache,
        stacker_llm: MagicMock,
        fallback_stacker_llm: MagicMock,
        parser_llm: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When BOTH primary and fallback stall past their deadlines, the
        median fallback (M3) takes over — but errors record both timeouts.
        """
        from metaculus_bot.ablation import run_stacker as run_stacker_module

        monkeypatch.setattr(run_stacker_module, "STACKER_SOFT_DEADLINE", 1)
        monkeypatch.setattr(run_stacker_module, "STACKER_FALLBACK_SOFT_DEADLINE", 1)

        async def _stall(*_args: Any, **_kwargs: Any) -> tuple[float, str]:
            await asyncio.sleep(5)
            return 0.99, "never"

        with (
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.run_tools_for_forecaster",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.build_cross_model_aggregation",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.stacking.run_stacking_binary",
                new=AsyncMock(side_effect=_stall),
            ),
        ):
            start = asyncio.get_event_loop().time()
            payload = _run(
                run_stacker_for_arm(
                    question=_make_binary_q(qid=802),
                    research_blob="R",
                    forecaster_payloads=_three_binary_forecasters(),
                    arm=ARM_STACK,
                    cache=cache,
                    stacker_llm=stacker_llm,
                    fallback_stacker_llm=fallback_stacker_llm,
                    parser_llm=parser_llm,
                )
            )
            elapsed = asyncio.get_event_loop().time() - start

        assert elapsed < 5.0, f"runner did not honor soft deadlines; elapsed={elapsed:.1f}s"
        assert any("TimeoutError" in e for e in payload["errors"]), payload["errors"]


# ===========================================================================
# Window patch active during stacker call
# ===========================================================================


class TestWindowPatchActive:
    def test_window_patch_active_during_stacker_invocation(
        self,
        cache: AblationCache,
        stacker_llm: MagicMock,
        fallback_stacker_llm: MagicMock,
        parser_llm: MagicMock,
    ) -> None:
        from metaculus_bot.ablation import window_patch as wp

        observed_active: list[bool] = []

        def _fake_stacker(*_args: Any, **_kwargs: Any) -> tuple[float, str]:
            observed_active.append(wp._window_patch_active)
            return 0.5, "meta"

        with (
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.run_tools_for_forecaster",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.build_cross_model_aggregation",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.stacking.run_stacking_binary",
                new=AsyncMock(side_effect=_fake_stacker),
            ),
        ):
            _run(
                run_stacker_for_arm(
                    question=_make_binary_q(qid=1),
                    research_blob="R",
                    forecaster_payloads=_three_binary_forecasters(),
                    arm=ARM_STACK,
                    cache=cache,
                    stacker_llm=stacker_llm,
                    fallback_stacker_llm=fallback_stacker_llm,
                    parser_llm=parser_llm,
                )
            )
        assert observed_active == [True]
        # And after the call, no longer active
        assert wp._window_patch_active is False


# ===========================================================================
# Question-type dispatch
# ===========================================================================


class TestQuestionTypeDispatch:
    def test_binary_dispatches_to_run_stacking_binary(
        self,
        cache: AblationCache,
        stacker_llm: MagicMock,
        fallback_stacker_llm: MagicMock,
        parser_llm: MagicMock,
    ) -> None:
        binary_called = False
        mc_called = False
        numeric_called = False

        def _fake_binary(*_args: Any, **_kwargs: Any) -> tuple[float, str]:
            nonlocal binary_called
            binary_called = True
            return 0.5, "meta"

        def _fake_mc(*_args: Any, **_kwargs: Any) -> tuple[Any, str]:
            nonlocal mc_called
            mc_called = True
            return PredictedOptionList(predicted_options=[PredictedOption(option_name="Red", probability=1.0)]), "meta"

        def _fake_numeric(*_args: Any, **_kwargs: Any) -> tuple[Any, str]:
            nonlocal numeric_called
            numeric_called = True
            return [Percentile(percentile=0.5, value=42.0)], "meta"

        with (
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.run_tools_for_forecaster",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.build_cross_model_aggregation",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.stacking.run_stacking_binary",
                new=AsyncMock(side_effect=_fake_binary),
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.stacking.run_stacking_mc",
                new=AsyncMock(side_effect=_fake_mc),
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.stacking.run_stacking_numeric",
                new=AsyncMock(side_effect=_fake_numeric),
            ),
        ):
            _run(
                run_stacker_for_arm(
                    question=_make_binary_q(qid=1),
                    research_blob="R",
                    forecaster_payloads=_three_binary_forecasters(),
                    arm=ARM_STACK,
                    cache=cache,
                    stacker_llm=stacker_llm,
                    fallback_stacker_llm=fallback_stacker_llm,
                    parser_llm=parser_llm,
                )
            )
        assert binary_called is True
        assert mc_called is False
        assert numeric_called is False

    def test_mc_dispatches_to_run_stacking_mc(
        self,
        cache: AblationCache,
        stacker_llm: MagicMock,
        fallback_stacker_llm: MagicMock,
        parser_llm: MagicMock,
    ) -> None:
        binary_called = False
        mc_called = False

        def _fake_binary(*_args: Any, **_kwargs: Any) -> tuple[float, str]:
            nonlocal binary_called
            binary_called = True
            return 0.5, "meta"

        def _fake_mc(*_args: Any, **_kwargs: Any) -> tuple[Any, str]:
            nonlocal mc_called
            mc_called = True
            return PredictedOptionList(
                predicted_options=[
                    PredictedOption(option_name="Red", probability=0.6),
                    PredictedOption(option_name="Blue", probability=0.4),
                ]
            ), "meta"

        with (
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.run_tools_for_forecaster",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.build_cross_model_aggregation",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.stacking.run_stacking_binary",
                new=AsyncMock(side_effect=_fake_binary),
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.stacking.run_stacking_mc",
                new=AsyncMock(side_effect=_fake_mc),
            ),
        ):
            payload = _run(
                run_stacker_for_arm(
                    question=_make_mc_q(qid=2),
                    research_blob="R",
                    forecaster_payloads=_three_mc_forecasters(),
                    arm=ARM_STACK,
                    cache=cache,
                    stacker_llm=stacker_llm,
                    fallback_stacker_llm=fallback_stacker_llm,
                    parser_llm=parser_llm,
                )
            )
        assert mc_called is True
        assert binary_called is False
        assert payload["success"] is True

    def test_numeric_dispatches_with_bound_messages(
        self,
        cache: AblationCache,
        stacker_llm: MagicMock,
        fallback_stacker_llm: MagicMock,
        parser_llm: MagicMock,
    ) -> None:
        # Post-Phase-A.1-v4: ``run_stacking_numeric`` returns ``tuple[list[Percentile], str]``
        # (matching production at main.py:436). ``_dispatch_stacker`` then mirrors
        # production main.py:450-465 by piping the percentile list through
        # ``sanitize_percentiles`` + ``detect_unit_mismatch`` + ``build_numeric_distribution``
        # before serialization. The fake here returns the raw list to mimic that
        # production contract; the wrapping is what we're verifying gets exercised.
        _percentiles = [
            Percentile(percentile=0.025, value=20.0),
            Percentile(percentile=0.05, value=25.0),
            Percentile(percentile=0.10, value=30.0),
            Percentile(percentile=0.20, value=38.0),
            Percentile(percentile=0.40, value=45.0),
            Percentile(percentile=0.50, value=50.0),
            Percentile(percentile=0.60, value=55.0),
            Percentile(percentile=0.80, value=62.0),
            Percentile(percentile=0.90, value=70.0),
            Percentile(percentile=0.95, value=75.0),
            Percentile(percentile=0.975, value=80.0),
        ]
        captured_args: list[Any] = []

        def _fake_numeric(*args: Any, **_kwargs: Any) -> tuple[Any, str]:
            captured_args.extend(args)
            return _percentiles, "meta"

        with (
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.run_tools_for_forecaster",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.build_cross_model_aggregation",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.stacking.run_stacking_numeric",
                new=AsyncMock(side_effect=_fake_numeric),
            ),
        ):
            payload = _run(
                run_stacker_for_arm(
                    question=_make_numeric_q(qid=3),
                    research_blob="R",
                    forecaster_payloads=_three_numeric_forecasters(),
                    arm=ARM_STACK,
                    cache=cache,
                    stacker_llm=stacker_llm,
                    fallback_stacker_llm=fallback_stacker_llm,
                    parser_llm=parser_llm,
                )
            )
        # Position 5 = lower_bound_message; position 6 = upper_bound_message (per stacking.run_stacking_numeric
        # signature: stacker_llm, parser_llm, question, research, base_texts, lower_bound_message, upper_bound_message)
        assert len(captured_args) >= 7, f"expected >=7 positional args, got {len(captured_args)}"
        lower_msg = captured_args[5]
        upper_msg = captured_args[6]
        assert isinstance(lower_msg, str)
        assert isinstance(upper_msg, str)
        # Validate bound messages mention the bounds
        assert "0.0" in lower_msg or "0" in lower_msg
        assert "100" in upper_msg
        assert payload["success"] is True

    def test_dispatch_stacker_wraps_numeric_with_sanitize_and_build(
        self,
        cache: AblationCache,
        stacker_llm: MagicMock,
        fallback_stacker_llm: MagicMock,
        parser_llm: MagicMock,
    ) -> None:
        """``_dispatch_stacker`` must mirror production main.py:450-465 by piping
        ``stacking.run_stacking_numeric``'s ``list[Percentile]`` through
        ``sanitize_percentiles`` + ``detect_unit_mismatch`` + ``build_numeric_distribution``
        before the canonical full-CDF serializer runs.

        We feed an *unsorted, duplicate-laden* 11-percentile list. If wrapping is
        absent, the raw list goes straight to ``serialize_prediction_value``,
        which raises ``TypeError`` (Bucket 1 contract: numeric requires
        ``NumericDistribution``). If wrapping is present, ``sanitize_percentiles``
        sorts by percentile + clamps + deduplicates, and the resulting payload
        carries a sorted ``declared_percentiles`` list with all 11 standard
        percentiles preserved.
        """
        # 11 standard percentiles, deliberately unsorted and with one near-duplicate
        # value cluster that ``apply_jitter_for_duplicates`` should spread.
        unsorted_with_dupes = [
            Percentile(percentile=0.50, value=50.0),  # out of order
            Percentile(percentile=0.025, value=20.0),
            Percentile(percentile=0.10, value=30.0),
            Percentile(percentile=0.05, value=25.0),
            Percentile(percentile=0.40, value=45.0),
            Percentile(percentile=0.20, value=38.0),
            Percentile(percentile=0.60, value=50.0),  # duplicate value with p=0.5
            Percentile(percentile=0.80, value=62.0),
            Percentile(percentile=0.975, value=80.0),
            Percentile(percentile=0.90, value=70.0),
            Percentile(percentile=0.95, value=75.0),
        ]

        def _fake_numeric(*_args: Any, **_kwargs: Any) -> tuple[Any, str]:
            return unsorted_with_dupes, "meta"

        with (
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.run_tools_for_forecaster",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.build_cross_model_aggregation",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.stacking.run_stacking_numeric",
                new=AsyncMock(side_effect=_fake_numeric),
            ),
        ):
            payload = _run(
                run_stacker_for_arm(
                    question=_make_numeric_q(qid=42),
                    research_blob="R",
                    forecaster_payloads=_three_numeric_forecasters(),
                    arm=ARM_STACK,
                    cache=cache,
                    stacker_llm=stacker_llm,
                    fallback_stacker_llm=fallback_stacker_llm,
                    parser_llm=parser_llm,
                )
            )
        assert payload["success"] is True, f"stacker dispatch should succeed; errors={payload.get('errors')}"
        sp = payload["stacker_prediction"]
        # Wrapping ran: serialized payload is a NumericDistribution-shaped dict.
        assert isinstance(sp, dict)
        assert sp["type"] == "numeric"
        # All 11 standard percentiles survive (filter_to_standard_percentiles
        # keeps the canonical set; sanitize_percentiles validates count).
        assert len(sp["declared_percentiles"]) == 11
        # sort_percentiles_by_value reorders by ``percentile`` ascending.
        percentile_keys = [round(float(p["percentile"]), 6) for p in sp["declared_percentiles"]]
        assert percentile_keys == sorted(percentile_keys), (
            f"sanitize_percentiles should sort by percentile; got {percentile_keys}"
        )
        # apply_jitter_for_duplicates / ensure_strictly_increasing_bounded:
        # value-axis must be strictly increasing after sanitization.
        values = [float(p["value"]) for p in sp["declared_percentiles"]]
        assert all(v_next > v_prev for v_prev, v_next in zip(values, values[1:])), (
            f"sanitize_percentiles should produce strictly increasing values; got {values}"
        )
        # Full 201-point CDF ran through build_numeric_distribution.
        assert len(sp["cdf_probabilities"]) == 201


# ===========================================================================
# Payload shape
# ===========================================================================


class TestPayloadShape:
    def test_success_payload_has_expected_keys(
        self,
        cache: AblationCache,
        stacker_llm: MagicMock,
        fallback_stacker_llm: MagicMock,
        parser_llm: MagicMock,
    ) -> None:
        def _fake_stacker(*_args: Any, **_kwargs: Any) -> tuple[float, str]:
            return 0.62, "stacker meta text"

        with (
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.run_tools_for_forecaster",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.build_cross_model_aggregation",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.stacking.run_stacking_binary",
                new=AsyncMock(side_effect=_fake_stacker),
            ),
        ):
            payload = _run(
                run_stacker_for_arm(
                    question=_make_binary_q(qid=1),
                    research_blob="R",
                    forecaster_payloads=_three_binary_forecasters(),
                    arm=ARM_STACK,
                    cache=cache,
                    stacker_llm=stacker_llm,
                    fallback_stacker_llm=fallback_stacker_llm,
                    parser_llm=parser_llm,
                )
            )
        expected_keys = {
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
        assert expected_keys.issubset(payload.keys())
        assert payload["success"] is True
        assert payload["arm"] == ARM_STACK
        assert payload["stacker_prediction"] == {"type": "binary", "prob": 0.62}
        assert payload["stacker_meta_reasoning"] == "stacker meta text"
        assert payload["stacker_model_used"] == "primary"
        assert payload["n_forecasters_used"] == 3
        assert payload["tools_enabled_at_runtime"] is False
        assert payload["errors"] == []

    def test_binary_prediction_value_is_serialized_as_canonical_dict(
        self,
        cache: AblationCache,
        stacker_llm: MagicMock,
        fallback_stacker_llm: MagicMock,
        parser_llm: MagicMock,
    ) -> None:
        def _fake_stacker(*_args: Any, **_kwargs: Any) -> tuple[float, str]:
            return 0.42, "meta"

        with (
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.run_tools_for_forecaster",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.build_cross_model_aggregation",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.stacking.run_stacking_binary",
                new=AsyncMock(side_effect=_fake_stacker),
            ),
        ):
            payload = _run(
                run_stacker_for_arm(
                    question=_make_binary_q(qid=1),
                    research_blob="R",
                    forecaster_payloads=_three_binary_forecasters(),
                    arm=ARM_STACK,
                    cache=cache,
                    stacker_llm=stacker_llm,
                    fallback_stacker_llm=fallback_stacker_llm,
                    parser_llm=parser_llm,
                )
            )
        # binary: stored using canonical forecasters.serialize_prediction_value format
        import json

        assert payload["stacker_prediction"] == {"type": "binary", "prob": 0.42}
        # JSON-roundtrippable
        assert json.loads(json.dumps(payload["stacker_prediction"])) == {"type": "binary", "prob": 0.42}

    def test_numeric_prediction_value_is_serialized_as_canonical_dict(
        self,
        cache: AblationCache,
        stacker_llm: MagicMock,
        fallback_stacker_llm: MagicMock,
        parser_llm: MagicMock,
    ) -> None:
        """Stacker numeric output is serialized via the post-Bucket-1 full-CDF schema.

        Production ``stacking.run_stacking_numeric`` returns
        ``tuple[list[Percentile], str]`` (the raw parser output + meta-reasoning).
        ``_dispatch_stacker`` mirrors main.py:450-465 by piping the list through
        ``sanitize_percentiles`` → ``detect_unit_mismatch`` → ``build_numeric_distribution``
        before the canonical full-CDF serializer runs. The fake here returns the
        raw 11-Percentile list so we exercise that wrapping, then assert the
        serialized payload still carries declared_percentiles + cdf_probabilities
        + bounds + zero_point + cdf_size.
        """
        question = _make_numeric_q(qid=3)
        # 11 standard percentiles in canonical order — what production
        # ``stacking.run_stacking_numeric`` emits after the parser LLM.
        declared = [
            Percentile(percentile=0.025, value=20.0),
            Percentile(percentile=0.05, value=25.0),
            Percentile(percentile=0.10, value=30.0),
            Percentile(percentile=0.20, value=38.0),
            Percentile(percentile=0.40, value=45.0),
            Percentile(percentile=0.50, value=50.0),
            Percentile(percentile=0.60, value=55.0),
            Percentile(percentile=0.80, value=62.0),
            Percentile(percentile=0.90, value=70.0),
            Percentile(percentile=0.95, value=75.0),
            Percentile(percentile=0.975, value=80.0),
        ]
        cdf_size = int(question.cdf_size or 201)

        def _fake_numeric(*_args: Any, **_kwargs: Any) -> tuple[Any, str]:
            return declared, "meta"

        with (
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.run_tools_for_forecaster",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.build_cross_model_aggregation",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.stacking.run_stacking_numeric",
                new=AsyncMock(side_effect=_fake_numeric),
            ),
        ):
            payload = _run(
                run_stacker_for_arm(
                    question=question,
                    research_blob="R",
                    forecaster_payloads=_three_numeric_forecasters(),
                    arm=ARM_STACK,
                    cache=cache,
                    stacker_llm=stacker_llm,
                    fallback_stacker_llm=fallback_stacker_llm,
                    parser_llm=parser_llm,
                )
            )
        import json

        # numeric (post-Bucket-1): full-CDF schema; JSON-roundtrippable.
        sp = payload["stacker_prediction"]
        assert isinstance(sp, dict)
        assert sp["type"] == "numeric"
        # declared_percentiles: the 11 standard percentiles round-trip after
        # sanitize_percentiles + build_numeric_distribution.
        assert isinstance(sp["declared_percentiles"], list)
        assert len(sp["declared_percentiles"]) == 11
        assert all("percentile" in p and "value" in p for p in sp["declared_percentiles"])
        # cdf_probabilities: 201 monotonic floats
        assert isinstance(sp["cdf_probabilities"], list)
        assert len(sp["cdf_probabilities"]) == cdf_size
        assert all(isinstance(p, float) for p in sp["cdf_probabilities"])
        # Bounds + zero_point + cdf_size present.
        assert sp["lower_bound"] == question.lower_bound
        assert sp["upper_bound"] == question.upper_bound
        assert sp["open_lower_bound"] is question.open_lower_bound
        assert sp["open_upper_bound"] is question.open_upper_bound
        assert sp["cdf_size"] == cdf_size
        rt = json.loads(json.dumps(sp))
        assert rt == sp

    def test_mc_prediction_value_is_serialized_as_canonical_dict(
        self,
        cache: AblationCache,
        stacker_llm: MagicMock,
        fallback_stacker_llm: MagicMock,
        parser_llm: MagicMock,
    ) -> None:
        pol = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name="Red", probability=0.6),
                PredictedOption(option_name="Blue", probability=0.4),
            ]
        )

        def _fake_mc(*_args: Any, **_kwargs: Any) -> tuple[Any, str]:
            return pol, "meta"

        with (
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.run_tools_for_forecaster",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.build_cross_model_aggregation",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.stacking.run_stacking_mc",
                new=AsyncMock(side_effect=_fake_mc),
            ),
        ):
            payload = _run(
                run_stacker_for_arm(
                    question=_make_mc_q(qid=2),
                    research_blob="R",
                    forecaster_payloads=_three_mc_forecasters(),
                    arm=ARM_STACK,
                    cache=cache,
                    stacker_llm=stacker_llm,
                    fallback_stacker_llm=fallback_stacker_llm,
                    parser_llm=parser_llm,
                )
            )
        import json

        sp = payload["stacker_prediction"]
        assert isinstance(sp, dict)
        assert sp["type"] == "multiple_choice"
        assert "options" in sp
        assert isinstance(sp["options"], list)
        # JSON-roundtrippable
        rt = json.loads(json.dumps(sp))
        assert rt == sp


# ===========================================================================
# run_stacker_batch
# ===========================================================================


class TestRunStackerBatch:
    def test_batch_returns_dict_keyed_by_qid(
        self,
        cache: AblationCache,
        stacker_llm: MagicMock,
        fallback_stacker_llm: MagicMock,
        parser_llm: MagicMock,
    ) -> None:
        qid_to_data = {
            10: {
                "question": _make_binary_q(qid=10),
                "research": "R10",
                "forecaster_payloads": _three_binary_forecasters(),
            },
            20: {
                "question": _make_binary_q(qid=20),
                "research": "R20",
                "forecaster_payloads": _three_binary_forecasters(),
            },
        }

        def _fake_stacker(*_args: Any, **_kwargs: Any) -> tuple[float, str]:
            return 0.5, "meta"

        with (
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.run_tools_for_forecaster",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.build_cross_model_aggregation",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.stacking.run_stacking_binary",
                new=AsyncMock(side_effect=_fake_stacker),
            ),
        ):
            results = _run(
                run_stacker_batch(
                    qid_to_data=qid_to_data,
                    arm=ARM_STACK,
                    cache=cache,
                    stacker_llm=stacker_llm,
                    fallback_stacker_llm=fallback_stacker_llm,
                    parser_llm=parser_llm,
                )
            )
        assert set(results.keys()) == {10, 20}
        assert all(r["success"] for r in results.values())

    def test_batch_per_question_failure_does_not_kill_batch(
        self,
        cache: AblationCache,
        stacker_llm: MagicMock,
        fallback_stacker_llm: MagicMock,
        parser_llm: MagicMock,
    ) -> None:
        qid_to_data = {
            10: {
                "question": _make_binary_q(qid=10),
                "research": "R10",
                "forecaster_payloads": _three_binary_forecasters(),
            },
            20: {
                "question": _make_binary_q(qid=20),
                "research": "R20",
                "forecaster_payloads": {  # only 1 valid -> insufficient
                    model_slug_to_filename("openrouter/test/m1"): _binary_payload("openrouter/test/m1", 0.5),
                },
            },
            30: {
                "question": _make_binary_q(qid=30),
                "research": "R30",
                "forecaster_payloads": _three_binary_forecasters(),
            },
        }

        def _fake_stacker(*_args: Any, **_kwargs: Any) -> tuple[float, str]:
            return 0.5, "meta"

        with (
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.run_tools_for_forecaster",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.build_cross_model_aggregation",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.stacking.run_stacking_binary",
                new=AsyncMock(side_effect=_fake_stacker),
            ),
        ):
            results = _run(
                run_stacker_batch(
                    qid_to_data=qid_to_data,
                    arm=ARM_STACK,
                    cache=cache,
                    stacker_llm=stacker_llm,
                    fallback_stacker_llm=fallback_stacker_llm,
                    parser_llm=parser_llm,
                )
            )
        assert set(results.keys()) == {10, 20, 30}
        assert results[10]["success"] is True
        assert results[20]["success"] is False
        assert results[30]["success"] is True

    def test_batch_uses_passed_llms_directly(
        self,
        cache: AblationCache,
        stacker_llm: MagicMock,
        fallback_stacker_llm: MagicMock,
        parser_llm: MagicMock,
    ) -> None:
        """Passing the same llm objects across multiple qids is the contract — they are reused."""
        qid_to_data = {
            10: {
                "question": _make_binary_q(qid=10),
                "research": "R10",
                "forecaster_payloads": _three_binary_forecasters(),
            },
            20: {
                "question": _make_binary_q(qid=20),
                "research": "R20",
                "forecaster_payloads": _three_binary_forecasters(),
            },
        }

        seen_stacker_llms: list[Any] = []

        def _fake_stacker(*args: Any, **_kwargs: Any) -> tuple[float, str]:
            seen_stacker_llms.append(args[0])
            return 0.5, "meta"

        with (
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.run_tools_for_forecaster",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.build_cross_model_aggregation",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.stacking.run_stacking_binary",
                new=AsyncMock(side_effect=_fake_stacker),
            ),
        ):
            _run(
                run_stacker_batch(
                    qid_to_data=qid_to_data,
                    arm=ARM_STACK,
                    cache=cache,
                    stacker_llm=stacker_llm,
                    fallback_stacker_llm=fallback_stacker_llm,
                    parser_llm=parser_llm,
                )
            )
        assert len(seen_stacker_llms) == 2
        assert all(s is stacker_llm for s in seen_stacker_llms)


# ===========================================================================
# Concurrent stacker calls — window-patch lock serializes patched section
# ===========================================================================


class TestConcurrentStackerLock:
    def test_concurrent_stacker_calls_serialized_under_lock(
        self,
        cache: AblationCache,
        stacker_llm: MagicMock,
        fallback_stacker_llm: MagicMock,
        parser_llm: MagicMock,
    ) -> None:
        """Two concurrent run_stacker_for_arm calls must NOT both be inside
        ``patched_window_for_question`` simultaneously.

        ``patched_window_for_question`` is a global monkey-patch that raises
        ``RuntimeError`` on nested entry. Without the module-level
        ``_WINDOW_PATCH_LOCK`` serializing patched-section entry, two
        concurrent stacker calls would race and the second to enter would
        crash. The lock keeps each call inside its own patched region.
        """
        from metaculus_bot.ablation import window_patch as wp

        observed_active_during_call: list[bool] = []
        max_concurrent_in_patch = 0
        currently_in_patch = 0

        async def _fake_stacker(*_args: Any, **_kwargs: Any) -> tuple[float, str]:
            nonlocal max_concurrent_in_patch, currently_in_patch
            observed_active_during_call.append(wp._window_patch_active)
            currently_in_patch += 1
            max_concurrent_in_patch = max(max_concurrent_in_patch, currently_in_patch)
            # Yield so a second concurrent call gets a chance to interleave
            # if the lock isn't doing its job. The test only catches the bug
            # if there's a real opportunity for concurrent entry.
            await asyncio.sleep(0.01)
            currently_in_patch -= 1
            return 0.5, "meta"

        async def _drive() -> tuple[dict, dict]:
            with (
                patch(
                    "metaculus_bot.ablation.run_stacker.tool_runner.run_tools_for_forecaster",
                    return_value="",
                ),
                patch(
                    "metaculus_bot.ablation.run_stacker.tool_runner.build_cross_model_aggregation",
                    return_value="",
                ),
                patch(
                    "metaculus_bot.ablation.run_stacker.stacking.run_stacking_binary",
                    new=AsyncMock(side_effect=_fake_stacker),
                ),
            ):
                return await asyncio.gather(
                    run_stacker_for_arm(
                        question=_make_binary_q(qid=101),
                        research_blob="R101",
                        forecaster_payloads=_three_binary_forecasters(),
                        arm=ARM_STACK,
                        cache=cache,
                        stacker_llm=stacker_llm,
                        fallback_stacker_llm=fallback_stacker_llm,
                        parser_llm=parser_llm,
                    ),
                    run_stacker_for_arm(
                        question=_make_binary_q(qid=102),
                        research_blob="R102",
                        forecaster_payloads=_three_binary_forecasters(),
                        arm=ARM_STACK,
                        cache=cache,
                        stacker_llm=stacker_llm,
                        fallback_stacker_llm=fallback_stacker_llm,
                        parser_llm=parser_llm,
                    ),
                )

        results = _run(_drive())
        assert all(r["success"] for r in results)
        # Patch was active during every stacker call (both succeeded)
        assert observed_active_during_call == [True, True]
        # The lock kept patched-section entry serialized — at most one stacker
        # call inside patched_window_for_question at a time.
        assert max_concurrent_in_patch == 1, (
            f"window patch lock failed to serialize concurrent calls "
            f"(max_concurrent_in_patch={max_concurrent_in_patch})"
        )
        # And after the calls, no longer active.
        assert wp._window_patch_active is False


# ===========================================================================
# Real tool_runner integration — proves arm A vs arm B genuinely differ
# ===========================================================================


def _binary_rationale_with_valid_json(posterior_prob: float = 0.42) -> str:
    """Synthetic forecaster rationale containing a valid binary structured JSON block.

    Built to exercise the real ``tool_runner.parse_structured_block`` path —
    no mocking. The block declares prior, base_rate, evidence, and posterior
    so multiple tools fire (Beta-binomial, Prior→posterior, Prior+k/n combine).
    """
    return textwrap.dedent(
        f"""
        Model: openrouter/test/foo

        I think the answer is yes because [analysis].

        ```json
        {{
          "question_type": "binary",
          "prior": {{"prob": 0.15, "source": "annual incidence"}},
          "base_rate": {{"k": 3, "n": 12, "ref_class": "comparable years"}},
          "evidence": [{{"summary": "policy shift", "direction": "up", "strength": "moderate"}}],
          "posterior_prob": {posterior_prob}
        }}
        ```

        Probability: {int(posterior_prob * 100)}%
        """
    ).strip()


def _numeric_rationale_with_valid_json(median: float = 50.0) -> str:
    """Synthetic numeric rationale with mixture_components + standard percentiles.

    The mixture_components block triggers _render_mixture_section in the real
    tool_runner, producing a "### Mixture-of-normals" subsection with CDF
    samples. The declared_percentiles also trigger family-consistency and
    out-of-bounds-mass tools.
    """
    return textwrap.dedent(
        f"""
        Model: openrouter/test/foo

        Reasoning about percentiles.

        ```json
        {{
          "question_type": "numeric",
          "declared_percentiles": {{
            "0.025": {median - 30}, "0.05": {median - 25}, "0.1": {median - 20},
            "0.2": {median - 12}, "0.4": {median - 5}, "0.5": {median},
            "0.6": {median + 5}, "0.8": {median + 12}, "0.9": {median + 20},
            "0.95": {median + 25}, "0.975": {median + 30}
          }},
          "distribution_family_hint": "normal",
          "mixture_components": [
            {{"weight": 0.3, "mean": {median - 20}, "sd": 7.0}},
            {{"weight": 0.4, "mean": {median}, "sd": 10.0}},
            {{"weight": 0.3, "mean": {median + 20}, "sd": 8.0}}
          ]
        }}
        ```

        Percentile 50: {median}
        """
    ).strip()


def _mc_rationale_with_valid_json() -> str:
    """Synthetic MC rationale with valid option_probs + other_mass + concentration.

    Triggers MC tools: residual-mass line + Dirichlet-with-Other CIs.
    """
    return textwrap.dedent(
        """
        Model: openrouter/test/foo

        Reasoning about options.

        ```json
        {
          "question_type": "multiple_choice",
          "option_probs": {"Red": 0.6, "Blue": 0.4},
          "other_mass": 0.1,
          "concentration": 20.0
        }
        ```

        Red: 60%
        Blue: 40%
        """
    ).strip()


class TestRealToolRunnerIntegration:
    """End-to-end tests with REAL tool_runner (no mocks).

    These tests exercise tool_runner.run_tools_for_forecaster and
    tool_runner.build_cross_model_aggregation against synthetic forecaster
    rationales that contain valid structured JSON blocks. They prove the
    A/B contrast actually fires in arm B — the failure mode the plan calls
    out at "verification step 5: if cross_model_aggregation is empty
    everywhere, debug parse_structured_block on free-model rationales".
    """

    def test_arm_b_with_real_tool_runner_produces_computed_quantities_for_binary(
        self,
        cache: AblationCache,
        stacker_llm: MagicMock,
        fallback_stacker_llm: MagicMock,
        parser_llm: MagicMock,
    ) -> None:
        """Arm B + real tool_runner + valid binary JSON block → non-empty
        per-forecaster computed_quantities AND non-empty cross_model_aggregation."""
        # Two forecasters with valid but distinct binary structured blocks.
        # Cross-model aggregation needs >= 2 forecasters to emit pool lines.
        forecasters = {
            model_slug_to_filename("openrouter/test/m1"): {
                "model": "openrouter/test/m1",
                "prediction_value": {"type": "binary", "prob": 0.42},
                "reasoning": _binary_rationale_with_valid_json(0.42),
                "errors": [],
            },
            model_slug_to_filename("openrouter/test/m2"): {
                "model": "openrouter/test/m2",
                "prediction_value": {"type": "binary", "prob": 0.55},
                "reasoning": _binary_rationale_with_valid_json(0.55),
                "errors": [],
            },
        }

        # Mock ONLY the stacker LLM call — tool_runner runs for real.
        with patch(
            "metaculus_bot.ablation.run_stacker.stacking.run_stacking_binary",
            new=AsyncMock(return_value=(0.5, "stacker meta")),
        ):
            payload = _run(
                run_stacker_for_arm(
                    question=_make_binary_q(qid=1),
                    research_blob="research",
                    forecaster_payloads=forecasters,
                    arm=ARM_STACK_AUG,
                    cache=cache,
                    stacker_llm=stacker_llm,
                    fallback_stacker_llm=fallback_stacker_llm,
                    parser_llm=parser_llm,
                )
            )

        # Critical assertions.
        assert payload["success"] is True
        assert payload["computed_quantities"], (
            "Real tool_runner produced empty output for arm B — tools not firing! "
            "Check that PROBABILISTIC_TOOLS_ENABLED is set inside run_stacker_for_arm."
        )
        # Both forecasters' rationales should have produced output.
        assert len(payload["computed_quantities"]) == 2
        # Each forecaster's computed-quantities markdown contains real tool output.
        for slug, md in payload["computed_quantities"].items():
            assert "Beta-binomial" in md, f"Missing Beta-binomial for {slug}: {md!r}"
            assert "Prior → posterior" in md, f"Missing prior→posterior for {slug}: {md!r}"
            assert "Bayesian combine" in md, f"Missing Bayesian combine for {slug}: {md!r}"

        # Cross-model aggregation also fired (linear pool, log pool, Satopää).
        assert payload["cross_model_aggregation"], (
            "Real build_cross_model_aggregation produced empty output! "
            "Check that PROBABILISTIC_TOOLS_ENABLED was set during the call."
        )
        agg = payload["cross_model_aggregation"]
        assert "Pools over 2 forecasters" in agg, agg
        assert "linear" in agg.lower(), agg
        assert "Blended base rate" in agg, agg

    def test_arm_a_with_real_tool_runner_produces_empty_for_binary(
        self,
        cache: AblationCache,
        stacker_llm: MagicMock,
        fallback_stacker_llm: MagicMock,
        parser_llm: MagicMock,
    ) -> None:
        """Arm A + real tool_runner + valid binary JSON block → STILL empty
        because the env-flag is unset.

        This proves arm A and arm B genuinely differ when the JSON is parseable.
        Without this test, an arm B that silently degrades to arm A would look
        identical to a working arm A.
        """
        forecasters = {
            model_slug_to_filename("openrouter/test/m1"): {
                "model": "openrouter/test/m1",
                "prediction_value": {"type": "binary", "prob": 0.42},
                "reasoning": _binary_rationale_with_valid_json(0.42),
                "errors": [],
            },
            model_slug_to_filename("openrouter/test/m2"): {
                "model": "openrouter/test/m2",
                "prediction_value": {"type": "binary", "prob": 0.55},
                "reasoning": _binary_rationale_with_valid_json(0.55),
                "errors": [],
            },
        }

        with patch(
            "metaculus_bot.ablation.run_stacker.stacking.run_stacking_binary",
            new=AsyncMock(return_value=(0.5, "stacker meta")),
        ):
            payload = _run(
                run_stacker_for_arm(
                    question=_make_binary_q(qid=2),
                    research_blob="research",
                    forecaster_payloads=forecasters,
                    arm=ARM_STACK,
                    cache=cache,
                    stacker_llm=stacker_llm,
                    fallback_stacker_llm=fallback_stacker_llm,
                    parser_llm=parser_llm,
                )
            )

        assert payload["success"] is True
        # Arm A: env flag is unset, so tool_runner returns "" everywhere.
        assert payload["computed_quantities"] == {}, (
            "Arm A leaked tool output! computed_quantities should be empty when "
            f"PROBABILISTIC_TOOLS_ENABLED is unset. Got: {payload['computed_quantities']!r}"
        )
        assert payload["cross_model_aggregation"] == "", (
            f"Arm A leaked cross-model aggregation! Got: {payload['cross_model_aggregation']!r}"
        )
        assert payload["tools_enabled_at_runtime"] is False

    def test_arm_b_with_invalid_json_silently_produces_empty_computed_quantities(
        self,
        cache: AblationCache,
        stacker_llm: MagicMock,
        fallback_stacker_llm: MagicMock,
        parser_llm: MagicMock,
    ) -> None:
        """Arm B + INVALID JSON (missing required posterior_prob) → tool_runner
        returns empty for that forecaster, even though the env flag is on.

        This documents the silent-degradation failure mode: when free models
        emit malformed JSON, parse_structured_block returns None → tool runner
        returns "" → per-forecaster computed_quantities is empty. The stacker
        still runs (success=True) but the treatment effect collapses.

        If a smoke run shows ~0 measured effect from arm A → arm B, this is
        the first thing to check: are the free-model rationales parseable?
        """
        # Two forecasters, both with INVALID JSON: missing required `posterior_prob`.
        bad_rationale = textwrap.dedent(
            """
            Model: openrouter/test/foo

            Analysis here.

            ```json
            {
              "question_type": "binary",
              "prior": {"prob": 0.15, "source": "annual incidence"}
            }
            ```

            Probability: 35%
            """
        ).strip()

        forecasters = {
            model_slug_to_filename("openrouter/test/m1"): {
                "model": "openrouter/test/m1",
                "prediction_value": {"type": "binary", "prob": 0.42},
                "reasoning": bad_rationale,
                "errors": [],
            },
            model_slug_to_filename("openrouter/test/m2"): {
                "model": "openrouter/test/m2",
                "prediction_value": {"type": "binary", "prob": 0.55},
                "reasoning": bad_rationale,
                "errors": [],
            },
        }

        with patch(
            "metaculus_bot.ablation.run_stacker.stacking.run_stacking_binary",
            new=AsyncMock(return_value=(0.5, "stacker meta")),
        ):
            payload = _run(
                run_stacker_for_arm(
                    question=_make_binary_q(qid=3),
                    research_blob="research",
                    forecaster_payloads=forecasters,
                    arm=ARM_STACK_AUG,
                    cache=cache,
                    stacker_llm=stacker_llm,
                    fallback_stacker_llm=fallback_stacker_llm,
                    parser_llm=parser_llm,
                )
            )

        # Stacker still ran successfully — the arm just got no real tool output.
        assert payload["success"] is True
        assert payload["tools_enabled_at_runtime"] is True

        # Per-forecaster computed_quantities: empty because parse_structured_block
        # rejected both rationales (missing required field).
        assert payload["computed_quantities"] == {}, (
            f"Expected empty computed_quantities on invalid JSON; got {payload['computed_quantities']!r}"
        )

        # Cross-model aggregation: NOT empty because aggregate_binary_values
        # pools the prediction_values directly (doesn't depend on JSON parse).
        # The "Pools over N forecasters" line always fires when N >= 2.
        # Other lines (Blended base rate, Prior/posterior snapshot) DO depend
        # on parsed blocks — they should be absent here.
        agg = payload["cross_model_aggregation"]
        assert "Pools over 2 forecasters" in agg, f"Expected pool line from prediction values alone; got {agg!r}"
        # Lines that need parsed structured blocks must be absent — proving
        # the JSON parse failed on both forecasters.
        assert "Blended base rate" not in agg, f"Blended base rate appeared but JSON should not have parsed: {agg!r}"
        assert "Prior/posterior snapshot" not in agg, (
            f"Prior/posterior snapshot appeared but JSON should not have parsed: {agg!r}"
        )

    def test_arm_b_with_real_tool_runner_produces_mixture_section_for_numeric(
        self,
        cache: AblationCache,
        stacker_llm: MagicMock,
        fallback_stacker_llm: MagicMock,
        parser_llm: MagicMock,
    ) -> None:
        """Arm B + real tool_runner + valid numeric JSON with mixture_components
        → per-forecaster Mixture-of-normals subsection AND cross-model medians."""
        # Post-Bucket-1: forecaster numeric payloads use the full-CDF schema.
        # Synthesize a monotone linear CDF that spans the bounds for both forecasters.
        _cdf_probs = [0.001 + (0.998 * i / 200) for i in range(201)]

        def _numeric_pred_payload(declared_pcts: list[dict]) -> dict:
            return {
                "type": "numeric",
                "declared_percentiles": declared_pcts,
                "cdf_probabilities": _cdf_probs,
                "lower_bound": 0.0,
                "upper_bound": 100.0,
                "open_lower_bound": False,
                "open_upper_bound": False,
                "zero_point": None,
                "cdf_size": 201,
            }

        forecasters = {
            model_slug_to_filename("openrouter/test/m1"): {
                "model": "openrouter/test/m1",
                "prediction_value": _numeric_pred_payload(
                    [
                        {"percentile": 0.025, "value": 20},
                        {"percentile": 0.05, "value": 25},
                        {"percentile": 0.1, "value": 30},
                        {"percentile": 0.2, "value": 38},
                        {"percentile": 0.4, "value": 45},
                        {"percentile": 0.5, "value": 50},
                        {"percentile": 0.6, "value": 55},
                        {"percentile": 0.8, "value": 62},
                        {"percentile": 0.9, "value": 70},
                        {"percentile": 0.95, "value": 75},
                        {"percentile": 0.975, "value": 80},
                    ]
                ),
                "reasoning": _numeric_rationale_with_valid_json(50.0),
                "errors": [],
            },
            model_slug_to_filename("openrouter/test/m2"): {
                "model": "openrouter/test/m2",
                "prediction_value": _numeric_pred_payload(
                    [
                        {"percentile": 0.025, "value": 25},
                        {"percentile": 0.05, "value": 30},
                        {"percentile": 0.1, "value": 35},
                        {"percentile": 0.2, "value": 43},
                        {"percentile": 0.4, "value": 50},
                        {"percentile": 0.5, "value": 55},
                        {"percentile": 0.6, "value": 60},
                        {"percentile": 0.8, "value": 67},
                        {"percentile": 0.9, "value": 75},
                        {"percentile": 0.95, "value": 80},
                        {"percentile": 0.975, "value": 85},
                    ]
                ),
                "reasoning": _numeric_rationale_with_valid_json(55.0),
                "errors": [],
            },
        }

        # ``stacking.run_stacking_numeric`` returns ``tuple[list[Percentile], str]``
        # in production; ``_dispatch_stacker`` wraps the list with sanitize +
        # build_numeric_distribution before serialization.
        _stacker_percentiles = [
            Percentile(percentile=0.025, value=22.0),
            Percentile(percentile=0.05, value=27.0),
            Percentile(percentile=0.10, value=32.0),
            Percentile(percentile=0.20, value=40.0),
            Percentile(percentile=0.40, value=47.0),
            Percentile(percentile=0.50, value=52.0),
            Percentile(percentile=0.60, value=57.0),
            Percentile(percentile=0.80, value=64.0),
            Percentile(percentile=0.90, value=72.0),
            Percentile(percentile=0.95, value=77.0),
            Percentile(percentile=0.975, value=82.0),
        ]

        with patch(
            "metaculus_bot.ablation.run_stacker.stacking.run_stacking_numeric",
            new=AsyncMock(return_value=(_stacker_percentiles, "stacker meta")),
        ):
            payload = _run(
                run_stacker_for_arm(
                    question=_make_numeric_q(qid=4),
                    research_blob="research",
                    forecaster_payloads=forecasters,
                    arm=ARM_STACK_AUG,
                    cache=cache,
                    stacker_llm=stacker_llm,
                    fallback_stacker_llm=fallback_stacker_llm,
                    parser_llm=parser_llm,
                )
            )

        assert payload["success"] is True
        assert len(payload["computed_quantities"]) == 2
        # Each forecaster's per-rationale output contains real tool sections.
        for slug, md in payload["computed_quantities"].items():
            assert "Percentile-family consistency" in md, f"Missing family check for {slug}: {md!r}"
            assert "Out-of-bounds mass" in md, f"Missing OOB mass for {slug}: {md!r}"
            assert "### Mixture-of-normals" in md, f"Missing mixture section for {slug}: {md!r}"

        # Cross-model agg: medians + declared families.
        assert payload["cross_model_aggregation"], "Real build_cross_model_aggregation produced empty numeric output!"
        agg = payload["cross_model_aggregation"]
        assert "Forecaster medians" in agg, agg
        assert "Declared distribution families" in agg, agg

    def test_arm_b_with_real_tool_runner_produces_dirichlet_for_mc(
        self,
        cache: AblationCache,
        stacker_llm: MagicMock,
        fallback_stacker_llm: MagicMock,
        parser_llm: MagicMock,
    ) -> None:
        """Arm B + real tool_runner + valid MC JSON with option_probs + other_mass
        → per-forecaster Dirichlet-with-Other CIs AND cross-model linear pool."""
        forecasters = {
            model_slug_to_filename("openrouter/test/m1"): {
                "model": "openrouter/test/m1",
                "prediction_value": {
                    "type": "multiple_choice",
                    "options": [
                        {"option_name": "Red", "probability": 0.6},
                        {"option_name": "Blue", "probability": 0.4},
                    ],
                },
                "reasoning": _mc_rationale_with_valid_json(),
                "errors": [],
            },
            model_slug_to_filename("openrouter/test/m2"): {
                "model": "openrouter/test/m2",
                "prediction_value": {
                    "type": "multiple_choice",
                    "options": [
                        {"option_name": "Red", "probability": 0.5},
                        {"option_name": "Blue", "probability": 0.5},
                    ],
                },
                "reasoning": _mc_rationale_with_valid_json(),
                "errors": [],
            },
        }

        mc_result = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name="Red", probability=0.55),
                PredictedOption(option_name="Blue", probability=0.45),
            ]
        )
        with patch(
            "metaculus_bot.ablation.run_stacker.stacking.run_stacking_mc",
            new=AsyncMock(return_value=(mc_result, "stacker meta")),
        ):
            payload = _run(
                run_stacker_for_arm(
                    question=_make_mc_q(qid=5),
                    research_blob="research",
                    forecaster_payloads=forecasters,
                    arm=ARM_STACK_AUG,
                    cache=cache,
                    stacker_llm=stacker_llm,
                    fallback_stacker_llm=fallback_stacker_llm,
                    parser_llm=parser_llm,
                )
            )

        assert payload["success"] is True
        assert len(payload["computed_quantities"]) == 2
        for slug, md in payload["computed_quantities"].items():
            assert "Other / residual mass" in md, f"Missing residual mass for {slug}: {md!r}"
            assert "Dirichlet-with-Other" in md, f"Missing Dirichlet for {slug}: {md!r}"
            assert "80% CI" in md, f"Missing CI for {slug}: {md!r}"

        # Cross-model: linear pool over named options.
        assert payload["cross_model_aggregation"], "Real build_cross_model_aggregation produced empty MC output!"
        agg = payload["cross_model_aggregation"]
        assert "Linear pool across 2 forecasters" in agg, agg
        assert "Red=" in agg, agg


# ===========================================================================
# Default stacker LLM construction — donated-key wrapper
# ===========================================================================


class TestDefaultStackerWiredViaDonatedKey:
    """When callers pass ``stacker_llm=None`` we construct claude-opus-4.5
    (primary) and gpt-5.5 (fallback) routed via ``build_llm_with_openrouter_fallback``
    so the Metaculus-donated key is tried before the operator's paid key.

    This mirrors production STACKER_LLM / STACKER_FALLBACK_LLM in
    ``llm_configs.py``. The earlier iteration tried gpt-5.5 as primary, but
    the operator's local-`.env` donated key data-policy blocks gpt-5.5;
    Anthropic models work cleanly. Production with a different
    ``OAI_ANTH_OPENROUTER_KEY`` GitHub-secret value behaved differently.
    """

    def test_default_stacker_uses_opus_4_5_via_donated_key_wrapper(
        self, cache: AblationCache, parser_llm: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from metaculus_bot.ablation.run_stacker import (
            DEFAULT_STACKER_FALLBACK_MODEL,
            DEFAULT_STACKER_MODEL,
        )
        from metaculus_bot.fallback_openrouter import FallbackOpenRouterLlm

        # Both keys present + distinct → wrapper chooses FallbackOpenRouterLlm.
        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "fake_donated")
        monkeypatch.setenv("OPENROUTER_API_KEY", "fake_paid")

        # Pin the new defaults at the constant level — primary is opus-4.5,
        # fallback stays gpt-5.5 (different provider for independent failure mode).
        assert DEFAULT_STACKER_MODEL == "openrouter/anthropic/claude-opus-4.5"
        assert DEFAULT_STACKER_FALLBACK_MODEL == "openrouter/openai/gpt-5.5"

        captured_llms: list[Any] = []

        def _fake_stacker(*args: Any, **_kwargs: Any) -> tuple[float, str]:
            captured_llms.append(args[0])
            return 0.5, "meta"

        with (
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.run_tools_for_forecaster",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.build_cross_model_aggregation",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.stacking.run_stacking_binary",
                new=AsyncMock(side_effect=_fake_stacker),
            ),
        ):
            payload = _run(
                run_stacker_for_arm(
                    question=_make_binary_q(qid=1),
                    research_blob="R",
                    forecaster_payloads=_three_binary_forecasters(),
                    arm=ARM_STACK,
                    cache=cache,
                    stacker_llm=None,  # exercise default construction
                    fallback_stacker_llm=None,
                    parser_llm=parser_llm,
                )
            )
        assert payload["success"] is True
        assert len(captured_llms) == 1
        primary = captured_llms[0]
        assert isinstance(primary, FallbackOpenRouterLlm), (
            f"Default stacker should be FallbackOpenRouterLlm; got {type(primary).__name__}"
        )
        assert primary.model == "openrouter/anthropic/claude-opus-4.5"

    def test_default_stacker_in_batch_uses_opus_4_5_via_donated_key_wrapper(
        self, cache: AblationCache, parser_llm: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from metaculus_bot.fallback_openrouter import FallbackOpenRouterLlm

        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "fake_donated")
        monkeypatch.setenv("OPENROUTER_API_KEY", "fake_paid")

        captured_llms: list[Any] = []

        def _fake_stacker(*args: Any, **_kwargs: Any) -> tuple[float, str]:
            captured_llms.append(args[0])
            return 0.5, "meta"

        qid_to_data = {
            10: {
                "question": _make_binary_q(qid=10),
                "research": "R10",
                "forecaster_payloads": _three_binary_forecasters(),
            },
        }

        with (
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.run_tools_for_forecaster",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.build_cross_model_aggregation",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.stacking.run_stacking_binary",
                new=AsyncMock(side_effect=_fake_stacker),
            ),
        ):
            results = _run(
                run_stacker_batch(
                    qid_to_data=qid_to_data,
                    arm=ARM_STACK,
                    cache=cache,
                    stacker_llm=None,  # exercise default construction
                    fallback_stacker_llm=None,
                    parser_llm=parser_llm,
                )
            )
        assert results[10]["success"] is True
        assert len(captured_llms) == 1
        primary = captured_llms[0]
        assert isinstance(primary, FallbackOpenRouterLlm)
        assert primary.model == "openrouter/anthropic/claude-opus-4.5"


# ===========================================================================
# M3 — Tertiary MEDIAN fallback when both stackers fail
#
# Production at main.py:1287-1323 has a final MEDIAN aggregation when both
# the primary and fallback stackers raise. The ablation previously just
# recorded success=False and lost the question for both arms. With this
# fix, a both-stackers-fail outcome yields a degraded-but-publishable
# MEDIAN forecast tagged stacker_model_used="median_fallback" so the
# confounder analysis can distinguish it from the regular primary/fallback
# outcomes.
# ===========================================================================


class TestNoStackerFallback:
    """Fail-fast contract for ``--no-stacker-fallback`` (paid prod-ish runs).

    When ``fallback_stacker_llm=None`` is passed explicitly (the new sentinel
    semantics), a primary-stacker failure must:
      1. Write a failure payload to cache (so resume-from-cache sees the state).
      2. Raise ``RuntimeError`` so the orchestrator aborts the run.

    This is true fail-fast: a borked-key scenario aborts at qid #1 instead of
    silently writing failure payloads for all 88 questions. Prior behavior
    (``fallback_stacker_llm=None`` meant "build default fallback") is now
    triggered by omitting the kwarg or passing the ``_UNSET`` sentinel.
    """

    def test_primary_failure_with_no_fallback_raises_and_writes_failure_payload(
        self,
        cache: AblationCache,
        stacker_llm: MagicMock,
        parser_llm: MagicMock,
    ) -> None:
        forecasters = {
            model_slug_to_filename("openrouter/test/m1"): _binary_payload("openrouter/test/m1", 0.6),
            model_slug_to_filename("openrouter/test/m2"): _binary_payload("openrouter/test/m2", 0.5),
            model_slug_to_filename("openrouter/test/m3"): _binary_payload("openrouter/test/m3", 0.4),
        }

        def _primary_fails(*_args: Any, **_kwargs: Any) -> tuple[float, str]:
            raise RuntimeError("primary boom")

        with (
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.run_tools_for_forecaster",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.build_cross_model_aggregation",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.stacking.run_stacking_binary",
                new=AsyncMock(side_effect=_primary_fails),
            ),
            pytest.raises(RuntimeError, match="--no-stacker-fallback"),
        ):
            _run(
                run_stacker_for_arm(
                    question=_make_binary_q(qid=3001),
                    research_blob="R",
                    forecaster_payloads=forecasters,
                    arm=ARM_STACK,
                    cache=cache,
                    stacker_llm=stacker_llm,
                    fallback_stacker_llm=None,  # explicit None = --no-stacker-fallback
                    parser_llm=parser_llm,
                )
            )

        # Failure payload must be on disk for resume-from-cache to see it.
        cached = cache.read_stacker_output(qid=3001, arm=ARM_STACK)
        assert cached is not None
        assert cached["success"] is False
        assert cached["reason"] == "stacker_failed_no_fallback"
        assert cached["stacker_prediction"] is None


class TestMedianFallback:
    def test_both_stackers_fail_falls_back_to_median_binary(
        self,
        cache: AblationCache,
        stacker_llm: MagicMock,
        fallback_stacker_llm: MagicMock,
        parser_llm: MagicMock,
    ) -> None:
        forecasters = {
            model_slug_to_filename("openrouter/test/m1"): _binary_payload("openrouter/test/m1", 0.6),
            model_slug_to_filename("openrouter/test/m2"): _binary_payload("openrouter/test/m2", 0.5),
            model_slug_to_filename("openrouter/test/m3"): _binary_payload("openrouter/test/m3", 0.4),
        }

        def _both_fail(*_args: Any, **_kwargs: Any) -> tuple[float, str]:
            raise RuntimeError("both stackers boom")

        with (
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.run_tools_for_forecaster",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.build_cross_model_aggregation",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.stacking.run_stacking_binary",
                new=AsyncMock(side_effect=_both_fail),
            ),
        ):
            payload = _run(
                run_stacker_for_arm(
                    question=_make_binary_q(qid=2001),
                    research_blob="R",
                    forecaster_payloads=forecasters,
                    arm=ARM_STACK,
                    cache=cache,
                    stacker_llm=stacker_llm,
                    fallback_stacker_llm=fallback_stacker_llm,
                    parser_llm=parser_llm,
                )
            )

        assert payload["success"] is True
        assert payload["stacker_model_used"] == "median_fallback"
        # Median of [0.4, 0.5, 0.6] = 0.5
        assert payload["stacker_prediction"]["type"] == "binary"
        assert payload["stacker_prediction"]["prob"] == pytest.approx(0.5)
        # Both failures recorded
        joined_errors = " | ".join(payload["errors"])
        assert "primary" in joined_errors and "fallback" in joined_errors

    def test_both_stackers_fail_falls_back_to_median_mc(
        self,
        cache: AblationCache,
        stacker_llm: MagicMock,
        fallback_stacker_llm: MagicMock,
        parser_llm: MagicMock,
    ) -> None:
        def _both_fail(*_args: Any, **_kwargs: Any) -> tuple[PredictedOptionList, str]:
            raise RuntimeError("both stackers boom")

        with (
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.run_tools_for_forecaster",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.build_cross_model_aggregation",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.stacking.run_stacking_mc",
                new=AsyncMock(side_effect=_both_fail),
            ),
        ):
            payload = _run(
                run_stacker_for_arm(
                    question=_make_mc_q(qid=2002),
                    research_blob="R",
                    forecaster_payloads=_three_mc_forecasters(),
                    arm=ARM_STACK,
                    cache=cache,
                    stacker_llm=stacker_llm,
                    fallback_stacker_llm=fallback_stacker_llm,
                    parser_llm=parser_llm,
                )
            )

        assert payload["success"] is True
        assert payload["stacker_model_used"] == "median_fallback"
        assert payload["stacker_prediction"]["type"] == "multiple_choice"
        # Median of three identical _mc_payload outputs gives same option probs.
        options = {o["option_name"]: o["probability"] for o in payload["stacker_prediction"]["options"]}
        assert set(options.keys()) == {"Red", "Blue"}
        assert sum(options.values()) == pytest.approx(1.0)

    def test_both_stackers_fail_falls_back_to_median_numeric(
        self,
        cache: AblationCache,
        stacker_llm: MagicMock,
        fallback_stacker_llm: MagicMock,
        parser_llm: MagicMock,
    ) -> None:
        def _both_fail(*_args: Any, **_kwargs: Any) -> tuple[Any, str]:
            raise RuntimeError("both stackers boom")

        with (
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.run_tools_for_forecaster",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.build_cross_model_aggregation",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.stacking.run_stacking_numeric",
                new=AsyncMock(side_effect=_both_fail),
            ),
        ):
            payload = _run(
                run_stacker_for_arm(
                    question=_make_numeric_q(qid=2003),
                    research_blob="R",
                    forecaster_payloads=_three_numeric_forecasters(),
                    arm=ARM_STACK,
                    cache=cache,
                    stacker_llm=stacker_llm,
                    fallback_stacker_llm=fallback_stacker_llm,
                    parser_llm=parser_llm,
                )
            )

        assert payload["success"] is True
        assert payload["stacker_model_used"] == "median_fallback"
        assert payload["stacker_prediction"]["type"] == "numeric"
        cdf = payload["stacker_prediction"]["cdf_probabilities"]
        assert len(cdf) == 201
        assert all(math.isfinite(p) for p in cdf)


# ===========================================================================
# M2 — Stacker prompt size guard
#
# 4 forecaster rationales each at 200k chars + a long research blob can
# exceed Claude/GPT context windows. The runner must truncate per-rationale
# (preserving the LAST chars — most likely to hold the conclusion) and
# WARN. Without the guard, the primary stacker fails with
# context_length_exceeded, fallback inherits the same prompt and fails
# too, and both arms lose the question.
# ===========================================================================


class TestStackerPromptSizeGuard:
    def test_oversized_rationales_are_truncated_and_warned(
        self,
        cache: AblationCache,
        stacker_llm: MagicMock,
        fallback_stacker_llm: MagicMock,
        parser_llm: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """4 rationales x 200k chars -> WARNING log + each truncated to a
        per-rationale share of the budget. The truncation must keep the
        LAST chars (conclusion), not the head."""
        import logging  # noqa: PLC0415  - test-local

        big_chunk = "a" * 200_000
        # Distinctive end marker so we can confirm the tail survived truncation.
        rationales: dict[str, dict] = {}
        for idx, model in enumerate(["m1", "m2", "m3", "m4"]):
            payload = _binary_payload(model, 0.5)
            payload["reasoning"] = f"Model: {model}\n\n{big_chunk}\n[END-{idx}]"
            rationales[model_slug_to_filename(f"openrouter/test/{model}")] = payload

        captured_base_texts: list[list[str]] = []

        def _fake_stacker(*args: Any, **kwargs: Any) -> tuple[float, str]:
            captured_base_texts.append(_capture_base_texts(args, kwargs))
            return 0.5, "meta"

        with (
            caplog.at_level(logging.WARNING, logger="metaculus_bot.ablation.run_stacker"),
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.run_tools_for_forecaster",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.build_cross_model_aggregation",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.stacking.run_stacking_binary",
                new=AsyncMock(side_effect=_fake_stacker),
            ),
        ):
            payload = _run(
                run_stacker_for_arm(
                    question=_make_binary_q(qid=1001),
                    research_blob="research",
                    forecaster_payloads=rationales,
                    arm=ARM_STACK,
                    cache=cache,
                    stacker_llm=stacker_llm,
                    fallback_stacker_llm=fallback_stacker_llm,
                    parser_llm=parser_llm,
                )
            )

        assert payload["success"] is True
        assert any("prompt size" in r.message and "truncat" in r.message.lower() for r in caplog.records), (
            f"expected a WARNING about prompt size truncation. Records: {[r.message for r in caplog.records]}"
        )
        assert captured_base_texts, "stacker should have been invoked"
        passed_to_stacker = captured_base_texts[0]
        # Each rationale ended up shorter than the original 200k char body.
        assert all(len(t) < 200_000 for t in passed_to_stacker)
        # Tail-preserving truncation keeps the [END-N] marker visible.
        assert all("[END-" in t for t in passed_to_stacker), (
            "truncation must preserve the conclusion (last chars), not the head"
        )

    def test_small_rationales_pass_through_unchanged(
        self,
        cache: AblationCache,
        stacker_llm: MagicMock,
        fallback_stacker_llm: MagicMock,
        parser_llm: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import logging  # noqa: PLC0415

        captured_base_texts: list[list[str]] = []

        def _fake_stacker(*args: Any, **kwargs: Any) -> tuple[float, str]:
            captured_base_texts.append(_capture_base_texts(args, kwargs))
            return 0.5, "meta"

        with (
            caplog.at_level(logging.WARNING, logger="metaculus_bot.ablation.run_stacker"),
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.run_tools_for_forecaster",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.build_cross_model_aggregation",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.stacking.run_stacking_binary",
                new=AsyncMock(side_effect=_fake_stacker),
            ),
        ):
            _run(
                run_stacker_for_arm(
                    question=_make_binary_q(qid=1002),
                    research_blob="research",
                    forecaster_payloads=_three_binary_forecasters(),
                    arm=ARM_STACK,
                    cache=cache,
                    stacker_llm=stacker_llm,
                    fallback_stacker_llm=fallback_stacker_llm,
                    parser_llm=parser_llm,
                )
            )

        assert not any("prompt size" in r.message for r in caplog.records), (
            "small rationales should not trip the size guard"
        )


# ===========================================================================
# M4 — NaN/inf screen on forecaster + stacker output
#
# A forecaster whose parser emits NaN slips through the existing
# prediction_value=None / errors=[] filter — Python's max(0.02, min(0.98, NaN))
# returns NaN (NaN propagates through min/max). _surviving_forecasters must
# also reject NaN-valued payloads. Same screen on stacker output before
# serialize-to-disk so a NaN result doesn't poison the cache.
# ===========================================================================


class TestNaNFiltering:
    def test_surviving_forecasters_filters_binary_nan(self) -> None:
        from metaculus_bot.ablation.run_stacker import _surviving_forecasters

        forecasters = {
            "m1": _binary_payload("m1", float("nan")),
            "m2": _binary_payload("m2", 0.5),
            "m3": _binary_payload("m3", 0.6),
        }
        surviving = _surviving_forecasters(forecasters)
        assert "m1" not in surviving
        assert set(surviving.keys()) == {"m2", "m3"}

    def test_surviving_forecasters_filters_binary_infinity(self) -> None:
        from metaculus_bot.ablation.run_stacker import _surviving_forecasters

        forecasters = {
            "m1": _binary_payload("m1", float("inf")),
            "m2": _binary_payload("m2", 0.5),
            "m3": _binary_payload("m3", 0.6),
        }
        surviving = _surviving_forecasters(forecasters)
        assert "m1" not in surviving

    def test_surviving_forecasters_filters_mc_nan_option(self) -> None:
        from metaculus_bot.ablation.run_stacker import _surviving_forecasters

        bad_mc = _mc_payload("m1")
        bad_mc["prediction_value"]["options"][0]["probability"] = float("nan")
        forecasters = {
            "m1": bad_mc,
            "m2": _mc_payload("m2"),
            "m3": _mc_payload("m3"),
        }
        surviving = _surviving_forecasters(forecasters)
        assert "m1" not in surviving

    def test_surviving_forecasters_filters_numeric_nan_in_cdf(self) -> None:
        from metaculus_bot.ablation.run_stacker import _surviving_forecasters

        bad_numeric = _numeric_payload("m1", median=50.0)
        bad_numeric["prediction_value"]["cdf_probabilities"][100] = float("nan")
        forecasters = {
            "m1": bad_numeric,
            "m2": _numeric_payload("m2", median=55.0),
            "m3": _numeric_payload("m3", median=60.0),
        }
        surviving = _surviving_forecasters(forecasters)
        assert "m1" not in surviving

    def test_surviving_forecasters_keeps_finite_values(self) -> None:
        from metaculus_bot.ablation.run_stacker import _surviving_forecasters

        forecasters = _three_binary_forecasters()
        surviving = _surviving_forecasters(forecasters)
        assert len(surviving) == 3

    def test_stacker_nan_output_is_recorded_as_failure(
        self,
        cache: AblationCache,
        stacker_llm: MagicMock,
        fallback_stacker_llm: MagicMock,
        parser_llm: MagicMock,
    ) -> None:
        """If both stackers emit NaN, the runner must NOT cache a success
        payload. Per M4 spec we treat NaN stacker output as failure and
        either fall through to the median-fallback path or record an
        error payload."""

        def _nan_stacker(*_args: Any, **_kwargs: Any) -> tuple[float, str]:
            return float("nan"), "meta nan"

        with (
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.run_tools_for_forecaster",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.build_cross_model_aggregation",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.stacking.run_stacking_binary",
                new=AsyncMock(side_effect=_nan_stacker),
            ),
        ):
            payload = _run(
                run_stacker_for_arm(
                    question=_make_binary_q(qid=950),
                    research_blob="R",
                    forecaster_payloads=_three_binary_forecasters(),
                    arm=ARM_STACK,
                    cache=cache,
                    stacker_llm=stacker_llm,
                    fallback_stacker_llm=fallback_stacker_llm,
                    parser_llm=parser_llm,
                )
            )

        assert payload["success"] is False, "NaN stacker output must not be cached as success"
        assert payload["stacker_prediction"] is None, "NaN stacker output must not be persisted"
        # The reason field carries the diagnostic so audit can bucket NaN-vs-other failures.
        assert payload.get("reason") == "stacker_nonfinite_output"
        # Defensive: nothing cached should re-introduce a NaN in any numeric field.
        cached_prob = (payload.get("stacker_prediction") or {}).get("prob")
        assert cached_prob is None or math.isfinite(cached_prob)
