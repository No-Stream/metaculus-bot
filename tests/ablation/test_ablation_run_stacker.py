# HARNESS-SCAN-EXEMPT-monolithic-file-loc
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

This module keeps the arm/treatment contract itself: the env-var context manager, the
per-arm env-var visibility, and the two places tool output enters the stacker prompt
(per-forecaster "Computed quantities" and the cross-model aggregation block). The rest of
the runner's surface lives in sibling modules, all sharing the factories in
``tests/ablation_stacker_fakes.py`` and the fixtures in ``tests/ablation/conftest.py``:

* ``test_ablation_run_stacker_cache.py`` — cache short-circuit / force + per-stacker slug keying
* ``test_ablation_run_stacker_dispatch.py`` — question-type dispatch + serialized payload shape
* ``test_ablation_run_stacker_fallbacks.py`` — primary→fallback chain, soft deadlines, MEDIAN fallback
* ``test_ablation_run_stacker_guards.py`` — insufficient-forecaster, NaN/inf and prompt-size guards
* ``test_ablation_run_stacker_batch.py`` — ``run_stacker_batch``, window-patch lock, default stacker LLMs
* ``test_ablation_run_stacker_tool_runner.py`` — same contract against the REAL ``tool_runner``
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from metaculus_bot.ablation.cache import AblationCache
from metaculus_bot.ablation.run_stacker import (
    ARM_STACK,
    ARM_STACK_AUG,
    probabilistic_tools_enabled,
    run_stacker_for_arm,
)
from tests.ablation_stacker_fakes import (
    FEATURE_FLAG,
    _capture_base_texts,
    _make_binary_q,
    _run,
    _three_binary_forecasters,
)

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
        with pytest.raises(RuntimeError, match="boom"), probabilistic_tools_enabled(True):  # noqa: PT012  # in-block assert must run inside the live context manager
            assert os.environ[FEATURE_FLAG] == "1"
            raise RuntimeError("boom")
        assert FEATURE_FLAG not in os.environ

    def test_env_var_restored_on_exception_when_previously_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(FEATURE_FLAG, "preset")
        with pytest.raises(ValueError, match="boom"), probabilistic_tools_enabled(False):  # noqa: PT012  # in-block assert must run inside the live context manager
            assert FEATURE_FLAG not in os.environ
            raise ValueError("boom")
        assert os.environ[FEATURE_FLAG] == "preset"


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
            assert not base_text.startswith("Model: "), (
                f"Expected stripped, got: {base_text[:60]!r}"
            )  # HARNESS-SCAN-EXEMPT-subsampling  # display truncation in assert message, not data subsampling

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
