"""Input/output guards of the ablation stacker runner.

Split out of ``test_ablation_run_stacker.py``. Three screens live here: the
minimum-forecaster guard that refuses to stack a single survivor, the NaN/inf screen on
both forecaster payloads and stacker output, and the prompt-size guard that
tail-truncates oversized rationales. Factories come from ``tests/ablation_stacker_fakes.py``,
fixtures from ``tests/ablation_stacker_fixtures.py``.
"""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from metaculus_bot.ablation.cache import AblationCache, model_slug_to_filename
from metaculus_bot.ablation.run_stacker import ARM_STACK, run_stacker_for_arm
from tests import ablation_stacker_fixtures as _fixtures
from tests.ablation_stacker_fakes import (
    _binary_payload,
    _capture_base_texts,
    _make_binary_q,
    _mc_payload,
    _numeric_payload,
    _run,
    _three_binary_forecasters,
)

# pytest registers a fixture under the module attribute name it finds it at, so the shared
# fixtures are RE-BOUND here rather than imported: `import cache as _cache` would register
# `_cache` and leave `cache` falling through to pytest's builtin cache fixture, while a
# plain `import cache` trips ruff F811 against the same-named test-method parameters.
cache = _fixtures.cache
stacker_llm = _fixtures.stacker_llm
fallback_stacker_llm = _fixtures.fallback_stacker_llm
parser_llm = _fixtures.parser_llm
_ensure_flag_unset = _fixtures._ensure_flag_unset


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
        import logging  # HARNESS-SCAN-EXEMPT-function-level-import  # test-local

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
        import logging  # HARNESS-SCAN-EXEMPT-function-level-import

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
        from metaculus_bot.ablation.run_stacker import (  # HARNESS-SCAN-EXEMPT-function-level-import
            _surviving_forecasters,
        )

        forecasters = {
            "m1": _binary_payload("m1", float("nan")),
            "m2": _binary_payload("m2", 0.5),
            "m3": _binary_payload("m3", 0.6),
        }
        surviving = _surviving_forecasters(forecasters)
        assert "m1" not in surviving
        assert set(surviving.keys()) == {"m2", "m3"}

    def test_surviving_forecasters_filters_binary_infinity(self) -> None:
        from metaculus_bot.ablation.run_stacker import (  # HARNESS-SCAN-EXEMPT-function-level-import
            _surviving_forecasters,
        )

        forecasters = {
            "m1": _binary_payload("m1", float("inf")),
            "m2": _binary_payload("m2", 0.5),
            "m3": _binary_payload("m3", 0.6),
        }
        surviving = _surviving_forecasters(forecasters)
        assert "m1" not in surviving

    def test_surviving_forecasters_filters_mc_nan_option(self) -> None:
        from metaculus_bot.ablation.run_stacker import (  # HARNESS-SCAN-EXEMPT-function-level-import
            _surviving_forecasters,
        )

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
        from metaculus_bot.ablation.run_stacker import (  # HARNESS-SCAN-EXEMPT-function-level-import
            _surviving_forecasters,
        )

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
        from metaculus_bot.ablation.run_stacker import (  # HARNESS-SCAN-EXEMPT-function-level-import
            _surviving_forecasters,
        )

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
