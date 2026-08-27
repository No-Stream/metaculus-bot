"""Degradation paths of the ablation stacker runner: fallback chain, soft deadlines, MEDIAN.

Split out of ``test_ablation_run_stacker.py``. Covers primary→fallback hand-off (on both a
raised error and a ``STACKER_SOFT_DEADLINE`` timeout), the ``--no-stacker-fallback``
fail-fast contract, and the tertiary MEDIAN fallback that keeps a question publishable when
both stackers fail. Factories come from ``tests/ablation_stacker_fakes.py``,
fixtures from ``tests/ablation_stacker_fixtures.py``.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from forecasting_tools import PredictedOptionList

from metaculus_bot.ablation.cache import AblationCache, model_slug_to_filename
from metaculus_bot.ablation.run_stacker import ARM_STACK, run_stacker_for_arm
from tests import ablation_stacker_fixtures as _fixtures
from tests.ablation_stacker_fakes import (
    _binary_payload,
    _make_binary_q,
    _make_mc_q,
    _make_numeric_q,
    _run,
    _three_binary_forecasters,
    _three_mc_forecasters,
    _three_numeric_forecasters,
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
        from metaculus_bot.ablation import (
            run_stacker as run_stacker_module,  # HARNESS-SCAN-EXEMPT-function-level-import
        )

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
        from metaculus_bot.ablation import (
            run_stacker as run_stacker_module,  # HARNESS-SCAN-EXEMPT-function-level-import
        )

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
        assert "primary" in joined_errors
        assert "fallback" in joined_errors

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
