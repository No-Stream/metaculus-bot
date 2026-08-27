"""Cache behavior of the ablation stacker runner: short-circuit, ``force``, per-stacker keying.

Split out of ``test_ablation_run_stacker.py``. Two aspects live here: the plain
cache-hit/``force=True`` contract, and the ``stacker_slug`` keying that keeps one stacker's
``stack``/``stack_aug`` payloads from clobbering another's. Factories come from ``tests/ablation_stacker_fakes.py``,
fixtures from ``tests/ablation/conftest.py``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from metaculus_bot.ablation.cache import AblationCache, model_slug_to_filename
from metaculus_bot.ablation.run_stacker import (
    ARM_STACK,
    ARM_STACK_AUG,
    run_stacker_batch,
    run_stacker_for_arm,
)
from tests.ablation_stacker_fakes import _make_binary_q, _run, _three_binary_forecasters

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
# Per-stacker cache keying (stacker_slug)
#
# stack / stack_aug payloads are keyed by the active stacker so a stacker swap
# (opus-4.5 free-tier vs opus-4.8 prod) never overwrites another stacker's
# results while deterministic arms + forecaster outputs stay shared. The slug
# is supplied by the CLI (derived from the stacker model via
# model_slug_to_filename); these tests pass it explicitly to the runner.
# ===========================================================================


class TestPerStackerCacheKeying:
    def test_stack_arm_writes_slugged_file(
        self,
        cache: AblationCache,
        stacker_llm: MagicMock,
        fallback_stacker_llm: MagicMock,
        parser_llm: MagicMock,
    ) -> None:
        """A slugged stack_aug run lands at arm_stack_aug__<slug>.json, not the bare name."""
        slug = model_slug_to_filename("openrouter/anthropic/claude-opus-4.8")

        def _fake_stacker(*_args: Any, **_kwargs: Any) -> tuple[float, str]:
            return 0.58, "meta"

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
                    question=_make_binary_q(qid=500),
                    research_blob="R",
                    forecaster_payloads=_three_binary_forecasters(),
                    arm=ARM_STACK_AUG,
                    cache=cache,
                    stacker_llm=stacker_llm,
                    fallback_stacker_llm=fallback_stacker_llm,
                    parser_llm=parser_llm,
                    stacker_slug=slug,
                )
            )
        assert payload["success"] is True
        slugged_path = cache.root / "stacker_outputs" / "500" / f"arm_{ARM_STACK_AUG}__{slug}.json"
        assert slugged_path.exists(), "slugged stack_aug run must write the slugged filename"
        # The bare unslugged filename must NOT have been written.
        assert not (cache.root / "stacker_outputs" / "500" / f"arm_{ARM_STACK_AUG}.json").exists()
        # And the runner reads its own slugged cell back on a cache hit.
        assert cache.read_stacker_output(qid=500, arm=ARM_STACK_AUG, stacker_slug=slug) is not None

    def test_different_stacker_models_do_not_clobber(
        self,
        cache: AblationCache,
        stacker_llm: MagicMock,
        fallback_stacker_llm: MagicMock,
        parser_llm: MagicMock,
    ) -> None:
        """Two runs with distinct stacker slugs leave both cells intact.

        Mirrors the prod scenario: a free-tier opus-4.5 run and a prod opus-4.8
        run share one cache dir; each stacker's stack_aug output must survive.
        """
        slug_opus45 = model_slug_to_filename("openrouter/anthropic/claude-opus-4.5")
        slug_opus48 = model_slug_to_filename("openrouter/anthropic/claude-opus-4.8")

        def _run_one_stacker(qid: int, prob: float, meta: str, slug: str) -> None:
            def _fake_stacker(*_args: Any, **_kwargs: Any) -> tuple[float, str]:
                return prob, meta

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
                        question=_make_binary_q(qid=qid),
                        research_blob="R",
                        forecaster_payloads=_three_binary_forecasters(),
                        arm=ARM_STACK_AUG,
                        cache=cache,
                        stacker_llm=stacker_llm,
                        fallback_stacker_llm=fallback_stacker_llm,
                        parser_llm=parser_llm,
                        stacker_slug=slug,
                    )
                )

        _run_one_stacker(qid=501, prob=0.45, meta="opus45", slug=slug_opus45)
        _run_one_stacker(qid=501, prob=0.48, meta="opus48", slug=slug_opus48)

        out_45 = cache.read_stacker_output(qid=501, arm=ARM_STACK_AUG, stacker_slug=slug_opus45)
        out_48 = cache.read_stacker_output(qid=501, arm=ARM_STACK_AUG, stacker_slug=slug_opus48)
        assert out_45 is not None
        assert out_48 is not None
        assert out_45["stacker_prediction"] == {"type": "binary", "prob": 0.45}
        assert out_48["stacker_prediction"] == {"type": "binary", "prob": 0.48}

    def test_slugged_run_does_not_hit_unslugged_cache(
        self,
        cache: AblationCache,
        stacker_llm: MagicMock,
        fallback_stacker_llm: MagicMock,
        parser_llm: MagicMock,
    ) -> None:
        """A pre-existing unslugged free-tier cell must NOT be returned for a slugged run.

        This is the load-bearing footgun the slug eliminates: without keying,
        a fresh prod stacker would cache-HIT a stale free-tier arm_stack_aug.json.
        """
        slug = model_slug_to_filename("openrouter/anthropic/claude-opus-4.8")
        # Seed a stale unslugged payload (the free-tier run's artifact).
        cache.write_stacker_output(
            qid=502,
            arm=ARM_STACK_AUG,
            payload={"success": True, "arm": ARM_STACK_AUG, "stacker_prediction": {"type": "binary", "prob": 0.11}},
        )

        ran = []

        def _fresh_stacker(*_args: Any, **_kwargs: Any) -> tuple[float, str]:
            ran.append(True)
            return 0.77, "fresh prod"

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
                new=AsyncMock(side_effect=_fresh_stacker),
            ),
        ):
            payload = _run(
                run_stacker_for_arm(
                    question=_make_binary_q(qid=502),
                    research_blob="R",
                    forecaster_payloads=_three_binary_forecasters(),
                    arm=ARM_STACK_AUG,
                    cache=cache,
                    stacker_llm=stacker_llm,
                    fallback_stacker_llm=fallback_stacker_llm,
                    parser_llm=parser_llm,
                    stacker_slug=slug,
                )
            )
        # The slugged run computed fresh (did NOT cache-hit the stale unslugged cell).
        assert ran == [True]
        assert payload["stacker_prediction"] == {"type": "binary", "prob": 0.77}
        # The stale free-tier cell is untouched.
        stale = cache.read_stacker_output(qid=502, arm=ARM_STACK_AUG)
        assert stale is not None
        assert stale["stacker_prediction"] == {"type": "binary", "prob": 0.11}

    def test_batch_threads_slug_to_per_question_writes(
        self,
        cache: AblationCache,
        stacker_llm: MagicMock,
        fallback_stacker_llm: MagicMock,
        parser_llm: MagicMock,
    ) -> None:
        """run_stacker_batch forwards stacker_slug so each per-question cell is slugged."""
        slug = model_slug_to_filename("openrouter/anthropic/claude-opus-4.8")
        qid_to_data = {
            60: {
                "question": _make_binary_q(qid=60),
                "research": "R60",
                "forecaster_payloads": _three_binary_forecasters(),
            },
            61: {
                "question": _make_binary_q(qid=61),
                "research": "R61",
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
                    arm=ARM_STACK_AUG,
                    cache=cache,
                    stacker_llm=stacker_llm,
                    fallback_stacker_llm=fallback_stacker_llm,
                    parser_llm=parser_llm,
                    stacker_slug=slug,
                )
            )
        assert set(results.keys()) == {60, 61}
        for qid in (60, 61):
            assert (cache.root / "stacker_outputs" / str(qid) / f"arm_{ARM_STACK_AUG}__{slug}.json").exists()
            assert not (cache.root / "stacker_outputs" / str(qid) / f"arm_{ARM_STACK_AUG}.json").exists()

    @pytest.mark.parametrize(("scenario", "qid"), [("nonfinite", 960), ("median_failed", 961)])
    def test_error_paths_write_only_the_slugged_cell(
        self,
        cache: AblationCache,
        stacker_llm: MagicMock,
        fallback_stacker_llm: MagicMock,
        parser_llm: MagicMock,
        scenario: str,
        qid: int,
    ) -> None:
        """Both ``write_error`` triggers must write ONLY the slugged cell.

        Regression pin for the bug recorded in ``_StackerArmCell``'s docstring: the
        non-finite path once omitted ``stacker_slug`` and wrote the legacy unslugged
        filename, so a resume cache-MISSed the slugged cell and re-spent the paid
        stacker call. ``nonfinite`` drives ``_payload_from_stacker_result`` (the
        stacker returns NaN); ``median_failed`` drives ``_median_fallback_payload``
        (both stackers raise and the median fallback itself raises). The
        directory-equality assertion also catches any EXTRA file a future write
        site might drop beside the cell.
        """
        slug = model_slug_to_filename("openrouter/anthropic/claude-opus-4.8")

        if scenario == "nonfinite":

            def _stacker(*_args: Any, **_kwargs: Any) -> tuple[float, str]:
                return float("nan"), "meta nan"
        else:

            def _stacker(*_args: Any, **_kwargs: Any) -> tuple[float, str]:
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
                new=AsyncMock(side_effect=_stacker),
            ),
            # Only reached in the median_failed scenario (the nonfinite stacker
            # "succeeds", so no fallback fires); harmless otherwise.
            patch(
                "metaculus_bot.ablation.run_stacker._median_fallback_prediction",
                side_effect=RuntimeError("median boom"),
            ),
        ):
            payload = _run(
                run_stacker_for_arm(
                    question=_make_binary_q(qid=qid),
                    research_blob="R",
                    forecaster_payloads=_three_binary_forecasters(),
                    arm=ARM_STACK,
                    cache=cache,
                    stacker_llm=stacker_llm,
                    fallback_stacker_llm=fallback_stacker_llm,
                    parser_llm=parser_llm,
                    stacker_slug=slug,
                )
            )

        assert payload["success"] is False
        expected_reason = "stacker_nonfinite_output" if scenario == "nonfinite" else "stacker_failed"
        assert payload["reason"] == expected_reason
        qdir = cache.root / "stacker_outputs" / str(qid)
        assert sorted(path.name for path in qdir.iterdir()) == [f"arm_{ARM_STACK}__{slug}.json"]
