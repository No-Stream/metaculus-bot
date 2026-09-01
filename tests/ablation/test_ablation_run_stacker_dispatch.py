"""Question-type dispatch and serialized-payload shape for the ablation stacker runner.

Split out of ``test_ablation_run_stacker.py``. Covers which ``stacking.run_stacking_*``
entry point each question type reaches (including the numeric wrapping through
``sanitize_percentiles`` + ``build_numeric_distribution``) and the canonical dict the
runner writes for binary / MC / numeric predictions. Factories come from
``tests/ablation_stacker_fakes.py``, fixtures from ``tests/ablation/conftest.py``.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from forecasting_tools import PredictedOptionList
from forecasting_tools.data_models.multiple_choice_report import PredictedOption
from forecasting_tools.data_models.numeric_report import Percentile

from metaculus_bot.ablation.cache import AblationCache
from metaculus_bot.ablation.run_stacker import ARM_STACK, run_stacker_for_arm
from metaculus_bot.numeric.pipeline import build_numeric_distribution, sanitize_percentiles
from tests.ablation_stacker_fakes import (
    _make_binary_q,
    _make_mc_q,
    _make_numeric_q,
    _run,
    _three_binary_forecasters,
    _three_mc_forecasters,
    _three_numeric_forecasters,
)

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
            Percentile(percentile=0.01, value=15.0),
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
            Percentile(percentile=0.99, value=85.0),
        ]
        captured_kwargs: dict[str, Any] = {}

        def _fake_numeric(*_args: Any, **kwargs: Any) -> tuple[Any, str]:
            captured_kwargs.update(kwargs)
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
            patch(
                "metaculus_bot.numeric.pipeline.sanitize_percentiles",
                wraps=sanitize_percentiles,
            ) as sanitize_spy,
            patch(
                "metaculus_bot.numeric.pipeline.build_numeric_distribution",
                wraps=build_numeric_distribution,
            ) as build_spy,
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
        # ``stacking.run_stacking_numeric`` takes the bound messages keyword-only, so the
        # dispatcher must name them rather than rely on their slot.
        assert {"lower_bound_message", "upper_bound_message"} <= captured_kwargs.keys(), captured_kwargs.keys()
        lower_msg = captured_kwargs["lower_bound_message"]
        upper_msg = captured_kwargs["upper_bound_message"]
        assert isinstance(lower_msg, str)
        assert isinstance(upper_msg, str)
        # Validate bound messages mention the bounds
        assert "0.0" in lower_msg or "0" in lower_msg
        assert "100" in upper_msg
        assert payload["success"] is True
        # NUMERIC_DEGENERATE_DECLARATION attributes by model_name and the archive reads
        # model=unknown as "a caller forgot to pass it" — the ablation stacker path must
        # wire its own model through, like the prod stacker and forecaster paths do.
        assert sanitize_spy.call_args.kwargs["model_name"] is stacker_llm.model
        # A lost model_name kwarg attributes CDF_MAXSTEP_CLIP to model=unknown silently.
        assert build_spy.call_args.kwargs["model_name"] is stacker_llm.model

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

        We feed an *unsorted, duplicate-laden* 13-percentile list. If wrapping is
        absent, the raw list goes straight to ``serialize_prediction_value``,
        which raises ``TypeError`` (Bucket 1 contract: numeric requires
        ``NumericDistribution``). If wrapping is present, ``sanitize_percentiles``
        sorts by percentile + clamps + deduplicates, and the resulting payload
        carries a sorted ``declared_percentiles`` list with all 13 standard
        percentiles preserved.
        """
        # 13 standard percentiles, deliberately unsorted and with one near-duplicate
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
            Percentile(percentile=0.01, value=15.0),
            Percentile(percentile=0.99, value=85.0),
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
        # All 13 standard percentiles survive (filter_to_standard_percentiles
        # keeps the canonical set; sanitize_percentiles validates count).
        assert len(sp["declared_percentiles"]) == 13
        # sort_percentiles_by_value reorders by ``percentile`` ascending.
        percentile_keys = [
            round(float(p["percentile"]), 6) for p in sp["declared_percentiles"]
        ]  # HARNESS-SCAN-EXEMPT-object-explosion  # tiny test frame (13 percentiles)
        assert percentile_keys == sorted(percentile_keys), (
            f"sanitize_percentiles should sort by percentile; got {percentile_keys}"
        )
        # apply_jitter_for_duplicates / ensure_strictly_increasing_bounded:
        # value-axis must be strictly increasing after sanitization.
        values = [
            float(p["value"]) for p in sp["declared_percentiles"]
        ]  # HARNESS-SCAN-EXEMPT-object-explosion  # tiny test frame (13 percentiles)
        assert all(v_next > v_prev for v_prev, v_next in pairwise(values)), (
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
        import json  # HARNESS-SCAN-EXEMPT-function-level-import

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
        raw 13-Percentile list so we exercise that wrapping, then assert the
        serialized payload still carries declared_percentiles + cdf_probabilities
        + bounds + zero_point + cdf_size.
        """
        question = _make_numeric_q(qid=3)
        # 13 standard percentiles in canonical order — what production
        # ``stacking.run_stacking_numeric`` emits after the parser LLM.
        declared = [
            Percentile(percentile=0.01, value=15.0),
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
            Percentile(percentile=0.99, value=85.0),
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
        import json  # HARNESS-SCAN-EXEMPT-function-level-import

        # numeric (post-Bucket-1): full-CDF schema; JSON-roundtrippable.
        sp = payload["stacker_prediction"]
        assert isinstance(sp, dict)
        assert sp["type"] == "numeric"
        # declared_percentiles: the 13 standard percentiles round-trip after
        # sanitize_percentiles + build_numeric_distribution.
        assert isinstance(sp["declared_percentiles"], list)
        assert len(sp["declared_percentiles"]) == 13
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
        import json  # HARNESS-SCAN-EXEMPT-function-level-import

        sp = payload["stacker_prediction"]
        assert isinstance(sp, dict)
        assert sp["type"] == "multiple_choice"
        assert "options" in sp
        assert isinstance(sp["options"], list)
        # JSON-roundtrippable
        rt = json.loads(json.dumps(sp))
        assert rt == sp
