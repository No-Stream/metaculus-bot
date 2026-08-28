"""The arm A vs arm B contrast against the REAL ``tool_runner`` (no mocks on it).

Split out of ``test_ablation_run_stacker.py``. Everywhere else ``tool_runner`` is patched;
here the synthetic forecaster rationales below carry genuine structured JSON blocks so
``run_tools_for_forecaster`` / ``build_cross_model_aggregation`` actually parse and compute.
That is what proves arm B differs from arm A rather than silently degrading to it. The
rationale builders stay in this module — it is their only consumer; question/payload factories
come from ``tests/ablation_stacker_fakes.py``, fixtures from ``tests/ablation/conftest.py``.
"""

from __future__ import annotations

import textwrap
from unittest.mock import AsyncMock, MagicMock, patch

from forecasting_tools import PredictedOptionList
from forecasting_tools.data_models.multiple_choice_report import PredictedOption
from forecasting_tools.data_models.numeric_report import Percentile

from metaculus_bot.ablation.cache import AblationCache, model_slug_to_filename
from metaculus_bot.ablation.run_stacker import ARM_STACK, ARM_STACK_AUG, run_stacker_for_arm
from tests.ablation_stacker_fakes import _make_binary_q, _make_mc_q, _make_numeric_q, _run

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
    """Synthetic numeric rationale with standard percentiles.

    The declared_percentiles trigger family-consistency and
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
            "0.01": {median - 35}, "0.025": {median - 30}, "0.05": {median - 25},
            "0.1": {median - 20}, "0.2": {median - 12}, "0.4": {median - 5},
            "0.5": {median}, "0.6": {median + 5}, "0.8": {median + 12},
            "0.9": {median + 20}, "0.95": {median + 25}, "0.975": {median + 30},
            "0.99": {median + 35}
          }}
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

    def test_arm_b_with_real_tool_runner_produces_family_and_oob_sections_for_numeric(
        self,
        cache: AblationCache,
        stacker_llm: MagicMock,
        fallback_stacker_llm: MagicMock,
        parser_llm: MagicMock,
    ) -> None:
        """Arm B + real tool_runner + valid numeric JSON with declared_percentiles
        → per-forecaster Percentile-family consistency + Out-of-bounds mass
        subsections AND a cross-model Forecaster-medians aggregation block."""
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
                        {"percentile": 0.01, "value": 15},
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
                        {"percentile": 0.99, "value": 85},
                    ]
                ),
                "reasoning": _numeric_rationale_with_valid_json(50.0),
                "errors": [],
            },
            model_slug_to_filename("openrouter/test/m2"): {
                "model": "openrouter/test/m2",
                "prediction_value": _numeric_pred_payload(
                    [
                        {"percentile": 0.01, "value": 20},
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
                        {"percentile": 0.99, "value": 90},
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
            Percentile(percentile=0.01, value=17.0),
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
            Percentile(percentile=0.99, value=87.0),
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

        # Cross-model agg: medians + declared families.
        assert payload["cross_model_aggregation"], "Real build_cross_model_aggregation produced empty numeric output!"
        agg = payload["cross_model_aggregation"]
        assert "Forecaster medians" in agg, agg

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
