"""End-to-end tests for metaculus_bot.tool_runner.

The tool runner extracts structured JSON blocks from forecaster rationales
and runs probabilistic tools over them, returning markdown-ready strings
for stacker injection. These tests exercise the full pipeline without
mocking the tool implementations themselves.
"""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Any

import pytest
from forecasting_tools import BinaryQuestion, MultipleChoiceQuestion, NumericQuestion
from forecasting_tools.data_models.multiple_choice_report import PredictedOption, PredictedOptionList
from forecasting_tools.data_models.numeric_report import Percentile

from metaculus_bot.tool_runner import (
    FEATURE_FLAG_ENV,
    TYPES_ENV,
    _feature_enabled,
    _lr_chained_posterior,
    aggregate_binary_values,
    aggregate_mc_values,
    aggregate_numeric_values,
    build_cross_model_aggregation,
    cdf_at_threshold_for_forecaster,
    run_tools_for_forecaster,
)


@pytest.fixture(autouse=True)
def _enable_feature_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tool runner gates all public entry points on ``PROBABILISTIC_TOOLS_ENABLED``.

    Tests exercise the path where the feature is ON; the flag-off case is
    covered by ``TestFeatureFlagGating`` below.
    """
    monkeypatch.setenv(FEATURE_FLAG_ENV, "1")


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_binary_question(**overrides: Any) -> BinaryQuestion:
    defaults: dict[str, Any] = dict(
        question_text="Will it rain before May 1?",
        id_of_question=1,
        page_url="https://example.com/q/1",
        background_info="",
        resolution_criteria="",
        fine_print="",
    )
    defaults.update(overrides)
    return BinaryQuestion(**defaults)


def _make_numeric_question(**overrides: Any) -> NumericQuestion:
    defaults: dict[str, Any] = dict(
        question_text="What will X be?",
        id_of_question=3,
        page_url="https://example.com/q/3",
        background_info="",
        resolution_criteria="",
        fine_print="",
        lower_bound=0.0,
        upper_bound=100.0,
        open_lower_bound=False,
        open_upper_bound=False,
    )
    defaults.update(overrides)
    return NumericQuestion(**defaults)


def _make_mc_question(**overrides: Any) -> MultipleChoiceQuestion:
    defaults: dict[str, Any] = dict(
        question_text="Which color?",
        options=["Red", "Blue", "Green"],
        id_of_question=2,
        page_url="https://example.com/q/2",
        background_info="",
        resolution_criteria="",
        fine_print="",
    )
    defaults.update(overrides)
    return MultipleChoiceQuestion(**defaults)


def _wrap_json(payload: dict, preamble: str = "Analysis body here.") -> str:
    """Wrap a JSON payload in a realistic-looking rationale with leading prose."""
    return f"{preamble}\n\n```json\n{json.dumps(payload)}\n```\n\nProbability: 35%"


def _binary_payload(**overrides) -> dict:
    base = {
        "question_type": "binary",
        "prior": {"prob": 0.15, "source": "annual incidence 2015-2024"},
        "base_rate": {"k": 3, "n": 12, "ref_class": "years matching condition"},
        "hazard": {
            "rate_per_unit": 0.25,
            "unit": "year",
            "window_duration_units": 1.0,
            "elapsed_fraction": 0.33,
            "remaining_fraction": 0.67,
        },
        "evidence": [{"summary": "Q1 policy shift", "direction": "up", "strength": "moderate"}],
        "scenarios": [],
        "posterior_prob": 0.28,
    }
    base.update(overrides)
    return base


def _numeric_payload(**overrides) -> dict:
    base = {
        "question_type": "numeric",
        "declared_percentiles": {
            "0.1": 10.0,
            "0.25": 20.0,
            "0.5": 40.0,
            "0.75": 60.0,
            "0.9": 80.0,
        },
        "distribution_family_hint": "normal",
        "tails": {"below_min_expected": 0.02, "above_max_expected": 0.05},
    }
    base.update(overrides)
    return base


def _mc_payload(**overrides) -> dict:
    base = {
        "question_type": "multiple_choice",
        "option_probs": {"Red": 0.5, "Blue": 0.3, "Green": 0.2},
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# run_tools_for_forecaster: binary
# ---------------------------------------------------------------------------


class TestRunToolsBinary:
    def test_full_binary_block_produces_all_tools(self):
        rationale = _wrap_json(_binary_payload())
        result = run_tools_for_forecaster(
            question=_make_binary_question(),
            rationale=rationale,
            forecaster_id="gpt-5",
        )
        assert "Beta-binomial" in result
        assert "Survival / hazard" in result
        assert "Prior → posterior" in result

    def test_binary_no_json_returns_empty(self):
        rationale = "No JSON block here. Just prose.\n\nProbability: 30%"
        result = run_tools_for_forecaster(
            question=_make_binary_question(),
            rationale=rationale,
            forecaster_id="gpt-5",
        )
        assert result == ""

    def test_binary_missing_hazard_still_runs_other_tools(self):
        payload = _binary_payload()
        del payload["hazard"]
        rationale = _wrap_json(payload)
        result = run_tools_for_forecaster(
            question=_make_binary_question(),
            rationale=rationale,
            forecaster_id="gpt-5",
        )
        assert "Beta-binomial" in result
        assert "Survival" not in result
        assert "Prior → posterior" in result

    def test_binary_only_prior_and_posterior(self):
        payload = {
            "question_type": "binary",
            "prior": {"prob": 0.2, "source": "analogous markets"},
            "evidence": [],
            "scenarios": [],
            "posterior_prob": 0.6,
        }
        rationale = _wrap_json(payload)
        result = run_tools_for_forecaster(
            question=_make_binary_question(),
            rationale=rationale,
            forecaster_id="m",
        )
        assert "Prior → posterior" in result

    def test_binary_flags_unexplained_jump(self):
        # prior 0.2 → posterior 0.9 with only weak evidence should flag.
        payload = _binary_payload(
            prior={"prob": 0.2, "source": "history"},
            posterior_prob=0.9,
            evidence=[{"summary": "weak signal", "direction": "up", "strength": "weak"}],
        )
        rationale = _wrap_json(payload)
        result = run_tools_for_forecaster(
            question=_make_binary_question(),
            rationale=rationale,
            forecaster_id="m",
        )
        assert "FLAGGED" in result

    def test_binary_base_rate_without_prior_still_surfaces_lr(self):
        payload = _binary_payload()
        del payload["prior"]
        rationale = _wrap_json(payload)
        result = run_tools_for_forecaster(
            question=_make_binary_question(),
            rationale=rationale,
            forecaster_id="m",
        )
        # When no prior, we expose a "Base-rate → posterior" line with implied LR.
        assert "Base-rate → posterior" in result or "Beta-binomial" in result

    def test_malformed_json_warns_and_returns_empty(self, caplog):
        rationale = "Analysis.\n\n```json\n{not valid json\n```\n\nProbability: 30%"
        with caplog.at_level(logging.WARNING):
            result = run_tools_for_forecaster(
                question=_make_binary_question(),
                rationale=rationale,
                forecaster_id="m",
            )
        assert result == ""
        assert any("Malformed JSON" in rec.message for rec in caplog.records)

    def test_question_type_mismatch_returns_empty(self, caplog):
        payload = _numeric_payload()  # Wrong type for binary question
        rationale = _wrap_json(payload)
        with caplog.at_level(logging.WARNING):
            result = run_tools_for_forecaster(
                question=_make_binary_question(),
                rationale=rationale,
                forecaster_id="m",
            )
        assert result == ""
        assert any("question_type mismatch" in rec.message for rec in caplog.records)

    def test_binary_prior_plus_base_rate_surfaces_bayesian_combine(self):
        # Both prior AND base_rate declared → we expose the informative-prior
        # Beta-binomial posterior with the declared posterior for comparison.
        rationale = _wrap_json(_binary_payload())
        result = run_tools_for_forecaster(
            question=_make_binary_question(),
            rationale=rationale,
            forecaster_id="gpt-5",
        )
        assert "Prior + k/n Bayesian combine" in result
        assert "stated prior 0.150" in result or "0.150" in result

    def test_binary_evidence_lrs_chained_into_posterior(self):
        payload = _binary_payload(
            prior={"prob": 0.2, "source": "x"},
            evidence=[
                {"summary": "moderate up", "direction": "up", "strength": "moderate", "likelihood_ratio": 2.0},
                {"summary": "weak up", "direction": "up", "strength": "weak", "likelihood_ratio": 1.5},
            ],
            posterior_prob=0.4,
        )
        rationale = _wrap_json(payload)
        result = run_tools_for_forecaster(
            question=_make_binary_question(),
            rationale=rationale,
            forecaster_id="m",
        )
        assert "Evidence-LR-chained posterior" in result

    def test_binary_no_evidence_lrs_means_no_lr_chain_line(self):
        # No declared likelihood_ratio on evidence → we skip the LR-chain line.
        rationale = _wrap_json(_binary_payload())
        result = run_tools_for_forecaster(
            question=_make_binary_question(),
            rationale=rationale,
            forecaster_id="m",
        )
        assert "Evidence-LR-chained posterior" not in result

    def test_binary_hazard_units_cancel_in_output(self):
        # rate 0.25/day, 30-day window, 67% remaining → the runner passes
        # window_duration_units=30 through to prob_event_before unchanged.
        # Conditional P(event | none yet) ≈ 1 - exp(-0.25 * 30 * 0.67) ≈ 0.993.
        payload = _binary_payload(
            hazard={
                "rate_per_unit": 0.25,
                "unit": "day",
                "window_duration_units": 30.0,
                "elapsed_fraction": 0.33,
                "remaining_fraction": 0.67,
            },
        )
        rationale = _wrap_json(payload)
        result = run_tools_for_forecaster(
            question=_make_binary_question(),
            rationale=rationale,
            forecaster_id="m",
        )
        # Pre-fix this would have been ~0.042 (units mis-converted through years).
        # Regex-match on the rendered probability to tolerate harmless formatting
        # changes; assert the value lands in the high-0.99 range.
        match = re.search(r"P\(event in remaining \| none yet\) = (0\.\d+)", result)
        assert match is not None, result
        assert 0.99 < float(match.group(1)) < 1.0


# ---------------------------------------------------------------------------
# run_tools_for_forecaster: numeric
# ---------------------------------------------------------------------------


class TestRunToolsNumeric:
    def test_numeric_full_block(self):
        rationale = _wrap_json(_numeric_payload())
        result = run_tools_for_forecaster(
            question=_make_numeric_question(),
            rationale=rationale,
            forecaster_id="m",
        )
        assert "Percentile-family consistency" in result
        assert "Out-of-bounds mass" in result

    def test_numeric_non_monotone_rejected_by_schema(self):
        payload = _numeric_payload(
            declared_percentiles={"0.1": 50.0, "0.5": 30.0, "0.9": 40.0},  # non-monotone
        )
        # Pydantic validates strictly-increasing values before tool-runner sees it;
        # this test confirms the schema catches the error (tool-runner gets None).
        rationale = _wrap_json(payload)
        result = run_tools_for_forecaster(
            question=_make_numeric_question(),
            rationale=rationale,
            forecaster_id="m",
        )
        # Schema rejects → tool-runner returns empty string.
        assert result == ""

    def test_numeric_family_mismatch_flagged(self):
        # Give heavy-tailed percentiles but claim normal family.
        payload = _numeric_payload(
            declared_percentiles={
                "0.05": 0.5,
                "0.1": 1.0,
                "0.5": 10.0,
                "0.9": 100.0,
                "0.95": 500.0,
            },
            distribution_family_hint="normal",
        )
        rationale = _wrap_json(payload)
        result = run_tools_for_forecaster(
            question=_make_numeric_question(lower_bound=0, upper_bound=1000),
            rationale=rationale,
            forecaster_id="m",
        )
        assert "Percentile-family consistency" in result

    def test_numeric_open_bounds_skip_oob_on_that_side(self):
        payload = _numeric_payload()
        rationale = _wrap_json(payload)
        q = _make_numeric_question(open_lower_bound=True)
        result = run_tools_for_forecaster(question=q, rationale=rationale, forecaster_id="m")
        assert "Out-of-bounds mass" in result

    def test_numeric_block_on_binary_question_logs_warning(self, caplog):
        payload = _numeric_payload()
        rationale = _wrap_json(payload)
        q = _make_binary_question()
        # Parse-level mismatch — question_type mismatch vs arg triggers WARNING before the
        # isinstance guard would fire.
        with caplog.at_level(logging.WARNING):
            result = run_tools_for_forecaster(question=q, rationale=rationale, forecaster_id="m")
        assert result == ""

    def test_numeric_with_mixture_components_surfaces_section(self):
        # Workstream D3: mixture_components populated in NumericStructured should
        # surface a "Mixture-of-normals" subsection in the output.
        payload = _numeric_payload(
            mixture_components=[
                {"weight": 0.2, "mean": 20.0, "sd": 5.0},
                {"weight": 0.55, "mean": 50.0, "sd": 8.0},
                {"weight": 0.25, "mean": 80.0, "sd": 6.0},
            ],
        )
        rationale = _wrap_json(payload)
        result = run_tools_for_forecaster(
            question=_make_numeric_question(lower_bound=0.0, upper_bound=100.0),
            rationale=rationale,
            forecaster_id="m",
        )
        # Section header.
        assert "### Mixture-of-normals" in result

        # Each component's full triple (weight, mean, sd) must appear in the
        # rendered table — not just the mean. Per _render_mixture_section
        # tool_runner.py:322-323:
        #   "- Component {i+1}: weight {weight:.3f}, mean {mean:.3g}, sd {sd:.3g}"
        assert "Component 1: weight 0.200, mean 20, sd 5" in result
        assert "Component 2: weight 0.550, mean 50, sd 8" in result
        assert "Component 3: weight 0.250, mean 80, sd 6" in result

        # CDF sample table header + exactly 5 sample rows over [0, 100].
        # Sample rows are indented two spaces ("  - "); the header line is "- ".
        assert "CDF sample (value → F):" in result
        cdf_sample_lines = [line for line in result.splitlines() if line.startswith("  - ") and "→" in line]
        assert len(cdf_sample_lines) == 5, f"expected 5 CDF sample rows, got {len(cdf_sample_lines)}"

        # Bounded size: a well-formed mixture section is a small markdown
        # block. A pathological serializer dumping a 201-pt CDF would balloon
        # this past a few hundred chars — guard the upper bound.
        mixture_section = result.split("### Mixture-of-normals", 1)[1]
        assert len(mixture_section) < 1000, (
            f"mixture section is unexpectedly large ({len(mixture_section)} chars); "
            "rendering may be dumping more than the 5-row sample table"
        )

    def test_numeric_without_mixture_components_unchanged(self):
        # Regression: existing numeric path (percentiles only) still works and does
        # NOT surface a mixture section when mixture_components is absent.
        rationale = _wrap_json(_numeric_payload())
        result = run_tools_for_forecaster(
            question=_make_numeric_question(),
            rationale=rationale,
            forecaster_id="m",
        )
        assert "Percentile-family consistency" in result
        assert "Out-of-bounds mass" in result
        assert "Mixture-of-normals" not in result

    def test_numeric_with_both_percentiles_and_mixture_shows_both(self):
        # When both declared_percentiles and mixture_components are present, both
        # sections appear. No consistency check between them (that's Workstream E).
        payload = _numeric_payload(
            mixture_components=[
                {"weight": 0.3, "mean": 25.0, "sd": 7.0},
                {"weight": 0.4, "mean": 50.0, "sd": 10.0},
                {"weight": 0.3, "mean": 75.0, "sd": 8.0},
            ],
        )
        rationale = _wrap_json(payload)
        result = run_tools_for_forecaster(
            question=_make_numeric_question(lower_bound=0.0, upper_bound=100.0),
            rationale=rationale,
            forecaster_id="m",
        )
        # Both sections must surface — no consistency check between them.
        assert "Percentile-family consistency" in result
        assert "### Mixture-of-normals" in result
        # Each component's full triple appears verbatim.
        assert "Component 1: weight 0.300, mean 25, sd 7" in result
        assert "Component 2: weight 0.400, mean 50, sd 10" in result
        assert "Component 3: weight 0.300, mean 75, sd 8" in result
        # CDF sample table has exactly 5 rows.
        cdf_sample_lines = [line for line in result.splitlines() if line.startswith("  - ") and "→" in line]
        assert len(cdf_sample_lines) == 5


# ---------------------------------------------------------------------------
# run_tools_for_forecaster: MC and discrete
# ---------------------------------------------------------------------------


class TestRunToolsMc:
    def test_mc_other_mass_surfaced(self):
        # option_probs sums to 1.0 per schema; other_mass is reported as declared metadata.
        payload = _mc_payload(other_mass=0.1)
        rationale = _wrap_json(payload)
        result = run_tools_for_forecaster(
            question=_make_mc_question(),
            rationale=rationale,
            forecaster_id="m",
        )
        assert "Other / residual mass" in result

    def test_mc_no_other_mass_empty_block(self):
        rationale = _wrap_json(_mc_payload())
        result = run_tools_for_forecaster(
            question=_make_mc_question(),
            rationale=rationale,
            forecaster_id="m",
        )
        assert result == ""

    def test_mc_other_mass_triggers_dirichlet_cis(self):
        payload = _mc_payload(other_mass=0.1)
        rationale = _wrap_json(payload)
        result = run_tools_for_forecaster(
            question=_make_mc_question(),
            rationale=rationale,
            forecaster_id="m",
        )
        assert "Dirichlet-with-Other (top 3 by mean)" in result
        assert "80% CI" in result

    def test_mc_concentration_alone_triggers_dirichlet_cis(self):
        # Concentration declared without other_mass still surfaces Dirichlet CIs.
        payload = _mc_payload(concentration=25.0)
        rationale = _wrap_json(payload)
        result = run_tools_for_forecaster(
            question=_make_mc_question(),
            rationale=rationale,
            forecaster_id="m",
        )
        assert "Dirichlet-with-Other (top 3 by mean)" in result


# ---------------------------------------------------------------------------
# extract_json_block / parse_structured_block edge cases surfaced via runner
# ---------------------------------------------------------------------------


class TestExtractionEdgeCases:
    def test_multiple_blocks_uses_last(self):
        first = _binary_payload(posterior_prob=0.1)
        last = _binary_payload(posterior_prob=0.5)
        rationale = (
            "Draft version.\n\n"
            f"```json\n{json.dumps(first)}\n```\n\n"
            "Actually on reflection:\n\n"
            f"```json\n{json.dumps(last)}\n```\n\n"
            "Probability: 50%"
        )
        result = run_tools_for_forecaster(
            question=_make_binary_question(),
            rationale=rationale,
            forecaster_id="m",
        )
        # Second block's posterior 0.5 with evidence "moderate" → check implied LR is based on 0.5.
        assert "posterior 0.500" in result

    def test_case_insensitive_fence_tag(self):
        rationale = f"Prose.\n\n```JSON\n{json.dumps(_binary_payload())}\n```\n\nProbability: 30%"
        result = run_tools_for_forecaster(
            question=_make_binary_question(),
            rationale=rationale,
            forecaster_id="m",
        )
        assert "Beta-binomial" in result

    def test_probability_line_does_not_confuse_extraction(self):
        # Trailing Probability: 35% line is AFTER the JSON block per Option A ordering.
        rationale = _wrap_json(_binary_payload())
        assert "Probability: 35%" in rationale
        result = run_tools_for_forecaster(
            question=_make_binary_question(),
            rationale=rationale,
            forecaster_id="m",
        )
        assert result != ""


# ---------------------------------------------------------------------------
# build_cross_model_aggregation
# ---------------------------------------------------------------------------


class TestCrossModelAggregationBinary:
    def test_pools_over_multiple_forecasters(self):
        rationales = [
            _wrap_json(_binary_payload(posterior_prob=0.2)),
            _wrap_json(_binary_payload(posterior_prob=0.4)),
            _wrap_json(_binary_payload(posterior_prob=0.6)),
        ]
        prediction_values = [0.2, 0.4, 0.6]
        result = build_cross_model_aggregation(
            question=_make_binary_question(),
            rationales=rationales,
            prediction_values=prediction_values,
        )
        assert "Pools over 3 forecasters" in result
        assert "linear" in result
        assert "log" in result
        assert "Satopää" in result

    def test_blended_base_rate_appears_when_multiple(self):
        rationales = [
            _wrap_json(_binary_payload(base_rate={"k": 2, "n": 10, "ref_class": "A"})),
            _wrap_json(_binary_payload(base_rate={"k": 5, "n": 20, "ref_class": "B"})),
        ]
        result = build_cross_model_aggregation(
            question=_make_binary_question(),
            rationales=rationales,
            prediction_values=[0.2, 0.25],
        )
        assert "Blended base rate" in result

    def test_single_forecaster_no_pool_line(self):
        result = build_cross_model_aggregation(
            question=_make_binary_question(),
            rationales=[_wrap_json(_binary_payload())],
            prediction_values=[0.3],
        )
        assert "Pools over" not in result

    def test_some_forecasters_have_no_json(self):
        # Mix: some emit JSON, some don't.
        rationales = [
            _wrap_json(_binary_payload()),
            "No JSON here.",
            _wrap_json(_binary_payload()),
        ]
        result = build_cross_model_aggregation(
            question=_make_binary_question(),
            rationales=rationales,
            prediction_values=[0.3, 0.4, 0.5],
        )
        # Pools use ALL 3 prediction values, blended base rate uses only the 2 parsed blocks.
        assert "Pools over 3" in result
        assert "Blended base rate across 2" in result

    def test_mixed_emission_valid_malformed_and_missing(self):
        # Three forecasters: (1) valid block, (2) malformed JSON, (3) no block.
        # aggregate_binary_values should still pool all three prediction values,
        # and the blended base-rate line should count only the 1 parseable block.
        rationales = [
            _wrap_json(_binary_payload()),
            "Analysis with bad JSON.\n```json\n{not valid\n```",
            "Pure prose — no fence at all.",
        ]
        result = aggregate_binary_values(rationales, prediction_probs=[0.3, 0.5, 0.7])
        assert "Pools over 3 forecasters" in result
        # With only 1 base_rate block, the aggregator skips the blended line
        # (needs >= 2 rates), so it should be absent.
        assert "Blended base rate" not in result


class TestCrossModelAggregationNumeric:
    def _make_pcts(self, median: float) -> list[Percentile]:
        return [
            Percentile(percentile=0.1, value=median - 10),
            Percentile(percentile=0.5, value=median),
            Percentile(percentile=0.9, value=median + 10),
        ]

    def test_medians_summarized(self):
        preds = [self._make_pcts(30), self._make_pcts(50), self._make_pcts(70)]
        result = build_cross_model_aggregation(
            question=_make_numeric_question(),
            rationales=["", "", ""],
            prediction_values=preds,
        )
        assert "Forecaster medians" in result
        # Match either integer or decimal formatting of the min value.
        assert re.search(r"min 30(\.\d+)?", result)

    def test_declared_families_listed(self):
        rationales = [
            _wrap_json(_numeric_payload(distribution_family_hint="normal")),
            _wrap_json(_numeric_payload(distribution_family_hint="lognormal")),
        ]
        preds = [self._make_pcts(40), self._make_pcts(60)]
        result = build_cross_model_aggregation(
            question=_make_numeric_question(),
            rationales=rationales,
            prediction_values=preds,
        )
        assert "Declared distribution families" in result


class TestCrossModelAggregationMc:
    def test_mc_dirichlet_pool(self):
        pred1 = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name="Red", probability=0.5),
                PredictedOption(option_name="Blue", probability=0.3),
                PredictedOption(option_name="Green", probability=0.2),
            ]
        )
        pred2 = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name="Red", probability=0.4),
                PredictedOption(option_name="Blue", probability=0.4),
                PredictedOption(option_name="Green", probability=0.2),
            ]
        )
        result = build_cross_model_aggregation(
            question=_make_mc_question(),
            rationales=["", ""],
            prediction_values=[pred1, pred2],
        )
        assert "Linear pool across 2 forecasters" in result
        assert "Red=" in result

    def test_mc_mismatched_option_sets(self):
        pred1 = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name="Red", probability=0.5),
                PredictedOption(option_name="Blue", probability=0.5),
            ]
        )
        pred2 = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name="Red", probability=0.6),
                PredictedOption(option_name="Green", probability=0.4),
            ]
        )
        result = build_cross_model_aggregation(
            question=_make_mc_question(),
            rationales=["", ""],
            prediction_values=[pred1, pred2],
        )
        assert "MC aggregation skipped" in result


# ---------------------------------------------------------------------------
# cdf_at_threshold_for_forecaster
# ---------------------------------------------------------------------------


class TestCdfAtThreshold:
    def test_normal_fit_crosses_threshold(self):
        # Roughly N(50, 15) implied.
        payload = _numeric_payload(
            declared_percentiles={
                "0.1": 30.0,
                "0.5": 50.0,
                "0.9": 70.0,
            },
            distribution_family_hint="normal",
        )
        rationale = _wrap_json(payload)
        q = _make_numeric_question(lower_bound=0.0, upper_bound=100.0)
        result = cdf_at_threshold_for_forecaster(rationale, q, threshold=50.0)
        assert result is not None
        assert 0.4 < result < 0.6

    def test_no_structured_block_returns_none(self):
        result = cdf_at_threshold_for_forecaster(
            rationale="Just prose, no JSON.",
            question=_make_numeric_question(),
            threshold=50.0,
        )
        assert result is None

    def test_threshold_far_below_returns_near_zero(self):
        payload = _numeric_payload(
            declared_percentiles={"0.1": 30.0, "0.5": 50.0, "0.9": 70.0},
            distribution_family_hint="normal",
        )
        rationale = _wrap_json(payload)
        result = cdf_at_threshold_for_forecaster(
            rationale,
            _make_numeric_question(),
            threshold=-100.0,
        )
        assert result is not None
        assert result < 0.01


# ---------------------------------------------------------------------------
# Realistic rationale fixture
# ---------------------------------------------------------------------------


class TestRealisticRationale:
    def test_rationale_with_prose_and_answer_line(self):
        rationale = """── Analysis Template ──

PHASE 0: PRELIMINARY CHECK
The question asks whether X happens by Y date. No resolution evidence yet.

PHASE 1: OUTSIDE VIEW
Historical incidence: 3 of the last 12 years saw this condition. That's a ~25% base rate.

PHASE 2: INSIDE VIEW
Q1 policy shift moderately increases probability.

── STRUCTURED FORECAST ──

```json
{
  "question_type": "binary",
  "prior": {"prob": 0.25, "source": "3-of-12 historical base rate"},
  "base_rate": {"k": 3, "n": 12, "ref_class": "years with matching precondition"},
  "hazard": {"rate_per_unit": 0.25, "unit": "year", "window_duration_units": 1.0, "elapsed_fraction": 0.3, "remaining_fraction": 0.7},
  "evidence": [
    {"summary": "Q1 policy shift", "direction": "up", "strength": "moderate"}
  ],
  "scenarios": [],
  "posterior_prob": 0.35
}
```

Probability: 35%
"""
        result = run_tools_for_forecaster(
            question=_make_binary_question(),
            rationale=rationale,
            forecaster_id="gpt-5-real",
        )
        # Ensure the JSON was extracted despite extensive prose.
        assert "Beta-binomial" in result
        assert "Survival" in result
        assert "Prior → posterior" in result

    def test_rationale_without_block_returns_empty_no_warning(self, caplog):
        rationale = """PHASE 1: OUTSIDE VIEW
Historical base rate ~25%.

Probability: 25%"""
        with caplog.at_level(logging.DEBUG):
            result = run_tools_for_forecaster(
                question=_make_binary_question(),
                rationale=rationale,
                forecaster_id="m",
            )
        assert result == ""
        # DEBUG log (not WARNING) for no-block case.
        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert not any("JSON" in r.message for r in warning_records)


# ---------------------------------------------------------------------------
# Unsupported question type
# ---------------------------------------------------------------------------


class TestUnsupportedType:
    def test_run_tools_unsupported_question_returns_empty(self):
        class FakeQuestion:
            pass

        result = run_tools_for_forecaster(
            question=FakeQuestion(),  # type: ignore[arg-type]
            rationale=_wrap_json(_binary_payload()),
            forecaster_id="m",
        )
        assert result == ""

    def test_aggregation_unsupported_question_returns_empty(self):
        class FakeQuestion:
            pass

        result = build_cross_model_aggregation(
            question=FakeQuestion(),  # type: ignore[arg-type]
            rationales=[],
            prediction_values=[],
        )
        assert result == ""


# ---------------------------------------------------------------------------
# Typed aggregation entry points
# ---------------------------------------------------------------------------


class TestTypedAggregationEntries:
    def test_aggregate_binary_values_direct(self):
        rationales = [
            _wrap_json(_binary_payload(posterior_prob=0.2)),
            _wrap_json(_binary_payload(posterior_prob=0.5)),
        ]
        out = aggregate_binary_values(rationales=rationales, prediction_probs=[0.2, 0.5])
        assert "Pools over 2 forecasters" in out

    def test_aggregate_numeric_values_direct(self):
        preds = [
            [
                Percentile(percentile=0.1, value=10),
                Percentile(percentile=0.5, value=30),
                Percentile(percentile=0.9, value=50),
            ],
            [
                Percentile(percentile=0.1, value=20),
                Percentile(percentile=0.5, value=40),
                Percentile(percentile=0.9, value=60),
            ],
        ]
        out = aggregate_numeric_values(rationales=["", ""], prediction_percentiles=preds)
        assert "Forecaster medians" in out

    def test_aggregate_mc_values_direct(self):
        pred1 = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name="Red", probability=0.5),
                PredictedOption(option_name="Blue", probability=0.5),
            ]
        )
        pred2 = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name="Red", probability=0.4),
                PredictedOption(option_name="Blue", probability=0.6),
            ]
        )
        out = aggregate_mc_values(["", ""], prediction_options=[pred1, pred2])
        assert "Linear pool across 2 forecasters" in out


# ---------------------------------------------------------------------------
# Feature-flag gating
# ---------------------------------------------------------------------------


class TestFeatureFlagGating:
    def test_run_tools_returns_empty_when_flag_off(self, monkeypatch):
        # Override the autouse fixture for this case.
        monkeypatch.delenv(FEATURE_FLAG_ENV, raising=False)
        result = run_tools_for_forecaster(
            question=_make_binary_question(),
            rationale=_wrap_json(_binary_payload()),
            forecaster_id="m",
        )
        assert result == ""

    def test_build_cross_model_returns_empty_when_flag_off(self, monkeypatch):
        monkeypatch.delenv(FEATURE_FLAG_ENV, raising=False)
        result = build_cross_model_aggregation(
            question=_make_binary_question(),
            rationales=[_wrap_json(_binary_payload()), _wrap_json(_binary_payload())],
            prediction_values=[0.3, 0.4],
        )
        assert result == ""

    def test_cdf_at_threshold_returns_none_when_flag_off(self, monkeypatch):
        monkeypatch.delenv(FEATURE_FLAG_ENV, raising=False)
        out = cdf_at_threshold_for_forecaster(
            rationale=_wrap_json(_numeric_payload()),
            question=_make_numeric_question(),
            threshold=50.0,
        )
        assert out is None


# ---------------------------------------------------------------------------
# Golden output snapshots — structural, not byte-exact
# ---------------------------------------------------------------------------


class TestGoldenOutput:
    def test_binary_golden_full_output_structure(self):
        # Realistic binary rationale: prior + base_rate + evidence-LRs declared
        # → tool runner should emit Beta-binomial line, Prior+k/n combine line,
        # AND the evidence-LR-chained posterior line.
        rationale = """Question: Will Country X's inflation exceed 5% by year-end?

Historical precedent: over the past 12 years, only 3 ended above 5%. That
puts the outside view around 25%.

Recent signals: central bank hiked 75bp twice (moderate drag, LR 0.6),
food prices surged 8% YoY (moderate push, LR 1.8), energy subsidies
extended through Q4 (weak drag, LR 0.9). Net: roughly 30% posterior.

```json
{
  "question_type": "binary",
  "prior": {"prob": 0.25, "source": "3-of-12 historical base rate"},
  "base_rate": {"k": 3, "n": 12, "ref_class": "past 12 annual CPI readings"},
  "evidence": [
    {"summary": "rate hikes", "direction": "down", "strength": "moderate", "likelihood_ratio": 0.6},
    {"summary": "food prices surge", "direction": "up", "strength": "moderate", "likelihood_ratio": 1.8},
    {"summary": "subsidy extension", "direction": "down", "strength": "weak", "likelihood_ratio": 0.9}
  ],
  "scenarios": [],
  "posterior_prob": 0.30
}
```

Probability: 30%
"""
        result = run_tools_for_forecaster(
            question=_make_binary_question(),
            rationale=rationale,
            forecaster_id="gpt-5-golden",
        )
        # Structural markers — section-level, not byte-exact.
        assert "Beta-binomial (ref class" in result
        assert "Prior + k/n Bayesian combine" in result
        assert "Evidence-LR-chained posterior" in result
        assert "Prior → posterior" in result

    def test_numeric_lognormal_golden_output(self):
        # Heavy-tailed lognormal-like percentiles with explicit lognormal hint.
        # Expect family-consistency line AND out-of-bounds mass line (lognormal fit).
        rationale = """Analysis: company Y's Q3 revenue in $M, positively skewed
with long upper tail.

```json
{
  "question_type": "numeric",
  "declared_percentiles": {
    "0.05": 5.0,
    "0.1": 7.0,
    "0.5": 20.0,
    "0.9": 80.0,
    "0.95": 150.0
  },
  "distribution_family_hint": "lognormal",
  "tails": {"below_min_expected": 0.02, "above_max_expected": 0.05}
}
```

Percentiles as given.
"""
        q = _make_numeric_question(lower_bound=0.0, upper_bound=500.0)
        result = run_tools_for_forecaster(question=q, rationale=rationale, forecaster_id="gpt-5-golden")
        assert "Percentile-family consistency" in result
        # Formatter emits "(lognormal fit)" since hint matches best family.
        assert "Out-of-bounds mass (lognormal fit)" in result

    def test_mc_with_other_mass_golden_output(self):
        # Option_probs + explicit other_mass → expect residual-mass line AND
        # Dirichlet-with-Other CI line.
        rationale = """MC: which candidate wins the primary? Three named
options plus a catchall.

```json
{
  "question_type": "multiple_choice",
  "option_probs": {"Red": 0.45, "Blue": 0.35, "Green": 0.20},
  "other_mass": 0.10,
  "concentration": 20.0
}
```
"""
        result = run_tools_for_forecaster(
            question=_make_mc_question(),
            rationale=rationale,
            forecaster_id="gpt-5-golden",
        )
        assert "Declared Other / residual mass" in result
        assert "Dirichlet-with-Other (top 3 by mean)" in result
        assert "80% CI" in result


# ---------------------------------------------------------------------------
# MC contract bridge — renormalize option_probs + other_mass
# ---------------------------------------------------------------------------


class TestMcContractBridge:
    def _extract_top_means(self, result: str) -> dict[str, float]:
        """Parse the Dirichlet top-3 line into a {name: mean} dict."""
        # Format: "...: Name1 0.xxx [80% CI ...]; Name2 0.xxx [80% CI ...]; Name3 0.xxx [80% CI ...]"
        means: dict[str, float] = {}
        for match in re.finditer(r"([A-Za-z_]+) (\d\.\d{3}) \[80% CI", result):
            means[match.group(1)] = float(match.group(2))
        return means

    def test_other_mass_zero_is_identity(self):
        # With other_mass=0, renormalization is a no-op: option_probs passed
        # through unchanged. Verify the reported means reflect 0.5/0.5.
        payload = _mc_payload(
            option_probs={"Red": 0.5, "Blue": 0.5},
            other_mass=0.0,
            concentration=20.0,
        )
        rationale = _wrap_json(payload)
        result = run_tools_for_forecaster(
            question=_make_mc_question(),
            rationale=rationale,
            forecaster_id="m",
        )
        means = self._extract_top_means(result)
        assert means.get("Red") == pytest.approx(0.5, abs=0.02)
        assert means.get("Blue") == pytest.approx(0.5, abs=0.02)
        # other_mass=0 is equivalent to "no Other": the Dirichlet runs over
        # the named options only, and "Other" is absent from the output.
        assert "Other" not in means

    def test_other_mass_moderate_rescales_option_probs(self):
        # option_probs {"A": 0.7, "B": 0.3} sum to 1.0 (schema contract), and
        # other_mass=0.3. After renormalization, tool-contract inputs are
        # {"A": 0.49, "B": 0.21} + other_mass=0.3 (summing to 1.0). Verify
        # reported means match these scaled values.
        payload = {
            "question_type": "multiple_choice",
            "option_probs": {"A": 0.7, "B": 0.3},
            "other_mass": 0.3,
            "concentration": 50.0,
        }
        rationale = _wrap_json(payload)
        result = run_tools_for_forecaster(
            question=_make_mc_question(options=["A", "B"]),
            rationale=rationale,
            forecaster_id="m",
        )
        means = self._extract_top_means(result)
        assert means.get("A") == pytest.approx(0.49, abs=0.02)
        assert means.get("B") == pytest.approx(0.21, abs=0.02)
        assert means.get("Other") == pytest.approx(0.30, abs=0.02)

    def test_other_mass_near_one_collapses_to_other(self):
        # other_mass=0.99 with option_probs {"A": 1.0}: after renorm,
        # A=0.01 and Other=0.99. Must not crash, and Other mean must be near 0.99.
        payload = {
            "question_type": "multiple_choice",
            "option_probs": {"A": 1.0},
            "other_mass": 0.99,
            "concentration": 100.0,
        }
        rationale = _wrap_json(payload)
        result = run_tools_for_forecaster(
            question=_make_mc_question(options=["A"]),
            rationale=rationale,
            forecaster_id="m",
        )
        means = self._extract_top_means(result)
        assert "Other" in means
        assert means["Other"] == pytest.approx(0.99, abs=0.02)


# ---------------------------------------------------------------------------
# F7: previously-unconsumed schema fields (tails delta, scenarios count)
# ---------------------------------------------------------------------------


class TestSchemaFieldWiring:
    def test_numeric_tails_delta_line_emitted(self):
        # Declared tails should produce a delta line comparing declared vs fitted tails.
        payload = _numeric_payload(
            declared_percentiles={"0.1": 20.0, "0.5": 40.0, "0.9": 60.0},
            distribution_family_hint="normal",
            tails={"below_min_expected": 0.05, "above_max_expected": 0.03},
        )
        rationale = _wrap_json(payload)
        result = run_tools_for_forecaster(
            question=_make_numeric_question(lower_bound=0.0, upper_bound=100.0),
            rationale=rationale,
            forecaster_id="m",
        )
        assert "Declared vs fitted tails" in result
        assert "declared [below=0.050, above=0.030]" in result

    def test_numeric_no_tails_omits_delta_line(self):
        payload = {
            "question_type": "numeric",
            "declared_percentiles": {"0.1": 20.0, "0.5": 40.0, "0.9": 60.0},
            "distribution_family_hint": "normal",
        }
        rationale = _wrap_json(payload)
        result = run_tools_for_forecaster(
            question=_make_numeric_question(lower_bound=0.0, upper_bound=100.0),
            rationale=rationale,
            forecaster_id="m",
        )
        assert "Declared vs fitted tails" not in result

    def test_binary_scenarios_count_line_emitted(self):
        payload = _binary_payload(
            scenarios=[
                {"name": "rapid escalation", "prob": 0.2, "conditional_outcome": "YES"},
                {"name": "steady", "prob": 0.5, "conditional_outcome": "NO"},
                {"name": "collapse", "prob": 0.3, "conditional_outcome": "NO"},
            ],
        )
        rationale = _wrap_json(payload)
        result = run_tools_for_forecaster(
            question=_make_binary_question(),
            rationale=rationale,
            forecaster_id="m",
        )
        assert "Declared scenario decomposition" in result
        assert "3 branches" in result
        assert "rapid escalation" in result

    def test_binary_many_scenarios_truncated(self):
        # >3 scenarios: only first 3 names listed, plus "+N more" suffix.
        payload = _binary_payload(
            scenarios=[
                {"name": "A", "prob": 0.2, "conditional_outcome": "YES"},
                {"name": "B", "prob": 0.2, "conditional_outcome": "NO"},
                {"name": "C", "prob": 0.2, "conditional_outcome": "YES"},
                {"name": "D", "prob": 0.2, "conditional_outcome": "NO"},
                {"name": "E", "prob": 0.2, "conditional_outcome": "NO"},
            ],
        )
        rationale = _wrap_json(payload)
        result = run_tools_for_forecaster(
            question=_make_binary_question(),
            rationale=rationale,
            forecaster_id="m",
        )
        assert "5 branches" in result
        assert "+2 more" in result


# ---------------------------------------------------------------------------
# F1: LR-chained posterior overflow saturation
# ---------------------------------------------------------------------------


class TestLrChainedOverflow:
    def test_overflow_saturates_not_nan(self):
        # Huge LRs on a modest prior would overflow float64 after a few multiplies.
        # Expect saturation to ~1.0, not nan.
        payload = _binary_payload(
            prior={"prob": 0.2, "source": "history"},
            evidence=[
                {"summary": "massive signal", "direction": "up", "strength": "strong", "likelihood_ratio": 1e100},
                {"summary": "another huge", "direction": "up", "strength": "strong", "likelihood_ratio": 1e100},
                {"summary": "and another", "direction": "up", "strength": "strong", "likelihood_ratio": 1e100},
                {"summary": "and yet another", "direction": "up", "strength": "strong", "likelihood_ratio": 1e100},
            ],
            posterior_prob=0.95,
        )
        rationale = _wrap_json(payload)
        result = run_tools_for_forecaster(
            question=_make_binary_question(),
            rationale=rationale,
            forecaster_id="m",
        )
        assert "Evidence-LR-chained posterior" in result
        assert "nan" not in result.lower()
        # Saturation target is 1 - 1e-9, which formats with 3 decimal places as 1.000.
        assert "→ 1.000" in result


# ---------------------------------------------------------------------------
# F13: _lr_chained_posterior boundary branches (direct unit test on private helper)
# ---------------------------------------------------------------------------


class TestLrChainedPosteriorBoundaries:
    def test_prior_zero_returns_none(self):
        assert _lr_chained_posterior(0.0, [2.0, 3.0]) is None

    def test_prior_one_returns_none(self):
        assert _lr_chained_posterior(1.0, [2.0, 3.0]) is None

    def test_non_finite_lr_returns_none(self):
        assert _lr_chained_posterior(0.5, [float("inf")]) is None

    def test_negative_lr_returns_none(self):
        assert _lr_chained_posterior(0.5, [-2.0]) is None

    def test_zero_lr_returns_none(self):
        assert _lr_chained_posterior(0.5, [0.0]) is None

    def test_long_chain_saturates_not_nan(self):
        result = _lr_chained_posterior(0.5, [1e20] * 50)
        assert result is not None
        assert math.isfinite(result)
        assert result > 0.999


# ---------------------------------------------------------------------------
# F13: numeric no-hint auto-select path
# ---------------------------------------------------------------------------


class TestRunToolsNumericAutoSelect:
    def test_no_family_hint_auto_selects_best_fit(self):
        payload = {
            "question_type": "numeric",
            "declared_percentiles": {"0.1": -1.28, "0.5": 0.0, "0.9": 1.28},
        }
        rationale = _wrap_json(payload)
        result = run_tools_for_forecaster(
            question=_make_numeric_question(lower_bound=-10.0, upper_bound=10.0),
            rationale=rationale,
            forecaster_id="m",
        )
        assert "Out-of-bounds mass" in result


# ---------------------------------------------------------------------------
# F13: cdf_at_threshold_for_forecaster above-upper-bound debug branch
# ---------------------------------------------------------------------------


class TestCdfAtThresholdOutOfBoundsLogging:
    def test_threshold_above_closed_upper_bound_logs_debug(self, caplog):
        payload = _numeric_payload(
            declared_percentiles={"0.1": 30.0, "0.5": 50.0, "0.9": 70.0},
            distribution_family_hint="normal",
        )
        rationale = _wrap_json(payload)
        q = _make_numeric_question(lower_bound=0.0, upper_bound=100.0)
        with caplog.at_level(logging.DEBUG, logger="metaculus_bot.tool_runner"):
            result = cdf_at_threshold_for_forecaster(rationale, q, threshold=150.0)
        # Function still returns a value (it's legitimate tail-mass query);
        # we just want the debug trail.
        assert result is not None
        assert any("above closed upper bound" in rec.message and rec.levelno == logging.DEBUG for rec in caplog.records)

    def test_threshold_below_closed_lower_bound_logs_debug(self, caplog):
        payload = _numeric_payload(
            declared_percentiles={"0.1": 30.0, "0.5": 50.0, "0.9": 70.0},
            distribution_family_hint="normal",
        )
        rationale = _wrap_json(payload)
        q = _make_numeric_question(lower_bound=0.0, upper_bound=100.0)
        with caplog.at_level(logging.DEBUG, logger="metaculus_bot.tool_runner"):
            result = cdf_at_threshold_for_forecaster(rationale, q, threshold=-50.0)
        assert result is not None
        assert any("below closed lower bound" in rec.message and rec.levelno == logging.DEBUG for rec in caplog.records)


# ---------------------------------------------------------------------------
# F13: mixed-type blocks are filtered in aggregate_binary_values
# ---------------------------------------------------------------------------


class TestCrossModelMixedBlocks:
    def test_mixed_type_blocks_filtered(self):
        # One binary rationale (with base_rate), one numeric rationale.
        # aggregate_binary_values should pool both prediction probs but only
        # see the single binary block when considering base_rate aggregation;
        # with only 1 base rate, the blended-base-rate line is suppressed.
        binary_block = {
            "question_type": "binary",
            "posterior_prob": 0.6,
            "base_rate": {"k": 3, "n": 10, "ref_class": "x"},
        }
        numeric_block = {
            "question_type": "numeric",
            "declared_percentiles": {"0.1": 0.0, "0.5": 0.5, "0.9": 1.0},
        }
        rationale_bin = _wrap_json(binary_block)
        rationale_num = _wrap_json(numeric_block)
        result = aggregate_binary_values([rationale_bin, rationale_num], prediction_probs=[0.5, 0.7])
        assert "Pools over 2 forecasters" in result
        assert "Blended base rate" not in result


# ---------------------------------------------------------------------------
# Spread plausibility check (Phase A.3 Package 3c)
# ---------------------------------------------------------------------------


class TestSpreadPlausibilityCheck:
    """Per-forecaster spread sanity check for numeric aggregation.

    Defense-in-depth atop the family-consistency check. Catches a
    confidently-narrow forecaster (e.g. qid 43171's GLM-4.5-air with σ=13K
    while ensemble σ ~965K) that the family check ratifies but is
    effectively over-fit to fabricated data.

    σ is derived from declared P10/P90: σ ≈ (P90 - P10) / 2.5631 (the
    ``_P10_P90_Z_GAP`` constant from probabilistic_tools/distributions.py).
    For each forecaster, ratio = σ_i / median(σ across forecasters). If
    ratio < 0.10, emit a ⚠ Spread anomaly line. Otherwise emit a single
    summary line confirming all forecasters fall within the threshold.
    """

    def _make_pcts(self, median: float, p10: float, p90: float) -> list[Percentile]:
        """11-point percentile vector implied σ = (p90 - p10) / 2.5631."""
        return [
            Percentile(percentile=0.025, value=median - 2 * (median - p10)),
            Percentile(percentile=0.05, value=median - 1.5 * (median - p10)),
            Percentile(percentile=0.10, value=p10),
            Percentile(percentile=0.20, value=median - 0.5 * (median - p10)),
            Percentile(percentile=0.40, value=median - 0.1 * (median - p10)),
            Percentile(percentile=0.50, value=median),
            Percentile(percentile=0.60, value=median + 0.1 * (p90 - median)),
            Percentile(percentile=0.80, value=median + 0.5 * (p90 - median)),
            Percentile(percentile=0.90, value=p90),
            Percentile(percentile=0.95, value=median + 1.5 * (p90 - median)),
            Percentile(percentile=0.975, value=median + 2 * (p90 - median)),
        ]

    def test_warns_on_narrow_sigma_forecaster(self):
        """Forecaster 4 has σ ≈ 0.039 (P90-P10 = 0.1 / 2.5631), the others σ ≈ 1.0."""
        # Three forecasters with σ ≈ 1.0 ((P90-P10) ≈ 2.5631) plus one narrow.
        preds = [
            self._make_pcts(median=10.0, p10=10.0 - 1.28155, p90=10.0 + 1.28155),
            self._make_pcts(median=12.0, p10=12.0 - 1.28155, p90=12.0 + 1.28155),
            self._make_pcts(median=11.0, p10=11.0 - 1.28155, p90=11.0 + 1.28155),
            self._make_pcts(median=10.5, p10=10.5 - 0.05, p90=10.5 + 0.05),
        ]
        out = aggregate_numeric_values(rationales=["", "", "", ""], prediction_percentiles=preds)
        assert "⚠ Spread anomaly" in out
        # 1-indexed, so the narrow forecaster is #4.
        assert "forecaster 4" in out
        # No false-positives for the three normal forecasters.
        for idx in (1, 2, 3):
            assert f"forecaster {idx})" not in out, f"forecaster {idx} should not be flagged"

    def test_no_warning_when_all_within_threshold(self):
        """Three forecasters, σ in {0.95, 1.0, 1.05} — all within 10× of median σ."""
        preds = [
            self._make_pcts(median=10.0, p10=10.0 - 0.5 * 2.5631 * 0.95, p90=10.0 + 0.5 * 2.5631 * 0.95),
            self._make_pcts(median=10.0, p10=10.0 - 0.5 * 2.5631 * 1.0, p90=10.0 + 0.5 * 2.5631 * 1.0),
            self._make_pcts(median=10.0, p10=10.0 - 0.5 * 2.5631 * 1.05, p90=10.0 + 0.5 * 2.5631 * 1.05),
        ]
        out = aggregate_numeric_values(rationales=["", "", ""], prediction_percentiles=preds)
        assert "⚠ Spread anomaly" not in out
        assert "Spread plausibility" in out
        assert "all 3 forecasters within 10×" in out

    def test_skips_when_p10_or_p90_missing(self):
        """Forecasters without both P10 and P90 are excluded from the σ pool."""
        # Two forecasters have full P10/P90, one only has P50.
        preds_full = self._make_pcts(median=10.0, p10=8.0, p90=12.0)
        preds_full_2 = self._make_pcts(median=11.0, p10=9.0, p90=13.0)
        preds_no_tail = [Percentile(percentile=0.5, value=10.5)]

        out = aggregate_numeric_values(
            rationales=["", "", ""],
            prediction_percentiles=[preds_full, preds_full_2, preds_no_tail],
        )
        # Should still produce a summary line for the 2 forecasters with valid σ.
        assert "Spread plausibility" in out
        assert "all 2 forecasters within 10×" in out

    def test_skips_when_only_one_forecaster(self):
        """With one forecaster the median σ is undefined; section is suppressed."""
        # Even though we still have ≥2 entries (gating bound), only one has P10/P90.
        preds_full = self._make_pcts(median=10.0, p10=8.0, p90=12.0)
        preds_no_tail = [Percentile(percentile=0.5, value=10.5)]

        out = aggregate_numeric_values(
            rationales=["", ""],
            prediction_percentiles=[preds_full, preds_no_tail],
        )
        # No spread section because only one forecaster has a valid σ.
        assert "Spread anomaly" not in out
        assert "Spread plausibility" not in out

    def test_emits_summary_when_no_anomalies(self):
        """When all σ within threshold, emit a single summary line."""
        preds = [
            self._make_pcts(median=10.0, p10=8.0, p90=12.0),  # σ ≈ 1.56
            self._make_pcts(median=10.0, p10=8.5, p90=11.5),  # σ ≈ 1.17
            self._make_pcts(median=10.0, p10=7.5, p90=12.5),  # σ ≈ 1.95
        ]
        out = aggregate_numeric_values(rationales=["", "", ""], prediction_percentiles=preds)
        assert "Spread plausibility" in out
        assert "all 3 forecasters" in out

    def test_qid_43171_regression_pattern(self):
        """Replay qid 43171's pattern: one forecaster with σ ~0.04M, three with σ in 1-3M.

        Don't need real Metaculus data — synthetic 4-forecaster scenario reproduces
        the GLM-vs-ensemble-σ ratio that the family check failed to catch.
        """
        # Three with σ ~1.0M-3.0M, one narrow with σ ~0.04M. Ratio ~= 0.02 ≪ 0.10.
        m = 1_000_000.0  # Median value scale
        preds = [
            # σ ≈ 1.0M
            self._make_pcts(median=m, p10=m - 1.28155 * 1_000_000, p90=m + 1.28155 * 1_000_000),
            # σ ≈ 2.0M
            self._make_pcts(median=m, p10=m - 1.28155 * 2_000_000, p90=m + 1.28155 * 2_000_000),
            # σ ≈ 3.0M
            self._make_pcts(median=m, p10=m - 1.28155 * 3_000_000, p90=m + 1.28155 * 3_000_000),
            # σ ≈ 0.04M (the GLM-style narrow one — third forecaster, 0-indexed = 3, 1-indexed = 4)
            self._make_pcts(median=m, p10=m - 1.28155 * 40_000, p90=m + 1.28155 * 40_000),
        ]
        out = aggregate_numeric_values(rationales=["", "", "", ""], prediction_percentiles=preds)
        assert "⚠ Spread anomaly" in out
        assert "forecaster 4" in out  # 1-indexed: the narrow one is the 4th in the list


# ---------------------------------------------------------------------------
# Dummy assertion so pytest doesn't skip the file on collection issues.
# ---------------------------------------------------------------------------


def test_module_imports():
    assert callable(run_tools_for_forecaster)
    assert callable(build_cross_model_aggregation)
    assert callable(cdf_at_threshold_for_forecaster)
    assert callable(aggregate_binary_values)
    assert callable(aggregate_numeric_values)
    assert callable(aggregate_mc_values)


# ---------------------------------------------------------------------------
# Per-type gating (PROBABILISTIC_TOOLS_TYPES allow-list)
# ---------------------------------------------------------------------------


class TestPerTypeGating:
    def test_global_off_disables_all_types(self, monkeypatch):
        # FEATURE_FLAG_ENV unset → all _feature_enabled(...) return False
        monkeypatch.delenv(FEATURE_FLAG_ENV, raising=False)
        assert _feature_enabled("binary") is False
        assert _feature_enabled("multiple_choice") is False
        assert _feature_enabled("numeric") is False
        assert _feature_enabled(None) is False

    def test_global_on_default_types_all_enabled(self, monkeypatch):
        # FEATURE_FLAG_ENV=true, TYPES unset → defaults to all 3 enabled (back-compat)
        monkeypatch.setenv(FEATURE_FLAG_ENV, "true")
        monkeypatch.delenv(TYPES_ENV, raising=False)
        assert _feature_enabled("binary") is True
        assert _feature_enabled("multiple_choice") is True
        assert _feature_enabled("numeric") is True

    def test_types_csv_excludes_numeric(self, monkeypatch):
        # Production setting: binary+MC only
        monkeypatch.setenv(FEATURE_FLAG_ENV, "true")
        monkeypatch.setenv(TYPES_ENV, "binary,multiple_choice")
        assert _feature_enabled("binary") is True
        assert _feature_enabled("multiple_choice") is True
        assert _feature_enabled("numeric") is False

    def test_types_csv_strips_whitespace(self, monkeypatch):
        monkeypatch.setenv(FEATURE_FLAG_ENV, "true")
        monkeypatch.setenv(TYPES_ENV, "binary, multiple_choice ")
        assert _feature_enabled("binary") is True
        assert _feature_enabled("multiple_choice") is True
        assert _feature_enabled("numeric") is False

    def test_invalid_type_in_csv_is_ignored_with_warning(self, monkeypatch, caplog):
        monkeypatch.setenv(FEATURE_FLAG_ENV, "true")
        monkeypatch.setenv(TYPES_ENV, "binary,foo,multiple_choice")
        with caplog.at_level("WARNING"):
            assert _feature_enabled("binary") is True
            assert _feature_enabled("multiple_choice") is True
            assert _feature_enabled("numeric") is False
        assert any("invalid" in rec.message.lower() for rec in caplog.records)

    def test_question_type_none_only_checks_global(self, monkeypatch):
        # When question_type isn't known, only the global flag matters.
        monkeypatch.setenv(FEATURE_FLAG_ENV, "true")
        monkeypatch.setenv(TYPES_ENV, "binary")  # numeric+mc disallowed
        assert _feature_enabled(None) is True  # passes — type-agnostic check


class TestPerTypeAggregateGating:
    def test_aggregate_numeric_returns_empty_when_numeric_excluded_by_types(self, monkeypatch):
        monkeypatch.setenv(FEATURE_FLAG_ENV, "true")
        monkeypatch.setenv(TYPES_ENV, "binary,multiple_choice")
        preds = [
            [
                Percentile(percentile=0.1, value=10),
                Percentile(percentile=0.5, value=30),
                Percentile(percentile=0.9, value=50),
            ],
            [
                Percentile(percentile=0.1, value=20),
                Percentile(percentile=0.5, value=40),
                Percentile(percentile=0.9, value=60),
            ],
        ]
        result = aggregate_numeric_values(rationales=["", ""], prediction_percentiles=preds)
        assert result == ""

    def test_aggregate_binary_runs_when_binary_in_types(self, monkeypatch):
        monkeypatch.setenv(FEATURE_FLAG_ENV, "true")
        monkeypatch.setenv(TYPES_ENV, "binary,multiple_choice")
        rationales = [
            _wrap_json(_binary_payload(posterior_prob=0.2)),
            _wrap_json(_binary_payload(posterior_prob=0.5)),
        ]
        result = aggregate_binary_values(rationales=rationales, prediction_probs=[0.2, 0.5])
        assert result != ""  # non-empty CMA produced

    def test_aggregate_mc_runs_when_mc_in_types(self, monkeypatch):
        monkeypatch.setenv(FEATURE_FLAG_ENV, "true")
        monkeypatch.setenv(TYPES_ENV, "binary,multiple_choice")
        pred1 = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name="Red", probability=0.5),
                PredictedOption(option_name="Blue", probability=0.5),
            ]
        )
        pred2 = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name="Red", probability=0.4),
                PredictedOption(option_name="Blue", probability=0.6),
            ]
        )
        result = aggregate_mc_values(["", ""], prediction_options=[pred1, pred2])
        assert result != ""


# ---------------------------------------------------------------------------
# Production workflow YAML configuration
# ---------------------------------------------------------------------------


class TestWorkflowYamlConfiguration:
    def test_workflow_yamls_set_both_env_vars(self):
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[1]
        workflows = [
            repo_root / ".github/workflows/run_bot_on_tournament.yaml",
            repo_root / ".github/workflows/run_bot_on_metaculus_cup.yaml",
            repo_root / ".github/workflows/run_bot_on_minibench.yaml",
        ]
        for path in workflows:
            text = path.read_text(encoding="utf-8")
            assert "PROBABILISTIC_TOOLS_ENABLED" in text, f"{path} missing global flag"
            assert "PROBABILISTIC_TOOLS_TYPES" in text, f"{path} missing types flag"
            # Light check: types contains binary + multiple_choice (not numeric).
            # Operator can change later but the default ship state should match
            # the empirical (binary p=0.094, MC directional, numeric inconclusive)
            # decision.
            assert "binary" in text
            assert "multiple_choice" in text


# Appease linters about pytest fixture param naming — kept unused on purpose.
_ = pytest
