"""Pinning tests for ``metaculus_bot.scoring_patches`` — the community-baseline
scoring formulas that ``backtest.py`` / ``analyze_correlations.py`` depend on.

Seam: on stock forecasting-tools 0.2.54, ``MultipleChoiceReport`` and
``NumericReport`` inherit an ``expected_baseline_score`` that only ever raises
``NotImplementedError``, and ``ForecastReport.calculate_average_expected_baseline_score``
crashes on any ``None`` score. ``apply_scoring_patches`` monkey-patches all three:
it installs concrete community-log-scoring formulas on the two report types and a
``None``-tolerant averager on the base class. Our backtest scoring reads those
patched properties, so the exact numbers they emit are load-bearing.

These tests apply the patches through the real ``apply_scoring_patches`` entry
point, build minimal reports with known community predictions and a known
forecast, and pin ``expected_baseline_score`` to values derived by hand from the
patched formulas (derivations inline). They are green on 0.2.54. If the 0.2.92
upgrade moves where these are consumed (e.g. upstream finally implements
``expected_baseline_score`` itself, or renames the averager), this pin plus the
migration investigation decides whether the patch is retired or retargeted — a
silent formula change would otherwise shift every backtest score.

Non-vacuity comes from the negative controls: a forecast that matches the
community scores exactly at the formula's calibration anchor (MC → 0.0, numeric →
-50.0) while a divergent forecast scores materially lower, and the averager keeps
a valid score while dropping a ``None`` one instead of crashing.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import pytest
from forecasting_tools.data_models.forecast_report import ForecastReport
from forecasting_tools.data_models.multiple_choice_report import (
    MultipleChoiceReport,
    PredictedOption,
    PredictedOptionList,
)
from forecasting_tools.data_models.numeric_report import (
    NumericDistribution,
    NumericReport,
    Percentile,
)
from forecasting_tools.data_models.questions import MultipleChoiceQuestion, NumericQuestion
from pydantic import PrivateAttr

from metaculus_bot import scoring_patches

# --- Hand-derived expected scores (from the patched formulas in scoring_patches.py) ---
#
# MC — calculate_multiple_choice_baseline_score:
#   score = 100 * (Σ_i c_i · ln(p_i) / ln(K) + 1),  K = number of options.
#   Options A,B (sorted by name); community c = [0.5, 0.5]; bot p = [0.25, 0.75]
#   (both already in-range and summing to 1, so the clamp/renormalize is a no-op):
#     Σ c_i·ln(p_i) = 0.5·ln(0.25) + 0.5·ln(0.75)
#                   = 0.5·(-1.3862944) + 0.5·(-0.2876821) = -0.8369882
#     ln(K) = ln(2) = 0.6931472
#     score = 100·(-0.8369882 / 0.6931472 + 1) = 100·(-1.2075187 + 1) = -20.7518750
_MC_MISMATCH_SCORE = -20.751874963942196
#   Negative control — bot p = community c = [0.5, 0.5]: Σ c_i·ln(p_i) = ln(0.5) = -ln(2),
#   so score = 100·(-ln(2)/ln(2) + 1) = 100·0 = 0.0 exactly (the formula's anchor).
_MC_MATCH_SCORE = 0.0

# Numeric — calculate_numeric_baseline_score PMF path → _calculate_relative_numeric_score:
#   model CDF = [0.0, 0.2, 1.0]      → bot_pmf       = diff = [0.2, 0.8]
#   community CDF = [0.0, 0.5, 1.0]  → community_pmf = diff = [0.5, 0.5]
#   bot_scored = 0.99·bot_pmf + 0.01·uniform([0.5, 0.5])
#              = [0.99·0.2 + 0.005, 0.99·0.8 + 0.005] = [0.203, 0.797]
#   E_c[ln(bot_scored)] = 0.5·ln(0.203) + 0.5·ln(0.797)
#                       = 0.5·(-1.5945938) + 0.5·(-0.2268499) = -0.9107219
#   num_bins = 2 → normalization = ln(2)/1.5 = 0.4620981
#   score = 100·(-0.9107219 / 0.4620981 + 1) = 100·(-1.9708476 + 1) = -97.0847554
_NUMERIC_DIVERGENT_SCORE = -97.08475536113947
#   Negative control — model CDF = community CDF = [0.0, 0.5, 1.0]:
#   bot_pmf = community_pmf = [0.5, 0.5]; bot_scored = [0.5, 0.5];
#   E_c[ln(bot_scored)] = ln(0.5) = -ln(2); normalization = ln(2)/1.5;
#   score = 100·(-ln(2) / (ln(2)/1.5) + 1) = 100·(-1.5 + 1) = -50.0 (the anchor).
_NUMERIC_MATCH_SCORE = -50.0

# Aggregation — patched calculate_average_expected_baseline_score is a plain mean of the
# non-None scores: (_MC_MISMATCH_SCORE + _NUMERIC_DIVERGENT_SCORE) / 2.
_AVERAGE_SCORE = (_MC_MISMATCH_SCORE + _NUMERIC_DIVERGENT_SCORE) / 2.0

_MISSING = object()
_PATCHED_TARGETS = (
    (MultipleChoiceReport, "expected_baseline_score"),
    (NumericReport, "expected_baseline_score"),
    (ForecastReport, "calculate_average_expected_baseline_score"),
)


@pytest.fixture(autouse=True)
def _patched_scoring() -> Iterator[None]:
    """Install the scoring patches via the real entry point, then restore.

    ``apply_scoring_patches`` mutates the forecasting-tools classes in place.
    Snapshot each target's own-class descriptor (``_MISSING`` if it is only
    inherited, e.g. ``NumericReport.expected_baseline_score`` lives on
    ``ForecastReport``) so teardown either restores the original descriptor or
    removes the patch-added one, leaving the classes as the suite found them.
    """

    originals = {
        (cls, name): cls.__dict__.get(name, _MISSING)  # noqa: HARNESS-SCAN-EXEMPT-mapping-get-in-loop  # bounded: 3 patch targets
        for cls, name in _PATCHED_TARGETS
    }
    scoring_patches.apply_scoring_patches()
    yield
    for (cls, name), original in originals.items():
        if original is _MISSING:
            if name in cls.__dict__:
                delattr(cls, name)
        else:
            setattr(cls, name, original)


def _mc_question(community: list[float]) -> MultipleChoiceQuestion:
    """MC question whose community CP is ``community`` aligned to options [A, B]."""
    return MultipleChoiceQuestion(
        question_text="Which option resolves?",
        id_of_question=1111,
        id_of_post=1111,
        options=["A", "B"],
        api_json={
            "question": {
                "type": "multiple_choice",
                "options": ["A", "B"],
                "aggregations": {"recency_weighted": {"latest": {"forecast_values": community}}},
            }
        },
    )


def _mc_report(question: MultipleChoiceQuestion, prob_a: float, prob_b: float) -> MultipleChoiceReport:
    prediction = PredictedOptionList(
        predicted_options=[
            PredictedOption(option_name="A", probability=prob_a),
            PredictedOption(option_name="B", probability=prob_b),
        ]
    )
    return MultipleChoiceReport(question=question, prediction=prediction, explanation="# Pin test")


class _FixedCdfDistribution(NumericDistribution):
    """NumericDistribution whose ``.cdf`` returns a caller-supplied, hand-checkable CDF.

    Mirrors the ``PchipNumericDistribution`` override pattern (a private CDF slot
    plus a ``.cdf`` property) so the numeric scorer reads a known 3-point CDF
    instead of a 201-point PCHIP build we could not derive by hand. The CDF is a
    Pydantic ``PrivateAttr`` set after construction, keeping the base
    ``__init__`` signature intact.
    """

    _cdf_probabilities: list[float] = PrivateAttr(default_factory=list)

    @property
    def cdf(self) -> list[Percentile]:
        return [Percentile(percentile=prob, value=float(index)) for index, prob in enumerate(self._cdf_probabilities)]


def _numeric_question(community_cdf: list[float]) -> NumericQuestion:
    return NumericQuestion(
        question_text="How many will there be?",
        id_of_question=2222,
        id_of_post=2222,
        lower_bound=0.0,
        upper_bound=100.0,
        open_lower_bound=False,
        open_upper_bound=False,
        zero_point=None,
        cdf_size=201,
        api_json={
            "question": {
                "aggregations": {"recency_weighted": {"latest": {"forecast_values": community_cdf}}},
            }
        },
    )


def _numeric_report(question: NumericQuestion, model_cdf: list[float]) -> NumericReport:
    distribution = _FixedCdfDistribution(
        declared_percentiles=[
            Percentile(percentile=0.1, value=10.0),
            Percentile(percentile=0.9, value=90.0),
        ],
        open_upper_bound=False,
        open_lower_bound=False,
        upper_bound=100.0,
        lower_bound=0.0,
        zero_point=None,
        cdf_size=201,
    )
    distribution._cdf_probabilities = model_cdf
    return NumericReport(question=question, prediction=distribution, explanation="# Pin test")


class TestMultipleChoiceBaselineScore:
    """Pin MultipleChoiceReport.expected_baseline_score to the patched MC formula."""

    def test_mismatch_forecast_scores_exact_patched_value(self) -> None:
        report = _mc_report(_mc_question([0.5, 0.5]), prob_a=0.25, prob_b=0.75)
        assert report.expected_baseline_score == pytest.approx(_MC_MISMATCH_SCORE)

    def test_matching_forecast_hits_the_zero_anchor(self) -> None:
        # Negative control: a forecast equal to the community CP must score the
        # formula's anchor (0.0), proving the score above isn't a constant.
        report = _mc_report(_mc_question([0.5, 0.5]), prob_a=0.5, prob_b=0.5)
        assert report.expected_baseline_score == pytest.approx(_MC_MATCH_SCORE, abs=1e-12)

    def test_missing_community_prediction_returns_none(self) -> None:
        # No aggregations node → the patch cannot read a community CP and returns
        # None (which the averager must then tolerate — see the aggregation test).
        question = MultipleChoiceQuestion(
            question_text="Which option resolves?",
            id_of_question=1313,
            id_of_post=1313,
            options=["A", "B"],
            api_json={"question": {"type": "multiple_choice", "options": ["A", "B"]}},
        )
        report = _mc_report(question, prob_a=0.25, prob_b=0.75)
        assert report.expected_baseline_score is None


class TestNumericBaselineScore:
    """Pin NumericReport.expected_baseline_score to the patched numeric PMF formula."""

    def test_divergent_forecast_scores_exact_patched_value(self) -> None:
        report = _numeric_report(_numeric_question([0.0, 0.5, 1.0]), model_cdf=[0.0, 0.2, 1.0])
        # The subclass must survive Pydantic construction, or the .cdf override
        # (and thus the whole PMF path) is silently lost before scoring.
        assert type(report.prediction).__name__ == "_FixedCdfDistribution"
        assert report.expected_baseline_score == pytest.approx(_NUMERIC_DIVERGENT_SCORE)

    def test_matching_pmf_hits_the_minus_fifty_anchor(self) -> None:
        # Negative control: model PMF == community PMF must land on the calibrated
        # -50.0 anchor, proving the divergent score reflects the divergence.
        report = _numeric_report(_numeric_question([0.0, 0.5, 1.0]), model_cdf=[0.0, 0.5, 1.0])
        assert report.expected_baseline_score == pytest.approx(_NUMERIC_MATCH_SCORE)


class TestAverageBaselineScoreAggregation:
    """Pin ForecastReport.calculate_average_expected_baseline_score to the patched averager."""

    def test_averages_valid_scores(self) -> None:
        mc_report = _mc_report(_mc_question([0.5, 0.5]), prob_a=0.25, prob_b=0.75)
        numeric_report = _numeric_report(_numeric_question([0.0, 0.5, 1.0]), model_cdf=[0.0, 0.2, 1.0])
        average = ForecastReport.calculate_average_expected_baseline_score([mc_report, numeric_report])
        assert average == pytest.approx(_AVERAGE_SCORE)

    def test_drops_none_scores_instead_of_crashing(self) -> None:
        # The patch's reason for existing: a None score (missing community CP)
        # must be filtered, leaving the valid score as the average — stock 0.2.54
        # raises ValueError here instead.
        valid = _mc_report(_mc_question([0.5, 0.5]), prob_a=0.25, prob_b=0.75)
        missing = MultipleChoiceReport(
            question=MultipleChoiceQuestion(
                question_text="Which option resolves?",
                id_of_question=1414,
                id_of_post=1414,
                options=["A", "B"],
                api_json={"question": {"type": "multiple_choice", "options": ["A", "B"]}},
            ),
            prediction=PredictedOptionList(
                predicted_options=[
                    PredictedOption(option_name="A", probability=0.25),
                    PredictedOption(option_name="B", probability=0.75),
                ]
            ),
            explanation="# Pin test",
        )
        assert missing.expected_baseline_score is None
        average = ForecastReport.calculate_average_expected_baseline_score([valid, missing])
        assert average == pytest.approx(_MC_MISMATCH_SCORE)

    def test_all_none_scores_average_to_zero(self) -> None:
        # Patched averager returns 0.0 when every score is None (stock raises).
        missing_questions = [
            MultipleChoiceQuestion(
                question_text="Which option resolves?",
                id_of_question=qid,
                id_of_post=qid,
                options=["A", "B"],
                api_json={"question": {"type": "multiple_choice", "options": ["A", "B"]}},
            )
            for qid in (1515, 1616)
        ]
        reports = [_mc_report(question, prob_a=0.25, prob_b=0.75) for question in missing_questions]
        assert all(report.expected_baseline_score is None for report in reports)
        assert ForecastReport.calculate_average_expected_baseline_score(reports) == 0.0


def test_binary_reference_sanity() -> None:
    """Cross-check the MC/numeric anchors against the binary log-score they mimic.

    The patched formulas are designed to sit on the same scale as the framework's
    binary baseline score. A perfectly-calibrated binary forecast (bot == community)
    yields 0.0, matching the MC anchor; this documents the intended scale and guards
    the derivation comments against a copy-paste drift in the anchor constants.
    """
    community, forecast = 0.5, 0.5
    binary_score = 100.0 * (
        community * (math.log2(forecast) + 1.0) + (1.0 - community) * (math.log2(1.0 - forecast) + 1.0)
    )
    assert binary_score == pytest.approx(_MC_MATCH_SCORE, abs=1e-12)
