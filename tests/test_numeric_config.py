"""Regression tests pinning numeric_config defaults to their empirically chosen values.

These tests guard against accidental reversion of the tail-widening defaults flipped
2026-05-12 in response to `scratch_docs_and_planning/tail_widening_empirical_calibration.md`.
On 43 resolved numerics (Feb-May 2026), k_tail=1.0 produced PIT std closest to the
ideal 0.289 in every segment; k_tail=1.25 moved away from ideal in every segment.
The span_floor_gamma floor never bound on real ensemble-averaged declared percentiles,
so the default was dropped to 0.0 (the floor enforcement at tail_widening.py:171/178
stays correctly gated on `> 0` and re-enables if a forecaster sets it back).
"""

import pytest
from forecasting_tools.data_models.questions import DateQuestion, DiscreteQuestion

from metaculus_bot.constants import MANTIC_SITE_URL, PLATFORM_MANTIC, PLATFORM_METACULUS
from metaculus_bot.numeric import config as numeric_config
from metaculus_bot.numeric.date_axis import as_epoch_question
from tests.mantic_fakes import load_legacy_date_question, load_preseason_date_question
from tests.pipeline_test_helpers import make_real_numeric_question


def test_standard_percentiles_is_13_with_p1_and_p99():
    """The standard set is the 13 percentiles incl. P1 (0.01) and P99 (0.99), sorted ascending.

    P1/P99 were added (11 -> 13) to give forecasters finer tail anchors so they can
    express probability mass below an open lower bound (the Minions & Monsters miss).
    """
    expected = [0.01, 0.025, 0.05, 0.10, 0.20, 0.40, 0.50, 0.60, 0.80, 0.90, 0.95, 0.975, 0.99]
    assert expected == numeric_config.STANDARD_PERCENTILES
    assert numeric_config.EXPECTED_PERCENTILE_COUNT == 13
    assert len(numeric_config.STANDARD_PERCENTILES) == 13
    assert sorted(numeric_config.STANDARD_PERCENTILES) == numeric_config.STANDARD_PERCENTILES
    assert 0.01 in numeric_config.STANDARD_PERCENTILES
    assert 0.99 in numeric_config.STANDARD_PERCENTILES


def test_standard_percentiles_csv_is_generated_from_constant():
    """The CSV label string used in prompts/errors is derived from STANDARD_PERCENTILES, not hardcoded."""
    assert numeric_config.STANDARD_PERCENTILES_CSV == "1,2.5,5,10,20,40,50,60,80,90,95,97.5,99"
    # Must track the constant: every label is its percentile * 100 formatted with %g.
    assert (
        ",".join(f"{p * 100:g}" for p in numeric_config.STANDARD_PERCENTILES) == numeric_config.STANDARD_PERCENTILES_CSV
    )


def test_tail_widen_k_tail_default_is_one():
    """TAIL_WIDEN_K_TAIL default must be 1.0 (no widening) per empirical calibration.

    See scratch_docs_and_planning/tail_widening_empirical_calibration.md.
    """
    assert numeric_config.TAIL_WIDEN_K_TAIL == 1.0


def test_tail_widen_span_floor_gamma_default_is_zero():
    """TAIL_WIDEN_SPAN_FLOOR_GAMMA default must be 0.0 (floor check disabled).

    Floor enforcement at tail_widening.py:171/178 is gated on `> 0`; the floor
    never bound on 2026 data. See
    scratch_docs_and_planning/tail_widening_empirical_calibration.md section 3.
    """
    assert numeric_config.TAIL_WIDEN_SPAN_FLOOR_GAMMA == 0.0


def test_tail_widening_enable_flag_still_present():
    """The enable flag stays available so tests and env overrides can re-enable widening."""
    assert hasattr(numeric_config, "TAIL_WIDENING_ENABLE")
    assert isinstance(numeric_config.TAIL_WIDENING_ENABLE, bool)


# --- Per-bin elicitation gate: ``elicit_per_bin`` (the numeric and date runners branch on it) ---


def _mantic_url(qid: int) -> str:
    return f"{MANTIC_SITE_URL}/questions/{qid}/"


def _metaculus_url(qid: int) -> str:
    return f"https://www.metaculus.com/questions/{qid}/"


def _on_mantic(question: DateQuestion) -> DateQuestion:
    """The framework parses every payload with a metaculus.com ``page_url``; ``ManticClient`` rewrites it to the host."""
    return question.model_copy(update={"page_url": _mantic_url(question.id_of_post or 0)})


def _count_question(bins: int, *, page_url: str) -> DiscreteQuestion:
    """A count question with ``bins`` integer bins from 0 upward, in the platform's half-step convention."""
    return DiscreteQuestion(
        id_of_question=700,
        id_of_post=700,
        page_url=page_url,
        question_text="How many?",
        background_info="",
        resolution_criteria="",
        fine_print="",
        published_time=None,
        close_time=None,
        lower_bound=-0.5,
        upper_bound=bins - 0.5,
        open_lower_bound=False,
        open_upper_bound=True,
        unit_of_measure="",
        zero_point=None,
        cdf_size=bins + 1,
    )


class TestElicitPerBinDefaults:
    """The gate is three facts at once: outcome-space grid, bin count at most the threshold, platform in the set."""

    def test_the_threshold_is_a_month_of_daily_bins(self) -> None:
        assert numeric_config.PMF_ELICITATION_MAX_BINS == 31

    def test_only_mantic_is_elicited_per_bin_on_this_landing(self) -> None:
        assert set(numeric_config.PMF_ELICITATION_PLATFORMS) == {PLATFORM_MANTIC}


class TestElicitPerBinOnRecordedManticQuestions:
    def test_the_twelve_bin_date_question_651_is_elicited_per_bin(self) -> None:
        assert numeric_config.elicit_per_bin(as_epoch_question(_on_mantic(load_preseason_date_question()))) is True

    def test_the_legacy_200_bin_date_question_500_stays_on_percentiles(self) -> None:
        assert numeric_config.elicit_per_bin(as_epoch_question(_on_mantic(load_legacy_date_question()))) is False

    def test_the_same_payload_without_the_clients_url_rewrite_reads_as_metaculus(self) -> None:
        assert numeric_config.elicit_per_bin(as_epoch_question(load_preseason_date_question())) is False


class TestElicitPerBinOnGridShapes:
    def test_a_201_point_mantic_numeric_question_stays_on_percentiles(self) -> None:
        question = make_real_numeric_question().model_copy(update={"page_url": _mantic_url(2001)})
        assert question.cdf_size == numeric_config.PCHIP_CDF_POINTS
        assert numeric_config.elicit_per_bin(question) is False

    def test_a_200_bin_mantic_discrete_question_stays_on_percentiles(self) -> None:
        question = _count_question(200, page_url=_mantic_url(700))
        assert numeric_config.grid_is_outcome_space(question)
        assert numeric_config.elicit_per_bin(question) is False

    @pytest.mark.parametrize(
        ("bins", "expected"),
        [
            pytest.param(11, True, id="eleven-bins"),
            pytest.param(31, True, id="at-the-threshold"),
            pytest.param(32, False, id="one-past-the-threshold"),
        ],
    )
    def test_the_threshold_is_inclusive_on_a_mantic_count_question(self, bins: int, expected: bool) -> None:
        assert numeric_config.elicit_per_bin(_count_question(bins, page_url=_mantic_url(700))) is expected

    def test_a_30_bin_mantic_numeric_question_with_nominal_bounds_on_the_range_is_elicited_per_bin(self) -> None:
        """Post 560's shape: a plain NumericQuestion (not discrete) on a 30-bin grid is an outcome-space grid too."""
        question = make_real_numeric_question(
            lower_bound=100.0, upper_bound=6100.0, open_lower_bound=True, open_upper_bound=True
        ).model_copy(
            update={
                "page_url": _mantic_url(560),
                "cdf_size": 31,
                "nominal_lower_bound": 100.0,
                "nominal_upper_bound": 6100.0,
            }
        )
        assert numeric_config.elicit_per_bin(question) is True


class TestElicitPerBinIsManticOnlyByOneConstant:
    def test_a_metaculus_discrete_question_stays_on_percentiles(self) -> None:
        assert numeric_config.elicit_per_bin(_count_question(11, page_url=_metaculus_url(700))) is False

    def test_adding_metaculus_to_the_platform_set_is_the_whole_switch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            numeric_config, "PMF_ELICITATION_PLATFORMS", frozenset({PLATFORM_MANTIC, PLATFORM_METACULUS})
        )
        assert numeric_config.elicit_per_bin(_count_question(11, page_url=_metaculus_url(700))) is True
        assert numeric_config.elicit_per_bin(_count_question(32, page_url=_metaculus_url(700))) is False
