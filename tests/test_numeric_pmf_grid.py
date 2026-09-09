"""Bin labels for per-bin elicitation (``metaculus_bot/numeric/pmf_grid.py``).

On an enumerable grid each forecaster is asked for one probability per bin, keyed by a label the
model can read and the extraction ladder can match back. The label style is chosen per grid: the
calendar day a bin covers on a day or week date grid (post 651's twelve days), the bin centre on a
centre-aligned quantity grid (the platform's discrete convention, ``range_min = nominal_min -
step / 2``), and a right-closed ``a to b`` interval otherwise. The reserved ``below_range`` and
``above_range`` keys appear only where the bound is open. Every quantity shape here is a real
Mantic grid, recorded 2026-09-08 from the public corpus; the two date shapes are the recorded
payloads in ``tests/mantic_fakes.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, NamedTuple

import pytest
from forecasting_tools.data_models.questions import DateQuestion, DiscreteQuestion, NumericQuestion

from metaculus_bot.constants import MANTIC_SITE_URL, PMF_ABOVE_RANGE_KEY, PMF_BELOW_RANGE_KEY
from metaculus_bot.numeric.date_axis import as_epoch_question, format_epoch, to_epoch
from metaculus_bot.numeric.pchip_cdf import build_cdf_value_grid
from metaculus_bot.numeric.pmf_grid import PmfGrid, fold_bin_label, format_bin_value, pmf_grid
from tests.mantic_fakes import (
    DATE_POST_ID,
    load_legacy_date_question,
    load_preseason_date_question,
    load_preseason_post,
)
from tests.pipeline_test_helpers import make_real_date_question


class _CorpusShape(NamedTuple):
    post_id: int
    cdf_size: int
    lower_bound: float
    upper_bound: float
    nominal_lower: float
    nominal_upper: float
    open_lower: bool
    open_upper: bool


# "How many counts in James Comey's April 2026 indictment will remain active": bins 0, 1, 2.
_POST_253 = _CorpusShape(253, 4, -0.5, 2.5, 0.0, 2.0, False, False)
# "How many public releases will U.S. Central Command publish": counts 0..20, open ceiling.
_POST_643 = _CorpusShape(643, 22, -0.5, 20.5, 0.0, 20.0, False, True)
# "What seasonally adjusted labor-force participation rate will the BLS report": step 0.1, both open.
_POST_619 = _CorpusShape(619, 14, 77.15, 78.45, 77.2, 78.4, True, True)
# "What will the price of bitcoin be on each day from 8 to 18 September 2026?": 450 bins of $100, both open.
_POST_650 = _CorpusShape(650, 451, 54950.0, 99950.0, 55000.0, 100000.0, True, True)
# 'How many CVEs will be "Awaiting Enrichment"': a NumericQuestion whose nominal bounds ARE the range edges.
_POST_560 = _CorpusShape(560, 31, 100.0, 6100.0, 100.0, 6100.0, True, True)
# Post 650's off-by-one on a coarse count grid: the writer's max declared one step above the last bin's centre.
_COUNT_GRID_WITH_650S_TOP = _CorpusShape(998, 21, -0.5, 19.5, 0.0, 20.0, False, True)


def _question_fields(shape: _CorpusShape, *, zero_point: float | None = None) -> dict[str, Any]:
    return {
        "id_of_question": shape.post_id,
        "id_of_post": shape.post_id,
        "page_url": f"{MANTIC_SITE_URL}/questions/{shape.post_id}/",
        "question_text": "Corpus question",
        "background_info": "",
        "resolution_criteria": "",
        "fine_print": "",
        "published_time": None,
        "close_time": None,
        "lower_bound": shape.lower_bound,
        "upper_bound": shape.upper_bound,
        "open_lower_bound": shape.open_lower,
        "open_upper_bound": shape.open_upper,
        "unit_of_measure": "",
        "zero_point": zero_point,
        "cdf_size": shape.cdf_size,
        "nominal_lower_bound": shape.nominal_lower,
        "nominal_upper_bound": shape.nominal_upper,
    }


def _discrete(shape: _CorpusShape) -> DiscreteQuestion:
    return DiscreteQuestion(**_question_fields(shape))


def _numeric(shape: _CorpusShape, *, zero_point: float | None = None) -> NumericQuestion:
    return NumericQuestion(**_question_fields(shape, zero_point=zero_point))


@pytest.fixture
def q651() -> DateQuestion:
    return load_preseason_date_question()


@pytest.fixture
def q500() -> DateQuestion:
    return load_legacy_date_question()


class TestDateGrids:
    def test_question_651_labels_are_its_twelve_calendar_days(self, q651: DateQuestion) -> None:
        grid = pmf_grid(as_epoch_question(q651))
        assert grid.style == "day"
        assert grid.labels == tuple(f"2026-09-{day:02d}" for day in range(8, 20))

    def test_the_last_label_is_the_apis_nominal_max(self, q651: DateQuestion) -> None:
        """Mantic labels a date bin by its LEFT edge: ``nominal_max`` is the last answer date, not ``range_max``."""
        view = as_epoch_question(q651)
        grid = pmf_grid(view)
        assert view.nominal_upper_bound is not None
        assert grid.labels[-1] == format_epoch(view.nominal_upper_bound, "day") == "2026-09-19"

    def test_edges_are_the_platforms_continuous_range(self, q651: DateQuestion) -> None:
        recorded = load_preseason_post(DATE_POST_ID)["question"]["scaling"]["continuous_range"]
        expected = tuple(to_epoch(datetime.fromisoformat(edge)) for edge in recorded)
        grid = pmf_grid(as_epoch_question(q651))
        assert grid.edges == pytest.approx(expected)
        assert len(grid.edges) == q651.cdf_size
        assert len(grid.labels) == q651.cdf_size - 1

    def test_closed_bounds_add_no_reserved_keys(self, q651: DateQuestion) -> None:
        grid = pmf_grid(as_epoch_question(q651))
        assert grid.open_lower_bound is False
        assert grid.open_upper_bound is False
        assert grid.keys == grid.labels

    def test_a_week_grid_labels_the_first_day_of_each_seven_day_bin(self) -> None:
        start = datetime(2026, 9, 8, tzinfo=UTC)
        question = make_real_date_question(
            lower_bound=start, upper_bound=start + timedelta(weeks=4), cdf_size=5, date_granularity="week"
        )
        grid = pmf_grid(as_epoch_question(question))
        assert grid.style == "week"
        assert grid.labels == ("2026-09-08", "2026-09-15", "2026-09-22", "2026-09-29")

    def test_the_legacy_timestamp_grid_labels_each_bin_as_an_interval(self, q500: DateQuestion) -> None:
        """Post 500's 200 bins are 6h42m18s wide and start at 15:00Z, so a calendar-day label cannot name them."""
        grid = pmf_grid(as_epoch_question(q500))
        assert grid.style == "timestamp"
        assert len(grid.labels) == 200
        assert grid.labels[0] == "2026-06-17T15:00:00Z to 2026-06-17T21:42:18Z"
        assert grid.labels[-1].endswith(" to 2026-08-12T12:00:00Z")
        assert grid.keys == (*grid.labels, PMF_ABOVE_RANGE_KEY)


class TestQuantityGrids:
    def test_post_253_labels_are_its_three_counts(self) -> None:
        grid = pmf_grid(_discrete(_POST_253))
        assert grid.style == "center"
        assert grid.labels == ("0", "1", "2")
        assert grid.keys == grid.labels

    def test_post_619_labels_are_the_bin_centres_at_one_decimal(self) -> None:
        """``77.15 + 0.05`` and its successors carry float residue; the labels are the platform's displayed values."""
        grid = pmf_grid(_discrete(_POST_619))
        assert grid.style == "center"
        expected = (
            "77.2",
            "77.3",
            "77.4",
            "77.5",
            "77.6",
            "77.7",
            "77.8",
            "77.9",
            "78",
            "78.1",
            "78.2",
            "78.3",
            "78.4",
        )
        assert grid.labels == expected
        assert grid.labels[-1] == format_bin_value(_POST_619.nominal_upper)

    def test_post_650_labels_the_450_hundred_dollar_bins_by_centre(self) -> None:
        """Mantic declared 650's ``nominal_max`` (100000) one step above its last bin centre; the labels follow the bins."""
        grid = pmf_grid(_discrete(_POST_650))
        assert grid.style == "center"
        assert len(grid.labels) == 450
        assert grid.labels[:3] == ("55000", "55100", "55200")
        assert grid.labels[-1] == "99900"
        assert format_bin_value(_POST_650.nominal_upper) not in grid.labels

    def test_a_coarse_count_grid_with_650s_top_still_labels_its_counts(self) -> None:
        """The style is read off the lower bound alone, so the writer's off-by-one at the top cannot turn ``0`` into ``-0.5 to 0.5``."""
        grid = pmf_grid(_discrete(_COUNT_GRID_WITH_650S_TOP))
        assert grid.style == "center"
        assert grid.labels == tuple(str(count) for count in range(20))
        assert grid.keys == (*grid.labels, PMF_ABOVE_RANGE_KEY)

    def test_post_643_has_an_above_range_key_and_no_below_range_key(self) -> None:
        grid = pmf_grid(_discrete(_POST_643))
        assert grid.labels == tuple(str(count) for count in range(21))
        assert grid.keys == (*grid.labels, PMF_ABOVE_RANGE_KEY)
        assert PMF_BELOW_RANGE_KEY not in grid.keys

    def test_both_open_bounds_add_both_reserved_keys_in_prompt_order(self) -> None:
        grid = pmf_grid(_discrete(_POST_650))
        assert grid.keys[0] == PMF_BELOW_RANGE_KEY
        assert grid.keys[-1] == PMF_ABOVE_RANGE_KEY
        assert grid.keys[1:-1] == grid.labels
        assert len(grid.keys) == 452

    def test_post_560_nominal_bounds_on_the_range_edges_label_right_closed_intervals(self) -> None:
        grid = pmf_grid(_numeric(_POST_560))
        assert grid.style == "interval"
        assert len(grid.labels) == 30
        assert grid.labels[:2] == ("100 to 300", "300 to 500")
        assert grid.labels[-1] == "5900 to 6100"

    def test_a_zero_point_grid_never_labels_by_centre(self) -> None:
        """On a geometric axis the bins have no common width, so ``lower + (k + 0.5) * step`` names nothing."""
        shape = _CorpusShape(999, 11, 1.0, 1000.0, 1.0, 1000.0, False, False)
        grid = pmf_grid(_numeric(shape, zero_point=0.0))
        assert grid.style == "interval"
        assert grid.edges[1] != pytest.approx(1.0 + 999.0 / 10)
        assert grid.labels[0] == f"1 to {format_bin_value(grid.edges[1])}"

    def test_edges_are_the_repos_cdf_value_grid(self) -> None:
        grid = pmf_grid(_discrete(_POST_619))
        expected = build_cdf_value_grid(_POST_619.lower_bound, _POST_619.upper_bound, None, _POST_619.cdf_size)
        assert grid.edges == pytest.approx(tuple(expected))
        assert len(grid.edges) == _POST_619.cdf_size
        assert len(grid.labels) == _POST_619.cdf_size - 1


class TestFormatBinValue:
    @pytest.mark.parametrize(
        ("value", "rendered"),
        [
            pytest.param(55000.0, "55000", id="integer-valued-float"),
            pytest.param(55.1, "55.1", id="one-decimal"),
            pytest.param(0.25, "0.25", id="two-decimals"),
            pytest.param(77.35000000000001, "77.35", id="float-residue-is-rounded-away"),
            pytest.param(123456.7, "123456.7", id="seven-significant-digits-survive"),
            pytest.param(-0.0, "0", id="negative-zero"),
            pytest.param(0.0, "0", id="zero"),
            pytest.param(-1.5, "-1.5", id="negative"),
            pytest.param(77.15 + 0.05, "77.2", id="post-619-first-centre"),
        ],
    )
    def test_renders_the_displayed_value(self, value: float, rendered: str) -> None:
        assert format_bin_value(value) == rendered


class TestFoldBinLabel:
    @pytest.mark.parametrize(
        ("written", "canonical"),
        [
            pytest.param("7.0", "7", id="trailing-decimal-zero"),
            pytest.param(" 7 ", "7", id="padding"),
            pytest.param("55,000", "55000", id="thousands-separator"),
            pytest.param("7", "7", id="already-canonical"),
            pytest.param("2026-09-16", "2026-09-16", id="iso-date"),
            pytest.param(" 2026-09-16 ", "2026-09-16", id="padded-iso-date"),
            pytest.param("Below_Range", PMF_BELOW_RANGE_KEY, id="reserved-key-case"),
        ],
    )
    def test_folds_onto_the_canonical_label(self, written: str, canonical: str) -> None:
        assert fold_bin_label(written) == canonical

    @pytest.mark.parametrize("written", ["2026-9-16", "Sep 16", "2026-09-16T12:00:00Z", "16 September 2026"])
    def test_a_non_iso_date_does_not_fold_onto_the_canonical_day(self, written: str) -> None:
        assert fold_bin_label(written) != "2026-09-16"

    def test_a_folded_interval_stays_an_interval(self) -> None:
        assert fold_bin_label(" 100 to 300 ") == "100 to 300"


def _every_style() -> list[PmfGrid]:
    return [
        pmf_grid(as_epoch_question(load_preseason_date_question())),
        pmf_grid(as_epoch_question(load_legacy_date_question())),
        pmf_grid(_discrete(_POST_253)),
        pmf_grid(_discrete(_POST_619)),
        pmf_grid(_discrete(_POST_643)),
        pmf_grid(_discrete(_POST_650)),
        pmf_grid(_numeric(_POST_560)),
    ]


class TestTheKeysAreMatchableByTheFold:
    """What the block rung relies on: folding both sides maps a model's key onto exactly one grid key."""

    def test_distinct_keys_stay_distinct_after_folding(self) -> None:
        for grid in _every_style():
            folded = [fold_bin_label(key) for key in grid.keys]
            assert len(set(folded)) == len(grid.keys), grid.style

    def test_centre_labels_that_collapse_at_nine_decimals_fail_shut(self) -> None:
        """Three 1e-10-wide bins all render as ``0``; a grid that cannot name its bins raises rather than relabelling."""
        shape = _CorpusShape(1, 4, 0.0, 3e-10, 0.0, 3e-10, False, False)
        with pytest.raises(ValueError, match=r"fold onto the same key.*'0'"):
            pmf_grid(_numeric(shape))

    def test_interval_labels_that_collapse_at_nine_decimals_fail_shut(self) -> None:
        """Sub-nanometre bins on a log axis all render as ``1000000 to 1000000``."""
        shape = _CorpusShape(2, 4, 1e6, 1e6 + 3e-10, 1e6, 1e6 + 3e-10, False, False)
        with pytest.raises(ValueError, match="1000000 to 1000000"):
            pmf_grid(_numeric(shape, zero_point=0.0))

    def test_the_fold_is_idempotent(self) -> None:
        for grid in _every_style():
            for key in grid.keys:
                assert fold_bin_label(fold_bin_label(key)) == fold_bin_label(key)

    def test_every_key_is_its_own_canonical_form_except_on_a_timestamp_grid(self) -> None:
        """Numbers, dates, intervals and the reserved keys carry no letters to lowercase; ``T``/``Z`` do."""
        for grid in _every_style():
            if grid.style == "timestamp":
                continue
            assert all(fold_bin_label(key) == key for key in grid.keys), grid.style

    def test_the_reserved_keys_are_the_data_contract_tokens(self) -> None:
        assert PMF_BELOW_RANGE_KEY == "below_range"
        assert PMF_ABOVE_RANGE_KEY == "above_range"
