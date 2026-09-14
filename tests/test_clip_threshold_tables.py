"""Tests for the derived tables the report is built from.

The extreme-bin calibration counts, the insurance view's break-even rate, the shared Jeffreys
interval both of them quote, the thin-publish floor on single-survivor publishes, and the row that
prices the live clamp on the records published under the older one.
"""

from __future__ import annotations

import math

import pytest
from scipy.stats import beta

from metaculus_bot.constants import THIN_PUBLISH_BINARY_FLOOR
from metaculus_bot.performance_analysis.analysis import jeffreys_ci
from metaculus_bot.performance_analysis.clip_threshold_report import render_report
from metaculus_bot.performance_analysis.clip_threshold_selection import binomial_cdf
from metaculus_bot.performance_analysis.clip_threshold_sweep import BOOTSTRAP_CL, build_clip_records, sweep_row
from metaculus_bot.performance_analysis.clip_threshold_tables import (
    WINDOW_OLDER_REGIME,
    binary_extreme_bins,
    compute_report,
    insurance_row,
    jeffreys_interval,
    single_survivor_report,
)
from metaculus_bot.performance_analysis.clip_threshold_windows import (
    WINDOW_ALL,
    WINDOW_CURRENT_CLAMP,
    WINDOW_ERA_PRE_FLIP,
)
from metaculus_bot.performance_analysis.eras import B4E9DF0_MERGED_AT
from tests.clip_threshold_fakes import (
    AFTER_FT_0292,
    AFTER_WIDENING,
    AS_OF,
    BEFORE_WIDENING,
    binary_record,
    mc_record,
    one_binary,
    one_mc,
)


class TestExtremeBins:
    """The calibration table the whole decision rests on: how often did a 1-2% call hit?"""

    def test_low_bins_count_yes_resolutions(self):
        records = build_clip_records(
            [
                binary_record(question_id=7001, p_yes=0.01, resolution=False, created_at=BEFORE_WIDENING),
                binary_record(question_id=7002, p_yes=0.01, resolution=True, created_at=BEFORE_WIDENING),
                binary_record(question_id=7003, p_yes=0.04, resolution=False),
            ],
            "binary",
        ).records
        bins = {b.label: b for b in binary_extreme_bins(records, window=WINDOW_ALL)}
        assert bins["<= 0.01"].n == 2
        assert bins["<= 0.01"].hits == 1
        # Two records at p_yes 0.01: the bot's own prices implied 0.02 YES between them.
        assert bins["<= 0.01"].expected_hits == pytest.approx(0.02, abs=1e-12)
        assert bins["(0.03, 0.05]"].n == 1
        assert bins["(0.03, 0.05]"].hits == 0
        assert bins["(0.03, 0.05]"].implied_rate == pytest.approx(0.04, abs=1e-9)

    def test_high_bins_count_no_resolutions(self):
        records = build_clip_records(
            [
                binary_record(question_id=7010, p_yes=0.99, resolution=False, created_at=BEFORE_WIDENING),
                binary_record(question_id=7011, p_yes=0.985, resolution=True, created_at=BEFORE_WIDENING),
            ],
            "binary",
        ).records
        bins = {b.label: b for b in binary_extreme_bins(records, window=WINDOW_ALL)}
        assert bins[">= 0.99"].n == 1
        assert bins[">= 0.99"].hits == 1
        assert bins["[0.98, 0.99)"].n == 1
        assert bins["[0.98, 0.99)"].hits == 0
        # The implied rate is the rate of the counted event (a NO), not of p_yes.
        assert bins["[0.98, 0.99)"].implied_rate == pytest.approx(0.015, abs=1e-9)
        # And so is the expected-hits column: one publish at 0.985 implies 0.015 of a NO, not 0.985.
        assert bins["[0.98, 0.99)"].expected_hits == pytest.approx(0.015, abs=1e-12)

    def test_bins_are_computed_per_window_not_only_pooled(self):
        """The pooled table is dominated by the pre-flip era; the decision is about today.

        A 0.01 publish could only happen under the pre-flip clamp, so it belongs in the
        pooled bin and must NOT appear in the current-regime one; a 0.04 publish from after
        the flip belongs in both.
        """
        report = compute_report(
            [
                binary_record(question_id=7020, p_yes=0.01, resolution=False, created_at=BEFORE_WIDENING),
                binary_record(question_id=7021, p_yes=0.04, resolution=True, created_at=AFTER_WIDENING),
            ],
            dataset_path="synthetic",
            as_of=AS_OF,
            exclude_qids=frozenset(),
        ).type_report("binary")
        pooled = {b.label: b for b in report.extreme_bins_for(WINDOW_ALL)}
        current = {b.label: b for b in report.extreme_bins_for(WINDOW_CURRENT_CLAMP)}
        assert (pooled["<= 0.01"].n, pooled["<= 0.01"].hits) == (1, 0)
        assert (current["<= 0.01"].n, current["<= 0.01"].hits) == (0, 0)
        assert (pooled["(0.03, 0.05]"].n, pooled["(0.03, 0.05]"].hits) == (1, 1)
        assert (current["(0.03, 0.05]"].n, current["(0.03, 0.05]"].hits) == (1, 1)
        # Every populated window gets its own copy of the table, and each row says which.
        windows = {b.window for b in report.extreme_bins}
        assert windows == {w.label for w in report.populated_windows}
        assert all(b.window in windows for b in report.extreme_bins)


class TestInsuranceView:
    def test_break_even_and_binomial_on_a_binary_row(self):
        records = [
            one_binary(question_id=9080, p_yes=0.02, resolution=False),
            one_binary(question_id=9081, p_yes=0.02, resolution=False),
            one_binary(question_id=9082, p_yes=0.02, resolution=True),
        ]
        row = insurance_row(sweep_row(records, question_type="binary", side="floor_only", window=WINDOW_ALL, c=0.05))
        gain = 100.0 * math.log(0.05 / 0.02)
        loss = 100.0 * math.log(0.95 / 0.98)
        assert row.n_affected == 3
        assert row.hits == 1
        assert row.break_even_rate == pytest.approx(-3 * loss / (3 * gain - 3 * loss), abs=1e-9)
        assert row.p_hits_at_most_if_rate_c == pytest.approx(binomial_cdf(1, 3, 0.05), abs=1e-12)
        assert row.best_case_sum_delta == pytest.approx(3 * gain, abs=1e-9)
        assert row.ci_lo is not None
        assert row.ci_hi is not None
        assert row.ci_lo < 1 / 3 < row.ci_hi
        assert row.rejected_at_ci_upper is False

    def test_zero_hits_rejects_when_break_even_clears_the_interval(self):
        records = [
            one_binary(question_id=9090 + i, p_yes=0.01, resolution=False, created_at=BEFORE_WIDENING)
            for i in range(150)
        ]
        row = insurance_row(sweep_row(records, question_type="binary", side="floor_only", window=WINDOW_ALL, c=0.05))
        assert row.hits == 0
        assert row.p_hits_at_most_if_rate_c == pytest.approx(0.95**150, abs=1e-12)
        assert row.rejected_at_ci_upper is True

    def test_mc_rows_have_no_break_even(self):
        record = one_mc(question_id=9095, options=["A", "B", "C", "D"], probs=[0.70, 0.27, 0.02, 0.01], resolution="A")
        row = insurance_row(
            sweep_row([record], question_type="multiple_choice", side="floor_only", window=WINDOW_ALL, c=0.05)
        )
        assert row.n_affected == 1
        assert row.break_even_rate is None
        assert row.p_hits_at_most_if_rate_c is None
        assert row.rejected_at_ci_upper is None

    def test_binomial_cdf_closed_form(self):
        assert binomial_cdf(0, 10, 0.1) == pytest.approx(0.9**10, abs=1e-12)
        assert binomial_cdf(10, 10, 0.3) == pytest.approx(1.0, abs=1e-12)
        assert binomial_cdf(0, 0, 0.3) == 1.0


class TestJeffreysInterval:
    """The extreme-bin and insurance intervals delegate to the package's one Jeffreys implementation.

    ``rejected_at_ci_upper`` rides on ``ci_hi`` directly (the headline row holds by 4e-4), so the
    prior is pinned to a literal Beta(0.5, 0.5) rather than bracket-checked: a Beta(1, 1) upper
    bound on 0 of 150 is 0.0242 against the correct 0.0166 and would pass any bracket test.
    """

    @pytest.mark.parametrize(("k", "n"), [(0, 150), (1, 3), (3, 5), (9, 11)])
    def test_bounds_are_the_shared_jeffreys_ci_at_the_sweep_level(self, k: int, n: int):
        assert jeffreys_interval(k, n)[1:] == jeffreys_ci(k, n, cl=BOOTSTRAP_CL)[1:]

    @pytest.mark.parametrize(("k", "n"), [(0, 150), (1, 3)])
    def test_point_is_the_raw_rate_and_bounds_are_the_half_half_prior(self, k: int, n: int):
        rate, lo, hi = jeffreys_interval(k, n)
        assert rate == pytest.approx(k / n, abs=1e-12)
        assert lo == pytest.approx(float(beta.ppf(0.025, 0.5 + k, 0.5 + n - k)), abs=1e-12)
        assert hi == pytest.approx(float(beta.ppf(0.975, 0.5 + k, 0.5 + n - k)), abs=1e-12)

    def test_empty_bin_has_no_rate_or_interval(self):
        assert jeffreys_interval(0, 0) == (None, None, None)


class TestSingleSurvivorThinFloor:
    """The thin publish floor fires on exactly one cohort, and the detector needs a date guard."""

    def test_genuine_single_survivor_publishes_are_priced_and_artifacts_counted(self):
        records = build_clip_records(
            [
                # Genuine: one member, after the MIN_FORECASTERS_TO_PUBLISH merge, published 0.03 YES.
                binary_record(
                    question_id=9120, p_yes=0.03, resolution=True, created_at=AFTER_FT_0292, per_model={"g": "3.0%"}
                ),
                # Genuine but inside the floor: moves by exactly 0.
                binary_record(
                    question_id=9121, p_yes=0.50, resolution=True, created_at=AFTER_FT_0292, per_model={"g": "50.0%"}
                ),
                # One member BEFORE the merge: a trimmed-comment parse artifact, counted not priced.
                binary_record(
                    question_id=9122,
                    p_yes=0.03,
                    resolution=False,
                    created_at=BEFORE_WIDENING,
                    per_model={"Forecaster 1": "3.0%"},
                ),
                # A stacked one-member record is the stacker's output, not a survivor.
                binary_record(
                    question_id=9123,
                    p_yes=0.03,
                    resolution=True,
                    created_at=AFTER_FT_0292,
                    per_model={"s": "3.0%"},
                    stacker_outcome="primary",
                ),
                # Three members: not the thin floor's cohort at all.
                binary_record(
                    question_id=9124,
                    p_yes=0.03,
                    resolution=True,
                    created_at=AFTER_FT_0292,
                    per_model={"a": "2.0%", "b": "3.0%", "c": "4.0%"},
                ),
            ],
            "binary",
        ).records
        report = single_survivor_report(records)
        assert [row.question_id for row in report.rows] == ["9120", "9121"]
        assert report.rows[0].delta == pytest.approx(100.0 * math.log(THIN_PUBLISH_BINARY_FLOOR / 0.03), abs=1e-9)
        assert report.rows[1].delta == 0.0
        assert report.sum_delta == pytest.approx(report.rows[0].delta, abs=1e-12)
        assert report.n_single_member_before_boundary == 1
        assert report.boundary == B4E9DF0_MERGED_AT.isoformat()


class TestOlderRegimeRow:
    """The live floor priced on the records published under the OLDER clamp, per type's own boundary.

    The binary clamp changed at the widening flip, so for binary the older regime IS
    ``era_pre_flip``; the MC clamp changed two months later at the ft 0.2.92 unfreeze, so for MC
    the older regime also holds the post-flip records. The row is keyed on the current-regime
    window's complement, which is right for both, where an ``era_pre_flip`` lookup was right for
    one (it dropped 26 of 91 MC records on the real archive).
    """

    def test_mc_older_regime_is_the_current_window_complement_not_era_pre_flip(self):
        """Both older records were published under the 0.005 floor and both carry a 0.007 option, so
        both are in the cohort and both move at the live 0.01 floor.
        """
        ballot = [0.60, 0.393, 0.007]
        report = compute_report(
            [
                mc_record(
                    question_id=9190, options=["A", "B", "C"], probs=ballot, resolution="A", created_at=BEFORE_WIDENING
                ),
                mc_record(
                    question_id=9191, options=["A", "B", "C"], probs=ballot, resolution="A", created_at=AFTER_WIDENING
                ),
                mc_record(
                    question_id=9192,
                    options=["A", "B", "C"],
                    probs=[0.60, 0.39, 0.01],
                    resolution="A",
                    created_at=AFTER_FT_0292,
                ),
            ],
            dataset_path="synthetic",
            as_of=AS_OF,
            exclude_qids=frozenset(),
        )
        mc = report.type_report("multiple_choice")
        row = mc.older_regime
        assert row is not None
        assert (row.window, row.c) == (WINDOW_OLDER_REGIME, 0.01)
        assert (row.n, row.n_affected) == (2, 2)
        pre_flip = next(w for w in mc.windows if w.label == WINDOW_ERA_PRE_FLIP)
        assert len(pre_flip.records) == 1
        assert f"priced on the {row.n} records published before the clamp in force went live" in render_report(report)

    def test_binary_older_regime_coincides_with_era_pre_flip(self):
        report = compute_report(
            [
                binary_record(question_id=9193, p_yes=0.01, resolution=False, created_at=BEFORE_WIDENING),
                binary_record(question_id=9194, p_yes=0.30, resolution=False, created_at=AFTER_WIDENING),
            ],
            dataset_path="synthetic",
            as_of=AS_OF,
            exclude_qids=frozenset(),
        ).type_report("binary")
        row = report.older_regime
        assert row is not None
        assert (row.c, row.n, row.n_affected) == (0.02, 1, 1)
        assert row.sum_delta == pytest.approx(100.0 * math.log(0.98 / 0.99), abs=1e-9)
        pre_flip = next(w for w in report.windows if w.label == WINDOW_ERA_PRE_FLIP)
        assert len(pre_flip.records) == row.n

    def test_no_older_records_means_no_row(self):
        report = compute_report(
            [binary_record(question_id=9195, p_yes=0.30, resolution=False, created_at=AFTER_WIDENING)],
            dataset_path="synthetic",
            as_of=AS_OF,
            exclude_qids=frozenset(),
        ).type_report("binary")
        assert report.older_regime is None
