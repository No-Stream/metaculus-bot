"""Tests for the counterfactual clip-threshold sweep.

The arithmetic is pinned against hand-computed ``100 * ln(new / old)`` spot-peer
figures on synthetic records, and the censoring rules are pinned on records
published exactly at their in-force clamp — the case where a LOOSER clip cannot be
priced because the raw member value was destroyed by the clamp that was live at
the time.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta

import pytest
from scipy.stats import beta

from metaculus_bot import bootstrap
from metaculus_bot.constants import MC_PROB_MAX, MC_PROB_MIN, THIN_PUBLISH_BINARY_FLOOR
from metaculus_bot.mc_processing import clamp_and_renormalize_probs
from metaculus_bot.performance_analysis.analysis import (
    B4E9DF0_MERGED_AT,
    FT_0292_MERGED_AT,
    WIDENING_FLIP_MERGED_AT,
    jeffreys_ci,
)
from metaculus_bot.performance_analysis.clip_threshold import main
from metaculus_bot.performance_analysis.clip_threshold_report import render_report
from metaculus_bot.performance_analysis.clip_threshold_selection import (
    binomial_cdf,
    censored_rows_at_argmax_score,
    oob_argmax,
)
from metaculus_bot.performance_analysis.clip_threshold_sweep import (
    BINARY_FLOOR_GRID,
    BOOTSTRAP_B,
    BOOTSTRAP_CL,
    BOOTSTRAP_SEED,
    DELTA_ATOL,
    MC_FLOOR_GRID,
    ClipRecord,
    argmax_row,
    argmax_rows,
    bootstrap_mean_ci,
    build_clip_records,
    clip_delta,
    in_force_bounds,
    member_censored,
    sweep_row,
)
from metaculus_bot.performance_analysis.clip_threshold_tables import (
    MIN_OOS_COMPLEMENT_N,
    WINDOW_OLDER_REGIME,
    binary_extreme_bins,
    compute_report,
    cross_check_row,
    insurance_row,
    jeffreys_interval,
    nesting_rows,
    oos_row,
    regime_span,
    replay_cohort,
    single_survivor_report,
)
from metaculus_bot.performance_analysis.clip_threshold_windows import (
    LOOKBACK_DAYS,
    WINDOW_ALL,
    WINDOW_CURRENT_CLAMP,
    WINDOW_ERA_POST_FLIP,
    WINDOW_ERA_PRE_FLIP,
    WINDOW_LAST_90D,
    WINDOW_TRIPLE_ERA,
    build_windows,
    nested_windows,
    window_labels,
)
from metaculus_bot.performance_analysis.width_monitor import WIDENING_FLIP

# A moment inside each clamp regime, so a factory default never straddles a boundary.
BEFORE_WIDENING = "2026-01-15T00:00:00Z"
AFTER_WIDENING = "2026-06-15T00:00:00Z"
AFTER_FT_0292 = "2026-08-01T00:00:00Z"
AS_OF = datetime(2026, 9, 2, tzinfo=UTC)


def _binary_record(
    *,
    question_id: int,
    p_yes: float,
    resolution: bool,
    created_at: str | None = AFTER_WIDENING,
    per_model: dict[str, str] | None = None,
    per_base_model: dict[str, str] | None = None,
    stacker_outcome: str | None = "skipped_config_off",
) -> dict:
    """A minimal binary performance record shaped the way the collector writes one."""
    return {
        "question_id": question_id,
        "type": "binary",
        "our_prob_yes": p_yes,
        "our_forecast_values": [1.0 - p_yes, p_yes],
        "resolution_parsed": resolution,
        "bot_comment_created_at": created_at,
        "per_model_forecasts": per_model or {},
        "per_base_model_forecasts": per_base_model or {},
        "stacker_outcome": stacker_outcome,
        "was_stacked": False,
        "metaculus_scores": {"spot_peer_score": 0.0},
    }


def _mc_record(
    *,
    question_id: int,
    options: list[str],
    probs: list[float],
    resolution: str,
    created_at: str | None = AFTER_FT_0292,
    per_model: dict[str, dict[str, float]] | None = None,
) -> dict:
    return {
        "question_id": question_id,
        "type": "multiple_choice",
        "our_prob_yes": None,
        "our_forecast_values": list(probs),
        "options": list(options),
        "resolution_parsed": resolution,
        "bot_comment_created_at": created_at,
        "per_model_forecasts": per_model or {},
        "stacker_outcome": "skipped_config_off",
        "was_stacked": False,
        "metaculus_scores": {"spot_peer_score": 0.0},
    }


def _one_binary(**kwargs):
    """The single :class:`ClipRecord` built from one hand-made binary dict."""
    [record] = build_clip_records([_binary_record(**kwargs)], "binary").records
    return record


def _one_mc(**kwargs):
    [record] = build_clip_records([_mc_record(**kwargs)], "multiple_choice").records
    return record


class TestTighteningArithmetic:
    """Group 1 — a tighter clip's spot-peer delta is exactly ``100 * ln(new / old)``."""

    def test_no_resolution_gains_when_the_floor_lifts_off_it(self):
        # Published 0.02 (the in-force floor after the widening flip) on a NO. A 0.05 floor
        # moves our NO mass 0.98 -> 0.95, which is a LOSS.
        record = _one_binary(question_id=1, p_yes=0.02, resolution=False)
        clip = clip_delta(record, 0.05, side="floor_only")
        assert clip.delta == pytest.approx(100.0 * math.log(0.95 / 0.98), abs=1e-9)
        assert clip.delta < 0.0
        assert clip.affected is True
        assert clip.censored is False

    def test_yes_resolution_gains_from_the_same_floor(self):
        record = _one_binary(question_id=2, p_yes=0.02, resolution=True)
        clip = clip_delta(record, 0.05, side="floor_only")
        assert clip.delta == pytest.approx(100.0 * math.log(0.05 / 0.02), abs=1e-9)
        assert clip.delta > 0.0

    def test_unaffected_record_is_exactly_zero(self):
        record = _one_binary(question_id=3, p_yes=0.30, resolution=True)
        clip = clip_delta(record, 0.05, side="floor_only")
        assert clip.delta == 0.0
        assert clip.affected is False

    def test_floor_only_leaves_the_high_side_alone(self):
        record = _one_binary(question_id=4, p_yes=0.98, resolution=True)
        assert clip_delta(record, 0.05, side="floor_only").delta == 0.0
        ceiling = clip_delta(record, 0.05, side="ceiling_only")
        assert ceiling.delta == pytest.approx(100.0 * math.log(0.95 / 0.98), abs=1e-9)

    def test_symmetric_moves_both_sides(self):
        low = _one_binary(question_id=5, p_yes=0.02, resolution=False)
        high = _one_binary(question_id=6, p_yes=0.98, resolution=False)
        assert clip_delta(low, 0.05, side="symmetric").delta == pytest.approx(100.0 * math.log(0.95 / 0.98), abs=1e-9)
        # A NO at 0.98 gains: the ceiling pulls our (tiny) NO mass 0.02 -> 0.05.
        assert clip_delta(high, 0.05, side="symmetric").delta == pytest.approx(100.0 * math.log(0.05 / 0.02), abs=1e-9)

    def test_hits_on_clipped_side_counts_the_records_the_floor_paid_on(self):
        records = [
            _one_binary(question_id=10, p_yes=0.02, resolution=True),
            _one_binary(question_id=11, p_yes=0.02, resolution=False),
            _one_binary(question_id=12, p_yes=0.60, resolution=True),
        ]
        row = sweep_row(records, question_type="binary", side="floor_only", window=WINDOW_ALL, c=0.05)
        assert row.n == 3
        assert row.n_affected == 2
        assert row.hits_on_clipped_side == 1
        assert row.sum_delta == pytest.approx(100.0 * math.log(0.05 / 0.02) + 100.0 * math.log(0.95 / 0.98), abs=1e-9)
        assert row.mean_delta == pytest.approx(row.sum_delta / 3, abs=1e-9)

    def test_mc_floor_renormalises_and_prices_the_resolving_option(self):
        record = _one_mc(
            question_id=20,
            options=["A", "B", "C", "D"],
            probs=[0.70, 0.27, 0.02, 0.01],
            resolution="A",
        )
        clip = clip_delta(record, 0.05, side="floor_only")
        expected_vector = clamp_and_renormalize_probs([0.70, 0.27, 0.02, 0.01], lo=0.05, hi=0.99)
        assert clip.delta == pytest.approx(100.0 * math.log(expected_vector[0] / 0.70), abs=1e-9)
        # Raising two floors steals mass from the leader, so the leader-resolving case loses.
        assert clip.delta < 0.0

    def test_mc_floor_pays_when_the_floored_option_resolves(self):
        record = _one_mc(
            question_id=21,
            options=["A", "B", "C", "D"],
            probs=[0.70, 0.27, 0.02, 0.01],
            resolution="D",
        )
        clip = clip_delta(record, 0.05, side="floor_only")
        expected_vector = clamp_and_renormalize_probs([0.70, 0.27, 0.02, 0.01], lo=0.05, hi=0.99)
        assert clip.delta == pytest.approx(100.0 * math.log(expected_vector[3] / 0.01), abs=1e-9)
        assert clip.delta > 0.0

    def test_top1_share_is_one_when_a_single_question_carries_the_row(self):
        records = [
            _one_binary(question_id=30, p_yes=0.02, resolution=True),
            _one_binary(question_id=31, p_yes=0.60, resolution=True),
        ]
        row = sweep_row(records, question_type="binary", side="floor_only", window=WINDOW_ALL, c=0.05)
        assert row.top1_share == pytest.approx(1.0, abs=1e-12)

    def test_top1_share_is_none_when_nothing_moved(self):
        records = [_one_binary(question_id=32, p_yes=0.60, resolution=True)]
        row = sweep_row(records, question_type="binary", side="floor_only", window=WINDOW_ALL, c=0.05)
        assert row.top1_share is None
        assert row.top1_question_id is None
        assert row.top1_spot_peer is None

    def test_mc_renormalisation_noise_is_not_a_driver(self):
        """A candidate looser than the clamp in force moves nothing, so it has no top1.

        The MC counterfactual runs the live clamp-and-renormalise, and a published vector
        whose floats sum to 1 + 2e-16 comes back perturbed by ~1e-14 points even when the
        bounds are exactly the ones in force. Computing a share over that noise reported a
        concentration of 0.07 and named a question the candidate never touched, on a row
        whose own n_affected was 0.
        """
        record = _one_mc(
            question_id=43052,
            options=["A", "B", "C", "D"],
            probs=[0.020833333333333336, 0.03125, 0.13541666666666669, 0.8125000000000001],
            resolution="A",
        )
        row = sweep_row([record], question_type="multiple_choice", side="ceiling_only", window=WINDOW_ALL, c=0.005)
        assert row.n_affected == 0
        assert row.sum_delta != 0.0
        assert abs(row.sum_delta) < DELTA_ATOL
        assert row.top1_share is None
        assert row.top1_question_id is None
        assert row.top1_spot_peer is None

    def test_top1_names_the_driving_question(self):
        """A top1_share near 1.0 is only actionable if the row says WHICH question it is."""
        records = [
            build_clip_records(
                [
                    {
                        **_binary_record(question_id=33, p_yes=0.02, resolution=True),
                        "metaculus_scores": {"spot_peer_score": -105.27},
                    }
                ],
                "binary",
            ).records[0],
            _one_binary(question_id=34, p_yes=0.60, resolution=True),
        ]
        row = sweep_row(records, question_type="binary", side="floor_only", window=WINDOW_ALL, c=0.05)
        assert row.top1_question_id == "33"
        assert row.top1_spot_peer == pytest.approx(-105.27)
        assert row.top1_share == pytest.approx(1.0, abs=1e-12)


class TestLoosenBounds:
    """Group 2 — below the in-force floor the raw member value is unobservable."""

    def test_record_at_the_floor_is_censored_and_bounded(self):
        record = _one_binary(question_id=40, p_yes=0.02, resolution=False)
        clip = clip_delta(record, 0.01, side="floor_only")
        assert clip.censored is True
        assert clip.affected is False
        assert clip.delta == 0.0
        # Lower bound: raw was exactly the floor, nothing moves. Upper: raw was <= 0.01.
        assert clip.loosen_at_c == pytest.approx(100.0 * math.log(0.99 / 0.98), abs=1e-9)

    def test_censored_yes_loses_under_the_upper_scenario(self):
        record = _one_binary(question_id=41, p_yes=0.02, resolution=True)
        clip = clip_delta(record, 0.01, side="floor_only")
        assert clip.censored is True
        assert clip.loosen_at_c == pytest.approx(100.0 * math.log(0.01 / 0.02), abs=1e-9)
        assert clip.loosen_at_c < 0.0

    def test_record_above_the_floor_contributes_zero_to_both_bounds(self):
        record = _one_binary(question_id=42, p_yes=0.30, resolution=False)
        clip = clip_delta(record, 0.01, side="floor_only")
        assert clip.censored is False
        assert clip.delta == 0.0
        assert clip.loosen_at_c == 0.0

    def test_row_reports_both_bounds_and_the_censored_count(self):
        records = [
            _one_binary(question_id=43, p_yes=0.02, resolution=False),
            _one_binary(question_id=44, p_yes=0.02, resolution=True),
            _one_binary(question_id=45, p_yes=0.30, resolution=False),
        ]
        row = sweep_row(records, question_type="binary", side="floor_only", window=WINDOW_ALL, c=0.01)
        assert row.censored_n == 2
        assert row.n_loosening == 3
        assert row.sum_delta == 0.0
        assert row.sum_delta_lower == 0.0
        gain = 100.0 * math.log(0.99 / 0.98)
        loss = 100.0 * math.log(0.01 / 0.02)
        assert row.sum_delta_upper == pytest.approx(gain + loss, abs=1e-9)
        # The identified bracket takes each censored record's own best/worst case, which is
        # why it is wider than the two named scenarios when the signs disagree.
        assert row.bracket_lo == pytest.approx(loss, abs=1e-9)
        assert row.bracket_hi == pytest.approx(gain, abs=1e-9)

    def test_pre_flip_record_tightens_at_a_candidate_above_its_own_floor(self):
        # In-force floor 0.01 before the widening flip, so c = 0.015 TIGHTENS this record
        # even though it sits at its floor — a tighter clip is exact regardless of the raw.
        record = _one_binary(question_id=46, p_yes=0.01, resolution=False, created_at=BEFORE_WIDENING)
        clip = clip_delta(record, 0.015, side="floor_only")
        assert clip.censored is False
        assert clip.delta == pytest.approx(100.0 * math.log(0.985 / 0.99), abs=1e-9)

    def test_mc_option_at_the_floor_censors_the_record(self):
        record = _one_mc(
            question_id=47,
            options=["A", "B", "C"],
            probs=[0.88, 0.11, 0.01],
            resolution="A",
        )
        clip = clip_delta(record, 0.005, side="floor_only")
        assert clip.censored is True
        assert clip.delta == 0.0
        assert clip.loosen_at_c > 0.0

    def test_mc_record_clear_of_the_floor_is_not_censored(self):
        record = _one_mc(
            question_id=48,
            options=["A", "B", "C"],
            probs=[0.50, 0.30, 0.20],
            resolution="A",
        )
        clip = clip_delta(record, 0.005, side="floor_only")
        assert clip.censored is False
        assert clip.loosen_at_c == 0.0

    def test_ceiling_only_loosening_is_counted_on_the_ceiling(self):
        """A publish at the in-force 0.98 ceiling swept on the ceiling side at c = 0.005 asks for
        a 0.995 ceiling: looser than the one in force, so the record is censored AND a loosening
        record. ``n_loosening`` used to inspect only the floor and read 0 on every ceiling-only
        row while ``censored_n`` on the same row read 1."""
        record = _one_binary(question_id=49, p_yes=0.98, resolution=True)
        row = sweep_row([record], question_type="binary", side="ceiling_only", window=WINDOW_ALL, c=0.005)
        assert row.censored_n == 1
        assert row.n_loosening == 1
        assert row.sum_delta == 0.0
        assert row.exact is False
        assert row.sum_delta_upper == pytest.approx(100.0 * math.log(0.995 / 0.98), abs=1e-9)
        # The same candidate on the floor side is a loosening too (0.005 < 0.02), but this publish
        # sits nowhere near the floor, so it is counted as loosening and NOT as censored.
        floor_side = sweep_row([record], question_type="binary", side="floor_only", window=WINDOW_ALL, c=0.005)
        assert (floor_side.n_loosening, floor_side.censored_n) == (1, 0)


class TestInForceClampLookup:
    """Group 3 — the clamp in force is looked up from the bot-comment timestamp."""

    def test_binary_boundary(self):
        assert in_force_bounds("binary", WIDENING_FLIP_MERGED_AT - timedelta(seconds=1)) == (0.01, 0.99)
        assert in_force_bounds("binary", WIDENING_FLIP_MERGED_AT) == (0.02, 0.98)
        assert in_force_bounds("binary", WIDENING_FLIP_MERGED_AT + timedelta(days=30)) == (0.02, 0.98)

    def test_mc_boundary(self):
        assert in_force_bounds("multiple_choice", FT_0292_MERGED_AT - timedelta(seconds=1)) == (0.005, 0.995)
        assert in_force_bounds("multiple_choice", FT_0292_MERGED_AT) == (0.01, 0.99)

    def test_no_timestamp_falls_back_to_the_widest_regime(self):
        # Undatable records get the LOOSEST historical clamp: it is the assumption that
        # claims the least, since a censoring claim needs to know which floor was live.
        assert in_force_bounds("binary", None) == (0.01, 0.99)
        assert in_force_bounds("multiple_choice", None) == (0.005, 0.995)

    def test_record_carries_the_bounds_its_timestamp_implies(self):
        assert _one_binary(question_id=50, p_yes=0.5, resolution=True, created_at=BEFORE_WIDENING).in_force_lo == 0.01
        assert _one_binary(question_id=51, p_yes=0.5, resolution=True, created_at=AFTER_WIDENING).in_force_lo == 0.02

    def test_unknown_question_type_raises(self):
        with pytest.raises(KeyError):
            in_force_bounds("numeric", AS_OF)

    def test_width_monitor_alias_is_the_shared_constant(self):
        assert WIDENING_FLIP is WIDENING_FLIP_MERGED_AT


class TestMcClampSignatureExtension:
    """Group 4 — the ``lo``/``hi`` kwargs are a pure extension of the live clamp."""

    def test_defaults_reproduce_the_module_globals(self):
        vector = [0.984, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002]
        assert clamp_and_renormalize_probs(vector) == clamp_and_renormalize_probs(
            vector, lo=MC_PROB_MIN, hi=MC_PROB_MAX
        )
        clamped = clamp_and_renormalize_probs(vector)
        assert sum(clamped) == pytest.approx(1.0, abs=1e-9)
        assert all(MC_PROB_MIN <= p <= MC_PROB_MAX for p in clamped)

    def test_default_still_reads_the_module_global_at_call_time(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The monkeypatch surface the existing suite relies on must survive the extension.

        A literal ``lo=MC_PROB_MIN`` default would bind at import, so patching the module
        global would silently stop working; the sentinel keeps the late lookup.
        """
        monkeypatch.setattr("metaculus_bot.mc_processing.MC_PROB_MIN", 0.20)
        clamped = clamp_and_renormalize_probs([0.90, 0.05, 0.05])
        assert min(clamped) >= 0.20 - 1e-12

    def test_explicit_floor_holds_on_a_four_option_vector(self):
        clamped = clamp_and_renormalize_probs([0.90, 0.06, 0.03, 0.01], lo=0.05, hi=0.95)
        assert sum(clamped) == pytest.approx(1.0, abs=1e-9)
        assert all(p >= 0.05 - 1e-12 for p in clamped)
        assert all(p <= 0.95 + 1e-12 for p in clamped)

    def test_explicit_floor_repairs_a_dominant_option(self):
        clamped = clamp_and_renormalize_probs([0.97, 0.01, 0.01, 0.01], lo=0.05, hi=0.99)
        assert sum(clamped) == pytest.approx(1.0, abs=1e-9)
        assert min(clamped) == pytest.approx(0.05, abs=1e-9)

    def test_exactly_feasible_floor_returns_the_uniform_vector(self):
        """``n * lo == 1.0`` has exactly one in-bounds sum-1 solution, the uniform vector, and
        the strict degenerate guard lets the normal path converge to it. The old ``>=`` sent this
        case down the degenerate branch and returned sub-floor values, which priced the sweep's
        MC c = 0.10 cell on the archive's two 10-option ballots at an effective floor of ~0.07."""
        vector = [0.30, 0.20, 0.10, 0.10, 0.10, 0.05, 0.05, 0.05, 0.03, 0.02]
        clamped = clamp_and_renormalize_probs(vector, lo=0.10, hi=0.90)
        assert sum(clamped) == pytest.approx(1.0, abs=1e-9)
        assert all(p == pytest.approx(0.10, abs=1e-9) for p in clamped)

    def test_infeasible_floor_takes_the_documented_degenerate_fallback(self):
        """Eleven options at a 0.10 floor already exceed 1, so no in-bounds sum-1 vector exists;
        the fallback keeps the sum at 1 and its minimum sits BELOW the requested floor."""
        vector = [0.30, 0.20, 0.10, 0.10, 0.10, 0.05, 0.05, 0.05, 0.03, 0.01, 0.01]
        clamped = clamp_and_renormalize_probs(vector, lo=0.10, hi=0.90)
        assert sum(clamped) == pytest.approx(1.0, abs=1e-9)
        assert min(clamped) < 0.10

    def test_binding_ceiling_pins_the_leader_and_rescales_the_rest(self):
        """The ``hi`` kwarg reaches the repair loop: a 0.5 ceiling pins the 0.9 leader and the
        freed mass rescales the other two, so an implementation reading ``MC_PROB_MAX`` instead
        of ``hi`` cannot pass."""
        clamped = clamp_and_renormalize_probs([0.9, 0.05, 0.05], lo=0.01, hi=0.5)
        assert clamped == pytest.approx([0.5, 0.25, 0.25], abs=1e-12)
        assert max(clamped) <= 0.5 + 1e-12
        assert sum(clamped) == pytest.approx(1.0, abs=1e-9)


class TestWindows:
    """Group 5 — suffix windows, the dated windows and the oversize disclosure."""

    @staticmethod
    def _dated_records(n: int) -> tuple[ClipRecord, ...]:
        base = datetime(2026, 1, 1, tzinfo=UTC)
        return build_clip_records(
            [
                _binary_record(
                    question_id=1000 + i,
                    p_yes=0.5,
                    resolution=True,
                    created_at=(base + timedelta(days=i)).isoformat().replace("+00:00", "Z"),
                )
                for i in range(n)
            ],
            "binary",
        ).records

    def _by_label(self, records, **kwargs) -> dict:
        return {w.label: w for w in build_windows(records, question_type="binary", as_of=AS_OF, **kwargs)}

    def test_last_n_takes_the_newest_n(self):
        records = self._dated_records(20)
        windows = {w.label: w for w in build_windows(records, question_type="binary", as_of=AS_OF, last_n=(10, 5))}
        assert [r.question_id for r in windows["last_5"].records] == ["1015", "1016", "1017", "1018", "1019"]
        assert len(windows["last_10"].records) == 10
        # The complement is exactly the records the fit may see.
        assert len(windows["last_5"].complement) == 15

    def test_oversize_window_reports_its_actual_n(self):
        records = self._dated_records(8)
        windows = {w.label: w for w in build_windows(records, question_type="binary", as_of=AS_OF, last_n=(300,))}
        window = windows["last_300"]
        assert window.oversize is True
        assert len(window.records) == 8
        assert window.requested_n == 300
        assert window.complement == ()

    def test_last_90d_honours_as_of(self):
        base = datetime(2026, 9, 1, tzinfo=UTC)
        records = build_clip_records(
            [
                _binary_record(
                    question_id=2000 + i,
                    p_yes=0.5,
                    resolution=True,
                    created_at=(base - timedelta(days=30 * i)).isoformat().replace("+00:00", "Z"),
                )
                for i in range(6)
            ],
            "binary",
        ).records
        windows = self._by_label(records)
        # 0, 30, 60 days back are inside a 90-day lookback from 2026-09-02; 90+ are not.
        assert len(windows[WINDOW_LAST_90D].records) == 3
        assert len(windows[WINDOW_LAST_90D].complement) == 3

    def test_clamp_regime_and_triple_era_use_the_shared_constants(self):
        records = build_clip_records(
            [
                _binary_record(question_id=3001, p_yes=0.5, resolution=True, created_at=BEFORE_WIDENING),
                _binary_record(question_id=3002, p_yes=0.5, resolution=True, created_at=AFTER_WIDENING),
                _binary_record(question_id=3003, p_yes=0.5, resolution=True, created_at=AFTER_FT_0292),
            ],
            "binary",
        ).records
        windows = self._by_label(records)
        assert windows[WINDOW_CURRENT_CLAMP].start == WIDENING_FLIP_MERGED_AT
        assert len(windows[WINDOW_CURRENT_CLAMP].records) == 2
        assert windows[WINDOW_TRIPLE_ERA].start == B4E9DF0_MERGED_AT
        assert [r.question_id for r in windows[WINDOW_TRIPLE_ERA].records] == ["3003"]

    def test_undated_records_are_in_all_and_out_of_every_dated_window(self):
        records = build_clip_records(
            [
                _binary_record(question_id=4001, p_yes=0.5, resolution=True, created_at=None),
                _binary_record(question_id=4002, p_yes=0.5, resolution=True, created_at=AFTER_FT_0292),
            ],
            "binary",
        ).records
        windows = self._by_label(records)
        assert len(windows[WINDOW_ALL].records) == 2
        for label in (WINDOW_LAST_90D, WINDOW_CURRENT_CLAMP, WINDOW_TRIPLE_ERA):
            assert [r.question_id for r in windows[label].records] == ["4002"]

    def test_mc_window_set_adds_last_50(self):
        assert "last_50" in window_labels("multiple_choice")
        assert "last_50" not in window_labels("binary")

    def test_the_lookback_label_is_derived_from_its_own_constant(self):
        """A change to LOOKBACK_DAYS must not leave a window labelled with the old span."""
        assert f"last_{LOOKBACK_DAYS}d" == WINDOW_LAST_90D


class TestBootstrap:
    """Group 6 — the CI is deterministic under the fixed seed and brackets the mean."""

    def test_deterministic_and_brackets_the_mean(self):
        """Re-seeding per call is what makes a row reproducible; the index cache is only a speed measure,
        so it is cleared between the two calls or the second would trivially return the first's array."""
        deltas = [-3.0, 0.0, 1.5, 8.0, -12.0, 4.0, 0.5, 2.0, -1.0, 6.0]
        first = bootstrap_mean_ci(deltas)
        bootstrap._INDEX_CACHE.clear()
        second = bootstrap_mean_ci(deltas)
        assert first == second
        lo, hi = first
        assert lo is not None
        assert hi is not None
        assert lo < sum(deltas) / len(deltas) < hi

    def test_index_cache_is_opt_in_and_the_sweep_opts_in(self):
        """The ablation harness derives a distinct seed per scoring group, so a cache that filled on
        every call would grow for the process lifetime with zero reuse; only the sweep, which repeats
        ``(n, B, seed)`` across cells, asks for it. Cached and uncached draws are the same matrix."""
        bootstrap._INDEX_CACHE.clear()
        uncached = bootstrap.bootstrap_indices(7, n_bootstrap=5, seed=3)
        assert bootstrap._INDEX_CACHE == {}
        cached = bootstrap.bootstrap_indices(7, n_bootstrap=5, seed=3, cache=True)
        assert list(bootstrap._INDEX_CACHE) == [(7, 5, 3)]
        assert (uncached == cached).all()
        bootstrap._INDEX_CACHE.clear()
        bootstrap_mean_ci([-3.0, 0.0, 1.5, 8.0])
        assert list(bootstrap._INDEX_CACHE) == [(4, BOOTSTRAP_B, BOOTSTRAP_SEED)]

    def test_degenerate_sample_gives_a_point_interval(self):
        lo, hi = bootstrap_mean_ci([0.0, 0.0, 0.0])
        assert (lo, hi) == (0.0, 0.0)

    def test_empty_sample_has_no_interval(self):
        assert bootstrap_mean_ci([]) == (None, None)


def _floor_rows(window, *, question_type: str = "binary", grid=BINARY_FLOOR_GRID):
    """The window's own floor-only sweep, one row per candidate, as ``_type_report`` hands it to ``oos_row``."""
    return [
        sweep_row(window.records, question_type=question_type, side="floor_only", window=window.label, c=c)
        for c in grid
    ]


class TestOutOfSample:
    """Group 7 — a thin complement cannot fit a clip level, and says so."""

    @staticmethod
    def _records(n: int, *, start: datetime) -> tuple[ClipRecord, ...]:
        return build_clip_records(
            [
                _binary_record(
                    question_id=5000 + i,
                    p_yes=0.02,
                    resolution=(i % 4 == 0),
                    created_at=(start + timedelta(days=i)).isoformat().replace("+00:00", "Z"),
                )
                for i in range(n)
            ],
            "binary",
        ).records

    def test_thin_complement_reports_no_fit(self):
        records = self._records(12, start=datetime(2026, 6, 1, tzinfo=UTC))
        [window] = [
            w for w in build_windows(records, question_type="binary", as_of=AS_OF, last_n=(5,)) if w.label == "last_5"
        ]
        assert len(window.complement) < MIN_OOS_COMPLEMENT_N
        row = oos_row(window, question_type="binary", in_window=_floor_rows(window))
        assert row.underpowered is True
        assert row.c_star is None
        assert row.carried_sum_delta is None
        # The in-window argmax is still reported: it is a description, not a fit.
        assert row.in_window_c_star is not None

    def test_fit_and_carry_are_both_reported_when_the_complement_is_thick(self):
        records = self._records(60, start=datetime(2026, 6, 1, tzinfo=UTC))
        [window] = [
            w for w in build_windows(records, question_type="binary", as_of=AS_OF, last_n=(20,)) if w.label == "last_20"
        ]
        row = oos_row(window, question_type="binary", in_window=_floor_rows(window))
        assert row.underpowered is False
        assert row.c_star in BINARY_FLOOR_GRID
        assert row.carried_sum_delta is not None
        assert row.in_window_c_star in BINARY_FLOOR_GRID


class TestArgmaxPlateau:
    """The winning candidate is usually a TIE, and the report has to say so.

    Every candidate at or below a window's in-force floor scores exactly 0 when no publish in
    that window was clamped, so reporting one representative as "the" argmax would read as a
    preference for loosening when the evidence is indifference.
    """

    def test_every_candidate_at_or_below_the_floor_ties_at_zero(self):
        # Published 0.30 post-flip: nothing at or below 0.02 can move it, and 0.025+ can't
        # either, so the tie runs from the smallest candidate up to 0.0200.
        records = [_one_binary(question_id=7100, p_yes=0.30, resolution=True)]
        rows = [
            sweep_row(records, question_type="binary", side="floor_only", window=WINDOW_ALL, c=c)
            for c in BINARY_FLOOR_GRID
        ]
        tied = argmax_rows(rows)
        assert [row.c for row in tied] == [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.10]
        assert argmax_row(rows) is tied[0]

    def test_censored_rows_never_win(self):
        # At 0.01 this record is censored (published AT the post-flip 0.02 floor), so that row
        # is excluded even though its sum_delta of 0 beats every tightening candidate.
        records = [_one_binary(question_id=7101, p_yes=0.02, resolution=False)]
        rows = [
            sweep_row(records, question_type="binary", side="floor_only", window=WINDOW_ALL, c=c)
            for c in BINARY_FLOOR_GRID
        ]
        tied = argmax_rows(rows)
        assert all(row.censored_n == 0 for row in tied)
        assert 0.005 not in {row.c for row in tied}
        best = argmax_row(rows)
        assert best is not None
        assert best.c == 0.02

    def test_no_exact_row_means_no_argmax(self):
        record = _one_binary(question_id=7102, p_yes=0.02, resolution=False)
        censored_only = [sweep_row([record], question_type="binary", side="floor_only", window=WINDOW_ALL, c=0.005)]
        assert argmax_rows(censored_only) == []
        assert argmax_row(censored_only) is None

    def test_member_censored_only_rows_never_win_either(self):
        """Members 0.02 / 0.03 publish 0.025 post-flip: the published-value rule sees nothing at
        the floor, but the 0.02 member is one of the two middle values, so every loosening
        candidate's ``sum_delta`` of 0 is unobservable. Such a row must lose the argmax on
        censoring and be named in the censored ties, not compete as if it were neutral."""
        records = [_one_binary(question_id=7103, p_yes=0.025, resolution=False, per_model={"a": "2.0%", "b": "3.0%"})]
        rows = [
            sweep_row(records, question_type="binary", side="floor_only", window=WINDOW_ALL, c=c)
            for c in BINARY_FLOOR_GRID
        ]
        loosening = {row.c for row in rows if row.n_loosening}
        assert loosening == {0.005, 0.01, 0.015}
        for row in rows:
            if row.c in loosening:
                assert (row.censored_n, row.member_censored_n, row.exact) == (0, 1, False)
        tied = argmax_rows(rows)
        assert {row.c for row in tied} == {0.02, 0.025}
        assert not ({row.c for row in tied} & loosening)
        assert {row.c for row in censored_rows_at_argmax_score(rows)} == loosening

    def test_carry_gap_is_zero_when_the_labels_disagree_but_the_scores_do_not(self):
        records = TestOutOfSample._records(60, start=datetime(2026, 6, 1, tzinfo=UTC))
        [window] = [
            w for w in build_windows(records, question_type="binary", as_of=AS_OF, last_n=(20,)) if w.label == "last_20"
        ]
        row = oos_row(window, question_type="binary", in_window=_floor_rows(window))
        assert row.carry_gap is not None
        assert row.carry_gap == pytest.approx(
            (row.in_window_sum_delta or 0.0) - (row.carried_sum_delta or 0.0), abs=1e-12
        )
        assert row.n_tied_in_window >= 1

    def test_carry_gap_is_none_when_no_fit_was_made(self):
        records = TestOutOfSample._records(12, start=datetime(2026, 6, 1, tzinfo=UTC))
        [window] = [
            w for w in build_windows(records, question_type="binary", as_of=AS_OF, last_n=(5,)) if w.label == "last_5"
        ]
        row = oos_row(window, question_type="binary", in_window=_floor_rows(window))
        assert row.carry_gap is None


class TestPerModelCrossCheck:
    """Group 9 — replaying the members through the clamp, then the median.

    ``median(clamp(members)) == clamp(median(members))`` holds exactly for an odd member
    count (the median is an order statistic and clamping is monotone). For an even count
    the published median averages the two middle members, so the two paths can differ —
    which is the point of reporting the gap rather than asserting it away.
    """

    def test_three_member_replay_agrees_with_the_published_vector_path(self):
        record = _one_binary(
            question_id=6001,
            p_yes=0.02,
            resolution=False,
            per_model={"a": "1.0%", "b": "2.0%", "c": "40.0%"},
        )
        row = cross_check_row([record], question_type="binary", window=WINDOW_ALL, c=0.05)
        assert replay_cohort([record], question_type="binary", window=WINDOW_ALL).n_replayable == 1
        assert row.n_disagree == 0
        assert row.max_abs_gap == pytest.approx(0.0, abs=1e-9)
        assert row.sum_delta_replay == pytest.approx(row.sum_delta_published, abs=1e-9)

    def test_two_member_replay_can_disagree_and_the_gap_is_reported(self):
        # Members 0.02 and 0.08 -> published median 0.05. A 0.06 floor lifts only the low
        # member, so the replayed median is 0.07 while clamping the published 0.05 gives
        # 0.06. That 0.01 gap is the even-count approximation, reported not hidden.
        record = _one_binary(
            question_id=6002,
            p_yes=0.05,
            resolution=True,
            per_model={"a": "2.0%", "b": "8.0%"},
        )
        row = cross_check_row([record], question_type="binary", window=WINDOW_ALL, c=0.06)
        assert replay_cohort([record], question_type="binary", window=WINDOW_ALL).n_even_members == 1
        assert row.max_abs_gap == pytest.approx(0.01, abs=1e-9)
        assert row.n_disagree == 1
        assert row.sum_delta_replay > row.sum_delta_published

    def test_stacked_records_are_not_replayable(self):
        record = _one_binary(
            question_id=6003,
            p_yes=0.02,
            resolution=False,
            per_model={"stacker": "2.0%"},
            stacker_outcome="primary",
        )
        cohort = replay_cohort([record], question_type="binary", window=WINDOW_ALL)
        assert (cohort.n_records, cohort.n_replayable) == (1, 0)

    def test_mean_era_record_is_replayed_with_the_mean(self):
        """A pre_flip record whose published value is the members' MEAN, not their median.

        q38797 is the real shape: members 0.55 / 0.60 / 0.85 published as 0.667. The replay
        detects the mean aggregator and rebuilds the publish exactly, so there is no baseline
        mismatch to disclose and nothing is charged to a clip that moves none of the members.
        Before aggregator detection this record read as a 0.067 baseline gap and a median
        replay put -122.33 points on candidates that move nothing at all.
        """
        record = _one_binary(
            question_id=6005,
            p_yes=0.667,
            resolution=True,
            created_at=BEFORE_WIDENING,
            per_model={"a": "55.0%", "b": "60.0%", "c": "85.0%"},
        )
        assert record.aggregator == "mean"
        cohort = replay_cohort([record], question_type="binary", window=WINDOW_ALL)
        assert cohort.n_replayable == 1
        assert cohort.n_mean_aggregator == 1
        assert cohort.n_unknown_aggregator == 0
        assert cohort.n_baseline_mismatch == 0
        assert cohort.max_baseline_gap == pytest.approx(0.0, abs=1e-9)
        row = cross_check_row([record], question_type="binary", window=WINDOW_ALL, c=0.05)
        assert row.sum_delta_replay == 0.0
        assert row.sum_delta_published == 0.0

    def test_mean_era_record_clips_along_the_mean_path(self):
        """Members 0.01 / 0.04 / 0.10 published as their mean 0.05 (median 0.04) under the 0.01 floor.

        A 0.03 floor lifts the low member to 0.03 and the MEAN moves to round(0.17/3, 3) = 0.057;
        the published-vector path sees a 0.05 publish that the 0.03 floor does not touch at all.
        """
        record = _one_binary(
            question_id=6006,
            p_yes=0.05,
            resolution=False,
            created_at=BEFORE_WIDENING,
            per_model={"a": "1.0%", "b": "4.0%", "c": "10.0%"},
        )
        assert record.aggregator == "mean"
        row = cross_check_row([record], question_type="binary", window=WINDOW_ALL, c=0.03)
        assert row.sum_delta_published == 0.0
        assert row.sum_delta_replay == pytest.approx(100.0 * math.log(0.943 / 0.95), abs=1e-9)
        assert row.n_disagree == 1

    def test_unknown_aggregator_is_the_baseline_mismatch_residue(self):
        # Published 0.20 from members whose median is 0.30 and mean is 0.40: neither rebuilds it.
        record = _one_binary(
            question_id=6007,
            p_yes=0.20,
            resolution=True,
            per_model={"a": "10.0%", "b": "30.0%", "c": "80.0%"},
        )
        assert record.aggregator == "unknown"
        cohort = replay_cohort([record], question_type="binary", window=WINDOW_ALL)
        assert cohort.n_unknown_aggregator == 1
        assert cohort.n_baseline_mismatch == 1
        assert cohort.max_baseline_gap == pytest.approx(0.10, abs=1e-9)

    def test_even_member_count_and_routing_flips_are_counted(self):
        # Members 0.02 / 0.20: spread 0.18 clears the 0.15 stacking threshold as published, but
        # a 0.10 floor squeezes it to 0.10, so the stacking route would have changed.
        record = _one_binary(
            question_id=6008,
            p_yes=0.11,
            resolution=False,
            per_model={"a": "2.0%", "b": "20.0%"},
        )
        row = cross_check_row([record], question_type="binary", window=WINDOW_ALL, c=0.10)
        assert replay_cohort([record], question_type="binary", window=WINDOW_ALL).n_even_members == 1
        assert row.n_routing_flips == 1
        untouched = cross_check_row([record], question_type="binary", window=WINDOW_ALL, c=0.03)
        assert untouched.n_routing_flips == 0

    def test_mc_rows_carry_no_routing_count(self):
        record = _one_mc(
            question_id=6009,
            options=["A", "B"],
            probs=[0.60, 0.40],
            resolution="A",
            per_model={"a": {"A": 0.60, "B": 0.40}},
        )
        row = cross_check_row([record], question_type="multiple_choice", window=WINDOW_ALL, c=0.05)
        assert row.n_routing_flips is None

    def test_mc_replay_runs_on_option_vectors(self):
        record = _one_mc(
            question_id=6004,
            options=["A", "B", "C"],
            probs=[0.70, 0.29, 0.01],
            resolution="C",
            per_model={
                "a": {"A": 0.70, "B": 0.29, "C": 0.01},
                "b": {"A": 0.70, "B": 0.29, "C": 0.01},
                "c": {"A": 0.70, "B": 0.29, "C": 0.01},
            },
        )
        row = cross_check_row([record], question_type="multiple_choice", window=WINDOW_ALL, c=0.05)
        assert replay_cohort([record], question_type="multiple_choice", window=WINDOW_ALL).n_replayable == 1
        assert row.n_disagree == 0


class TestExtremeBins:
    """The calibration table the whole decision rests on: how often did a 1-2% call hit?"""

    def test_low_bins_count_yes_resolutions(self):
        records = build_clip_records(
            [
                _binary_record(question_id=7001, p_yes=0.01, resolution=False, created_at=BEFORE_WIDENING),
                _binary_record(question_id=7002, p_yes=0.01, resolution=True, created_at=BEFORE_WIDENING),
                _binary_record(question_id=7003, p_yes=0.04, resolution=False),
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
                _binary_record(question_id=7010, p_yes=0.99, resolution=False, created_at=BEFORE_WIDENING),
                _binary_record(question_id=7011, p_yes=0.985, resolution=True, created_at=BEFORE_WIDENING),
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
                _binary_record(question_id=7020, p_yes=0.01, resolution=False, created_at=BEFORE_WIDENING),
                _binary_record(question_id=7021, p_yes=0.04, resolution=True, created_at=AFTER_WIDENING),
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


class TestCli:
    """Group 8 — the CLI writes JSON, prints markdown and honours ``--exclude-qids``."""

    def _dataset(self) -> list[dict]:
        records: list[dict] = []
        base = datetime(2026, 5, 1, tzinfo=UTC)
        for i in range(40):
            records.append(
                _binary_record(
                    question_id=8000 + i,
                    p_yes=0.02 if i % 3 == 0 else 0.45,
                    resolution=(i % 5 == 0),
                    created_at=(base + timedelta(days=2 * i)).isoformat().replace("+00:00", "Z"),
                    per_model={"a": "2.0%", "b": "3.0%", "c": "40.0%"},
                )
            )
        for i in range(12):
            records.append(
                _mc_record(
                    question_id=8500 + i,
                    options=["A", "B", "C"],
                    probs=[0.70, 0.29, 0.01],
                    resolution="A" if i % 2 else "C",
                    created_at=(base + timedelta(days=5 * i)).isoformat().replace("+00:00", "Z"),
                )
            )
        # A degraded_run cohort member, so --exclude-qids has something to drop.
        records.append(_binary_record(question_id=44870, p_yes=0.02, resolution=False))
        return records

    def _write(self, tmp_path, records: list[dict]) -> str:
        path = tmp_path / "data.json"
        path.write_text(json.dumps(records))
        return str(path)

    def test_writes_json_and_markdown(self, tmp_path, capsys):
        path = self._write(tmp_path, self._dataset())
        out_json = str(tmp_path / "sweep.json")
        main(["--cached", path, "--as-of", "2026-09-02T00:00:00Z", "--output-json", out_json])
        markdown = capsys.readouterr().out
        payload = json.loads((tmp_path / "sweep.json").read_text())
        assert {"binary", "multiple_choice"} <= set(payload)
        for label in window_labels("binary") + window_labels("multiple_choice"):
            assert label in markdown
        assert "2026-09-02" in markdown
        assert str(path) in markdown
        # Every grid level is rendered for both types.
        assert all(f"{c:.4f}" in markdown for c in BINARY_FLOOR_GRID)
        assert all(f"{c:.4f}" in markdown for c in MC_FLOOR_GRID)

    def test_exclude_qids_drops_the_cohort(self, tmp_path, capsys):
        path = self._write(tmp_path, self._dataset())
        main(["--cached", path, "--as-of", "2026-09-02T00:00:00Z"])
        baseline = capsys.readouterr().out
        main(["--cached", path, "--as-of", "2026-09-02T00:00:00Z", "--exclude-qids", "degraded_run"])
        excluded = capsys.readouterr().out
        assert "n=41" in baseline
        assert "n=40" in excluded

    def test_report_object_carries_the_same_numbers(self, tmp_path):
        records = self._dataset()
        report = compute_report(records, dataset_path="mem", as_of=AS_OF, exclude_qids=frozenset())
        payload = report.to_dict()
        assert payload["binary"]["n"] == 41
        assert payload["multiple_choice"]["n"] == 12
        assert payload["meta"]["as_of"].startswith("2026-09-02")
        binary_rows = payload["binary"]["sweep"]
        assert {row["c"] for row in binary_rows} == set(BINARY_FLOOR_GRID)

    def test_mc_sub_shippable_rows_are_labelled(self, tmp_path):
        report = compute_report(self._dataset(), dataset_path="mem", as_of=AS_OF, exclude_qids=frozenset())
        mc = next(t for t in report.types if t.question_type == "multiple_choice")
        below = [row for row in mc.sweep if row.c < MC_PROB_MIN]
        assert below
        assert all(row.shippable is False for row in below)
        assert all(row.shippable is True for row in mc.sweep if row.c >= MC_PROB_MIN)


class TestIdentityRows:
    """A candidate that moves nothing is the do-nothing map, and the report must say so."""

    def test_identity_flag_and_zero_interval(self):
        records = [_one_binary(question_id=9001, p_yes=0.30, resolution=True)]
        row = sweep_row(records, question_type="binary", side="floor_only", window=WINDOW_ALL, c=0.05)
        assert row.identity is True
        assert row.n_affected == 0
        assert (row.ci_lo, row.ci_hi) == (0.0, 0.0)
        moved = sweep_row(
            [_one_binary(question_id=9002, p_yes=0.02, resolution=False)],
            question_type="binary",
            side="floor_only",
            window=WINDOW_ALL,
            c=0.05,
        )
        assert moved.identity is False

    def test_rendered_ci_cell_reads_identity_not_an_interval(self):
        report = compute_report(
            [_binary_record(question_id=9003, p_yes=0.30, resolution=True)],
            dataset_path="synthetic",
            as_of=AS_OF,
            exclude_qids=frozenset(),
        )
        markdown = render_report(report)
        assert "| identity |" in markdown
        assert "[+0.000, +0.000]" not in markdown


class TestExpectedAndBestCase:
    """The properness cost and the insurance ceiling, pinned by hand on one record."""

    def test_binary_expected_is_minus_kl_and_best_worst_are_the_two_outcomes(self):
        record = _one_binary(question_id=9010, p_yes=0.02, resolution=False)
        clip = clip_delta(record, 0.05, side="floor_only")
        gain = 100.0 * math.log(0.05 / 0.02)
        loss = 100.0 * math.log(0.95 / 0.98)
        assert clip.best_case_delta == pytest.approx(gain, abs=1e-9)
        assert clip.worst_case_delta == pytest.approx(loss, abs=1e-9)
        assert clip.expected_delta == pytest.approx(0.02 * gain + 0.98 * loss, abs=1e-9)
        assert clip.expected_delta < 0.0

    def test_unaffected_record_carries_zero_everywhere(self):
        clip = clip_delta(_one_binary(question_id=9011, p_yes=0.30, resolution=True), 0.05, side="floor_only")
        assert (clip.expected_delta, clip.best_case_delta, clip.worst_case_delta) == (0.0, 0.0, 0.0)

    def test_mc_expected_is_never_positive(self):
        record = _one_mc(question_id=9012, options=["A", "B", "C", "D"], probs=[0.70, 0.27, 0.02, 0.01], resolution="A")
        clip = clip_delta(record, 0.05, side="floor_only")
        assert clip.expected_delta < 0.0
        assert clip.best_case_delta > 0.0 > clip.worst_case_delta

    def test_row_sums_the_per_record_quantities(self):
        records = [
            _one_binary(question_id=9013, p_yes=0.02, resolution=False),
            _one_binary(question_id=9014, p_yes=0.02, resolution=True),
        ]
        row = sweep_row(records, question_type="binary", side="floor_only", window=WINDOW_ALL, c=0.05)
        gain = 100.0 * math.log(0.05 / 0.02)
        loss = 100.0 * math.log(0.95 / 0.98)
        assert row.best_case_sum_delta == pytest.approx(2 * gain, abs=1e-9)
        assert row.worst_case_sum_delta == pytest.approx(2 * loss, abs=1e-9)
        assert row.expected_sum_delta == pytest.approx(2 * (0.02 * gain + 0.98 * loss), abs=1e-9)


class TestMemberCensoring:
    """A clamped MEMBER in a median position censors a looser clip even when the publish is above the floor."""

    def test_even_roster_with_a_floored_middle_member_is_member_censored(self):
        """Members 0.02 / 0.03 publish 0.025 post-flip: above the 0.02 floor, so the published
        rule says nothing is censored, yet the 0.02 member is one of the two middle values and a
        looser floor could have moved the publish."""
        record = _one_binary(
            question_id=9020,
            p_yes=0.025,
            resolution=False,
            per_model={"a": "2.0%", "b": "3.0%"},
        )
        clip = clip_delta(record, 0.01, side="floor_only")
        assert clip.censored is False
        assert clip.member_censored is True
        # Member replay: median(0.01, 0.03) = 0.02 against the baseline median 0.025, on a NO.
        assert clip.loosen_members_at_c == pytest.approx(100.0 * math.log(0.98 / 0.975), abs=1e-9)
        assert clip.loosen_at_c == 0.0

    def test_floored_member_outside_the_median_positions_cannot_move_the_publish(self):
        record = _one_binary(
            question_id=9021,
            p_yes=0.06,
            resolution=False,
            per_model={"a": "2.0%", "b": "6.0%", "c": "10.0%"},
        )
        assert member_censored(record, floor_side=True, ceiling_side=False) is False
        clip = clip_delta(record, 0.01, side="floor_only")
        assert clip.member_censored is False
        assert clip.loosen_members_at_c == 0.0

    def test_mean_aggregator_makes_every_floored_member_censoring(self):
        # Members 0.02 / 0.10 / 0.30 published as their mean 0.14 (median 0.10).
        record = _one_binary(
            question_id=9022,
            p_yes=0.14,
            resolution=False,
            per_model={"a": "2.0%", "b": "10.0%", "c": "30.0%"},
        )
        assert record.aggregator == "mean"
        assert member_censored(record, floor_side=True, ceiling_side=False) is True

    def test_no_members_falls_back_to_the_published_rule(self):
        at_floor = _one_binary(question_id=9023, p_yes=0.02, resolution=False)
        above = _one_binary(question_id=9024, p_yes=0.30, resolution=False)
        assert member_censored(at_floor, floor_side=True, ceiling_side=False) is True
        assert member_censored(above, floor_side=True, ceiling_side=False) is False
        clip = clip_delta(at_floor, 0.01, side="floor_only")
        assert clip.member_censored is True
        assert clip.loosen_members_at_c == pytest.approx(clip.loosen_at_c, abs=1e-12)

    def test_unknown_aggregator_falls_back_to_the_published_rule(self):
        """Members whose median and mean both miss the publish say nothing about member positions."""
        at_floor = _one_binary(
            question_id=9029, p_yes=0.02, resolution=False, per_model={"a": "10.0%", "b": "30.0%", "c": "80.0%"}
        )
        assert at_floor.aggregator == "unknown"
        assert member_censored(at_floor, floor_side=True, ceiling_side=False) is True
        clip = clip_delta(at_floor, 0.01, side="floor_only")
        assert clip.member_censored is True
        assert clip.loosen_members_at_c == pytest.approx(clip.loosen_at_c, abs=1e-12)

    def test_row_counts_both_rules_and_the_member_bound_is_never_narrower(self):
        records = [
            _one_binary(question_id=9025, p_yes=0.02, resolution=False),
            _one_binary(question_id=9026, p_yes=0.025, resolution=False, per_model={"a": "2.0%", "b": "3.0%"}),
            _one_binary(question_id=9027, p_yes=0.30, resolution=False),
        ]
        row = sweep_row(records, question_type="binary", side="floor_only", window=WINDOW_ALL, c=0.01)
        assert row.censored_n == 1
        assert row.member_censored_n == 2
        assert row.sum_delta_upper_members > row.sum_delta_upper > 0.0

    def test_mc_even_roster_with_a_floored_middle_option_is_member_censored(self):
        """Two MC ballots put option C at 0.01 and 0.03; the per-option median publishes 0.02,
        above the 0.01 floor, so the published rule sees nothing while the member rule does."""
        record = _one_mc(
            question_id=9028,
            options=["A", "B", "C"],
            probs=[0.60, 0.38, 0.02],
            resolution="A",
            per_model={
                "a": {"A": 0.60, "B": 0.39, "C": 0.01},
                "b": {"A": 0.60, "B": 0.37, "C": 0.03},
            },
        )
        assert record.aggregator == "median"
        clip = clip_delta(record, 0.005, side="floor_only")
        assert clip.censored is False
        assert clip.member_censored is True
        assert clip.loosen_members_at_c != 0.0


class TestMemberRecovery:
    """Which recovered forecasts the replay sees: base models over a collapsed aggregate, complete ballots only."""

    def test_per_base_model_forecasts_win_over_a_collapsed_per_model_aggregate(self):
        """On a record that carried a stacker, ``per_model_forecasts`` collapses to the single
        aggregate while ``per_base_model_forecasts`` keeps the roster; the sweep must read the
        roster, or every member-level reading is a 1-member replay of the publish itself."""
        record = _one_binary(
            question_id=9150,
            p_yes=0.30,
            resolution=True,
            per_model={"stacker": "30.0%"},
            per_base_model={"a": "10.0%", "b": "30.0%", "c": "80.0%"},
        )
        assert [member[1] for member in record.members] == [0.1, 0.3, 0.8]
        assert record.aggregator == "median"

    def test_mc_ballot_missing_an_option_is_dropped_not_padded(self):
        record = _one_mc(
            question_id=9151,
            options=["A", "B", "C"],
            probs=[0.60, 0.30, 0.10],
            resolution="A",
            per_model={
                "a": {"A": 0.60, "B": 0.30, "C": 0.10},
                "b": {"A": 0.70, "B": 0.30},
                "c": {"C": 0.10, "A": 0.60, "B": 0.30},
            },
        )
        # The partial ballot is gone; the complete ones survive in the question's option order.
        assert record.members == ((0.60, 0.30, 0.10), (0.60, 0.30, 0.10))


class TestAggregatorDetection:
    def test_median_mean_stacker_and_neither(self):
        median = _one_binary(
            question_id=9030, p_yes=0.30, resolution=True, per_model={"a": "10.0%", "b": "30.0%", "c": "80.0%"}
        )
        mean = _one_binary(
            question_id=9031, p_yes=0.40, resolution=True, per_model={"a": "10.0%", "b": "30.0%", "c": "80.0%"}
        )
        neither = _one_binary(
            question_id=9032, p_yes=0.20, resolution=True, per_model={"a": "10.0%", "b": "30.0%", "c": "80.0%"}
        )
        stacker = _one_binary(
            question_id=9033, p_yes=0.30, resolution=True, per_model={"stacker": "30.0%"}, stacker_outcome="primary"
        )
        assert (median.aggregator, mean.aggregator, neither.aggregator, stacker.aggregator) == (
            "median",
            "mean",
            "unknown",
            "unknown",
        )
        assert stacker.replayable is False


class TestEraSlices:
    """The disjoint config-era windows, beside the nested ones."""

    def _records(self):
        return build_clip_records(
            [
                _binary_record(question_id=9040, p_yes=0.5, resolution=True, created_at=BEFORE_WIDENING),
                _binary_record(question_id=9041, p_yes=0.5, resolution=True, created_at=AFTER_WIDENING),
                _binary_record(question_id=9042, p_yes=0.5, resolution=True, created_at=AFTER_FT_0292),
                _binary_record(question_id=9043, p_yes=0.5, resolution=True, created_at=None),
            ],
            "binary",
        ).records

    def test_era_slices_partition_the_dated_records_with_triple_era(self):
        windows = {w.label: w for w in build_windows(self._records(), question_type="binary", as_of=AS_OF)}
        pre = [r.question_id for r in windows[WINDOW_ERA_PRE_FLIP].records]
        post = [r.question_id for r in windows[WINDOW_ERA_POST_FLIP].records]
        triple = [r.question_id for r in windows[WINDOW_TRIPLE_ERA].records]
        assert (pre, post, triple) == (["9040"], ["9041"], ["9042"])
        assert windows[WINDOW_ERA_PRE_FLIP].end == WIDENING_FLIP_MERGED_AT
        assert windows[WINDOW_ERA_POST_FLIP].start == WIDENING_FLIP_MERGED_AT
        assert windows[WINDOW_ERA_POST_FLIP].end == B4E9DF0_MERGED_AT
        assert windows[WINDOW_ERA_PRE_FLIP].is_era_slice
        assert windows[WINDOW_ERA_POST_FLIP].is_era_slice
        assert not windows[WINDOW_TRIPLE_ERA].is_era_slice

    def test_era_complements_are_the_older_dated_records(self):
        windows = {w.label: w for w in build_windows(self._records(), question_type="binary", as_of=AS_OF)}
        assert windows[WINDOW_ERA_PRE_FLIP].complement == ()
        assert [r.question_id for r in windows[WINDOW_ERA_POST_FLIP].complement] == ["9040"]

    def test_nested_windows_exclude_the_era_slices_and_labels_include_them(self):
        windows = build_windows(self._records(), question_type="binary", as_of=AS_OF)
        nested = {w.label for w in nested_windows(windows)}
        assert WINDOW_ERA_PRE_FLIP not in nested
        assert WINDOW_ERA_POST_FLIP not in nested
        assert WINDOW_ALL in nested
        assert WINDOW_TRIPLE_ERA in nested
        assert {WINDOW_ERA_PRE_FLIP, WINDOW_ERA_POST_FLIP} <= set(window_labels("binary"))


class TestOutOfBagArgmax:
    """The selection-corrected value of "pick the best floor, then apply it"."""

    @staticmethod
    def _rows(records):
        return [
            sweep_row(records, question_type="binary", side="floor_only", window=WINDOW_ALL, c=c)
            for c in BINARY_FLOOR_GRID
        ]

    def test_identity_window_has_zero_oob_value(self):
        records = [_one_binary(question_id=9050 + i, p_yes=0.30 + 0.01 * i, resolution=True) for i in range(6)]
        result = oob_argmax(records, self._rows(records), side="floor_only", window=WINDOW_ALL)
        assert result.oob_mean_delta == pytest.approx(0.0, abs=1e-12)
        assert result.shrinkage == pytest.approx(0.0, abs=1e-12)
        assert result.n_iterations > 0

    def test_deterministic_and_bracketed(self):
        records = [_one_binary(question_id=9060 + i, p_yes=0.02, resolution=(i % 3 == 0)) for i in range(12)]
        rows = self._rows(records)
        first = oob_argmax(records, rows, side="floor_only", window=WINDOW_ALL)
        second = oob_argmax(records, rows, side="floor_only", window=WINDOW_ALL)
        assert first == second
        assert first.in_window_c is not None
        assert first.oob_ci_lo is not None
        assert first.oob_ci_hi is not None
        assert first.oob_mean_delta is not None
        assert first.oob_ci_lo <= first.oob_mean_delta <= first.oob_ci_hi
        # Only exact rows compete: the censored 0.005 candidate is never a fit.
        assert first.n_candidates == sum(1 for row in rows if row.exact)

    def test_selection_shrinks_a_lucky_small_window(self):
        """Six YES at 0.02 and five NO at 0.02: in-window the 0.10 floor looks like a gain, but a
        resample that draws mostly NOs fits a lower floor and scores it on the YES-heavy remainder
        (and vice versa), so the out-of-bag mean sits below the in-window mean."""
        records = [_one_binary(question_id=9070 + i, p_yes=0.02, resolution=(i < 6)) for i in range(11)]
        result = oob_argmax(records, self._rows(records), side="floor_only", window=WINDOW_ALL)
        assert result.in_window_mean_delta is not None
        assert result.in_window_mean_delta > 0.0
        assert result.shrinkage is not None
        assert result.shrinkage > 0.0


class TestInsuranceView:
    def test_break_even_and_binomial_on_a_binary_row(self):
        records = [
            _one_binary(question_id=9080, p_yes=0.02, resolution=False),
            _one_binary(question_id=9081, p_yes=0.02, resolution=False),
            _one_binary(question_id=9082, p_yes=0.02, resolution=True),
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
            _one_binary(question_id=9090 + i, p_yes=0.01, resolution=False, created_at=BEFORE_WIDENING)
            for i in range(150)
        ]
        row = insurance_row(sweep_row(records, question_type="binary", side="floor_only", window=WINDOW_ALL, c=0.05))
        assert row.hits == 0
        assert row.p_hits_at_most_if_rate_c == pytest.approx(0.95**150, abs=1e-12)
        assert row.rejected_at_ci_upper is True

    def test_mc_rows_have_no_break_even(self):
        record = _one_mc(question_id=9095, options=["A", "B", "C", "D"], probs=[0.70, 0.27, 0.02, 0.01], resolution="A")
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


class TestFloorFeasibility:
    """An MC floor with more options than ``1 / c`` cannot be delivered, and the row says so."""

    def test_eleven_options_cannot_take_a_tenth_floor(self):
        options = [f"O{i}" for i in range(11)]
        probs = [0.30, 0.20, 0.10, 0.10, 0.10, 0.05, 0.05, 0.05, 0.03, 0.01, 0.01]
        record = _one_mc(question_id=9160, options=options, probs=probs, resolution="O0")
        assert clip_delta(record, 0.10, side="floor_only").infeasible is True
        assert clip_delta(record, 0.05, side="floor_only").infeasible is False
        row = sweep_row([record], question_type="multiple_choice", side="floor_only", window=WINDOW_ALL, c=0.10)
        # Priced at the live clamp's sub-floor fallback, still counted as moved, and disclosed.
        assert (row.infeasible_n, row.n_affected) == (1, 1)
        assert (
            sweep_row(
                [record], question_type="multiple_choice", side="floor_only", window=WINDOW_ALL, c=0.05
            ).infeasible_n
            == 0
        )

    def test_ten_options_at_a_tenth_are_exactly_feasible(self):
        options = [f"O{i}" for i in range(10)]
        probs = [0.30, 0.20, 0.10, 0.10, 0.10, 0.05, 0.05, 0.05, 0.03, 0.02]
        record = _one_mc(question_id=9162, options=options, probs=probs, resolution="O0")
        clip = clip_delta(record, 0.10, side="floor_only")
        assert clip.infeasible is False
        # And the counterfactual really is the 0.10 floor: the uniform vector, not a sub-floor one.
        assert clip.delta == pytest.approx(100.0 * math.log(0.10 / 0.30), abs=1e-9)

    def test_binary_rows_are_never_infeasible(self):
        row = sweep_row(
            [_one_binary(question_id=9161, p_yes=0.02, resolution=False)],
            question_type="binary",
            side="floor_only",
            window=WINDOW_ALL,
            c=0.10,
        )
        assert row.infeasible_n == 0


class TestCohortAccounting:
    """The five ``ClipCohort`` counters are the sweep's only record of what it discarded.

    Split by question type because ``build_clip_records`` filters on the record's ``type`` BEFORE
    calling the builder, so a malformed MC shape can never reach a binary cohort's ``n_skipped``.
    """

    def test_binary_skips_and_counters(self):
        cohort = build_clip_records(
            [
                _binary_record(question_id=9170, p_yes=0.30, resolution=True),
                {**_binary_record(question_id=9171, p_yes=0.30, resolution=True), "our_prob_yes": None},
                {**_binary_record(question_id=9172, p_yes=0.30, resolution=True), "resolution_parsed": "annulled"},
                _binary_record(question_id=9173, p_yes=0.02, resolution=False),
                _binary_record(question_id=9174, p_yes=0.98, resolution=True),
                _binary_record(question_id=9175, p_yes=0.50, resolution=True, created_at=None),
            ],
            "binary",
        )
        assert [r.question_id for r in cohort.records] == ["9175", "9170", "9173", "9174"]
        # No members recovered, so the member rule falls back to the published one: 1, not 0.
        assert cohort.to_dict() == {
            "question_type": "binary",
            "n": 4,
            "n_skipped": 2,
            "n_no_timestamp": 1,
            "n_at_in_force_floor": 1,
            "n_at_in_force_ceiling": 1,
            "n_member_censored_floor": 1,
        }

    def test_mc_skips_count_only_records_of_the_cohorts_own_type(self):
        cohort = build_clip_records(
            [
                _mc_record(question_id=9180, options=["A", "B", "C"], probs=[0.6, 0.3, 0.1], resolution="A"),
                _mc_record(question_id=9181, options=["A"], probs=[1.0], resolution="A"),
                _mc_record(question_id=9182, options=["A", "B", "C"], probs=[0.6, 0.4], resolution="A"),
                _mc_record(question_id=9183, options=["A", "B", "C"], probs=[0.6, 0.3, 0.1], resolution="Z"),
                _binary_record(question_id=9184, p_yes=0.30, resolution=True),
            ],
            "multiple_choice",
        )
        assert [r.question_id for r in cohort.records] == ["9180"]
        # The stray binary is filtered by type, not counted as a skip: the header's "skipped N"
        # claim is about records of this cohort's own type.
        assert cohort.n_skipped == 3
        assert (
            build_clip_records(
                [_mc_record(question_id=9185, options=["A", "B"], probs=[0.5, 0.5], resolution="A")], "binary"
            ).n_skipped
            == 0
        )


class TestOlderRegimeRow:
    """The live floor priced on the records published under the OLDER clamp, per type's own boundary.

    The binary clamp changed at the widening flip, so for binary the older regime IS
    ``era_pre_flip``; the MC clamp changed two months later at the ft 0.2.92 unfreeze, so for MC
    the older regime also holds the post-flip records. The row is keyed on the current-regime
    window's complement, which is right for both, where an ``era_pre_flip`` lookup was right for
    one (it dropped 26 of 91 MC records on the real archive).
    """

    def test_mc_older_regime_is_the_current_window_complement_not_era_pre_flip(self):
        ballot = [0.60, 0.393, 0.007]
        report = compute_report(
            [
                _mc_record(
                    question_id=9190, options=["A", "B", "C"], probs=ballot, resolution="A", created_at=BEFORE_WIDENING
                ),
                _mc_record(
                    question_id=9191, options=["A", "B", "C"], probs=ballot, resolution="A", created_at=AFTER_WIDENING
                ),
                _mc_record(
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
        # Both older records were published under the 0.005 floor and both carry a 0.007 option,
        # so both are in the cohort and both move at the live 0.01 floor.
        assert (row.n, row.n_affected) == (2, 2)
        pre_flip = next(w for w in mc.windows if w.label == WINDOW_ERA_PRE_FLIP)
        assert len(pre_flip.records) == 1
        assert f"priced on the {row.n} records published before the clamp in force went live" in render_report(report)

    def test_binary_older_regime_coincides_with_era_pre_flip(self):
        report = compute_report(
            [
                _binary_record(question_id=9193, p_yes=0.01, resolution=False, created_at=BEFORE_WIDENING),
                _binary_record(question_id=9194, p_yes=0.30, resolution=False, created_at=AFTER_WIDENING),
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
            [_binary_record(question_id=9195, p_yes=0.30, resolution=False, created_at=AFTER_WIDENING)],
            dataset_path="synthetic",
            as_of=AS_OF,
            exclude_qids=frozenset(),
        ).type_report("binary")
        assert report.older_regime is None


class TestNestingAndRegimeSpan:
    def test_distinct_count_equals_the_all_count(self):
        records = build_clip_records(
            [
                _binary_record(question_id=9100, p_yes=0.02, resolution=False, created_at=BEFORE_WIDENING),
                _binary_record(question_id=9101, p_yes=0.03, resolution=False, created_at=AFTER_WIDENING),
                _binary_record(question_id=9102, p_yes=0.04, resolution=False, created_at=AFTER_FT_0292),
            ],
            "binary",
        ).records
        windows = build_windows(records, question_type="binary", as_of=AS_OF)
        rows = {row.c: row for row in nesting_rows(windows, grid=BINARY_FLOOR_GRID)}
        assert rows[0.05].n_distinct == 3 == rows[0.05].n_affected_by_window[WINDOW_ALL]
        assert rows[0.05].n_affected_by_window[WINDOW_TRIPLE_ERA] == 1
        assert WINDOW_ERA_PRE_FLIP not in rows[0.05].n_affected_by_window

    def test_regime_span_reports_whether_the_live_clamp_bound_anything(self):
        records = build_clip_records(
            [
                _binary_record(question_id=9110, p_yes=0.30, resolution=True, created_at=AFTER_WIDENING),
                _binary_record(question_id=9111, p_yes=0.90, resolution=True, created_at=AFTER_WIDENING),
            ],
            "binary",
        ).records
        [current] = [
            w for w in build_windows(records, question_type="binary", as_of=AS_OF) if w.label == WINDOW_CURRENT_CLAMP
        ]
        span = regime_span(current, question_type="binary")
        assert (span.floor, span.ceiling) == (0.02, 0.98)
        assert (span.min_value, span.max_value) == (0.30, 0.90)
        assert (span.n_at_or_below_floor, span.n_at_or_above_ceiling) == (0, 0)
        bound = build_clip_records([_binary_record(question_id=9112, p_yes=0.02, resolution=False)], "binary").records
        [current_bound] = [
            w for w in build_windows(bound, question_type="binary", as_of=AS_OF) if w.label == WINDOW_CURRENT_CLAMP
        ]
        assert regime_span(current_bound, question_type="binary").n_at_or_below_floor == 1


class TestSingleSurvivorThinFloor:
    """The thin publish floor fires on exactly one cohort, and the detector needs a date guard."""

    def test_genuine_single_survivor_publishes_are_priced_and_artifacts_counted(self):
        records = build_clip_records(
            [
                # Genuine: one member, after the MIN_FORECASTERS_TO_PUBLISH merge, published 0.03 YES.
                _binary_record(
                    question_id=9120, p_yes=0.03, resolution=True, created_at=AFTER_FT_0292, per_model={"g": "3.0%"}
                ),
                # Genuine but inside the floor: moves by exactly 0.
                _binary_record(
                    question_id=9121, p_yes=0.50, resolution=True, created_at=AFTER_FT_0292, per_model={"g": "50.0%"}
                ),
                # One member BEFORE the merge: a trimmed-comment parse artifact, counted not priced.
                _binary_record(
                    question_id=9122,
                    p_yes=0.03,
                    resolution=False,
                    created_at=BEFORE_WIDENING,
                    per_model={"Forecaster 1": "3.0%"},
                ),
                # A stacked one-member record is the stacker's output, not a survivor.
                _binary_record(
                    question_id=9123,
                    p_yes=0.03,
                    resolution=True,
                    created_at=AFTER_FT_0292,
                    per_model={"s": "3.0%"},
                    stacker_outcome="primary",
                ),
                # Three members: not the thin floor's cohort at all.
                _binary_record(
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


class TestCensoredTies:
    def test_censored_candidate_at_the_argmax_score_is_named(self):
        records = [_one_binary(question_id=9130, p_yes=0.02, resolution=False)]
        rows = [
            sweep_row(records, question_type="binary", side="floor_only", window=WINDOW_ALL, c=c)
            for c in BINARY_FLOOR_GRID
        ]
        censored = censored_rows_at_argmax_score(rows)
        assert [row.c for row in censored] == [0.005, 0.01, 0.015]
        report = compute_report(
            [_binary_record(question_id=9131, p_yes=0.02, resolution=False)],
            dataset_path="synthetic",
            as_of=AS_OF,
            exclude_qids=frozenset(),
        )
        assert "censored at the same score" in render_report(report)


class TestRenderedSections:
    """The report carries every new section and the JSON carries every new table."""

    def test_markdown_and_json_carry_the_new_tables(self, tmp_path):
        records = TestCli()._dataset()
        report = compute_report(records, dataset_path="mem", as_of=AS_OF, exclude_qids=frozenset())
        markdown = render_report(report)
        for heading in (
            "Out-of-bag value of the fitted argmax",
            "Affected-set nesting",
            "Insurance view",
            "Single-survivor publishes and the thin publish floor",
            "live clamp regime",
            "at_c (members)",
            "fit n_aff",
        ):
            assert heading in markdown
        payload = report.to_dict()
        for key in (
            "insurance",
            "nesting",
            "oob",
            "regime_span",
            "thin_floor",
            "argmax",
            "older_regime",
            "replay_cohorts",
            "cross_check_summary",
        ):
            assert key in payload["binary"]
        # Every selection-derived quantity the markdown prints is in the JSON: the plateau and the
        # censored ties per (side, window), and the sign counts behind the cross-check sentence.
        binary = report.type_report("binary")
        assert {(sel["side"], sel["window"]) for sel in payload["binary"]["argmax"]} == {
            (side, w.label) for side in ("floor_only", "ceiling_only", "symmetric") for w in binary.populated_windows
        }
        assert set(payload["binary"]["cross_check_summary"]) == {"n_differing", "n_replay_more_negative"}
        assert "n_replayable" not in payload["binary"]["cross_check"][0]
        assert {c["window"] for c in payload["binary"]["replay_cohorts"]} == {w.label for w in binary.populated_windows}
        assert payload["multiple_choice"]["thin_floor"] is None
        assert payload["meta"]["bootstrap"]["oob_B"] > 0
        assert {row["window"] for row in payload["binary"]["oob"]} == {
            w.label for w in report.type_report("binary").populated_windows
        }
