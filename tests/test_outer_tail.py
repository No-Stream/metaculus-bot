"""Tests for the starved-outer-tail scan (the open-bound p99 cliff).

Every case is built on the q45218 geometry, the record whose flat -219.5 zone motivated the
detector: a 72-point discrete grid with a dialable per-bin mass above the members' declared
p99, so a test states the starvation it wants as a multiple of the platform minimum step.
"""

import json

import numpy as np
import pytest

from metaculus_bot.performance_analysis.cohorts import KNOWN_BUG_QIDS
from metaculus_bot.performance_analysis.outer_tail import (
    STARVED_OUTER_TAIL_FLOOR_MULTIPLE,
    OuterTailReading,
    OuterTailVerdict,
    measure_outer_tails,
    render_starved_outer_tails,
    scan_outer_tails,
)
from metaculus_bot.performance_analysis.width_monitor import main


def _rig_count_record(
    *,
    band_bin_multiple: float,
    declared_top: tuple[float, ...] = (602.0, 603.0, 606.0),
    top_label: float = 99.0,
    lower_label: float = 1.0,
    declared_bottom: tuple[float, ...] = (566.0, 568.0, 570.0),
    open_upper: bool = True,
    open_lower: bool = True,
    question_id: object = 45218,
    created_at: str = "2026-08-13T06:11:38Z",
) -> dict:
    """The q45218 geometry: the record whose flat -219.5 zone motivated the detector.

    A 72-point discrete grid over [559.5, 630.5] (one rig per step), open on both ends, three
    members declaring p99 at 602 / 603 / 606 (median 603) and p1 at 566 / 568 / 570 (median
    568), ``cdf[0] = 0.01`` and ``cdf[-1] = 0.99`` — the canonical p1/p99 out-of-range mass.
    Each bin fully above the declared p99 carries ``band_bin_multiple`` times the platform's
    minimum step (``0.01 / 71``), so a caller dials starvation directly: the real record sat
    at ~1.12, every one of its 27 upper bins pinned at the structural floor.

    The mass below the declared p1 is left at a healthy density so a test can read one side at
    a time; ``declared_bottom`` mirrors the geometry when the low side is the subject.
    """
    n_points = 72
    grid = np.linspace(559.5, 630.5, n_points)
    min_step = 0.01 / (n_points - 1)
    cdf_start, cdf_end = 0.01, 0.99

    anchor_high = float(np.median(declared_top))
    first_band_bin = min(int(np.searchsorted(grid, anchor_high, side="left")), n_points - 1)
    band_bins = (n_points - 1) - first_band_bin

    if band_bins == 0:
        # The declared p99 sits at or past the displayed ceiling (the q44842 shape): there is no
        # in-range band to starve, so the CDF is a plain ramp across the whole displayed range.
        cdf = np.linspace(cdf_start, cdf_end, n_points)
    else:
        band_mass = band_bins * band_bin_multiple * min_step
        cdf = np.empty(n_points, dtype=float)
        cdf[: first_band_bin + 1] = np.linspace(cdf_start, cdf_end - band_mass, first_band_bin + 1)
        cdf[first_band_bin:] = cdf[first_band_bin] + np.arange(band_bins + 1) * (band_mass / band_bins)

    percentiles = {
        f"model-{i}": [[lower_label, low], [50.0, 586.0], [top_label, high]]
        for i, (low, high) in enumerate(zip(declared_bottom, declared_top, strict=True))
    }
    return {
        "type": "discrete",
        "question_id": question_id,
        "title": "How many active US drilling rigs will there be",
        "our_forecast_values": cdf.tolist(),
        "resolution_parsed": 588.0,
        "scaling": {"range_min": 559.5, "range_max": 630.5, "zero_point": None},
        "open_lower_bound": open_lower,
        "open_upper_bound": open_upper,
        "bot_comment_created_at": created_at,
        "per_model_numeric_percentiles": percentiles,
    }


def _high_side(record: dict) -> OuterTailReading:
    [reading] = [r for r in measure_outer_tails(record) if r.side == "high"]
    return reading


def _flat_zone_score(band_bin_multiple: float, *, n_points: int = 72, n_open_bounds: int = 2) -> float:
    """The closed-form Metaculus log score for a bin holding ``multiple`` x the min step.

    ``50 * ln(pmf / baseline)`` with ``pmf = multiple * 0.01 / N`` and
    ``baseline = (1 - 0.05 * open_bounds) / N``. At a multiple near 1.1 that is about -219 on
    ANY grid size, which is what makes a starved band a flat cliff rather than a gradient:
    q45218 measured -219.5 and q44182, the worst record on the board, -219.0.
    """
    n_inbound = n_points - 1
    pmf = band_bin_multiple * 0.01 / n_inbound
    baseline = (1.0 - 0.05 * n_open_bounds) / n_inbound
    return 50.0 * float(np.log(pmf / baseline))


class TestStarvedOuterTail:
    """The open-bound p99 cliff: mass beyond the declared outer anchor starved to the floor.

    Distinct from the shipped ``CDF_MAXSTEP_CLIP`` smear. On an open bound the declared tail
    can be routed out of the displayed range entirely, leaving every in-range bin above the
    declared p99 pinned at the platform's structural minimum step — so every resolution out
    there scores at the same floor (-219.5 on q45218, -219.0 on q44182, the worst record on
    the board), a cliff nobody declared and one no modest widening walks out of, because the
    defect is a step function in the declared p99.
    """

    def test_the_q45218_geometry_fires(self):
        reading = _high_side(_rig_count_record(band_bin_multiple=1.12))
        assert reading.verdict is OuterTailVerdict.STARVED
        assert reading.starved
        assert reading.members_used == 3
        assert reading.members_dropped == 0
        assert reading.declared_percentile == 99.0
        assert reading.declared_value == pytest.approx(603.0)
        assert reading.bound_value == pytest.approx(630.5)
        assert reading.band_bins == 27
        assert reading.tail_mass == pytest.approx(27 * 1.12 * 0.01 / 71)
        assert reading.beyond_bound_mass == pytest.approx(0.01)
        assert reading.floor_multiple == pytest.approx(1.12)
        # The whole point: any resolution in that band earns the platform's floor score, which
        # is what q45218 measured at -219.5 and q44182 at -219.0.
        assert reading.flat_zone_log_score == pytest.approx(_flat_zone_score(1.12), abs=0.5)

    def test_an_honest_tail_does_not_fire(self):
        reading = _high_side(_rig_count_record(band_bin_multiple=6.0))
        assert reading.verdict is OuterTailVerdict.HEALTHY
        assert not reading.starved
        assert reading.floor_multiple == pytest.approx(6.0)
        # Still a bad place to resolve, but ~84 points off the cliff rather than sitting on it.
        assert reading.flat_zone_log_score == pytest.approx(_flat_zone_score(6.0), abs=0.5)
        assert reading.flat_zone_log_score is not None
        assert reading.flat_zone_log_score - _flat_zone_score(1.12) > 80.0

    def test_the_threshold_is_the_named_constant(self):
        just_under = _high_side(_rig_count_record(band_bin_multiple=STARVED_OUTER_TAIL_FLOOR_MULTIPLE * 0.99))
        just_over = _high_side(_rig_count_record(band_bin_multiple=STARVED_OUTER_TAIL_FLOOR_MULTIPLE * 1.01))
        assert just_under.starved
        assert not just_over.starved
        # Calibrated over the archived performance dataset (271 numeric/discrete records, 417
        # measurable open-bound sides; receipts in scratch/next_season_bundle_2026-09/item19/):
        # q45218 reads 1.12 on both sides and q44182 1.46 high / 1.13 low, so both fire, while
        # q44842 is not measurable on either side because its declaration routes the tail past
        # the ceiling. The measured distribution is bimodal — 44 sides at [1.00, 1.25), then
        # ~8 per 0.25 bucket to 3.0, median 10.3 — so the cut is not load-bearing, and 2.0
        # keeps margin over q44182's 1.46.
        assert STARVED_OUTER_TAIL_FLOOR_MULTIPLE == 2.0

    def test_an_explicit_floor_multiple_overrides_the_default(self):
        record = _rig_count_record(band_bin_multiple=3.0)
        assert not _high_side(record).starved
        [strict] = [r for r in measure_outer_tails(record, floor_multiple=4.0) if r.side == "high"]
        assert strict.starved

    def test_the_q44842_shape_is_unmeasurable_rather_than_starved(self):
        # q44842 declared p99 at 20500 against a 14000 ceiling: the declaration itself routes
        # the outer tail past the displayed range, which is the honest open-bound shape (that
        # record won spot peer +24.4). There is no in-range band above the anchor to starve.
        reading = _high_side(_rig_count_record(band_bin_multiple=1.12, declared_top=(640.0, 660.0, 700.0)))
        assert reading.verdict is OuterTailVerdict.DECLARED_BEYOND_BOUND
        assert reading.starved is False
        assert reading.tail_mass is None

    def test_an_anchor_inside_the_last_bin_leaves_no_band_to_measure(self):
        # The declared anchor (median 630.0) sits below the displayed ceiling of 630.5, so this
        # is not the declared-beyond-bound shape — but no bin lies FULLY beyond it. A partially
        # covered bin would inflate the band's mass without inflating its bin count, and the
        # ratio between those two is the whole measurement, so the side reports EMPTY_BAND
        # rather than a floor multiple read off half a bin.
        reading = _high_side(_rig_count_record(band_bin_multiple=1.12, declared_top=(629.9, 630.0, 630.1)))
        assert reading.verdict is OuterTailVerdict.EMPTY_BAND
        assert reading.starved is False
        assert reading.band_bins is None
        assert reading.floor_multiple is None

    def test_the_low_side_mirror_fires_on_a_starved_floor(self):
        # The same geometry read from below: the bins under the declared p1 (568) at 1.2x the
        # min step, with the canonical 1% below the displayed floor.
        record = _rig_count_record(band_bin_multiple=6.0)
        cdf = np.asarray(record["our_forecast_values"])
        grid = np.linspace(559.5, 630.5, 72)
        last_bin = int(np.searchsorted(grid, 568.0, side="right")) - 1
        min_step = 0.01 / 71
        band_mass = last_bin * 1.2 * min_step
        cdf[: last_bin + 1] = 0.01 + np.arange(last_bin + 1) * (band_mass / last_bin)
        record["our_forecast_values"] = cdf.tolist()

        [low] = [r for r in measure_outer_tails(record) if r.side == "low"]
        assert low.verdict is OuterTailVerdict.STARVED
        assert low.declared_percentile == 1.0
        assert low.declared_value == pytest.approx(568.0)
        assert low.bound_value == pytest.approx(559.5)
        assert low.band_bins == last_bin
        assert low.beyond_bound_mass == pytest.approx(0.01)
        assert low.floor_multiple == pytest.approx(1.2)
        assert low.flat_zone_log_score == pytest.approx(_flat_zone_score(1.2), abs=0.5)

    def test_a_closed_bound_side_is_never_scanned(self):
        readings = measure_outer_tails(_rig_count_record(band_bin_multiple=1.12, open_upper=False))
        assert [r.side for r in readings] == ["low"]

    def test_a_record_without_member_curves_is_unmeasurable_not_healthy(self):
        # Stacked-era and trimmed comments lose the per-model percentile lines. The absence of
        # a declaration must not read as "this tail is fine" — that silent pass is exactly
        # what the verdict enum exists to prevent.
        record = _rig_count_record(band_bin_multiple=1.12)
        record["per_model_numeric_percentiles"] = {}
        assert _high_side(record).verdict is OuterTailVerdict.NO_MEMBER_CURVE

    def test_a_curve_whose_shared_anchor_is_not_extreme_is_unmeasurable(self):
        # A trimmed recovery can leave the members sharing only their p50. A band read as
        # "everything above the median" spans about half the grid, so its mean per-bin mass
        # lands tens of times above the platform minimum and the side would read HEALTHY
        # however starved its terminal bins are — the false clean bill of health this
        # requirement exists to refuse.
        record = _rig_count_record(band_bin_multiple=1.12)
        record["per_model_numeric_percentiles"] = {
            "model-0": [[10.0, 575.0], [50.0, 586.0]],
            "model-1": [[50.0, 586.0], [99.0, 603.0]],
        }
        assert _high_side(record).verdict is OuterTailVerdict.ANCHOR_NOT_EXTREME

    def test_members_sharing_no_percentile_label_are_unmeasurable(self):
        # Recovered curves need not agree on their labels: the canonical set changed mid-season
        # (11-point p2.5..p97.5 -> 13-point p1..p99), so an intersection can come back EMPTY.
        # Medianing p99 against p97.5 would average two different quantities, so the side
        # reports NO_SHARED_ANCHOR instead of picking one member's label.
        record = _rig_count_record(band_bin_multiple=1.12)
        record["per_model_numeric_percentiles"] = {
            "model-0": [[1.0, 566.0], [50.0, 586.0], [99.0, 603.0]],
            "model-1": [[2.5, 567.0], [40.0, 584.0], [97.5, 602.0]],
        }
        reading = _high_side(record)
        assert reading.verdict is OuterTailVerdict.NO_SHARED_ANCHOR
        assert reading.declared_percentile is None

    def test_anonymous_forecaster_keys_are_excluded_from_the_anchor(self):
        # On a stacker-fired record the positional `Forecaster N` bucket holds the stacker's
        # AGGREGATE, not a member; pooling it into the median-of-members moves the anchor.
        # Same rule as declared_percentile_pit and max_step_clamp_screen next door.
        record = _rig_count_record(band_bin_multiple=1.12)
        record["per_model_numeric_percentiles"] = {
            "Forecaster 1": [[1.0, 560.0], [50.0, 586.0], [99.0, 628.0]],
        }
        assert _high_side(record).verdict is OuterTailVerdict.NO_MEMBER_CURVE

    def test_a_record_without_a_usable_cdf_is_unmeasurable(self):
        record = _rig_count_record(band_bin_multiple=1.12)
        record["scaling"] = {}
        assert {r.verdict for r in measure_outer_tails(record)} == {OuterTailVerdict.NO_USABLE_CDF}

    def test_non_numeric_records_are_skipped_by_the_scan(self):
        scan = scan_outer_tails([{"type": "binary", "our_prob_yes": 0.5}])
        assert scan.readings == []
        assert scan.starved == []

    def test_the_scan_counts_verdicts_and_ranks_the_flagged_rows(self):
        starved = _rig_count_record(band_bin_multiple=1.12, question_id=45218)
        healthy = _rig_count_record(band_bin_multiple=6.0, question_id=44182)
        blind = _rig_count_record(band_bin_multiple=1.12, question_id=44553)
        blind["per_model_numeric_percentiles"] = {}
        scan = scan_outer_tails([starved, healthy, blind])

        assert scan.n_scanned == 6  # three records x two open sides
        assert [r.question_id for r in scan.starved] == [45218]
        assert scan.verdict_counts[OuterTailVerdict.STARVED] == 1
        assert scan.verdict_counts[OuterTailVerdict.NO_MEMBER_CURVE] == 2

    def test_excluded_qids_leave_the_scan(self):
        scan = scan_outer_tails(
            [_rig_count_record(band_bin_multiple=1.12, question_id=43913)],
            exclude_qids=KNOWN_BUG_QIDS,
        )
        assert scan.readings == []
        assert scan.n_excluded == 1

    def test_the_rendered_section_states_every_reported_field(self):
        scan = scan_outer_tails([_rig_count_record(band_bin_multiple=1.12)])
        md = render_starved_outer_tails(scan)
        assert "Starved outer tails" in md
        assert "45218" in md
        assert "603" in md  # the declared p99
        assert "630.5" in md  # the displayed ceiling
        assert "-219" in md  # the flat-zone log score
        assert "| 27 |" in md  # band bins
        assert "0.0043" in md  # tail mass, 27 bins x 1.12 x the min step

    def test_a_scan_with_nothing_starved_says_so(self):
        md = render_starved_outer_tails(scan_outer_tails([_rig_count_record(band_bin_multiple=6.0)]))
        assert "Starved outer tails" in md
        assert "none" in md.lower()

    def test_the_unmeasurable_tally_is_disclosed(self):
        blind = _rig_count_record(band_bin_multiple=1.12)
        blind["per_model_numeric_percentiles"] = {}
        md = render_starved_outer_tails(scan_outer_tails([blind]))
        assert "no_member_curve: 2" in md

    def test_the_cli_renders_the_section(self, tmp_path, capsys):
        path = tmp_path / "data.json"
        path.write_text(json.dumps([_rig_count_record(band_bin_multiple=1.12)]))
        main(["--cached", str(path)])
        out = capsys.readouterr().out
        assert "Numeric width / calibration monitor" in out
        assert "Starved outer tails" in out

    def test_the_cli_can_write_the_scan_as_json(self, tmp_path):
        data = tmp_path / "data.json"
        data.write_text(json.dumps([_rig_count_record(band_bin_multiple=1.12)]))
        out_json = tmp_path / "starved.json"
        main(["--cached", str(data), "--output-starved-json", str(out_json)])
        payload = json.loads(out_json.read_text())
        [high] = [row for row in payload["readings"] if row["side"] == "high"]
        assert high["verdict"] == "starved"
        assert high["declared_value"] == pytest.approx(603.0)
        assert high["flat_zone_log_score"] < -215.0
        assert payload["verdict_counts"]["starved"] == 1


class TestAnchorMemberCensus:
    """Who set the anchor, disclosed.

    The band the verdict is measured against starts at the members' shared outer anchor, and
    that anchor is a median over the members whose curve survived parsing. A partial loss
    therefore MOVES the boundary, so every reading states how many members it used and how many
    it dropped rather than reporting a boundary of unstated provenance.
    """

    def test_an_unparseable_member_is_counted_and_the_anchor_moves(self):
        # Archived per-model pairs come out of comment prose, so a garbled percentile label is a
        # real shape: `float("p99")` raises and the whole curve leaves the median. The two
        # survivors declared p99 at 603 and 606, so the anchor is their 604.5 — not the 603 all
        # three would have produced.
        record = _rig_count_record(band_bin_multiple=1.12)
        record["per_model_numeric_percentiles"] = {
            "model-0": [["p99", 602.0], [50.0, 586.0], [1.0, 566.0]],
            "model-1": [[1.0, 568.0], [50.0, 586.0], [99.0, 603.0]],
            "model-2": [[1.0, 570.0], [50.0, 586.0], [99.0, 606.0]],
        }
        reading = _high_side(record)
        assert reading.verdict is OuterTailVerdict.STARVED
        assert reading.members_used == 2
        assert reading.members_dropped == 1
        assert reading.declared_percentile == 99.0
        assert reading.declared_value == pytest.approx(604.5)

    def test_a_truncated_pair_is_counted_the_same_way(self):
        # The other malformed shape: a percentile line cut before its value arrives as a
        # one-element pair, which indexes out of range.
        record = _rig_count_record(band_bin_multiple=1.12)
        record["per_model_numeric_percentiles"] = {
            "model-0": [[1.0, 566.0], [50.0, 586.0], [99.0]],
            "model-1": [[1.0, 568.0], [50.0, 586.0], [99.0, 603.0]],
            "model-2": [[1.0, 570.0], [50.0, 586.0], [99.0, 606.0]],
        }
        reading = _high_side(record)
        assert reading.members_used == 2
        assert reading.members_dropped == 1
        assert reading.declared_value == pytest.approx(604.5)

    def test_a_curve_with_one_label_cannot_set_an_anchor_and_is_counted(self):
        record = _rig_count_record(band_bin_multiple=1.12)
        record["per_model_numeric_percentiles"] = {
            "model-0": [[99.0, 602.0]],
            "model-1": [[1.0, 568.0], [50.0, 586.0], [99.0, 603.0]],
            "model-2": [[1.0, 570.0], [50.0, 586.0], [99.0, 606.0]],
        }
        reading = _high_side(record)
        assert reading.members_used == 2
        assert reading.members_dropped == 1
        assert reading.declared_value == pytest.approx(604.5)

    def test_an_anonymous_bucket_beside_named_members_is_counted_as_dropped(self):
        # The only cause that actually fires on the archived cohort: 12 members over 12 records,
        # 6 of them partial losses. The exclusion is deliberate — a positional `Forecaster N`
        # bucket can hold the stacker's aggregate — but it moves the anchor exactly as a parse
        # failure does, so it is disclosed the same way.
        record = _rig_count_record(band_bin_multiple=1.12)
        record["per_model_numeric_percentiles"] = {
            "Forecaster 1": [[1.0, 560.0], [50.0, 586.0], [99.0, 628.0]],
            "model-1": [[1.0, 568.0], [50.0, 586.0], [99.0, 603.0]],
            "model-2": [[1.0, 570.0], [50.0, 586.0], [99.0, 606.0]],
        }
        reading = _high_side(record)
        assert reading.members_used == 2
        assert reading.members_dropped == 1
        # 628 would have dragged the median to 606; the anchor is the two named members' 604.5.
        assert reading.declared_value == pytest.approx(604.5)

    def test_an_intact_set_reports_no_drops(self):
        for reading in measure_outer_tails(_rig_count_record(band_bin_multiple=1.12)):
            assert reading.members_used == 3
            assert reading.members_dropped == 0

    def test_the_census_totals_the_records_members_on_every_verdict(self):
        # used + dropped is the record's member count, which makes the pair checkable rather
        # than indicative — including on the unmeasurable verdicts, where the members were
        # inspected and simply could not produce a shared extreme anchor.
        record = _rig_count_record(band_bin_multiple=1.12)
        record["per_model_numeric_percentiles"] = {
            "model-0": [[10.0, 575.0], [50.0, 586.0]],
            "model-1": [[50.0, 586.0], [99.0, 603.0]],
            "Forecaster 3": [[1.0, 560.0], [99.0, 628.0]],
        }
        reading = _high_side(record)
        assert reading.verdict is OuterTailVerdict.ANCHOR_NOT_EXTREME
        assert reading.members_used == 2
        assert reading.members_dropped == 1

    def test_a_side_that_never_inspected_members_reports_neither_count(self):
        # NO_USABLE_CDF precedes the member read, so an empty census would claim a measurement
        # that was never taken. None says "not inspected"; 0 would say "inspected, found none".
        record = _rig_count_record(band_bin_multiple=1.12)
        record["scaling"] = {}
        for reading in measure_outer_tails(record):
            assert reading.verdict is OuterTailVerdict.NO_USABLE_CDF
            assert reading.members_used is None
            assert reading.members_dropped is None

    def test_a_record_with_no_members_at_all_reports_an_empty_census(self):
        record = _rig_count_record(band_bin_multiple=1.12)
        record["per_model_numeric_percentiles"] = {}
        reading = _high_side(record)
        assert reading.verdict is OuterTailVerdict.NO_MEMBER_CURVE
        assert reading.members_used == 0
        assert reading.members_dropped == 0

    def test_the_flagged_row_states_the_census(self):
        record = _rig_count_record(band_bin_multiple=1.12)
        record["per_model_numeric_percentiles"] = {
            "Forecaster 1": [[1.0, 560.0], [50.0, 586.0], [99.0, 628.0]],
            "model-1": [[1.0, 568.0], [50.0, 586.0], [99.0, 603.0]],
            "model-2": [[1.0, 570.0], [50.0, 586.0], [99.0, 606.0]],
        }
        md = render_starved_outer_tails(scan_outer_tails([record]))
        assert "| members (used/dropped) |" in md
        assert "| 2/1 |" in md
        # Per SIDE, and said so: the one dropped member is counted on both of the record's open
        # bounds, which a bare "2 members dropped" would read as two distinct members.
        assert "Dropped an anchor member on 2 of 2 scanned side(s) (2 member-drop(s) in total" in md

    def test_the_section_says_so_when_no_member_was_dropped(self):
        md = render_starved_outer_tails(scan_outer_tails([_rig_count_record(band_bin_multiple=1.12)]))
        assert "Dropped an anchor member on no side." in md
        assert "| 3/0 |" in md

    def test_the_json_dump_carries_the_census_on_unflagged_sides_too(self, tmp_path):
        healthy = _rig_count_record(band_bin_multiple=6.0, question_id=44182)
        healthy["per_model_numeric_percentiles"] = {
            "Forecaster 1": [[1.0, 560.0], [50.0, 586.0], [99.0, 628.0]],
            "model-1": [[1.0, 568.0], [50.0, 586.0], [99.0, 603.0]],
            "model-2": [[1.0, 570.0], [50.0, 586.0], [99.0, 606.0]],
        }
        data = tmp_path / "data.json"
        data.write_text(json.dumps([healthy]))
        out_json = tmp_path / "starved.json"
        main(["--cached", str(data), "--output-starved-json", str(out_json)])
        payload = json.loads(out_json.read_text())
        [high] = [row for row in payload["readings"] if row["side"] == "high"]
        assert high["verdict"] == "healthy"
        assert high["members_used"] == 2
        assert high["members_dropped"] == 1
